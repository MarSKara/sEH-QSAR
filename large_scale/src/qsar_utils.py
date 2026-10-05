"""
qsar_utils.py
=============
Shared helper functions used by the three pipeline scripts.

Two families of helpers live here:

1. Molecule-level helpers (standardization, fingerprints, descriptors,
   scaffolds) -- used everywhere, at any scale.

2. Big-data helpers -- streaming file readers, a Parquet writer that never
   holds a full dataset in memory, and memory-bounded online aggregators
   (histogram / reservoir sample / top-K tracker). These exist so the
   pipeline can screen libraries of hundreds of millions of compounds
   (e.g. ZINC) without ever loading the whole thing into RAM or writing a
   feature matrix that wouldn't fit on disk. See the README section
   "Screening very large libraries" for the reasoning.
"""

from __future__ import annotations

import random
from heapq import heappush, heapreplace
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, SaltRemover, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

# RDKit prints its own parse/valence warnings straight to the terminal by
# default, independent of the try/except handling in standardize_mol()
# below -- at ZINC scale that can look like a wall of errors even when the
# pipeline is already catching every one of them correctly (see
# failed_standardization.parquet/.xlsx for the actual, tracked failures).
# This silences that console noise; it has no effect on which molecules
# succeed or fail.
RDLogger.DisableLog("rdApp.*")

# --------------------------------------------------------------------
# Column-name auto-detection (used by all three scripts)
# --------------------------------------------------------------------

ID_COL_CANDIDATES = ["id", "ID", "Id", "compound_id", "molecule_id", "zinc_id", "drugId", "drugbank_id"]
SMILES_COL_CANDIDATES = ["smiles", "SMILES", "Smiles", "canonical_smiles"]
NAME_COL_CANDIDATES = ["name", "Name", "compound_name", "molecule_name", "drugName"]


def find_column(columns, candidates: list[str]):
    """Return the first candidate name that exists in `columns`, else None."""
    for c in candidates:
        if c in columns:
            return c
    return None


# --------------------------------------------------------------------
# Molecule standardization
# --------------------------------------------------------------------

_remover = SaltRemover.SaltRemover()
_uncharger = rdMolStandardize.Uncharger()
_normalizer = rdMolStandardize.Normalizer()


def standardize_mol(smiles: str):
    """
    Strip salts, neutralize charges, normalize, and sanitize a molecule.

    Returns
    -------
    (mol, canonical_smiles, error_reason)
    On any failure this returns (None, None, reason_string).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, "invalid_smiles"

        mol = _remover.StripMol(mol, dontRemoveEverything=True)
        if mol is None or mol.GetNumAtoms() == 0:
            return None, None, "empty_after_salt_removal"

        mol = _uncharger.uncharge(mol)
        if mol is None:
            return None, None, "failed_uncharge"

        mol = _normalizer.normalize(mol)
        if mol is None:
            return None, None, "failed_normalize"

        try:
            Chem.SanitizeMol(mol)
        except Exception as e:  # noqa: BLE001
            return None, None, f"sanitize_error: {e}"

        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return mol, canon, None
    except Exception as e:  # noqa: BLE001
        return None, None, f"unexpected_error: {e}"


# --------------------------------------------------------------------
# Fingerprints / descriptors
# --------------------------------------------------------------------

def get_morgan_generator(radius: int = 2, fp_size: int = 1024, chirality: bool = True):
    """Return an RDKit Morgan fingerprint generator with the given settings."""
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=fp_size, includeChirality=chirality
    )


def smiles_to_fp(smiles: str, generator):
    """Convert a SMILES string to an RDKit fingerprint object, or None on failure."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return generator.GetFingerprint(mol)
    except Exception:  # noqa: BLE001
        return None


def compute_descriptors(mol) -> list[float]:
    """All RDKit descriptors (Descriptors._descList), with NaN/inf replaced by 0."""
    return [
        np.nan_to_num(f(mol), nan=0.0, posinf=0.0, neginf=0.0)
        for _, f in Descriptors._descList
    ]


DESCRIPTOR_NAMES = [name for name, _ in Descriptors._descList]


def max_tanimoto_to_reference(fp, reference_fps: list) -> float:
    """Max Tanimoto similarity of `fp` against a list of reference fingerprints."""
    if fp is None or not reference_fps:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
    return float(max(sims))


def applicability_domain_summary(n_total: int, n_inside: int, threshold: float) -> dict:
    """
    Summarize how much of a screening set falls inside/outside the
    applicability domain (AD), plus a plain-language warning when a large
    fraction of the library is chemically far from the training data.

    Works from running counts (n_total, n_inside) rather than a full array,
    so it's safe to use even when the library is too large to hold in memory.
    """
    n_outside = n_total - n_inside
    frac_outside = (n_outside / n_total) if n_total else 0.0

    warning = None
    if frac_outside >= 0.5:
        warning = (
            f"Over half of this library ({frac_outside * 100:.0f}%) is outside the "
            f"applicability domain (max Tanimoto similarity to training < {threshold}). "
            "This model was not trained on chemistry like most of this library -- "
            "treat its predictions on this set with substantial caution."
        )
    elif frac_outside >= 0.25:
        warning = (
            f"A notable share of this library ({frac_outside * 100:.0f}%) is outside the "
            f"applicability domain (max Tanimoto similarity to training < {threshold}). "
            "Predictions on those compounds are extrapolations -- treat them as lower-confidence."
        )

    return {
        "n_total": n_total,
        "n_inside": n_inside,
        "n_outside": n_outside,
        "frac_outside": frac_outside,
        "warning": warning,
    }


# --------------------------------------------------------------------
# Scaffolds
# --------------------------------------------------------------------

def get_murcko_scaffold(smiles: str) -> str:
    """
    Canonical Murcko scaffold SMILES for a molecule.
    Returns 'no_scaffold' for acyclic molecules and 'invalid' for bad SMILES.
    """
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return "invalid"
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return "no_scaffold"
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:  # noqa: BLE001
        return "no_scaffold"


# --------------------------------------------------------------------
# Labeling helpers
# --------------------------------------------------------------------

def confidence_category(conf: float) -> str:
    if conf >= 0.8:
        return "High"
    elif conf >= 0.5:
        return "Medium"
    return "Low"


def activity_category(pic50: float) -> str:
    if pic50 >= 8.0:
        return "Highly active (pIC50>=8)"
    elif pic50 >= 7.0:
        return "Active (7<=pIC50<8)"
    elif pic50 >= 6.0:
        return "Moderately active (6<=pIC50<7)"
    return "Weakly active (pIC50<6)"


# ======================================================================
# BIG-DATA HELPERS -- streaming I/O + memory-bounded aggregation
# ======================================================================

def iter_input_files(path):
    """Yield individual file paths given either a single file or a directory.
    Directories are searched RECURSIVELY (subfolders included), which is
    exactly ZINC's layout: a top-level download folder containing many
    subfolders, each containing many tranche files. Point --input at the
    top-level folder and every matching file underneath is picked up --
    no need to flatten or pre-process anything yourself."""
    path = Path(path)
    if path.is_file():
        yield path
        return
    if path.is_dir():
        exts = (
            "*.csv", "*.tsv", "*.txt", "*.smi", "*.parquet", "*.xlsx",
            "*.csv.gz", "*.tsv.gz", "*.txt.gz", "*.smi.gz",
        )
        seen = set()
        for ext in exts:
            for f in sorted(path.rglob(ext)):
                if f not in seen:
                    seen.add(f)
                    yield f
        return
    raise FileNotFoundError(f"Input path not found: {path}")


def _normalize_columns(df: pd.DataFrame, id_col, smiles_col, name_col, file_path) -> pd.DataFrame:
    resolved_smiles = smiles_col or find_column(df.columns, SMILES_COL_CANDIDATES)
    resolved_id = id_col or find_column(df.columns, ID_COL_CANDIDATES)
    resolved_name = name_col or find_column(df.columns, NAME_COL_CANDIDATES)

    if resolved_smiles is None or resolved_id is None:
        raise ValueError(
            f"Could not auto-detect id/smiles columns in {file_path} "
            f"(columns found: {list(df.columns)}). Pass --id-col / --smiles-col explicitly."
        )

    return pd.DataFrame({
        "name": df[resolved_name] if resolved_name else df[resolved_id].astype(str),
        "id": df[resolved_id].astype(str),
        "smiles": df[resolved_smiles],
    })


def _sniff_header_and_sep(file_path: Path):
    """
    Peek at a delimited text file's first line to figure out (a) the
    delimiter and (b) whether there's a header row at all. Handles
    .gz-compressed files transparently. This is what lets ZINC's tab-
    separated "smiles<TAB>zinc_id" files work whether or not they ship
    with a header line, with no manual pre-processing.
    """
    import gzip

    opener = gzip.open if file_path.name.lower().endswith(".gz") else open
    with opener(file_path, "rt", errors="replace") as fh:
        first_line = fh.readline()

    if "\t" in first_line:
        sep = "\t"
    elif "," in first_line:
        sep = ","
    else:
        sep = r"\s+"

    if sep == r"\s+":
        tokens = first_line.split()
    else:
        tokens = first_line.split(sep)
    tokens = [t.strip().strip('"') for t in tokens]

    known_names = {c.lower() for c in (ID_COL_CANDIDATES + SMILES_COL_CANDIDATES + NAME_COL_CANDIDATES)}
    has_header = any(t.lower() in known_names for t in tokens)

    return sep, has_header


def iter_library_chunks(input_path, chunksize: int = 100_000, id_col=None, smiles_col=None, name_col=None):
    """
    Stream (name, id, smiles) rows from a file or a directory of files (see
    `iter_input_files` -- directories are searched recursively), in chunks
    of `chunksize` rows, regardless of how large the total library is.
    Never loads more than one chunk into memory at a time, except for
    .xlsx (pandas can only read that whole -- fine for smaller libraries,
    but use .csv/.smi/.parquet for anything ZINC-sized).

    Supported formats: .csv, .tsv, .smi, .txt, and gzip-compressed versions
    of any of those (.csv.gz etc.), plus .parquet and .xlsx/.xls.

    Delimiter and header presence are auto-detected per file: a plain
    ZINC tranche file ("smiles<TAB>zinc_id", no header) and a header-ed
    CSV both just work, with column names auto-detected the same way as
    everywhere else in this project (or set explicitly with --id-col /
    --smiles-col if your columns don't match common naming).
    """
    for file_path in iter_input_files(input_path):
        name = file_path.name.lower()
        try:
            if name.endswith((".xlsx", ".xls")):
                df_full = pd.read_excel(file_path)
                for start in range(0, len(df_full), chunksize):
                    yield _normalize_columns(df_full.iloc[start:start + chunksize], id_col, smiles_col, name_col, file_path)

            elif name.endswith(".parquet"):
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(file_path)
                for batch in pf.iter_batches(batch_size=chunksize):
                    yield _normalize_columns(batch.to_pandas(), id_col, smiles_col, name_col, file_path)

            else:  # .csv / .tsv / .smi / .txt, optionally .gz-compressed
                sep, has_header = _sniff_header_and_sep(file_path)
                engine = "python" if sep == r"\s+" else "c"  # regex separators need the python engine; tab/comma use the faster C engine
                if has_header:
                    reader = pd.read_csv(file_path, sep=sep, header=0, chunksize=chunksize, engine=engine, comment="#")
                    for chunk_df in reader:
                        yield _normalize_columns(chunk_df, id_col, smiles_col, name_col, file_path)
                else:
                    # No header -- assume the standard "SMILES  ID" column order used by ZINC .smi/.txt exports.
                    reader = pd.read_csv(file_path, sep=sep, header=None, names=["smiles", "id"],
                                          chunksize=chunksize, engine=engine, comment="#")
                    for chunk_df in reader:
                        yield _normalize_columns(chunk_df, id_col or "id", smiles_col or "smiles", name_col, file_path)
        except Exception as e:  # noqa: BLE001
            # One corrupted/truncated/unreadable file (a bad download, a stray non-data file that
            # happens to share a recognized extension, etc.) should never take down a run that's
            # already made real progress on everything else. Any chunks already yielded from this
            # file before the error stay valid; we just skip whatever's left of THIS file and move on.
            print(f"  WARNING: skipping unreadable file '{file_path}' ({type(e).__name__}: {e})")
            continue


def iter_parquet_chunks(path, chunksize: int = 100_000):
    """
    Stream a Parquet file written by this pipeline (stage 1 or stage 2
    output) in chunks, preserving all columns exactly as written -- unlike
    `iter_library_chunks`, this does NOT re-run id/smiles column detection,
    since these files already have the pipeline's own fixed schema.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=chunksize):
        yield batch.to_pandas()


class ParquetChunkWriter:
    """
    Append pandas DataFrame chunks to a single Parquet file, one row-group at
    a time, without ever holding the whole dataset in memory. Schema is
    fixed from the first chunk written. Use as a context manager.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = None
        self.n_rows_written = 0

    def write(self, df: pd.DataFrame) -> None:
        if df is None or len(df) == 0:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(str(self.path), table.schema)
        self._writer.write_table(table)
        self.n_rows_written += len(df)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class OnlineHistogram:
    """Accumulate a histogram over a streamed sequence of values without
    ever storing every value -- memory is fixed at `bins` regardless of how
    many values are seen."""

    def __init__(self, bins: int = 30, value_range: tuple = (0.0, 1.0)):
        self.edges = np.linspace(value_range[0], value_range[1], bins + 1)
        self.counts = np.zeros(bins, dtype=np.int64)

    def update(self, values) -> None:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return
        idx = np.clip(np.digitize(values, self.edges) - 1, 0, len(self.counts) - 1)
        self.counts += np.bincount(idx, minlength=len(self.counts))

    def bin_centers(self) -> np.ndarray:
        return (self.edges[:-1] + self.edges[1:]) / 2


class ReservoirSample:
    """
    A uniform random sample of a stream of unknown (possibly huge) length,
    held in bounded memory (`size` items). Used purely for distribution
    plots -- exact aggregate counts (totals, AD fractions, etc.) are tracked
    separately and are exact, not sampled.
    """

    def __init__(self, size: int, seed: int = 42):
        self.size = size
        self.reservoir: list = []
        self.n_seen = 0
        self._rng = random.Random(seed)

    def update(self, values) -> None:
        for v in values:
            self.n_seen += 1
            if len(self.reservoir) < self.size:
                self.reservoir.append(v)
            else:
                j = self._rng.randint(0, self.n_seen - 1)
                if j < self.size:
                    self.reservoir[j] = v

    def values(self) -> np.ndarray:
        return np.asarray(self.reservoir)


class TopKTracker:
    """
    Keep the top-K rows (dicts) by a numeric score while streaming through
    an arbitrarily large dataset, using a bounded min-heap (memory is fixed
    at K rows). Used to build "top predicted actives" reports/plots and a
    candidate pool for scaffold-diverse selection without ever sorting the
    full library.
    """

    def __init__(self, k: int):
        self.k = k
        self._heap: list = []  # (score, tie_breaker, row)
        self._counter = 0

    def offer(self, score: float, row: dict) -> None:
        self._counter += 1
        entry = (score, self._counter, row)
        if len(self._heap) < self.k:
            heappush(self._heap, entry)
        elif score > self._heap[0][0]:
            heapreplace(self._heap, entry)

    def get_sorted(self, descending: bool = True) -> list:
        items = sorted(self._heap, key=lambda e: e[0], reverse=descending)
        return [row for _, _, row in items]

    def __len__(self):
        return len(self._heap)


# --------------------------------------------------------------------
# Multiprocessing workers
#
# These run in separate worker processes (via ProcessPoolExecutor), which
# is what makes screening hundreds of millions of compounds feasible in
# practice -- RDKit standardization/featurization is CPU-bound, and running
# many molecules across many cores at once turns a multi-week job on one
# core into a multi-day job on a typical multi-core machine.
#
# Each worker process builds its own RDKit objects once (via the
# initializer) rather than recreating them per molecule.
# --------------------------------------------------------------------

def map_chunksize(n_items: int, n_workers) -> int:
    """
    Heuristic batch size for ProcessPoolExecutor.map's internal `chunksize`
    argument (how many items get pickled and sent to a worker per task).
    Too small and you pay serialization/IPC overhead per item; too large
    and load balancing across workers gets lumpy. This aims for roughly
    4 tasks per worker per chunk, bounded to a sane range.
    """
    workers = max(int(n_workers or 1), 1)
    return max(50, min(2000, n_items // (workers * 4) or 50))


_worker_generator = None
_worker_train_fps = None


def standardize_one(smiles: str):
    """Worker function: standardize a single SMILES. Returns (canonical_smiles, error)."""
    _, canon, error = standardize_mol(str(smiles))
    return canon, error


def init_featurize_worker(train_canonical_smiles: list, fp_radius: int, fp_size: int, fp_chirality: bool) -> None:
    """
    Initializer for stage-2 workers: builds this process's Morgan generator
    once, and precomputes training-set fingerprints once (for the
    applicability-domain check) so every molecule doesn't redo that work.
    """
    global _worker_generator, _worker_train_fps
    _worker_generator = get_morgan_generator(fp_radius, fp_size, fp_chirality)
    _worker_train_fps = [
        fp for fp in (smiles_to_fp(s, _worker_generator) for s in train_canonical_smiles) if fp is not None
    ]


def init_featurize_only_worker(fp_radius: int, fp_size: int, fp_chirality: bool) -> None:
    """Initializer for workers that only need descriptors + fingerprint (no
    applicability-domain comparison) -- used by stage 3, where the AD flag
    was already computed in stage 2 and just needs to be carried through."""
    global _worker_generator
    _worker_generator = get_morgan_generator(fp_radius, fp_size, fp_chirality)


def featurize_only_one(canonical_smiles: str):
    """Worker function: descriptors + Morgan fingerprint bits for an
    already-standardized canonical SMILES. Returns None if unparseable."""
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return None
    fp_bits = list(_worker_generator.GetFingerprint(mol))
    descriptors = compute_descriptors(mol)
    return descriptors, fp_bits


def featurize_for_regression_one(canonical_smiles: str):
    """Worker function for stage 3: descriptors + Morgan fingerprint bits +
    Murcko scaffold for an already-standardized canonical SMILES, computed
    from a single mol parse. Returns None if unparseable."""
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return None
    fp_bits = list(_worker_generator.GetFingerprint(mol))
    descriptors = compute_descriptors(mol)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold_mol is None or scaffold_mol.GetNumAtoms() == 0:
        scaffold = "no_scaffold"
    else:
        scaffold = Chem.MolToSmiles(scaffold_mol, canonical=True)
    return descriptors, fp_bits, scaffold


def featurize_and_score_one(canonical_smiles: str):
    """
    Worker function: given an already-standardized canonical SMILES, compute
    its RDKit descriptors, Morgan fingerprint bits, and max Tanimoto
    similarity to the training set -- all in one pass, in one worker
    process. Returns None if the SMILES can't be parsed (shouldn't normally
    happen post-standardization, but stays defensive).
    """
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return None
    fp = _worker_generator.GetFingerprint(mol)
    descriptors = compute_descriptors(mol)
    fp_bits = list(fp)
    max_tan = max_tanimoto_to_reference(fp, _worker_train_fps)
    return descriptors, fp_bits, max_tan
