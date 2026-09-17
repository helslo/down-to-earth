"""
Loader for an OSSL-format SQLite `.db` file (Open Soil Spectroscopy
Library, as exported by the `soilspecdata` package). See the `_readme`
table inside the db itself for provenance notes.

Schema this expects (confirmed against your file's inspect_db.py output):
  - mir_spectra(sample_id TEXT, absorbance BLOB)
        float32 arrays, 1701 bands, 600-4000 cm-1 (matches the paper's
        Vertex 70/HTS-XT spectral range).
  - mir_wavenumbers(idx INTEGER, wavenumber REAL)
        the 1701 wavenumber values, in band order.
  - properties(sample_id TEXT, <~100 property columns>...)
        the full USDA/ISO lab-measured property table (Ca, S, etc. live
        here under names like ca_ext_usda_a722_cmolc_kg,
        s_tot_usda_a624_w_pct -- not present in guo_subset below).
  - guo_subset(sample_id TEXT, C, CEC, Clay, K, N, OC, P, pH, Sand, split)
        a ready-made benchmark subset (n=19,110) with a 'train'/'test'
        split column, reproducing Guo et al. 2025 (Comput. Electron.
        Agric. 237:110507) exactly. Covers 5 of the paper's 7 target
        properties (OC, N, pH, Sand, K) plus CEC, Clay, P instead of the
        paper's Ca and S.

This module also supports CSV-style feature tables that follow the same
structure as data/features_samples.csv, where the spectra live in z000,
z001, ... columns and the targets are stored in columns like oc_pct,
ph_h2o, sand_pct, etc.
"""
import csv
import sqlite3
import numpy as np

GUO_SUBSET_COLUMNS = ["C", "CEC", "Clay", "K", "N", "OC", "P", "pH", "Sand"]
CSV_TARGET_ALIASES = {
    "OC": "oc_pct",
    "N": "n_pct",
    "pH": "ph_h2o",
    "Sand": "sand_pct",
    "K": "k_cmolc_kg",
}


def _decode_spectrum(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _to_float_array(values) -> np.ndarray:
    # SQLite NULLs come back as Python None; convert those to NaN so
    # downstream code can filter with np.isnan rather than special-casing.
    return np.array([np.nan if v is None else v for v in values], dtype=np.float32)


def load_mir_wavenumbers(db_path: str) -> np.ndarray:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT wavenumber FROM mir_wavenumbers ORDER BY idx;")
    wn = np.array([row[0] for row in cur.fetchall()], dtype=np.float32)
    conn.close()
    return wn


def load_mir_spectra(db_path: str, sample_ids=None) -> dict:
    """Returns dict sample_id -> np.ndarray of shape (1701,).

    If sample_ids is given, only fetches those (much faster than pulling
    all 85k spectra when you only need the ~19k in guo_subset).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    if sample_ids is None:
        cur.execute("SELECT sample_id, absorbance FROM mir_spectra;")
        rows = cur.fetchall()
    else:
        rows = []
        sample_ids = list(sample_ids)
        chunk = 500  # stay well under SQLite's default 999-parameter limit
        for i in range(0, len(sample_ids), chunk):
            batch = sample_ids[i : i + chunk]
            placeholders = ",".join("?" for _ in batch)
            cur.execute(
                f"SELECT sample_id, absorbance FROM mir_spectra "
                f"WHERE sample_id IN ({placeholders});",
                batch,
            )
            rows.extend(cur.fetchall())
    conn.close()
    return {sample_id: _decode_spectrum(blob) for sample_id, blob in rows}


def load_guo_subset(db_path: str, target_cols=None) -> dict:
    """Loads the ready-made Guo et al. 2025 benchmark subset.

    Returns dict with keys 'sample_id' (list[str]), 'split'
    (list[str], 'train'/'test'), and one (n,) float32 array per
    requested target column (default: all 9 in GUO_SUBSET_COLUMNS).
    """
    target_cols = target_cols or GUO_SUBSET_COLUMNS
    for c in target_cols:
        assert c in GUO_SUBSET_COLUMNS, (
            f"'{c}' is not one of guo_subset's columns: {GUO_SUBSET_COLUMNS}"
        )

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cols_sql = ", ".join(["sample_id", "split"] + target_cols)
    cur.execute(f"SELECT {cols_sql} FROM guo_subset;")
    rows = cur.fetchall()
    conn.close()

    out = {
        "sample_id": [r[0] for r in rows],
        "split": [r[1] for r in rows],
    }
    for i, c in enumerate(target_cols):
        out[c] = _to_float_array([r[2 + i] for r in rows])
    return out


def _normalize_target_cols(target_cols):
    if target_cols is None:
        return list(GUO_SUBSET_COLUMNS)
    normalized = []
    for col in target_cols:
        if col in CSV_TARGET_ALIASES:
            normalized.append(col)
        elif col in CSV_TARGET_ALIASES.values():
            normalized.append(next(k for k, v in CSV_TARGET_ALIASES.items() if v == col))
        else:
            normalized.append(col)
    return normalized


def load_csv_dataset(csv_path: str, target_cols=None, downsample_to: int = None):
    """Loads a CSV table shaped like data/features_samples.csv.

    The file contains one row per sample with feature columns named z000,
    z001, ... and target columns like oc_pct, n_pct, ph_h2o, sand_pct,
    and k_cmolc_kg. This routine converts them to the same internal
    X/Y/split structure expected by the DB-based training path.
    """
    target_cols = _normalize_target_cols(target_cols or GUO_SUBSET_COLUMNS)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_path} is empty or missing a header row.")

        fieldnames = reader.fieldnames
        feature_cols = [name for name in fieldnames if name.startswith("z")]
        missing_targets = [
            col for col in target_cols if CSV_TARGET_ALIASES.get(col, col) not in fieldnames
        ]
        if missing_targets:
            available = [c for c in fieldnames if c in CSV_TARGET_ALIASES.values() or c.startswith("z")]
            raise ValueError(
                f"Target columns missing from CSV: {missing_targets}. "
                f"Available columns include: {available[:10]}"
            )

        xs, ys, splits, ids = [], {c: [] for c in target_cols}, [], []
        for row in reader:
            sample_id = row.get("labsampnum") or row.get("sample_id")
            split = (row.get("split") or "").strip()
            if sample_id is None or not sample_id:
                continue
            feature_values = []
            for col in feature_cols:
                value = row.get(col, "")
                if value in ("", "NA", "N/A", "nan", "NaN"):
                    value = np.nan
                else:
                    value = float(value)
                feature_values.append(value)
            if any(np.isnan(v) for v in feature_values):
                continue

            row_targets = []
            for col in target_cols:
                csv_name = CSV_TARGET_ALIASES.get(col, col)
                value = row.get(csv_name, "")
                if value in ("", "NA", "N/A", "nan", "NaN"):
                    value = np.nan
                else:
                    value = float(value)
                row_targets.append(value)
            if any(np.isnan(v) for v in row_targets):
                continue

            xs.append(np.asarray(feature_values, dtype=np.float32))
            for c, v in zip(target_cols, row_targets):
                ys[c].append(float(v))
            splits.append(split)
            ids.append(sample_id)

    if len(xs) == 0:
        raise ValueError(
            "No valid CSV rows had both a feature vector and complete target "
            "values -- nothing to train on."
        )

    X = np.stack(xs).astype(np.float32)

    if downsample_to is not None and downsample_to < X.shape[1]:
        bin_size = X.shape[1] // downsample_to
        usable = bin_size * downsample_to
        X = X[:, :usable].reshape(X.shape[0], downsample_to, bin_size).mean(axis=2)

    Y = {c: np.array(v, dtype=np.float32) for c, v in ys.items()}
    split = np.array(splits)
    return X, Y, split, ids


def build_dataset(db_path: str, target_cols=None, downsample_to: int = None,
                   min_overlap_fraction: float = 0.5):
    """
    Joins guo_subset's targets with MIR spectra on sample_id, drops any
    sample missing a spectrum or any requested target, and returns:

      X: (n_samples, n_features) float32 MIR absorbance spectra
      Y: dict[target_name -> (n_samples,) float32 array]
      split: (n_samples,) array of 'train'/'test' strings, taken as-is
             from guo_subset -- use this instead of a fresh random split
             so your results are comparable to the Guo et al. 2025 numbers.
      sample_ids: list[str], same order as X/Y/split

    downsample_to: if given, averages the 1701-band spectrum down to
        this many bins (simple non-overlapping bin averaging). This
        keeps the transformer's O(L^2) attention cost manageable without
        the paper's autoencoder compression step -- e.g. 170 keeps
        roughly the paper's ~10x compression ratio. Leave as None to
        keep the full 1701-band spectrum (slow on CPU for the default
        model size -- see train_model.py for a note on this).

    min_overlap_fraction: raises a clear error if fewer than this
        fraction of guo_subset's sample_ids are found in mir_spectra --
        a low overlap almost always means the two tables use different
        ID namespaces (this hasn't been verified against your actual
        file), and training on a tiny broken join would silently give
        meaningless results.
    """
    if str(db_path).lower().endswith(".csv"):
        return load_csv_dataset(db_path, target_cols=target_cols, downsample_to=downsample_to)

    targets = load_guo_subset(db_path, target_cols)
    target_cols = target_cols or GUO_SUBSET_COLUMNS

    spectra = load_mir_spectra(db_path, sample_ids=targets["sample_id"])

    overlap = len(spectra) / max(len(targets["sample_id"]), 1)
    if overlap < min_overlap_fraction:
        raise ValueError(
            f"Only {overlap:.1%} of guo_subset's {len(targets['sample_id'])} "
            f"sample_ids were found in mir_spectra ({len(spectra)} matched). "
            "This usually means the two tables use different sample_id "
            "namespaces in your specific file. Inspect a few actual "
            "guo_subset sample_ids vs. mir_spectra sample_ids directly "
            "(e.g. via inspect_db.py) before trusting this join."
        )

    xs, ys, splits, ids = [], {c: [] for c in target_cols}, [], []
    for i, sid in enumerate(targets["sample_id"]):
        spec = spectra.get(sid)
        if spec is None:
            continue
        row_targets = [targets[c][i] for c in target_cols]
        if any(np.isnan(v) for v in row_targets):
            continue
        xs.append(spec)
        for c, v in zip(target_cols, row_targets):
            ys[c].append(v)
        splits.append(targets["split"][i])
        ids.append(sid)

    if len(xs) == 0:
        raise ValueError(
            "No samples had both a matched spectrum and complete target "
            "values -- nothing to train on. Check target_cols for columns "
            "with heavy missingness in guo_subset."
        )

    X = np.stack(xs).astype(np.float32)

    if downsample_to is not None:
        n_bands = X.shape[1]
        if downsample_to >= n_bands:
            raise ValueError(
                f"downsample_to ({downsample_to}) must be smaller than the "
                f"number of bands ({n_bands})."
            )
        bin_size = n_bands // downsample_to
        usable = bin_size * downsample_to
        X = X[:, :usable].reshape(X.shape[0], downsample_to, bin_size).mean(axis=2)

    Y = {c: np.array(v, dtype=np.float32) for c, v in ys.items()}
    split = np.array(splits)
    return X, Y, split, ids
