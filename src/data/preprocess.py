#!/usr/bin/env python3
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def dataset_defaults(dataset: str) -> dict:
    dataset = dataset.lower().strip()

    # Target 4 classes (match prepare_lstm_data.py naming):
    # eating, standing, ruminating, walking
    if dataset == "japan_cow":
        return {
            "kind": "auto",  # dir or single csv
            "label_map": {
                # raw -> target
                "GRZ": "eating",
                "FES": "eating",
                "SLT": "eating",
                "DRN": "eating",
                "LCK": "eating",
                "MOV": "walking",
                "RES": "standing",
                "REL": "standing",
                "RUS": "ruminating",
                "RUL": "ruminating",
                # dashboard label pass-through / normalize
                "eating": "eating",
                "grazing": "eating",
                "standing": "standing",
                "resting": "standing",
                "walking": "walking",
                "ruminating": "ruminating",
            },
            "drop_codes": set(),
            "id_prefix": "",
            "min_purity": 0.80,
        }

    if dataset == "actbecalf":
        return {
            "kind": "single_csv_grouped",
            # IMPORTANT: prefer the 4-class column "Label" if present
            "label_col_prefer": ["Label", "label", "behaviour", "behavior", "activity"],
            # IMPORTANT: prefer "cow_id" (often already like calf_1408) if present
            "id_col_prefer": ["cow_id", "cowid", "animal_id", "calfId", "calf_id", "calfid", "id"],
            "label_map": {
                # standing-like
                "standing": "standing",
                "lying": "standing",
                "lying-down": "standing",
                "rising": "standing",

                # walking-like
                "walking": "walking",
                "backward": "walking",
                "sniff_walking": "walking",
                "running": "walking",

                # eating-like
                "eating_forage": "eating",
                "eating_concentrates": "eating",
                "eating_bedding": "eating",
                "eating": "eating",
                "grazing": "eating",

                # ruminating-like
                "rumination": "ruminating",
                "rumination_lying": "ruminating",
                "rumination_standing": "ruminating",
                "ruminating": "ruminating",
            },
            "drop_codes": set(),
            "id_prefix": "",  # do not force "cow" for calves
            "min_purity": 0.80,
        }

    if dataset == "nose_ring":
        return {
            "kind": "single_csv_grouped",
            "label_col_prefer": ["Label", "label", "behavior", "behaviour", "activity"],
            # IMPORTANT: prefer cow_id (already cow2) over cow_num (2)
            "id_col_prefer": ["cow_id", "cow", "cowid", "animal_id", "cow_num", "cownum", "id"],
            "label_map": {
                0: "eating",
                1: "ruminating",
                2: "standing",
                3: "standing",
                4: "walking",
                "eating": "eating",
                "grazing": "eating",
                "standing": "standing",
                "resting": "standing",
                "walking": "walking",
                "ruminating": "ruminating",
            },
            "drop_codes": set(),
            # keep IDs stable; many dashboard files already have cow1/cow2
            "id_prefix": "",
            "min_purity": 0.70,
        }

    raise SystemExit(f"FATAL: unknown dataset '{dataset}'")


def _normalize_name(s: str) -> str:
    return str(s).strip().lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename common column variants to canonical names:
      ax, ay, az, animal_id, label, datetime (or date/time)

    IMPORTANT FIX:
      - Many of your CSVs contain BOTH Label + behaviour (or Label + behavior),
        and/or BOTH cow_id + calfId / cow_num.
      - If we rename all of them to the same name, pandas will create duplicated columns,
        and df['label'] becomes a DataFrame -> code breaks / reads wrong labels.
      - So we pick ONE best label column + ONE best id column before renaming.
    """
    orig_cols = list(df.columns)
    lower = {c: _normalize_name(c) for c in orig_cols}

    # pick best label col: prefer exact "label" over behaviour/behavior/activity
    label_col = None
    for p in ["label", "labels"]:
        for c in orig_cols:
            if lower[c] == p:
                label_col = c
                break
        if label_col is not None:
            break
    if label_col is None:
        for p in ["behaviour", "behavior", "activity"]:
            for c in orig_cols:
                if lower[c] == p:
                    label_col = c
                    break
            if label_col is not None:
                break

    # pick best id col: prefer cow_id/animal_id/cow over calfId/cow_num/id
    id_col = None
    for p in ["cow_id", "cowid", "animal_id", "cow"]:
        for c in orig_cols:
            if lower[c] == p:
                id_col = c
                break
        if id_col is not None:
            break
    if id_col is None:
        for p in ["calfid", "calf_id", "id", "cow_num", "cownum"]:
            for c in orig_cols:
                if lower[c] == p:
                    id_col = c
                    break
            if id_col is not None:
                break

    col_map = {}
    for c in orig_cols:
        lc = lower[c]
        if lc in {"accx", "acc_x", "ax", "acc-x", "acc x"}:
            col_map[c] = "ax"
        elif lc in {"accy", "acc_y", "ay", "acc-y", "acc y"}:
            col_map[c] = "ay"
        elif lc in {"accz", "acc_z", "az", "acc-z", "acc z"}:
            col_map[c] = "az"
        elif lc in {"datetime", "date_time", "date-time", "timestamp", "time_stamp", "time-stamp", "datetimestamp",
                    "timestamp_unix", "timestamp_jst", "time_stamp_unix", "time_stamp_jst"}:
            col_map[c] = "datetime"
        elif lc == "date":
            col_map[c] = "date"
        elif lc == "time":
            col_map[c] = "time"
        elif label_col is not None and c == label_col:
            col_map[c] = "label"
        elif id_col is not None and c == id_col:
            col_map[c] = "animal_id"

    df = df.rename(columns=col_map)

    # safety: if somehow duplicates still exist, keep the first
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    return df


# ✅ ADDED: robust epoch datetime parsing (ms vs s)
def _parse_epoch_datetime(s: pd.Series) -> pd.Series:
    """
    Robust datetime parsing:
    - If numeric epoch: infer ms vs s
    - Else: parse as datetime string
    """
    if pd.api.types.is_numeric_dtype(s):
        v = s.dropna().astype(float)
        if len(v) == 0:
            return pd.to_datetime(s, errors="coerce")
        med = float(v.median())
        unit = "ms" if med > 1e12 else ("s" if med > 1e9 else None)
        if unit:
            return pd.to_datetime(s, unit=unit, errors="coerce")
        return pd.to_datetime(s, errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols_lower:
            return cols_lower[key]
    return None


def load_one_cow_csv(path: str, label_map: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = standardize_columns(df)

    # --- OPTIONAL BUT RECOMMENDED: keep time order stable ---
    if "datetime" in df.columns:
        df["_dt"] = _parse_epoch_datetime(df["datetime"])
        df = df.sort_values("_dt", kind="mergesort").drop(columns=["_dt"])
    elif ("date" in df.columns) and ("time" in df.columns):
        dt = df["date"].astype(str) + " " + df["time"].astype(str)
        df["_dt"] = pd.to_datetime(dt, errors="coerce")
        df = df.sort_values("_dt", kind="mergesort").drop(columns=["_dt"])

    required = {"ax", "ay", "az", "label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"FATAL: {Path(path).name} missing {sorted(missing)}. Found: {list(df.columns)}")

    for a in ("ax", "ay", "az"):
        df[a] = pd.to_numeric(df[a], errors="coerce")

    df["label"] = df["label"].apply(lambda x: x if not isinstance(x, str) else x.strip())

    # keep only mappable labels
    df = df[df["label"].isin(label_map.keys())].copy()
    if df.empty:
        raise SystemExit(f"FATAL: {Path(path).name} has 0 rows after filtering label keys")

    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["ax", "ay", "az", "label"]).reset_index(drop=True)
    return df


def load_japan_cow_dir(dataset_dir: str, label_map: dict) -> dict:
    files = sorted(glob.glob(str(Path(dataset_dir) / "cow*.csv")))
    if not files:
        raise SystemExit(f"FATAL: No cow*.csv found in {dataset_dir}")
    cows = {}
    for fp in files:
        cows[Path(fp).stem] = load_one_cow_csv(fp, label_map)
    return cows


def _normalize_test_ids(test_ids: set[str], available_ids: set[str]) -> set[str]:
    """
    Allow user to pass:
      - 1408 or calf_1408 (actbecalf)
      - 2 or cow2 (nose_ring/japan)
    We'll try common prefixes so it won't crash.
    """
    out = set()
    for tid in test_ids:
        t = str(tid).strip()
        if not t:
            continue
        if t in available_ids:
            out.add(t); continue
        if ("cow" + t) in available_ids:
            out.add("cow" + t); continue
        if t.startswith("cow") and t[3:] in available_ids:
            out.add(t[3:]); continue
        if ("calf_" + t) in available_ids:
            out.add("calf_" + t); continue
        if t.startswith("calf_") and t[5:] in available_ids:
            out.add(t[5:]); continue
        out.add(t)
    return out


def load_single_csv_grouped(dataset_path: str, spec: dict) -> dict:
    df = pd.read_csv(dataset_path)
    df = standardize_columns(df)

    label_col = _pick_first_existing(df, spec.get("label_col_prefer", ["label"]))
    id_col = _pick_first_existing(df, spec.get("id_col_prefer", ["animal_id"]))

    if label_col is None:
        raise SystemExit(f"FATAL: cannot find label column. Found: {list(df.columns)}")
    if id_col is None:
        raise SystemExit(f"FATAL: cannot find animal id column. Found: {list(df.columns)}")

    if label_col != "label":
        df = df.rename(columns={label_col: "label"})
    if id_col != "animal_id":
        df = df.rename(columns={id_col: "animal_id"})

    required = {"animal_id", "ax", "ay", "az", "label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"FATAL: missing columns {sorted(missing)}. Found: {list(df.columns)}")

    for a in ("ax", "ay", "az"):
        df[a] = pd.to_numeric(df[a], errors="coerce")

    label_map = spec["label_map"]
    df["label"] = df["label"].apply(lambda x: x if not isinstance(x, str) else x.strip())
    df = df[df["label"].isin(label_map.keys())].copy()
    df["label"] = df["label"].map(label_map)

    df = df.dropna(subset=["animal_id", "ax", "ay", "az", "label"]).reset_index(drop=True)

    df["animal_id"] = df["animal_id"].astype(str).str.strip()

    # ✅ ADDED: stable time order within each animal (important for window correctness)
    if "datetime" in df.columns:
        df["_dt"] = _parse_epoch_datetime(df["datetime"])
        df = df.sort_values(["animal_id", "_dt"], kind="mergesort").drop(columns=["_dt"])
    elif ("date" in df.columns) and ("time" in df.columns):
        dt = df["date"].astype(str) + " " + df["time"].astype(str)
        df["_dt"] = pd.to_datetime(dt, errors="coerce")
        df = df.sort_values(["animal_id", "_dt"], kind="mergesort").drop(columns=["_dt"])

    grouped = {}
    for animal_id, sub in df.groupby("animal_id", sort=False):
        grouped[animal_id] = sub.reset_index(drop=True)
    return grouped


def load_dataset(dataset: str, dataset_path: str) -> tuple[dict, dict]:
    spec = dataset_defaults(dataset)

    # japan_cow: support BOTH directory of cow*.csv and merged dashboard csv
    p = Path(dataset_path)
    if dataset == "japan_cow" and p.is_file() and p.suffix.lower() == ".csv":
        return load_single_csv_grouped(dataset_path, spec), spec

    if spec["kind"] == "auto" and Path(dataset_path).is_dir():
        return load_japan_cow_dir(dataset_path, spec["label_map"]), spec

    return load_single_csv_grouped(dataset_path, spec), spec


def make_features(win: pd.DataFrame) -> dict:
    out = {}
    ax = win["ax"].astype(float).to_numpy()
    ay = win["ay"].astype(float).to_numpy()
    az = win["az"].astype(float).to_numpy()
    mag = np.sqrt(ax * ax + ay * ay + az * az)

    def stats(prefix: str, arr: np.ndarray):
        out[f"{prefix}_mean"] = float(arr.mean())
        out[f"{prefix}_std"] = float(arr.std(ddof=0))
        out[f"{prefix}_min"] = float(arr.min())
        out[f"{prefix}_max"] = float(arr.max())
        out[f"{prefix}_var"] = float(arr.var(ddof=0))

    stats("ax", ax)
    stats("ay", ay)
    stats("az", az)
    stats("mag", mag)

    out["sma"] = float(np.mean(np.abs(ax) + np.abs(ay) + np.abs(az)))
    out["energy"] = float(np.mean(mag ** 2))
    return out


def windows_from_index(df: pd.DataFrame, window_size: int, step_size: int, min_purity: float):
    wins, labs = [], []
    n = len(df)
    if n < window_size:
        return wins, labs

    for i in range(0, n - window_size + 1, step_size):
        w = df.iloc[i : i + window_size]
        counts = w["label"].value_counts()
        top = counts.index[0]
        purity = float(counts.iloc[0]) / float(window_size)
        if purity >= min_purity:
            wins.append(w)
            labs.append(top)
    return wins, labs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["japan_cow", "actbecalf", "nose_ring"], default="japan_cow")
    ap.add_argument("--dataset-path", default=None, help="dir (japan_cow) or csv file")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--split", choices=["cow", "random"], default="cow")
    ap.add_argument("--test-cows", default=None, help="Comma-separated IDs, e.g. cow2 or 1408 or calf_1408")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--window-size", type=int, default=50)
    ap.add_argument("--step-size", type=int, default=25)
    ap.add_argument("--min-label-purity", type=float, default=0.80)
    args = ap.parse_args()

    spec = dataset_defaults(args.dataset)

    if args.dataset_path is None:
        if args.dataset == "japan_cow":
            args.dataset_path = "data/raw/japan_cows_1to6_merged_dashboard.csv"
        elif args.dataset == "actbecalf":
            args.dataset_path = "data/raw/AcTBeCalf_dashboard.csv"
        else:
            args.dataset_path = "data/raw/nose_ring_dashboard.csv"

    cows, spec = load_dataset(args.dataset, args.dataset_path)

    if args.test_cows is None:
        args.test_cows = "cow2" if args.dataset != "actbecalf" else "1408"
    test_ids_raw = {c.strip() for c in str(args.test_cows).split(",") if c.strip()}
    test_ids = _normalize_test_ids(test_ids_raw, set(cows.keys()))

    rows = []
    for animal_id, df in cows.items():
        wins, labs = windows_from_index(df, args.window_size, args.step_size, float(args.min_label_purity))
        if not wins:
            print(f"[warn] {animal_id}: 0 windows (try lower --min-label-purity).")
            continue
        for w, lab in zip(wins, labs):
            feats = make_features(w)
            feats["label"] = lab
            feats["animal_id"] = animal_id
            rows.append(feats)

    if not rows:
        raise SystemExit("FATAL: No windows created. Check CSVs or lower --min-label-purity.")

    feats_df = pd.DataFrame(rows).dropna()
    print(f"[info] dataset={args.dataset} windows={len(feats_df)} classes={feats_df['label'].nunique()}")
    print(feats_df["label"].value_counts())

    split_tag = "loco" if args.split == "cow" else "random"
    outdir = Path(args.processed_dir) / args.dataset / split_tag
    outdir.mkdir(parents=True, exist_ok=True)

    if args.split == "cow":
        missing = sorted(list(set(test_ids) - set(feats_df["animal_id"].unique())))
        if missing:
            raise SystemExit(f"FATAL: test id(s) not found in windows: {missing}")

        train_df = feats_df[~feats_df["animal_id"].isin(test_ids)].drop(columns=["animal_id"]).reset_index(drop=True)
        test_df  = feats_df[feats_df["animal_id"].isin(test_ids)].drop(columns=["animal_id"]).reset_index(drop=True)
        if train_df.empty or test_df.empty:
            raise SystemExit(f"FATAL: empty split. train={len(train_df)} test={len(test_df)}")
    else:
        from sklearn.model_selection import train_test_split
        X = feats_df.drop(columns=["label", "animal_id"])
        y = feats_df["label"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
        train_df = pd.concat([Xtr.reset_index(drop=True), ytr.reset_index(drop=True)], axis=1)
        test_df  = pd.concat([Xte.reset_index(drop=True), yte.reset_index(drop=True)], axis=1)

    train_path = outdir / "train.parquet"
    test_path  = outdir / "test.parquet"
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    print(f"[ok] saved {train_path}")
    print(f"[ok] saved {test_path}")


if __name__ == "__main__":
    main()

