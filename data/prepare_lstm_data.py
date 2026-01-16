#!/usr/bin/env python3

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


CANONICAL_LABELS = {"eating", "standing", "walking", "ruminating"}

LABEL_SYNONYMS = {
    # eating-like
    "grazing": "eating",
    "feeding": "eating",
    "feed": "eating",
    # standing/resting-like
    "resting": "standing",
    "lying": "standing",
    "lying-down": "standing",
    "rising": "standing",
    "idle": "standing",
    # rumination-like
    "rumination": "ruminating",
    "rumination_lying": "ruminating",
    "rumination_standing": "ruminating",
}

JAPAN_CODE_MAP = {
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
}


def _normalize_name(s: str) -> str:
    return str(s).strip().lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename common column variants to: animal_id, ax, ay, az, label, datetime.
    IMPORTANT: avoid duplicated columns by picking best label & id columns first.
    """
    orig_cols = list(df.columns)
    lower = {c: _normalize_name(c) for c in orig_cols}

    # label: prefer Label/label over behaviour/behavior/activity
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

    # id: prefer cow_id/animal_id/cow over calfId/cow_num/id
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
        elif lc in {
            "datetime", "date_time", "date-time",
            "timestamp", "time_stamp", "time-stamp", "datetimestamp",
            "timestamp_unix", "timestamp_jst", "time_stamp_unix", "time_stamp_jst"
        }:
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

    # Create datetime from date+time if needed
    if "datetime" not in df.columns and "date" in df.columns and "time" in df.columns:
        dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
        df = df.assign(datetime=dt)

    # safety: drop duplicate column names
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    return df


def normalize_label_value(v) -> str:
    if pd.isna(v):
        return ""

    # numeric labels (e.g., nose_ring raw)
    if isinstance(v, (int, np.integer)) or (isinstance(v, str) and v.strip().isdigit()):
        vv = int(v)
        if vv == 0:
            return "eating"
        if vv == 1:
            return "ruminating"
        if vv == 4:
            return "walking"
        if vv in (2, 3):
            return "standing"
        return ""

    s = str(v).strip()
    if not s:
        return ""

    if s in JAPAN_CODE_MAP:
        return JAPAN_CODE_MAP[s]

    s2 = s.lower().strip()
    if s2 in LABEL_SYNONYMS:
        return LABEL_SYNONYMS[s2]

    if s2 in CANONICAL_LABELS:
        return s2

    return ""


def _normalize_test_ids(test_ids: List[str], available_ids: List[str]) -> List[str]:
    avail = set(available_ids)
    out = []
    for tid in test_ids:
        t = str(tid).strip()
        if not t:
            continue
        if t in avail:
            out.append(t); continue
        if ("cow" + t) in avail:
            out.append("cow" + t); continue
        if t.startswith("cow") and t[3:] in avail:
            out.append(t[3:]); continue
        if ("calf_" + t) in avail:
            out.append("calf_" + t); continue
        if t.startswith("calf_") and t[5:] in avail:
            out.append(t[5:]); continue
        out.append(t)
    return out


def _parse_and_sort_datetime(df: pd.DataFrame, by_animal: bool = True) -> pd.DataFrame:
    """
    Helper: if df has 'datetime', parse it safely:
      - numeric epoch: guess ms vs s
      - string: pd.to_datetime
    Then stable-sort by (animal_id, datetime) or just datetime.
    """
    if "datetime" not in df.columns:
        # still keep stable order by animal_id if requested
        if by_animal and "animal_id" in df.columns:
            return df.sort_values(["animal_id"], kind="mergesort")
        return df

    s = df["datetime"]

    if pd.api.types.is_numeric_dtype(s):
        v = s.dropna().astype(float)
        if len(v):
            med = float(v.median())
            unit = "ms" if med > 1e12 else ("s" if med > 1e9 else None)
            if unit:
                df["datetime"] = pd.to_datetime(s, unit=unit, errors="coerce")
            else:
                df["datetime"] = pd.to_datetime(s, errors="coerce")
        else:
            df["datetime"] = pd.to_datetime(s, errors="coerce")
    else:
        df["datetime"] = pd.to_datetime(s, errors="coerce")

    if by_animal and "animal_id" in df.columns:
        return df.sort_values(["animal_id", "datetime"], kind="mergesort")
    return df.sort_values(["datetime"], kind="mergesort")


def load_grouped_from_one_csv(path: Path) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(path)
    df = standardize_columns(df)

    required = {"animal_id", "ax", "ay", "az", "label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"FATAL: {path.name} missing columns {sorted(missing)}. Found: {list(df.columns)}")

    for a in ("ax", "ay", "az"):
        df[a] = pd.to_numeric(df[a], errors="coerce")

    # correct indentation + robust epoch parsing + stable sort
    df = _parse_and_sort_datetime(df, by_animal=True)

    df["animal_id"] = df["animal_id"].astype(str).str.strip()

    df["label"] = df["label"].apply(normalize_label_value)
    df = df[df["label"].isin(CANONICAL_LABELS)].copy()

    df = df.dropna(subset=["animal_id", "ax", "ay", "az"]).reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"FATAL: {path.name} has 0 valid rows after filtering labels/axes")

    grouped = {}
    for animal_id, sub in df.groupby("animal_id", sort=False):
        grouped[animal_id] = sub.reset_index(drop=True)
    return grouped


def load_japan_dir(dir_path: Path) -> Dict[str, pd.DataFrame]:
    files = sorted(glob.glob(str(dir_path / "cow*.csv")))
    if not files:
        raise SystemExit(f"FATAL: No cow*.csv found in {dir_path}")

    cows = {}
    for fp in files:
        p = Path(fp)
        df = pd.read_csv(p)
        df = standardize_columns(df)

        required = {"ax", "ay", "az", "label"}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"FATAL: {p.name} missing columns {sorted(missing)}. Found: {list(df.columns)}")

        for a in ("ax", "ay", "az"):
            df[a] = pd.to_numeric(df[a], errors="coerce")

        # ✅ FIX: if datetime exists, sort it too (keeps windowing correct)
        df = _parse_and_sort_datetime(df, by_animal=False)

        df["label"] = df["label"].apply(normalize_label_value)
        df = df[df["label"].isin(CANONICAL_LABELS)].copy()
        df = df.dropna(subset=["ax", "ay", "az"]).reset_index(drop=True)
        if df.empty:
            raise SystemExit(f"FATAL: {p.name} has 0 valid rows after filtering labels/axes")

        cows[p.stem] = df

    return cows


def windows_from_series(df: pd.DataFrame, window_size: int, step_size: int, min_purity: float) -> Tuple[List[np.ndarray], List[str]]:
    X_list, y_list = [], []
    n = len(df)
    if n < window_size:
        return X_list, y_list

    labels = df["label"].to_numpy()
    acc = df[["ax", "ay", "az"]].to_numpy(dtype=np.float32)

    for i in range(0, n - window_size + 1, step_size):
        y_win = labels[i:i + window_size]
        vals, counts = np.unique(y_win, return_counts=True)
        top_idx = int(np.argmax(counts))
        top_lab = str(vals[top_idx])
        purity = float(counts[top_idx]) / float(window_size)
        if purity >= min_purity:
            X_list.append(acc[i:i + window_size])
            y_list.append(top_lab)

    return X_list, y_list


def build_splits(
    grouped: Dict[str, pd.DataFrame],
    split: str,
    test_ids: List[str],
    test_size: float,
    seed: int,
    window_size: int,
    step_size: int,
    min_purity: float,
):
    Xtr_chunks, ytr_chunks, Xte_chunks, yte_chunks = [], [], [], []

    for animal_id, df in grouped.items():
        X_list, y_list = windows_from_series(df, window_size, step_size, min_purity)
        if not X_list:
            print(f"[warn] {animal_id}: 0 windows (try lower --min-label-purity)")
            continue

        Xc = np.stack(X_list, axis=0)  # (N,T,3)
        yc = np.array(y_list, dtype=object)

        if split == "cow":
            if animal_id in test_ids:
                Xte_chunks.append(Xc); yte_chunks.append(yc)
            else:
                Xtr_chunks.append(Xc); ytr_chunks.append(yc)
        else:
            Xtr, Xte, ytr, yte = train_test_split(
                Xc, yc, test_size=test_size, random_state=seed, stratify=yc
            )
            Xtr_chunks.append(Xtr); ytr_chunks.append(ytr)
            Xte_chunks.append(Xte); yte_chunks.append(yte)

    if not Xtr_chunks or not Xte_chunks:
        raise SystemExit("FATAL: Not enough windows for train/test. "
                         "Lower --min-label-purity, use smaller window, or check --test-ids.")

    X_train = np.concatenate(Xtr_chunks, axis=0)
    y_train_text = np.concatenate(ytr_chunks, axis=0)
    X_test = np.concatenate(Xte_chunks, axis=0)
    y_test_text = np.concatenate(yte_chunks, axis=0)
    return X_train, y_train_text, X_test, y_test_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["japan_cow", "actbecalf", "nose_ring"], required=True)
    ap.add_argument("--dataset-path", default=None, help="CSV file (recommended) or Japan folder of cow*.csv")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--ckpt-dir", default="results/checkpoints")

    ap.add_argument("--split", choices=["cow", "random"], default="cow")
    ap.add_argument("--test-ids", default=None, help="Comma-separated test animal IDs (e.g., cow2, 2, calf_1408, 1408)")

    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--window-size", type=int, default=50)
    ap.add_argument("--step-size", type=int, default=25)
    ap.add_argument("--min-label-purity", type=float, default=0.80)
    args = ap.parse_args()

    if args.dataset_path is None:
        if args.dataset == "japan_cow":
            args.dataset_path = "data/raw/japan_cows_1to6_merged_dashboard.csv"
        elif args.dataset == "actbecalf":
            args.dataset_path = "data/raw/AcTBeCalf_dashboard.csv"
        else:
            args.dataset_path = "data/raw/nose_ring_dashboard.csv"

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise SystemExit(f"FATAL: dataset path not found: {dataset_path}")

    if dataset_path.is_dir():
        grouped = load_japan_dir(dataset_path)
    else:
        grouped = load_grouped_from_one_csv(dataset_path)

    if args.test_ids is None:
        args.test_ids = "cow2" if args.dataset != "actbecalf" else "1408"

    test_ids_raw = [s.strip() for s in str(args.test_ids).split(",") if s.strip()]
    test_ids = _normalize_test_ids(test_ids_raw, list(grouped.keys()))

    if args.split == "cow":
        missing = sorted(set(test_ids) - set(grouped.keys()))
        if missing:
            raise SystemExit(f"FATAL: test id(s) not found: {missing}\n"
                             f"Available IDs (first 15): {list(grouped.keys())[:15]}")

    X_train, y_train_text, X_test, y_test_text = build_splits(
        grouped=grouped,
        split=args.split,
        test_ids=test_ids,
        test_size=args.test_size,
        seed=args.seed,
        window_size=args.window_size,
        step_size=args.step_size,
        min_purity=args.min_label_purity,
    )

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_text)

    unseen = set(np.unique(y_test_text)) - set(le.classes_)
    if unseen:
        raise SystemExit(f"FATAL: test has unseen labels not in train: {sorted(unseen)}")
    y_test = le.transform(y_test_text)

    mean = X_train.reshape(-1, 3).mean(axis=0).astype(np.float32)
    std = X_train.reshape(-1, 3).std(axis=0).astype(np.float32)
    std[std == 0.0] = 1.0

    X_train = ((X_train - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)

    split_tag = "loco" if args.split == "cow" else "random"
    out_dir  = Path(args.processed_dir) / args.dataset / split_tag
    ckpt_dir = Path(args.ckpt_dir) / args.dataset / split_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)

    joblib.dump(le, ckpt_dir / "label_encoder.joblib")
    (ckpt_dir / "lstm_norm.json").write_text(json.dumps(
        {"mean": mean.tolist(), "std": std.tolist(), "axes": ["ax", "ay", "az"]}, indent=2
    ))

    manifest = {
        "dataset": args.dataset,
        "dataset_path": str(dataset_path),
        "split": args.split,
        "test_ids": test_ids,
        "seed": args.seed,
        "window_size": args.window_size,
        "step_size": args.step_size,
        "min_label_purity": args.min_label_purity,
        "classes": le.classes_.tolist(),
        "train_windows": int(X_train.shape[0]),
        "test_windows": int(X_test.shape[0]),
    }
    (ckpt_dir / "lstm_manifest.json").write_text(json.dumps(manifest, indent=2))

    def _counts(y):
        uniq, cnt = np.unique(y, return_counts=True)
        return {le.classes_[int(u)]: int(c) for u, c in zip(uniq, cnt)}

    print("[ok] Prepared LSTM tensors")
    print(" dataset     :", args.dataset)
    print(" dataset_path:", str(dataset_path))
    print(" split       :", args.split, ("test_ids=" + ",".join(test_ids) if args.split == "cow" else f"test_size={args.test_size}"))
    print(" X_train     :", X_train.shape, "class_counts:", _counts(y_train))
    print(" X_test      :", X_test.shape, "class_counts:", _counts(y_test))
    print(" classes     :", le.classes_.tolist())


if __name__ == "__main__":
    main()

