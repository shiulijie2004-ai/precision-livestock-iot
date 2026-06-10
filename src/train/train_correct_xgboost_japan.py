#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train the CORRECT XGBoost model for japan_cows.csv.

Correct meaning:
- Uses ONLY accelerometer 3-axis features: AccX, AccY, AccZ
- Drops time columns, cow_id, and label from features
- Uses a 50-sample window
- Converts each 50-sample window into 18 statistical features:
  3 sensor axes × 6 stats = 18 features
- Evaluates using:
  1) group split
  2) full LOCO cow-wise validation
- Saves final deployment model trained on ALL data:
  deployment_builds/XGBoost_final.joblib
- Saves feature metadata:
  deployment_builds/XGBoost_final_features.json

Run from your project root:
    cd ~/Farm/precision-livestock-iot
    python src/train/train_correct_xgboost_japan.py \
      --data data/japan_cows.csv \
      --window 50
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise RuntimeError(
        "xgboost is not installed. Install it first:\n"
        "    pip install xgboost\n"
        f"Original error: {e}"
    )


# ============================================================
# Fixed 4-class target
# ============================================================

TARGET_CLASSES = ["eating", "ruminating", "standing", "walking"]

LABEL2ID = {
    "eating": 0,
    "ruminating": 1,
    "standing": 2,
    "walking": 3,
}

ID2LABEL = {
    0: "eating",
    1: "ruminating",
    2: "standing",
    3: "walking",
}

FIXED_LABELS = [0, 1, 2, 3]


# ============================================================
# Japan cow label mapping
# ============================================================

JAPAN_COW_LABEL_MAP = {
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

    # Already-clean labels in your dashboard CSV
    "eating": "eating",
    "ruminating": "ruminating",
    "standing": "standing",
    "walking": "walking",
}


# ============================================================
# Feature configuration
# ============================================================

RAW_FEATURE_COLS = ["AccX", "AccY", "AccZ"]
DEPLOY_FEATURE_KEYS = ["ax_g", "ay_g", "az_g"]

TIME_COL = "TimeStamp_UNIX"
GROUP_COL = "cow_id"
LABEL_COL = "Label"

STAT_NAMES = ["mean", "std", "min", "max", "range", "energy"]


def normalize_label(x):
    """Map Japan cow labels or already-clean labels into 4 target classes."""
    if pd.isna(x):
        return None

    s = str(x).strip()
    upper = s.upper()
    lower = s.lower()

    if upper in JAPAN_COW_LABEL_MAP:
        return JAPAN_COW_LABEL_MAP[upper]

    if lower in JAPAN_COW_LABEL_MAP:
        return JAPAN_COW_LABEL_MAP[lower]

    # Safe fallback
    if "rumin" in lower:
        return "ruminating"
    if "graz" in lower or "feed" in lower or "eat" in lower:
        return "eating"
    if "walk" in lower or "mov" in lower:
        return "walking"
    if "stand" in lower or "rest" in lower or "lie" in lower or "lying" in lower:
        return "standing"

    return None


def make_feature_names(raw_cols):
    """Feature order MUST match build_window_stats()."""
    names = []
    for stat in STAT_NAMES:
        for col in raw_cols:
            names.append(f"{stat}_{col}")
    return names


def build_window_stats_one_group(X_group, y_group, group_name, window):
    """
    Build rolling window statistical features for ONE cow only.
    This avoids windows crossing from cow1 into cow2.

    Output shape:
        rows = len(group) - window + 1
        cols = 6 stats × 3 accelerometer axes = 18
    """
    if len(X_group) < window:
        return None, None, None

    X_group = np.asarray(X_group, dtype=np.float32)
    y_group = np.asarray(y_group, dtype=np.int64)

    df_x = pd.DataFrame(X_group, columns=RAW_FEATURE_COLS)

    mean = df_x.rolling(window=window, min_periods=window).mean().iloc[window - 1:].to_numpy(dtype=np.float32)
    std = df_x.rolling(window=window, min_periods=window).std(ddof=0).iloc[window - 1:].to_numpy(dtype=np.float32)
    mn = df_x.rolling(window=window, min_periods=window).min().iloc[window - 1:].to_numpy(dtype=np.float32)
    mx = df_x.rolling(window=window, min_periods=window).max().iloc[window - 1:].to_numpy(dtype=np.float32)
    rng = (mx - mn).astype(np.float32)

    # Energy = sum of squares in the window
    energy = (df_x * df_x).rolling(window=window, min_periods=window).sum().iloc[window - 1:].to_numpy(dtype=np.float32)

    Xw = np.concatenate([mean, std, mn, mx, rng, energy], axis=1).astype(np.float32)

    # Window label = label of the last row in the window
    yw = y_group[window - 1:]

    gw = np.full(shape=len(yw), fill_value=str(group_name), dtype=object)

    return Xw, yw, gw


def load_and_prepare(data_path, window):
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found: {data_path}")

    df = pd.read_csv(data_path)

    print("\n========== RAW CSV ==========")
    print(f"Path: {data_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    missing = [c for c in [TIME_COL, GROUP_COL, LABEL_COL] + RAW_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Your CSV is missing required columns:\n"
            f"Missing: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    # Keep only required columns to prevent leakage
    df = df[[TIME_COL, GROUP_COL, *RAW_FEATURE_COLS, LABEL_COL]].copy()

    # Convert time and features
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    for c in RAW_FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Map labels
    df["target_4class"] = df[LABEL_COL].apply(normalize_label)
    df = df[df["target_4class"].isin(TARGET_CLASSES)].copy()

    # Clean missing feature/time/group rows
    df = df.dropna(subset=[TIME_COL, GROUP_COL, *RAW_FEATURE_COLS, "target_4class"]).copy()

    # Sort by cow and time to make rolling windows correct
    df[GROUP_COL] = df[GROUP_COL].astype(str)
    df = df.sort_values([GROUP_COL, TIME_COL], kind="mergesort").reset_index(drop=True)

    df["y"] = df["target_4class"].map(LABEL2ID).astype(int)

    print("\n========== AFTER CLEANING ==========")
    print(f"Rows kept: {len(df):,}")
    print("Label distribution:")
    print(df["target_4class"].value_counts().reindex(TARGET_CLASSES, fill_value=0))
    print("\nCow distribution:")
    print(df[GROUP_COL].value_counts().sort_index())

    X_all = []
    y_all = []
    g_all = []

    for cow_id, part in df.groupby(GROUP_COL, sort=True):
        Xg = part[RAW_FEATURE_COLS].to_numpy(dtype=np.float32)
        yg = part["y"].to_numpy(dtype=np.int64)

        Xw, yw, gw = build_window_stats_one_group(Xg, yg, cow_id, window)

        if Xw is None:
            print(f"[SKIP] {cow_id}: rows={len(part):,} < window={window}")
            continue

        X_all.append(Xw)
        y_all.append(yw)
        g_all.append(gw)

        print(f"[WINDOW] {cow_id}: raw_rows={len(part):,}, windowed_rows={len(yw):,}")

    if not X_all:
        raise ValueError("No windowed samples were created. Check window size or dataset size.")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    groups = np.concatenate(g_all, axis=0)

    feature_names = make_feature_names(RAW_FEATURE_COLS)

    print("\n========== FINAL TRAINING MATRIX ==========")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"groups: {sorted(pd.Series(groups).unique().tolist())}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Expected feature count: 18")
    print(f"Feature names: {feature_names}")

    if X.shape[1] != 18:
        raise ValueError(f"Wrong feature count. Expected 18, got {X.shape[1]}")

    return X, y, groups, feature_names, df


def make_model(args):
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=args.seed,
        verbosity=1,
    )
    return model


def evaluate_model(model, X_test, y_test, title):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    macro_p = precision_score(y_test, preds, average="macro", labels=FIXED_LABELS, zero_division=0)
    macro_r = recall_score(y_test, preds, average="macro", labels=FIXED_LABELS, zero_division=0)
    macro_f1 = f1_score(y_test, preds, average="macro", labels=FIXED_LABELS, zero_division=0)
    weighted_f1 = f1_score(y_test, preds, average="weighted", labels=FIXED_LABELS, zero_division=0)

    print(f"\n========== {title} ==========")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision Macro : {macro_p:.4f}")
    print(f"Recall Macro    : {macro_r:.4f}")
    print(f"F1 Macro        : {macro_f1:.4f}")
    print(f"F1 Weighted     : {weighted_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            preds,
            labels=FIXED_LABELS,
            target_names=TARGET_CLASSES,
            zero_division=0,
            digits=4,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds, labels=FIXED_LABELS))

    return {
        "Title": title,
        "Accuracy": float(acc),
        "Precision (Macro)": float(macro_p),
        "Recall (Macro)": float(macro_r),
        "F1 (Macro)": float(macro_f1),
        "F1 (Weighted)": float(weighted_f1),
        "F1 (eating)": float(f1_score(y_test, preds, average=None, labels=FIXED_LABELS, zero_division=0)[0]),
        "F1 (ruminating)": float(f1_score(y_test, preds, average=None, labels=FIXED_LABELS, zero_division=0)[1]),
        "F1 (standing)": float(f1_score(y_test, preds, average=None, labels=FIXED_LABELS, zero_division=0)[2]),
        "F1 (walking)": float(f1_score(y_test, preds, average=None, labels=FIXED_LABELS, zero_division=0)[3]),
    }


def run_group_split(X, y, groups, args):
    print("\n\n############################################################")
    print("GROUP SPLIT EVALUATION")
    print("############################################################")

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_groups = sorted(pd.Series(groups[train_idx]).unique().tolist())
    test_groups = sorted(pd.Series(groups[test_idx]).unique().tolist())

    print(f"Train groups: {train_groups}")
    print(f"Test groups : {test_groups}")
    print(f"Train shape : {X_train.shape}")
    print(f"Test shape  : {X_test.shape}")

    model = make_model(args)

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    return evaluate_model(model, X_test, y_test, "Group split")


def run_full_loco(X, y, groups, args):
    print("\n\n############################################################")
    print("FULL LOCO EVALUATION")
    print("############################################################")

    results = []

    unique_groups = sorted(pd.Series(groups).unique().tolist())

    for test_group in unique_groups:
        test_mask = groups == test_group

        X_train = X[~test_mask]
        y_train = y[~test_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        print(f"\n--- LOCO test_group={test_group} ---")
        print(f"Train shape: {X_train.shape}")
        print(f"Test shape : {X_test.shape}")
        print("Test label distribution:")
        print(pd.Series(y_test).map(ID2LABEL).value_counts().reindex(TARGET_CLASSES, fill_value=0))

        model = make_model(args)
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_weight)

        res = evaluate_model(model, X_test, y_test, f"LOCO {test_group}")
        res["Test Group"] = test_group
        results.append(res)

    df_res = pd.DataFrame(results)

    print("\n========== FULL LOCO AVERAGE ==========")
    metric_cols = [
        "Accuracy",
        "Precision (Macro)",
        "Recall (Macro)",
        "F1 (Macro)",
        "F1 (Weighted)",
        "F1 (eating)",
        "F1 (ruminating)",
        "F1 (standing)",
        "F1 (walking)",
    ]
    print(df_res[metric_cols].mean(numeric_only=True).round(4))

    return df_res


def train_final_model(X, y, feature_names, args):
    print("\n\n############################################################")
    print("FINAL DEPLOYMENT MODEL: TRAIN ON ALL DATA")
    print("############################################################")

    model = make_model(args)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    model.fit(X, y, sample_weight=sample_weight)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "XGBoost_final.joblib"
    meta_path = out_dir / "XGBoost_final_features.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_name": "XGBoost_final",
        "dataset": "japan_cow",
        "raw_training_sensor_columns": RAW_FEATURE_COLS,
        "live_sensor_keys_expected": DEPLOY_FEATURE_KEYS,
        "window": args.window,
        "feature_mode": "stats",
        "stats_order": STAT_NAMES,
        "feature_names": feature_names,
        "n_features_expected": len(feature_names),
        "target_classes": TARGET_CLASSES,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "important_note": (
            "This model expects 18 features: 3 accelerometer axes × 6 rolling-window statistics. "
            "Do not feed 50×3 flatten features and do not include gyroscope features."
        ),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    loaded = joblib.load(model_path)

    print(f"\n[SAVED] Model: {model_path}")
    print(f"[SAVED] Metadata: {meta_path}")
    print(f"[CHECK] Loaded model type: {type(loaded)}")
    print(f"[CHECK] n_features_in_: {getattr(loaded, 'n_features_in_', None)}")

    if getattr(loaded, "n_features_in_", None) != 18:
        raise ValueError(
            f"Saved model has wrong n_features_in_: {getattr(loaded, 'n_features_in_', None)}. "
            "Expected 18."
        )

    print("\n✅ Correct XGBoost model saved successfully.")
    print("✅ This model is accelerometer-only.")
    print("✅ Expected input = 18 statistical features.")
    print("✅ Do NOT use gyro columns with this model.")

    return model_path, meta_path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="data/japan_cows_1to6_merged_dashboard.csv",
        help="Path to Japan cow CSV.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Rolling window size. Use 50 to match your FYP setting.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="deployment_builds",
        help="Where to save XGBoost_final.joblib.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)

    # XGBoost parameters
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=1.0)

    parser.add_argument(
        "--skip-loco",
        action="store_true",
        help="Skip LOCO evaluation to train faster.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    X, y, groups, feature_names, clean_df = load_and_prepare(args.data, args.window)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    group_res = run_group_split(X, y, groups, args)

    if not args.skip_loco:
        loco_df = run_full_loco(X, y, groups, args)
        loco_path = out_dir / "XGBoost_loco_results.csv"
        loco_df.to_csv(loco_path, index=False)
        print(f"\n[SAVED] LOCO results: {loco_path}")

    train_final_model(X, y, feature_names, args)

    print("\n========== HOW TO USE THIS MODEL ==========")
    print("For live prediction, the predictor must compute the SAME 18 stats features:")
    print("  mean/std/min/max/range/energy for ax_g, ay_g, az_g over a 50-sample window.")
    print("")
    print("Do NOT run this model with:")
    print("  --feature-keys ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps")
    print("")
    print("Use accelerometer only:")
    print("  --feature-keys ax_g,ay_g,az_g")
    print("")
    print("And make sure predictor feature mode is stats/auto, NOT flatten.")


if __name__ == "__main__":
    main()
