#!/usr/bin/env python3

import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["japan_cow", "actbecalf", "nose_ring"], required=True)
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--split", choices=["cow", "random"], default="cow")
    args = ap.parse_args()

    ds = args.dataset
    split_tag = "loco" if args.split == "cow" else "random"
    processed  = Path(args.processed_dir) / ds / split_tag
    models_dir = Path(args.models_dir) / ds / split_tag
    models_dir.mkdir(parents=True, exist_ok=True)

    train_p = processed / "train.parquet"
    test_p = processed / "test.parquet"
    if not train_p.exists() or not test_p.exists():
        raise SystemExit(f"FATAL: train/test parquet not found for {ds}. Run preprocess.py first.")

    train = pd.read_parquet(train_p)
    test = pd.read_parquet(test_p)

    if "label" not in train.columns or "label" not in test.columns:
        raise SystemExit("FATAL: parquet missing 'label' column")

    X_train = train.drop(columns=["label"])
    y_train_text = train["label"].astype(str)

    X_test = test.drop(columns=["label"])
    y_test_text = test["label"].astype(str)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_text)

    unseen = set(y_test_text.unique()) - set(le.classes_)
    if unseen:
        raise SystemExit(f"FATAL: test has unseen labels not in train: {sorted(unseen)}")
    y_test = le.transform(y_test_text)

    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    labels = list(range(len(le.classes_)))
    print(classification_report(
        y_test, y_pred,
        labels=labels,
        target_names=le.classes_.tolist(),
        digits=3,
        zero_division=0
    ))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion matrix label order:", le.classes_.tolist())
    print(cm)

    macro_f1 = f1_score(y_test, y_pred, average="macro", labels=labels, zero_division=0)
    print(f"[RF] macro-F1: {macro_f1:.4f}")

    joblib.dump(
        {"model": clf, "features": X_train.columns.tolist(), "label_encoder": le},
        models_dir / "baseline_rf.pkl"
    )
    print(f"[ok] saved → {(models_dir / 'baseline_rf.pkl').resolve()}")


if __name__ == "__main__":
    main()

