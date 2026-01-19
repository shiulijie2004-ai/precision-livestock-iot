#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# plotting (no GUI needed)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch.nn as nn

import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# ---------------- robust import for src.* ----------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm import CowActivityLSTM  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_cnn_input_shape(X: np.ndarray) -> np.ndarray:
    """
    Conv1d wants (N, C, T). Your X_test.npy is usually (N, T, C).
    If X is (N, T, C), transpose to (N, C, T).
    """
    if X.ndim != 3:
        raise ValueError(f"Expected 3D X, got {X.shape}")
    # common case: (N, T, C) where T > C
    if X.shape[1] > X.shape[2]:
        return np.transpose(X, (0, 2, 1))
    return X


class CNN1D(nn.Module):
    """Input: (B, C, T)"""
    def __init__(self, in_ch: int, n_classes: int, base: int = 64, dropout: float = 0.2):
        super().__init__()

        def block(cin, cout, k, p):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=k, padding=p, bias=False),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        self.backbone = nn.Sequential(
            block(in_ch, base,     k=7, p=3),
            nn.MaxPool1d(2),
            block(base, base*2,    k=5, p=2),
            nn.MaxPool1d(2),
            block(base*2, base*4,  k=3, p=1),
            nn.MaxPool1d(2),
            block(base*4, base*4,  k=3, p=1),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base*4, base*2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base*2, n_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


@torch.no_grad()
def cnn1d_predict(model_path: Path, X: np.ndarray, device: str = DEVICE, batch_size: int = 256) -> np.ndarray:
    """
    model_path: results/checkpoints/<dataset>/<split>/cnn1d_best_model.pth
    X: (N, C, T)
    """
    ckpt = torch.load(model_path, map_location=device)

    # our training script saves dict with model_state + meta
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    in_ch = int(ckpt.get("in_ch", X.shape[1])) if isinstance(ckpt, dict) else int(X.shape[1])
    n_classes = int(ckpt.get("n_classes")) if isinstance(ckpt, dict) and "n_classes" in ckpt else None
    base = int(ckpt.get("base", 64)) if isinstance(ckpt, dict) else 64
    dropout = float(ckpt.get("dropout", 0.2)) if isinstance(ckpt, dict) else 0.2

    if n_classes is None:
        raise ValueError("cnn1d checkpoint missing n_classes. Re-train with the provided trainer or save n_classes.")

    model = CNN1D(in_ch=in_ch, n_classes=n_classes, base=base, dropout=dropout).to(device)
    model.load_state_dict(state)
    model.eval()

    X_t = torch.from_numpy(X.astype(np.float32))
    preds = []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i:i+batch_size].to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(pred)
    return np.concatenate(preds)

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    out_path: Path,
    macro_f1: float | None = None,
) -> None:
    cm = np.asarray(cm, dtype=int)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)

    # --- make the colour like your example (magma) ---
    im = ax.imshow(cm, interpolation="nearest", cmap="magma", vmin=0)

    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)

    # --- bold numbers + auto black/white text like your example ---
    maxv = float(cm.max()) if cm.size else 0.0
    thresh = maxv * 0.45
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            color = "black" if v >= thresh else "white"
            ax.text(
                j, i, f"{v}",
                ha="center", va="center",
                fontsize=14, fontweight="bold",
                color=color
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Macro-F1 at bottom (like your example) ---
    if macro_f1 is not None:
        fig.text(0.5, 0.03, f"Macro-F1 = {macro_f1:.4f}", ha="center", fontsize=14)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_f1_per_class(
    f1s: np.ndarray,
    class_names: list[str],
    macro_f1: float,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(class_names))
    bars = ax.bar(x, f1s)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1", fontsize=14)
    ax.set_title(title, fontsize=18)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=12)

    # annotate each bar with F1-score=xxx
    for i, b in enumerate(bars):
        val = float(f1s[i])
        ax.text(
            b.get_x() + b.get_width() / 2,
            val + 0.02,
            f"F1-score={val:.3f}",
            ha="center",
            va="bottom",
            fontsize=12
        )

    fig.text(0.5, 0.01, f"Macro-F1 = {macro_f1:.4f}", ha="center", fontsize=16)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------- model helpers ----------------
@torch.no_grad()
def lstm_predict(model, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(DEVICE)
        logits = model(xb)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    labels = list(range(num_classes))
    return f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["japan_cow", "actbecalf", "nose_ring"], required=True)
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--ckpt-dir", default="results/checkpoints")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--split", choices=["cow", "random"], default="cow")
    ap.add_argument("--out-dir", default="results/eval")
    args = ap.parse_args()

    ds = args.dataset
    split_tag = "loco" if args.split == "cow" else "random"
    processed = Path(args.processed_dir) / ds / split_tag
    ckpt     = Path(args.ckpt_dir) / ds / split_tag
    models   = Path(args.models_dir) / ds / split_tag
    out      = Path(args.out_dir) / ds / split_tag
    out.mkdir(parents=True, exist_ok=True)


    warnings = []
    results = {"dataset": ds}

    # ---------- load test.parquet for classical models ----------
    test_parquet = processed / "test.parquet"
    test_df = None
    if test_parquet.exists():
        test_df = pd.read_parquet(test_parquet)
    else:
        warnings.append(f"[WARN] missing {test_parquet} (RF/SVM/XGB will be skipped)")

    # ============================================================
    # RF
    # ============================================================
    rf_f1 = None
    if test_df is not None:
        baseline_path = models / "baseline_rf.pkl"
        if not baseline_path.exists():
            warnings.append(f"[WARN] missing {baseline_path} (run train_baseline.py --dataset {ds})")
        else:
            obj = joblib.load(baseline_path)
            rf = obj["model"]
            feat_cols = obj.get("features", [])
            le_rf = obj["label_encoder"]

            X_test_rf = test_df.drop(columns=["label"])
            y_test_text = test_df["label"].astype(str)

            if feat_cols:
                X_test_rf = X_test_rf[feat_cols]

            unseen = set(y_test_text.unique()) - set(le_rf.classes_)
            if unseen:
                warnings.append(f"[WARN] RF unseen labels in test (skip RF): {sorted(unseen)}")
            else:
                y_test_rf = le_rf.transform(y_test_text)
                y_pred_rf = rf.predict(X_test_rf)

                labels_rf = list(range(len(le_rf.classes_)))
                rf_f1 = f1_score(y_test_rf, y_pred_rf, average="macro", labels=labels_rf, zero_division=0)
                rf_cm = confusion_matrix(y_test_rf, y_pred_rf, labels=labels_rf)
                rf_f1s = per_class_f1(y_test_rf, y_pred_rf, num_classes=len(labels_rf))

                print("\n[RF] report")
                print(classification_report(
                    y_test_rf, y_pred_rf,
                    target_names=le_rf.classes_.tolist(),
                    digits=3, zero_division=0
                ))

                plot_confusion_matrix(
                    rf_cm,
                    le_rf.classes_.tolist(),
                    title=f"RF Confusion Matrix ({ds})",
                    out_path=out / "rf_confusion_matrix.png",
                    macro_f1=float(rf_f1),
                )

                plot_f1_per_class(
                    rf_f1s,
                    le_rf.classes_.tolist(),
                    macro_f1=float(rf_f1),
                    title=f"RF F1 per Class ({ds})",
                    out_path=out / "rf_f1_per_class.png",
                )

    results["rf_macro_f1"] = None if rf_f1 is None else float(rf_f1)

    # ============================================================
    # SVM
    # ============================================================
    svm_f1 = None
    if test_df is not None:
        svm_path = models / "svm.pkl"
        if not svm_path.exists():
            warnings.append(f"[WARN] missing {svm_path} (run train_svm.py --dataset {ds})")
        else:
            try:
                obj = joblib.load(svm_path)
                svm = obj["model"]
                feat_cols = obj.get("features", [])
                le_svm = obj["label_encoder"]

                X_test = test_df.drop(columns=["label"])
                y_text = test_df["label"].astype(str)
                if feat_cols:
                    X_test = X_test[feat_cols]

                unseen = set(y_text.unique()) - set(le_svm.classes_)
                if unseen:
                    warnings.append(f"[WARN] SVM unseen labels in test (skip SVM): {sorted(unseen)}")
                else:
                    y_true = le_svm.transform(y_text)
                    y_pred = svm.predict(X_test)

                    labels_ = list(range(len(le_svm.classes_)))
                    svm_f1 = f1_score(y_true, y_pred, average="macro", labels=labels_, zero_division=0)
                    cm = confusion_matrix(y_true, y_pred, labels=labels_)
                    f1s = per_class_f1(y_true, y_pred, num_classes=len(labels_))

                    print("\n[SVM] report")
                    print(classification_report(
                        y_true, y_pred,
                        target_names=le_svm.classes_.tolist(),
                        digits=3, zero_division=0
                    ))

                    plot_confusion_matrix(
                        cm,
                        le_svm.classes_.tolist(),
                        f"SVM Confusion Matrix ({ds})",
                        out / "svm_confusion_matrix.png",
                        macro_f1=float(svm_f1),
                    )

                    plot_f1_per_class(
                        f1s,
                        le_svm.classes_.tolist(),
                        float(svm_f1),
                        f"SVM F1 per Class ({ds})",
                        out / "svm_f1_per_class.png",
                    )

            except Exception as e:
                warnings.append(f"[WARN] SVM failed: {type(e).__name__}: {e}")

    results["svm_macro_f1"] = None if svm_f1 is None else float(svm_f1)

    # ============================================================
    # XGBoost
    # ============================================================
    xgb_f1 = None
    if test_df is not None:
        xgb_path = models / "xgboost.pkl"
        if not xgb_path.exists():
            warnings.append(f"[WARN] missing {xgb_path} (run train_xgboost.py --dataset {ds})")
        else:
            try:
                import xgboost  # noqa: F401

                obj = joblib.load(xgb_path)
                xgb_model = obj["model"]
                feat_cols = obj.get("features", [])
                le_xgb = obj["label_encoder"]

                X_test = test_df.drop(columns=["label"])
                y_text = test_df["label"].astype(str)
                if feat_cols:
                    X_test = X_test[feat_cols]

                unseen = set(y_text.unique()) - set(le_xgb.classes_)
                if unseen:
                    warnings.append(f"[WARN] XGB unseen labels in test (skip XGB): {sorted(unseen)}")
                else:
                    y_true = le_xgb.transform(y_text)
                    y_pred = xgb_model.predict(X_test)

                    labels_ = list(range(len(le_xgb.classes_)))
                    xgb_f1 = f1_score(y_true, y_pred, average="macro", labels=labels_, zero_division=0)
                    cm = confusion_matrix(y_true, y_pred, labels=labels_)
                    f1s = per_class_f1(y_true, y_pred, num_classes=len(labels_))

                    print("\n[XGBoost] report")
                    print(classification_report(
                        y_true, y_pred,
                        target_names=le_xgb.classes_.tolist(),
                        digits=3, zero_division=0
                    ))

                    plot_confusion_matrix(
                        cm,
                        le_xgb.classes_.tolist(),
                        f"XGBoost Confusion Matrix ({ds})",
                        out / "xgb_confusion_matrix.png",
                        macro_f1=float(xgb_f1),
                    )

                    plot_f1_per_class(
                        f1s,
                        le_xgb.classes_.tolist(),
                        float(xgb_f1),
                        f"XGBoost F1 per Class ({ds})",
                        out / "xgb_f1_per_class.png",
                    )

            except ModuleNotFoundError:
                warnings.append("[WARN] xgboost not installed (skip XGBoost). Install inside venv: pip install xgboost")
            except Exception as e:
                warnings.append(f"[WARN] XGBoost failed: {type(e).__name__}: {e}")

    results["xgb_macro_f1"] = None if xgb_f1 is None else float(xgb_f1)

    # ============================================================
    # CNN1D
    # ============================================================
    cnn1d_f1 = None

    # reuse the SAME test npy + label encoder as LSTM pipeline
    X_path = processed / "X_test.npy"
    y_path = processed / "y_test.npy"
    le_path = ckpt / "label_encoder.joblib"
    cnn_path = ckpt / "cnn1d_best_model.pth"

    missing = [p for p in [X_path, y_path, le_path, cnn_path] if not p.exists()]
    if missing:
        warnings.append(f"[WARN] missing CNN1D files (skip CNN1D): {[str(p) for p in missing]}")
    else:
        try:
            le = joblib.load(le_path)
            X_test = np.load(X_path)
            y_true = np.load(y_path).astype(int)

            X_test = ensure_cnn_input_shape(X_test)  # -> (N, C, T)
            y_pred = cnn1d_predict(cnn_path, X_test, device=DEVICE)

            labels_ = list(range(len(le.classes_)))
            cnn1d_f1 = f1_score(y_true, y_pred, average="macro", labels=labels_, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=labels_)
            f1s = per_class_f1(y_true, y_pred, num_classes=len(labels_))

            print("\n[CNN1D] report")
            print(classification_report(
                y_true, y_pred,
                target_names=le.classes_.tolist(),
                digits=3, zero_division=0
            ))

            plot_confusion_matrix(
                cm,
                le.classes_.tolist(),
                f"CNN1D Confusion Matrix ({ds})",
                out / "cnn1d_confusion_matrix.png",
                macro_f1=float(cnn1d_f1),
           )

            plot_f1_per_class(
                f1s,
                le.classes_.tolist(),
                float(cnn1d_f1),
                f"CNN1D F1 per Class ({ds})",
                out / "cnn1d_f1_per_class.png",
            )

        except Exception as e:
            warnings.append(f"[WARN] CNN1D failed: {type(e).__name__}: {e}")

    results["cnn1d_macro_f1"] = None if cnn1d_f1 is None else float(cnn1d_f1)


    # ============================================================
    # LSTM
    # ============================================================
    lstm_f1 = None
    X_path = processed / "X_test.npy"
    y_path = processed / "y_test.npy"
    le_path = ckpt / "label_encoder.joblib"
    cfg_path = ckpt / "lstm_config.json"
    model_path = ckpt / "lstm_best_model.pth"

    missing = [p for p in [X_path, y_path, le_path, cfg_path, model_path] if not p.exists()]
    if missing:
        warnings.append(f"[WARN] missing LSTM files (skip LSTM): {[str(p) for p in missing]}")
    else:
        X_test = np.load(X_path).astype("float32")
        y_test = np.load(y_path).astype("int64")

        le = joblib.load(le_path)
        cfg = json.loads(cfg_path.read_text())
        num_classes = int(cfg.get("num_classes", len(le.classes_)))
        class_names = le.classes_.tolist()[:num_classes]

        model = CowActivityLSTM(
            input_size=int(cfg.get("input_size", 3)),
            hidden_size=int(cfg.get("hidden_size", 64)),
            num_layers=int(cfg.get("num_layers", 2)),
            num_classes=num_classes,
            dropout_prob=float(cfg.get("dropout_prob", 0.5)),
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        y_pred = lstm_predict(model, X_test, batch_size=512)

        labels_lstm = list(range(num_classes))
        lstm_f1 = f1_score(y_test, y_pred, average="macro", labels=labels_lstm, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=labels_lstm)
        f1s = per_class_f1(y_test, y_pred, num_classes=num_classes)

        print("\n[LSTM] report")
        print(classification_report(
            y_test, y_pred,
            target_names=class_names,
            digits=3, zero_division=0
        ))

        plot_confusion_matrix(
            cm,
            class_names,
            f"LSTM Confusion Matrix ({ds})",
            out / "lstm_confusion_matrix.png",
            macro_f1=float(lstm_f1),
        )

        plot_f1_per_class(
            f1s,
            class_names,
            float(lstm_f1),
            f"LSTM F1 per Class ({ds})",
            out / "lstm_f1_per_class.png",
        )

    results["lstm_macro_f1"] = None if lstm_f1 is None else float(lstm_f1)

    # ---------- Summary ----------
    print("\n=== Summary ===")
    print(f"Dataset: {ds}")
    print(f"RF      Macro F1: {results['rf_macro_f1']}")
    print(f"SVM     Macro F1: {results['svm_macro_f1']}")
    print(f"XGBoost  Macro F1: {results['xgb_macro_f1']}")
    print(f"CNN1D   Macro F1: {results.get('cnn1d_macro_f1')}")
    print(f"LSTM     Macro F1: {results['lstm_macro_f1']}")

    if warnings:
        print("\n=== Warnings (not fatal) ===")
        for w in warnings:
            print(w)

    # save metrics
    (out / "final_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\n[ok] saved → {(out / 'final_metrics.json').resolve()}")


if __name__ == "__main__":
    main()

