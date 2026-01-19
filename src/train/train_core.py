#!/usr/bin/env python3


import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# robust import when running as script
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm import CowActivityLSTM  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic-ish
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Oversample minority classes in the TRAIN loader (often improves macro-F1)."""
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y)
    counts[counts == 0] = 1
    class_w = 1.0 / counts.astype(np.float32)
    sample_w = class_w[y]
    sample_w = torch.tensor(sample_w, dtype=torch.float32)
    return WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)


@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    all_y, all_p = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu().numpy()
        all_p.append(pred)
        all_y.append(yb.numpy())
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    return float(f1_score(y, p, average="macro", zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["japan_cow", "actbecalf", "nose_ring"], required=True)
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--ckpt-dir", default="results/checkpoints")
    ap.add_argument("--split", choices=["cow", "random"], default="cow")

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    # model params (small tuning often helps)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)

    # training stability
    ap.add_argument("--grad-clip", type=float, default=1.0)
    args = ap.parse_args()

    set_seed(args.seed)

    split_tag = "loco" if args.split == "cow" else "random"
    data_dir = Path(args.processed_dir) / args.dataset / split_tag
    ckpt_dir = Path(args.ckpt_dir) / args.dataset / split_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    X_train_path = data_dir / "X_train.npy"
    y_train_path = data_dir / "y_train.npy"
    X_test_path = data_dir / "X_test.npy"
    y_test_path = data_dir / "y_test.npy"
    le_path = ckpt_dir / "label_encoder.joblib"

    missing = [str(p) for p in [X_train_path, y_train_path, X_test_path, y_test_path, le_path] if not p.exists()]
    if missing:
        raise SystemExit("FATAL: missing required file(s):\n  - " + "\n  - ".join(missing))

    X_train = np.load(X_train_path).astype(np.float32)
    y_train = np.load(y_train_path).astype(np.int64)
    X_test = np.load(X_test_path).astype(np.float32)
    y_test = np.load(y_test_path).astype(np.int64)

    le = joblib.load(le_path)
    num_classes = len(le.classes_)

    # val split INSIDE training set
    Xtr, Xval, ytr, yval = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=args.seed,
        stratify=y_train
    )

    # loaders
    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    val_ds = TensorDataset(torch.from_numpy(Xval), torch.from_numpy(yval))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    sampler = build_weighted_sampler(ytr)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # model
    model = CowActivityLSTM(
        input_size=3,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout_prob=args.dropout
    ).to(DEVICE)

    # loss + optimizer + scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val = -1.0
    best_path = ckpt_dir / "lstm_best_model.pth"
    bad = 0

    history = {"epoch": [], "train_loss": [], "val_macro_f1": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()
            total_loss += float(loss.item()) * xb.size(0)

        avg_loss = total_loss / max(1, len(train_loader.dataset))
        val_f1 = eval_model(model, val_loader)
        scheduler.step(val_f1)

        # record
        lr_now = float(optimizer.param_groups[0]["lr"])
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_macro_f1"].append(val_f1)
        history["lr"].append(lr_now)

        print(f"[epoch {epoch:02d}] dataset={args.dataset} loss={avg_loss:.4f} val_macro_f1={val_f1:.4f} lr={lr_now:.2e}")

        if val_f1 > best_val + 1e-6:
            best_val = val_f1
            bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print("[early-stop] no val improvement")
                break

    # save config (include manifest if it exists)
    manifest_path = ckpt_dir / "lstm_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    cfg: Dict = {
        "dataset": args.dataset,
        "device": DEVICE,
        "seed": args.seed,
        "input_size": 3,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_classes": num_classes,
        "dropout_prob": args.dropout,
        "classes": le.classes_.tolist(),
        "epochs_ran": int(history["epoch"][-1]) if history["epoch"] else 0,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "grad_clip": args.grad_clip,
        "data_manifest": manifest,
    }
    (ckpt_dir / "lstm_config.json").write_text(json.dumps(cfg, indent=2))
    (ckpt_dir / "lstm_history.json").write_text(json.dumps(history, indent=2))

    # final test
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_f1 = eval_model(model, test_loader)
    print(f"[final] dataset={args.dataset} test_macro_f1={test_f1:.4f}")
    print(f"[ok] saved best model → {best_path.resolve()}")


if __name__ == "__main__":
    main()
