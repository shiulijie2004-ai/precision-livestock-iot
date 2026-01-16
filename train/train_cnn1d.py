
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report


# ----------------------------
# Dataset
# ----------------------------
class NpyWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ----------------------------
# Model: 1D CNN
# ----------------------------
class CNN1D(nn.Module):
    """
    Input: (B, C, T)
    """
    def __init__(self, in_ch: int, n_classes: int, base: int = 64, dropout: float = 0.2):
        super().__init__()

        def block(cin, cout, k=7, s=1, p=3):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        self.backbone = nn.Sequential(
            block(in_ch, base, k=7, s=1, p=3),
            nn.MaxPool1d(kernel_size=2),

            block(base, base * 2, k=5, s=1, p=2),
            nn.MaxPool1d(kernel_size=2),

            block(base * 2, base * 4, k=3, s=1, p=1),
            nn.MaxPool1d(kernel_size=2),

            block(base * 4, base * 4, k=3, s=1, p=1),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # -> (B, base*4, 1)
            nn.Flatten(),              # -> (B, base*4)
            nn.Linear(base * 4, base * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base * 2, n_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


# ----------------------------
# Helpers
# ----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y = []
    all_p = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1)
        all_y.append(yb.cpu().numpy())
        all_p.append(pred.cpu().numpy())
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    acc = accuracy_score(y, p)
    macro_f1 = f1_score(y, p, average="macro", zero_division=0)
    return acc, macro_f1, y, p


def ensure_cnn_input_shape(X: np.ndarray) -> np.ndarray:
    """
    Make sure X is (N, C, T) for Conv1d.
    Accepts:
      - (N, T, C)  -> transpose to (N, C, T)
      - (N, C, T)  -> keep
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X to be 3D (N,T,C) or (N,C,T), got shape {X.shape}")
    n, a, b = X.shape
    # Heuristic: if "channels" dimension is small (<=64) and "time" bigger, assume (N,T,C)
    if b <= 64 and a > b:
        # (N, T, C) -> (N, C, T)
        return np.transpose(X, (0, 2, 1))
    else:
        # already (N, C, T) or ambiguous; keep
        return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", choices=["cow", "random"], default="cow")

    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--ckpt-dir", default="results/checkpoints")

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--base", type=int, default=64)

    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()
    set_seed(args.seed)

    split_tag = "loco" if args.split == "cow" else "random"

    data_dir = Path(args.processed_dir) / args.dataset / split_tag
    ckpt_dir = Path(args.ckpt_dir) / args.dataset / split_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load npy
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_test  = np.load(data_dir / "X_test.npy")
    y_test  = np.load(data_dir / "y_test.npy")

    # Ensure Conv1d shape
    X_train = ensure_cnn_input_shape(X_train)
    X_test  = ensure_cnn_input_shape(X_test)

    n_classes = int(np.max(y_train)) + 1
    in_ch = X_train.shape[1]  # (N, C, T)

    # Split train/val (保持你的 train_core 思路：在 train 内部分 val，不动 test)
    idx = np.arange(len(y_train))
    idx_tr, idx_va = train_test_split(
        idx,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_train
    )
    tr_ds = NpyWindowDataset(X_train[idx_tr], y_train[idx_tr])
    va_ds = NpyWindowDataset(X_train[idx_va], y_train[idx_va])
    te_ds = NpyWindowDataset(X_test, y_test)

    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    te_ld = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device(args.device)
    model = CNN1D(in_ch=in_ch, n_classes=n_classes, base=args.base, dropout=args.dropout).to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_macro_f1 = -1.0
    best_path = ckpt_dir / "cnn1d_best_model.pth"
    log_path = ckpt_dir / "cnn1d_train_log.json"

    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0

        for xb, yb in tr_ld:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            running += float(loss.item()) * len(yb)
            n_seen += len(yb)

        tr_loss = running / max(n_seen, 1)

        va_acc, va_f1, _, _ = evaluate(model, va_ld, device)

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_acc": va_acc,
            "val_macro_f1": va_f1,
        }
        history.append(row)
        print(f"[epoch {epoch:03d}] train_loss={tr_loss:.4f}  val_acc={va_acc:.4f}  val_macro_f1={va_f1:.4f}")

        # Early stopping on val macro-F1
        if va_f1 > best_macro_f1 + 1e-6:
            best_macro_f1 = va_f1
            patience_left = args.patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "in_ch": in_ch,
                    "n_classes": n_classes,
                    "base": args.base,
                    "dropout": args.dropout,
                    "split": args.split,
                    "dataset": args.dataset,
                },
                best_path
            )
            print(f"[ok] saved best -> {best_path} (val_macro_f1={best_macro_f1:.4f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("[stop] early stopping")
                break

    # Save log
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "best_val_macro_f1": best_macro_f1,
                "history": history,
            },
            f,
            indent=2
        )
    print(f"[ok] saved log -> {log_path}")

    # Final test using best model
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    te_acc, te_f1, y_true, y_pred = evaluate(model, te_ld, device)

    print("\n=== Test Summary (CNN1D) ===")
    print(f"Test Accuracy : {te_acc:.4f}")
    print(f"Test Macro-F1 : {te_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


if __name__ == "__main__":
    main()

