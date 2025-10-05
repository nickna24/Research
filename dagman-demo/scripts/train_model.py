#!/usr/bin/env python3
import argparse, os, json, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Dataset + Model
# ---------------------------
class NpDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x).squeeze(1)

# ---------------------------
# Helpers
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy_from_logits(logits, y_true):
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == y_true.squeeze(1)).float().mean().item()

# ---------------------------
# Main Training
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc_dir", default="data/preprocessed")
    ap.add_argument("--optimizer", choices=["adam","sgd"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--w_pred", type=float, default=1.0)
    ap.add_argument("--w_const", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint_path", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # --- Load preprocessed data ---
    X = np.load(os.path.join(args.preproc_dir, "train_proc.npy"))
    y = np.load(os.path.join(args.preproc_dir, "labels.npy"))
    with open(os.path.join(args.preproc_dir, "meta.json")) as f:
        meta = json.load(f)

    N = X.shape[0]
    val_frac = 0.2
    idx = np.arange(N)
    np.random.default_rng(args.seed).shuffle(idx)
    val_size = int(math.floor(N * val_frac))
    val_idx = idx[:val_size]
    tr_idx  = idx[val_size:]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    ds_tr = NpDataset(X_tr, y_tr)
    ds_val = NpDataset(X_val, y_val)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size)

    model = MLP(in_dim=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == "adam" else torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    bce = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred_loss = bce(logits, yb.squeeze(1))
            l2 = torch.tensor(0., device=device)
            if args.w_const > 0:
                l2 = sum((p**2).sum() for p in model.parameters())
            loss = args.w_pred * pred_loss + args.w_const * l2

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits, val_targets = [], []
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                val_logits.append(model(xb).cpu())
                val_targets.append(yb.cpu())
            val_logits = torch.cat(val_logits)
            val_targets = torch.cat(val_targets)
            val_loss = bce(val_logits, val_targets.squeeze(1)).item()
            val_acc = accuracy_from_logits(val_logits, val_targets)

        print(f"[E{epoch:02d}] train_loss={total_loss/len(ds_tr):.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # Save model + metadata
    ckpt = {
        "state_dict": model.state_dict(),
        "meta": {**meta, "optimizer": args.optimizer, "lr": args.lr, "w_pred": args.w_pred, "w_const": args.w_const, "val_indices": val_idx.tolist(), "best_val_acc": best_val_acc}
    }
    torch.save(ckpt, args.checkpoint_path)
    print(f"[INFO] Saved checkpoint -> {args.checkpoint_path}")

if __name__ == "__main__":
    main()
