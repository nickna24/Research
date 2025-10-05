#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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

def accuracy_from_logits(logits, y_true):
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == y_true.squeeze(1)).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--preproc_dir", default="data/preprocessed")
    ap.add_argument("--metrics_out", default="logs/metrics.txt")
    ap.add_argument("--write_submission", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    meta = ckpt["meta"]

    X = np.load(os.path.join(args.preproc_dir, "train_proc.npy"))
    y = np.load(os.path.join(args.preproc_dir, "labels.npy"))
    val_idx = np.array(meta["val_indices"], dtype=int)

    X_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y[val_idx])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(in_dim=X.shape[1]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(X_val.to(device))
        bce = nn.BCEWithLogitsLoss()
        val_loss = bce(logits, y_val.to(device).squeeze(1)).item()
        val_acc = accuracy_from_logits(logits.cpu(), y_val.cpu())

    with open(args.metrics_out, "w") as f:
        f.write(f"val_loss: {val_loss:.6f}\n")
        f.write(f"val_acc: {val_acc:.6f}\n")
    print(f"[EVAL] val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    if args.write_submission:
        X_test = np.load(os.path.join(args.preproc_dir, "test_proc.npy"))
        test_df = pd.read_csv("data/test.csv")
        with torch.no_grad():
            preds = (torch.sigmoid(model(torch.from_numpy(X_test).to(device))) >= 0.5).float().cpu().numpy().astype(int)
        sub = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": preds.squeeze()})
        sub_path = os.path.join(os.path.dirname(args.metrics_out), "submission.csv")
        sub.to_csv(sub_path, index=False)
        print(f"[EVAL] Wrote submission -> {sub_path}")

if __name__ == "__main__":
    main()
