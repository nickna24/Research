import os, glob, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from chamferdist import ChamferDistance
from torchdiffeq import odeint


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class PCSeqDataset(Dataset):
    def __init__(self, data_dir, max_samples=None, mmap=True, normalize=False):
        pattern = os.path.join(data_dir, "**", "*.npy")
        self.files = sorted(glob.glob(pattern, recursive=True))
        if not self.files:
            raise RuntimeError(f"No files in {data_dir}")
        loader = (lambda f: np.load(f, mmap_mode='r')) if mmap else np.load
        self.seq_list = [loader(f) for f in self.files]  # (T,N,3)

        self.idx_map = [(i, t) for i, seq in enumerate(self.seq_list) for t in range(len(seq)-1)]
        if max_samples is not None:
            self.idx_map = self.idx_map[:max(1, int(max_samples))]

        Ns = {arr.shape[1] for arr in self.seq_list}
        if len(Ns) != 1:
            raise RuntimeError(f"Number of points changes: {Ns}")
        self.N = next(iter(Ns))
        self.normalize = normalize

    def __len__(self): return len(self.idx_map)

    def _center_scale(self, x):
        x = x - x.mean(dim=1, keepdim=True)
        r = x.norm(dim=0).max().clamp_min(1e-12)
        return x / r

    def __getitem__(self, idx):
        i, t = self.idx_map[idx]
        arr = self.seq_list[i]           
        x_t    = torch.tensor(arr[t].T,    dtype=torch.float32)  
        x_next = torch.tensor(arr[t+1].T,  dtype=torch.float32)  
        if self.normalize:
            x_t    = self._center_scale(x_t)
            x_next = self._center_scale(x_next)
        return x_t, x_next


class GatedConv1d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.h = nn.Conv1d(in_ch, out_ch, 1)
        self.g = nn.Conv1d(in_ch, out_ch, 1)
    def forward(self, x):
        return self.h(x) * torch.sigmoid(self.g(x))

class ODEField(nn.Module):
    def __init__(self, hidden=128, vel_scale=0.5):
        super().__init__()
        self.vel_scale = vel_scale
        self.net = nn.Sequential(
            GatedConv1d(3+1, hidden), nn.ReLU(),
            GatedConv1d(hidden, hidden), nn.ReLU(),
            nn.Conv1d(hidden, 3, 1)
        )
    def forward(self, t, x):
        B, _, N = x.shape
        t_chan = t.view(1,1,1).expand(B, 1, N)
        v = self.net(torch.cat([x, t_chan], dim=1))
        return torch.tanh(v) * self.vel_scale

class DataNODE(nn.Module):
    def __init__(self, field: ODEField, T: float=1.0, solver='rk4', atol=1e-5, rtol=1e-5):
        super().__init__()
        self.field = field
        self.T = T; self.solver = solver; self.atol = atol; self.rtol = rtol
    def forward(self, x0):
        times = torch.tensor([0., self.T], device=x0.device)
        xt = odeint(self.field, x0, times, method=self.solver, atol=self.atol, rtol=self.rtol)
        return xt[-1]

def train(args):
    set_seed(args.seed)
    device = torch.device("cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print("Device: CPU")

    ds = PCSeqDataset(args.data_dir, max_samples=args.max_samples, mmap=True)
    train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=0)

    field = ODEField(hidden=args.hidden, vel_scale=args.vel_scale).to(device)
    model = DataNODE(field, T=args.T, solver='rk4').to(device)

    cham = ChamferDistance()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tr_hist = []

    for ep in range(1, args.epochs+1):
        model.train()
        tot = 0.0
        for x_t, x_t1 in train_loader:
            x_t, x_t1 = x_t.to(device), x_t1.to(device)
            opt.zero_grad()
            x_pred = model(x_t)                        # (B,3,N)
            pc_pred = x_pred.permute(0, 2, 1).contiguous()  # (B,N,3)
            pc_gt   = x_t1.permute(0, 2, 1).contiguous()
            loss = cham(pc_pred, pc_gt) + cham(pc_gt, pc_pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * x_t.size(0)

        tr = tot / len(train_loader.dataset)
        tr_hist.append(tr)
        print(f"{ep}/{args.epochs}  train {tr:.4f}")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "forecast.pth"))
    np.save(os.path.join(args.output_dir, "train_losses.npy"), np.array(tr_hist))
    print("saved to", args.output_dir)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--batch-size",  type=int, default=8)

    p.add_argument("--epochs",      type=int, default=40)
    p.add_argument("--lr",          type=float, default=1e-3)

    p.add_argument("--hidden",      type=int, default=256)
    p.add_argument("--vel-scale",   type=float, default=0.5)
    p.add_argument("--T",           type=float, default=1.0)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
