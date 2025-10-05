import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from chamferdist import ChamferDistance


class PointCloudAE(nn.Module):
    def __init__(self, point_size, latent_size):
        super(PointCloudAE, self).__init__()
        self.latent_size = latent_size
        self.point_size = point_size

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, self.latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)

        self.dec1 = nn.Linear(self.latent_size, 256)
        self.dec2 = nn.Linear(256, 256)
        self.dec3 = nn.Linear(256, self.point_size * 3)

    def encoder(self, x):  
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]   
        x = x.view(-1, self.latent_size)       
        return x

    def decoder(self, z):  
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, 3)  

    def forward(self, x):  
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec


class PCSeqDataset(Dataset):
    def __init__(self, data_dir, num_points=None):
        pattern = os.path.join(data_dir, "**", "*.npy")
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            raise RuntimeError(f"No .npy in {data_dir}")
        self.seq_list = [np.load(f) for f in files]
        self.idx_map = [
            (i, t)
            for i, seq in enumerate(self.seq_list)
            for t in range(len(seq) - 1)
        ]
        self.N = num_points  

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        i, t = self.idx_map[idx]
        seq = self.seq_list[i]       
        x_t = seq[t]                 
        x_next = seq[t + 1]          

        # -> (3,N)
        x_t = torch.tensor(x_t.T, dtype=torch.float32)
        x_next = torch.tensor(x_next.T, dtype=torch.float32)

        # Zentrieren
        cen_t = x_t.mean(dim=1, keepdim=True)
        cen_n = x_next.mean(dim=1, keepdim=True)
        x_t = x_t - cen_t
        x_next = x_next - cen_n

        # Normieren auf max-Abstand = 1
        furthest_t = x_t.norm(dim=0).max().clamp_min(1e-12)
        furthest_n = x_next.norm(dim=0).max().clamp_min(1e-12)
        x_t = x_t / furthest_t
        x_next = x_next / furthest_n

        return x_t, x_next  # (3,N), (3,N)



def to_chamfer(x):  
    return x.permute(0, 2, 1).contiguous()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/")
    p.add_argument("--output-dir", type=str, default="./output")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--ae-epochs", type=int, default=80)
    p.add_argument("--latent-size", type=int, default=128)
    p.add_argument("--lr-ae", type=float, default=5e-4)
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--flow-epochs", type=int, default=50)
    p.add_argument("--lr-flow", type=float, default=1e-3)
    p.add_argument("--forecast-pre-epochs", type=int, default=20)
    return p.parse_args()

#Main
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")

    ds = PCSeqDataset(args.data_dir)
    n_test = max(1, int(0.1 * len(ds)))
    n_train = len(ds) - n_test
    train_ds, test_ds = random_split(ds, [n_train, n_test])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    sample_x_t, _ = next(iter(train_loader))
    point_size = sample_x_t.shape[-1]

    AE = PointCloudAE(point_size, args.latent_size).to(device)
    opt_ae = optim.Adam(AE.parameters(), lr=args.lr_ae)
    cham_d = ChamferDistance()


    # Phase 1: Autoencoder-Pretraining 
    ae_train_losses, ae_test_losses = [], []
    for epoch in range(1, args.ae_epochs + 1):
        AE.train()
        run_loss = 0.0
        for x_t, _ in train_loader:  
            x_t = x_t.to(device)   
            opt_ae.zero_grad()
            x_rec = AE(x_t)   
            loss = cham_d(to_chamfer(x_t), x_rec)
            loss.backward()
            opt_ae.step()
            run_loss += loss.item() * x_t.size(0)

        AE.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_t, _ in test_loader:
                x_t = x_t.to(device)
                x_rec = AE(x_t)
                loss = cham_d(to_chamfer(x_t), x_rec)
                test_loss += loss.item() * x_t.size(0)

        ae_train_losses.append(run_loss / len(train_loader.dataset))
        ae_test_losses.append(test_loss / len(test_loader.dataset))
        print(f"[AE] {epoch}/{args.ae_epochs}  "
              f"Train {ae_train_losses[-1]:.6f}  Test {ae_test_losses[-1]:.6f}")

     
    torch.save(AE.state_dict(), os.path.join(args.output_dir, "ae.pth"))
    np.save(os.path.join(args.output_dir, "ae_train_losses.npy"), np.array(ae_train_losses))
    np.save(os.path.join(args.output_dir, "ae_test_losses.npy"), np.array(ae_test_losses))
    print("Phase 1 done!")


    # Phase 2: Forecaster-Pretraining
    forecaster = nn.Sequential(
        nn.Linear(args.latent_size, args.latent_size),
        nn.ReLU(),
        nn.Linear(args.latent_size, args.latent_size)
    ).to(device)
    opt_fore = optim.Adam(forecaster.parameters(), lr=args.lr_flow)

    # AE einfrieren
    for p in AE.parameters():
        p.requires_grad = False
    AE.eval()

    fore_train_losses, fore_test_losses = [], []
    for epoch in range(1, args.forecast_pre_epochs + 1):
        forecaster.train()
        run_loss = 0.0
        for x_t, x_t1 in train_loader:
            x_t, x_t1 = x_t.to(device), x_t1.to(device)
            opt_fore.zero_grad()
            with torch.no_grad():
                z_t = AE.encoder(x_t)
                z_true = AE.encoder(x_t1)
            z_pred = forecaster(z_t)
            loss = F.mse_loss(z_pred, z_true)
            loss.backward()
            opt_fore.step()
            run_loss += loss.item() * x_t.size(0)

        # Eval
        forecaster.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_t, x_t1 in test_loader:
                x_t, x_t1 = x_t.to(device), x_t1.to(device)
                z_t = AE.encoder(x_t)
                z_true = AE.encoder(x_t1)
                z_pred = forecaster(z_t)
                test_loss += F.mse_loss(z_pred, z_true).item() * x_t.size(0)

        fore_train_losses.append(run_loss / len(train_loader.dataset))
        fore_test_losses.append(test_loss / len(test_loader.dataset))
        print(f"[FORE] {epoch}/{args.forecast_pre_epochs}  "
              f"Train {fore_train_losses[-1]:.6f}  Test {fore_test_losses[-1]:.6f}")

    np.save(os.path.join(args.output_dir, "fore_train_losses.npy"), np.array(fore_train_losses))
    np.save(os.path.join(args.output_dir, "fore_test_losses.npy"), np.array(fore_test_losses))
    print("Phase 2 done!")



    # Phase 3: Joint-Training
    for p in AE.parameters():
        p.requires_grad = True

    opt_joint = optim.Adam(list(AE.parameters()) + list(forecaster.parameters()), lr=args.lr_flow)

    w_latent, w_cloud = 5.0, 1.0

    jt_train_total, jt_train_latent, jt_train_cloud = [], [], []
    jt_test_total, jt_test_latent, jt_test_cloud = [], [], []

    for epoch in range(1, args.flow_epochs + 1):
        AE.train()
        forecaster.train()
        sum_total = sum_lat = sum_cloud = 0.0

        for x_t, x_t1 in train_loader:
            x_t, x_t1 = x_t.to(device), x_t1.to(device)
            opt_joint.zero_grad()

            z_t = AE.encoder(x_t)      
            z_pred = forecaster(z_t)   
            with torch.no_grad():
                z_true = AE.encoder(x_t1)
            loss_lat = F.mse_loss(z_pred, z_true)

            x_pred = AE.decoder(z_pred)              
            loss_cf = cham_d(to_chamfer(x_t1), x_pred)

            loss = w_latent * loss_lat + w_cloud * loss_cf
            loss.backward()
            opt_joint.step()

            bsz = x_t.size(0)
            sum_lat += loss_lat.item() * bsz
            sum_cloud += loss_cf.item() * bsz
            sum_total += loss.item() * bsz

        n_train_samples = len(train_loader.dataset)
        jt_train_latent.append(sum_lat / n_train_samples)
        jt_train_cloud.append(sum_cloud / n_train_samples)
        jt_train_total.append(sum_total / n_train_samples)

        AE.eval()
        forecaster.eval()
        sum_total = sum_lat = sum_cloud = 0.0
        with torch.no_grad():
            for x_t, x_t1 in test_loader:
                x_t, x_t1 = x_t.to(device), x_t1.to(device)

                z_t = AE.encoder(x_t)
                z_pred = forecaster(z_t)
                z_true = AE.encoder(x_t1)
                loss_lat = F.mse_loss(z_pred, z_true)

                x_pred = AE.decoder(z_pred)
                loss_cf = cham_d(to_chamfer(x_t1), x_pred)

                loss = w_latent * loss_lat + w_cloud * loss_cf

                bsz = x_t.size(0)
                sum_lat += loss_lat.item() * bsz
                sum_cloud += loss_cf.item() * bsz
                sum_total += loss.item() * bsz

        n_test_samples = len(test_loader.dataset)
        jt_test_latent.append(sum_lat / n_test_samples)
        jt_test_cloud.append(sum_cloud / n_test_samples)
        jt_test_total.append(sum_total / n_test_samples)

        print(
            f"[JOINT] {epoch}/{args.flow_epochs}  "
            f"Train total={jt_train_total[-1]:.6f} "
            f"(lat={jt_train_latent[-1]:.6f}, cloud={jt_train_cloud[-1]:.6f})  "
            f"Test total={jt_test_total[-1]:.6f} "
            f"(lat={jt_test_latent[-1]:.6f}, cloud={jt_test_cloud[-1]:.6f})"
        )


    torch.save({
        "ae": AE.state_dict(),
        "forecaster": forecaster.state_dict(),
        "args": vars(args),
    }, os.path.join(args.output_dir, "ae_forecast.pth"))


    np.save(os.path.join(args.output_dir, "ae_train_losses.npy"), np.array(ae_train_losses))
    np.save(os.path.join(args.output_dir, "ae_test_losses.npy"), np.array(ae_test_losses))

    np.save(os.path.join(args.output_dir, "fore_train_losses.npy"), np.array(fore_train_losses))
    np.save(os.path.join(args.output_dir, "fore_test_losses.npy"), np.array(fore_test_losses))

    np.save(os.path.join(args.output_dir, "joint_train_total.npy"), np.array(jt_train_total))
    np.save(os.path.join(args.output_dir, "joint_test_total.npy"), np.array(jt_test_total))
    np.save(os.path.join(args.output_dir, "joint_train_latent.npy"), np.array(jt_train_latent))
    np.save(os.path.join(args.output_dir, "joint_test_latent.npy"), np.array(jt_test_latent))
    np.save(os.path.join(args.output_dir, "joint_train_cloud.npy"), np.array(jt_train_cloud))
    np.save(os.path.join(args.output_dir, "joint_test_cloud.npy"), np.array(jt_test_cloud))

    print("Training Done!")


if __name__ == "__main__":
    main()
