"""
Train FTIRNet on real OSSL data, using the ready-made guo_subset
benchmark targets and its predefined train/test split.

Usage:
    python train_model.py path/to/your_dataset.db
"""
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from model import FTIRNet
from losses import total_loss
from load_data import build_dataset


class SoilDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: dict, task_names: list):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = {t: torch.tensor(Y[t], dtype=torch.float32) for t in task_names}
        self.task_names = task_names

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        y = {t: self.Y[t][idx] for t in self.task_names}
        return self.X[idx], y


def main(data_path: str):
    # guo_subset's 5 properties that map onto the paper's target list.
    # Swap in 'CEC', 'Clay', 'P' here too/instead if that's more useful
    # for your farmer conversations -- they're in the same table.
    task_names = ["OC", "N", "pH", "Sand", "K"]
    fusion_edges = {"N": ["OC", "pH"]}  # paper's best-performing N config

    dataset_kind = "CSV" if str(data_path).lower().endswith(".csv") else "SQLite"
    print(f"Loading {dataset_kind} dataset from: {data_path}")

    # NOTE on downsample_to: the full spectrum is 1701 bands. A standard
    # transformer's self-attention is O(L^2) in sequence length, so
    # d_model=256 over 1701 positions is very slow on CPU. 170 keeps
    # roughly the paper's ~10x compression ratio (they went 3x~1500 -> 160
    # via autoencoder+PCA; here it's simple bin-averaging instead, which
    # is a much cruder form of compression -- fine for an initial pass,
    # worth replacing with something smarter once this end-to-end path works).
    X, Y, split, sample_ids = build_dataset(
        data_path, target_cols=task_names, downsample_to=170
    )
    print(f"Loaded {len(X)} samples with complete {task_names} targets.")
    print(f"Split counts: {dict(zip(*np.unique(split, return_counts=True)))}")

    train_mask = split == "train"
    test_mask = split == "test"

    x_scaler = StandardScaler().fit(X[train_mask])
    X_scaled = x_scaler.transform(X).astype(np.float32)

    y_scalers = {}
    Y_scaled = {}
    for t in task_names:
        scaler = StandardScaler()
        Y_scaled[t] = np.empty_like(Y[t])
        Y_scaled[t][train_mask] = scaler.fit_transform(
            Y[t][train_mask].reshape(-1, 1)
        ).ravel()
        Y_scaled[t][test_mask] = scaler.transform(
            Y[t][test_mask].reshape(-1, 1)
        ).ravel()
        y_scalers[t] = scaler

    train_ds = SoilDataset(
        X_scaled[train_mask], {t: Y_scaled[t][train_mask] for t in task_names}, task_names
    )
    test_ds = SoilDataset(
        X_scaled[test_mask], {t: Y_scaled[t][test_mask] for t in task_names}, task_names
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model size: this is a real dataset (thousands of samples), so this
    # is closer to the paper's own config than the small synthetic-data
    # smoke test -- but still trimmed down a bit as a first pass on CPU.
    # Scale d_model/dim_feedforward up once you've confirmed this runs
    # and you're ready for a longer/GPU training run.
    model = FTIRNet(
        task_names,
        in_features=X_scaled.shape[1],
        d_model=128,
        nhead=8,
        shared_layers=2,
        task_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        fusion_edges=fusion_edges,
        fixed_gamma=None,
    ).to(device)

    alpha_raw = torch.nn.Parameter(torch.tensor(0.5413, device=device))
    optimizer = torch.optim.Adam(list(model.parameters()) + [alpha_raw], lr=1e-3)

    n_epochs = 100
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = {t: y[t].to(device) for t in task_names}
            preds = model(x)
            loss, logs = total_loss(preds, y, model, task_names, epoch, n_epochs, alpha_raw)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [alpha_raw], max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(
                f"epoch {epoch:3d} | loss={epoch_loss / len(train_loader):.4f} "
                f"mse={logs['mse']:.4f} kl={logs['kl']:.4f} ec={logs['ec']:.2f} "
                f"alpha={logs['alpha']:.3f}"
            )

    # ---- held-out evaluation on guo_subset's own 'test' split ----
    model.eval()
    with torch.no_grad():
        for t in task_names:
            preds_all, targets_all = [], []
            for x, y in test_loader:
                x = x.to(device)
                p = model(x)[t].cpu().numpy()
                preds_all.append(p)
                targets_all.append(y[t].numpy())
            preds_all = np.concatenate(preds_all)
            targets_all = np.concatenate(targets_all)

            preds_unscaled = y_scalers[t].inverse_transform(preds_all.reshape(-1, 1)).ravel()
            targets_unscaled = y_scalers[t].inverse_transform(targets_all.reshape(-1, 1)).ravel()

            ss_res = np.sum((targets_unscaled - preds_unscaled) ** 2)
            ss_tot = np.sum((targets_unscaled - targets_unscaled.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot
            rmse = np.sqrt(ss_res / len(targets_unscaled))
            print(f"{t}: R2={r2:.4f}  RMSE={rmse:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_model.py path/to/your_dataset.db OR path/to/your_dataset.csv")
        sys.exit(1)
    main(sys.argv[1])
