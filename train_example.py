"""
End-to-end example training script for FTIRNet.

Out of the box this runs on synthetic random data purely so you can
confirm the pipeline runs on your machine. Replace `load_data()` with
your real spectra + lab-measured target values (see README.md, section
"Getting real data in").

Usage:
    python train_example.py
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import FTIRNet
from losses import total_loss


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


def load_data():
    """
    TODO: replace this with your real data loading.

    Must return:
      X: np.ndarray of shape (n_samples, in_features) -- your spectral
         features (see README for options on what "in_features" should
         contain if you don't have the paper's autoencoder available).
      Y: dict[task_name -> np.ndarray of shape (n_samples,)] -- lab
         measured target values for each soil property you're predicting.
    """
    rng = np.random.default_rng(0)
    n = 500
    in_features = 160
    X = rng.normal(size=(n, in_features)).astype(np.float32)
    # Synthetic targets loosely correlated with each other and with X,
    # just so the fusion edges have something non-trivial to exploit.
    base = X[:, :5].sum(axis=1)
    Y = {
        "OC": (base + rng.normal(scale=0.5, size=n)).astype(np.float32),
        "pH": (0.3 * base + rng.normal(scale=1.0, size=n) + 6.5).astype(np.float32),
        "N": (0.4 * base + 0.2 * (0.3 * base + 6.5) + rng.normal(scale=0.3, size=n)).astype(np.float32),
    }
    return X, Y


def main():
    task_names = ["OC", "N", "pH"]
    # Example science-informed edges: OC and pH both feed into N,
    # mirroring the paper's best-performing N configuration (Table II, d).
    fusion_edges = {"N": ["OC", "pH"]}

    X, Y = load_data()

    x_scaler = StandardScaler().fit(X)
    X = x_scaler.transform(X).astype(np.float32)

    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    y_scalers = {}
    Y_scaled = {}
    for t in task_names:
        scaler = StandardScaler()
        Y_scaled[t] = scaler.fit_transform(Y[t].reshape(-1, 1)).ravel().astype(np.float32)
        y_scalers[t] = scaler

    train_ds = SoilDataset(X[train_idx], {t: Y_scaled[t][train_idx] for t in task_names}, task_names)
    test_ds = SoilDataset(X[test_idx], {t: Y_scaled[t][test_idx] for t in task_names}, task_names)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # NOTE on sizing: the paper's own config (d_model=256, 3 shared +
    # 2 task-specific transformer layers -> several million parameters)
    # is tuned for their ~19k-sample KSSL dataset. Applied to a few
    # hundred samples (the synthetic smoke test here, or an early-stage
    # farmer dataset), that config is heavily overparameterized and can
    # produce noisy, non-converging training. The values below are a
    # much smaller config meant to make the *pipeline* easy to sanity
    # check; scale back up once you have thousands of real samples.
    model = FTIRNet(
        task_names,
        in_features=X.shape[1],
        d_model=64,
        nhead=4,
        shared_layers=2,
        task_layers=1,
        dim_feedforward=128,
        dropout=0.1,
        fusion_edges=fusion_edges,
        fixed_gamma=None,   # None = learned gamma; set e.g. 1.0 to fix it
    ).to(device)

    alpha_raw = torch.nn.Parameter(torch.tensor(0.5413, device=device))  # softplus(0.5413) ~= 1.0
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [alpha_raw], lr=1e-3
    )

    n_epochs = 50
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
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(
                f"epoch {epoch:3d} | loss={epoch_loss / len(train_loader):.4f} "
                f"mse={logs['mse']:.4f} kl={logs['kl']:.4f} ec={logs['ec']:.2f} "
                f"alpha={logs['alpha']:.3f}"
            )

    # ---- simple held-out evaluation (unscaled R^2 per task) ----
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
    main()
