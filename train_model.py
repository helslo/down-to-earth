"""
Train FTIRNet on real OSSL data, using the ready-made guo_subset
benchmark targets and its predefined train/test split.

Usage:
    python train_model.py path/to/your_dataset.db
"""
import argparse
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


def compute_calibration_factor(predicted: np.ndarray, sigma: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """Global temperature-style calibration for regression uncertainty.

    We scale the model's standard deviation by a single factor so that the
    average standardized residual matches 1 in the validation set.
    """
    sigma = np.clip(sigma, eps, None)
    z = np.abs(target - predicted) / sigma
    return float(np.sqrt(np.mean(z ** 2)))


def train_and_calibrate(data_path: str, epochs: int = 100, downsample_to: int = 170, max_samples: int = None):
    # guo_subset's 5 properties that map onto the paper's target list.
    # Swap in 'CEC', 'Clay', 'P' here too/instead if that's more useful
    # for your farmer conversations -- they're in the same table.
    task_names = ["OC", "N", "pH", "Sand", "K"]
    fusion_edges = {"N": ["OC", "pH"]}  # paper's best-performing N config

    dataset_kind = "CSV" if str(data_path).lower().endswith(".csv") else "SQLite"
    print(f"Loading {dataset_kind} dataset from: {data_path}")

    X, Y, split, sample_ids = build_dataset(
        data_path, target_cols=task_names, downsample_to=downsample_to
    )
    if max_samples is not None and max_samples < len(X):
        X = X[:max_samples]
        split = split[:max_samples]
        for t in task_names:
            Y[t] = Y[t][:max_samples]
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

    n_epochs = epochs
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
        if epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
            print(
                f"epoch {epoch:3d} | loss={epoch_loss / len(train_loader):.4f} "
                f"nll={logs['nll']:.4f} kl={logs['kl']:.4f} ec={logs['ec']:.2f} "
                f"alpha={logs['alpha']:.3f}"
            )

    calibration_factors = {}
    model.eval()
    with torch.no_grad():
        for t in task_names:
            preds_all, targets_all = [], []
            sigma_all = []
            for x, y in test_loader:
                x = x.to(device)
                mean, log_var = model(x)[t]
                preds_all.append(mean.cpu().numpy())
                sigma_all.append(torch.exp(0.5 * log_var).cpu().numpy())
                targets_all.append(y[t].numpy())
            preds_all = np.concatenate(preds_all)
            sigma_all = np.concatenate(sigma_all)
            targets_all = np.concatenate(targets_all)

            preds_unscaled = y_scalers[t].inverse_transform(preds_all.reshape(-1, 1)).ravel()
            targets_unscaled = y_scalers[t].inverse_transform(targets_all.reshape(-1, 1)).ravel()
            sigma_unscaled = sigma_all * y_scalers[t].scale_[0]
            calibration_factors[t] = compute_calibration_factor(
                preds_unscaled, sigma_unscaled, targets_unscaled
            )

            ss_res = np.sum((targets_unscaled - preds_unscaled) ** 2)
            ss_tot = np.sum((targets_unscaled - targets_unscaled.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot
            rmse = np.sqrt(ss_res / len(targets_unscaled))
            avg_sigma = np.mean(sigma_unscaled * calibration_factors[t])
            print(
                f"{t}: R2={r2:.4f}  RMSE={rmse:.4f}  "
                f"avg_sigma={avg_sigma:.4f}  calibration={calibration_factors[t]:.4f}"
            )

    return model, x_scaler, y_scalers, calibration_factors, task_names


def save_checkpoint(model, x_scaler, y_scalers, calibration_factors, task_names, downsample_to: int, path: str = "model_state.pt"):
    """Save the fitted model + preprocessing state for later pure prediction."""
    torch.save(
        {
            "model_state": model.state_dict(),
            "task_names": task_names,
            "fusion_edges": model.fusion_edges,
            "downsample_to": int(downsample_to),
            "x_scaler_mean": x_scaler.mean_.copy(),
            "x_scaler_scale": x_scaler.scale_.copy(),
            "y_scalers": {t: {"mean": s.mean_.copy(), "scale": s.scale_.copy()} for t, s in y_scalers.items()},
            "calibration_factors": calibration_factors,
        },
        path,
    )


def main(data_path: str, epochs: int = 100, downsample_to: int = 170, max_samples: int = None, save_path: str = "model_state.pt"):
    model, x_scaler, y_scalers, calibration_factors, task_names = train_and_calibrate(
        data_path,
        epochs=epochs,
        downsample_to=downsample_to,
        max_samples=max_samples,
    )
    save_checkpoint(model, x_scaler, y_scalers, calibration_factors, task_names, downsample_to, path=save_path)
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the FTIR soil-property model.")
    parser.add_argument("data_path", help="Path to a CSV or SQLite dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--downsample-to", type=int, default=170, help="Spectral downsampling target")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit rows for a quick smoke test")
    parser.add_argument("--save-path", default="model_state.pt", help="Location for the saved checkpoint")
    args = parser.parse_args()

    main(
        args.data_path,
        epochs=args.epochs,
        downsample_to=args.downsample_to,
        max_samples=args.max_samples,
        save_path=args.save_path,
    )
