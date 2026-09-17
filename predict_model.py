"""Pure prediction entry point for the FTIR soil model.

This script does not train. It expects a saved model checkpoint created by
train_model.py and then predicts on new rows from CSV or SQLite sources.

Usage:
    python predict_model.py model_state.pt data/features_samples.csv --output predictions.csv
"""
import csv
import sys

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from load_data import load_prediction_dataset
from model import FTIRNet


class PredictionDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def load_model_checkpoint(checkpoint_path: str):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    task_names = state["task_names"]
    fusion_edges = state["fusion_edges"]

    model = FTIRNet(
        task_names,
        in_features=state["x_scaler_mean"].shape[0],
        d_model=128,
        nhead=8,
        shared_layers=2,
        task_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        fusion_edges=fusion_edges,
        fixed_gamma=None,
    )
    model.load_state_dict(state["model_state"])
    model.eval()

    x_scaler = StandardScaler()
    x_scaler.mean_ = np.asarray(state["x_scaler_mean"], dtype=np.float32)
    x_scaler.scale_ = np.asarray(state["x_scaler_scale"], dtype=np.float32)
    x_scaler.var_ = np.square(x_scaler.scale_)

    y_scalers = {}
    for t, meta in state["y_scalers"].items():
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(meta["mean"], dtype=np.float32)
        scaler.scale_ = np.asarray(meta["scale"], dtype=np.float32)
        scaler.var_ = np.square(scaler.scale_)
        y_scalers[t] = scaler

    checkpoint_downsample = int(state.get("downsample_to", 170))
    return model, x_scaler, y_scalers, state["calibration_factors"], task_names, checkpoint_downsample


def predict_on_data(checkpoint_path: str, data_path: str, downsample_to: int = None, output_path: str = None):
    model, x_scaler, y_scalers, calibration_factors, task_names, checkpoint_downsample = load_model_checkpoint(checkpoint_path)
    if downsample_to is None:
        downsample_to = checkpoint_downsample
    X, ids = load_prediction_dataset(data_path, downsample_to=downsample_to)
    X_scaled = x_scaler.transform(X).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(PredictionDataset(X_scaled), batch_size=64, shuffle=False)

    predictions = {t: [] for t in task_names}
    sigmas = {t: [] for t in task_names}
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)
            outputs = model(x)
            for t in task_names:
                mean, log_var = outputs[t]
                predictions[t].append(mean.cpu().numpy())
                sigmas[t].append(torch.exp(0.5 * log_var).cpu().numpy())

    rows = []
    for i, sample_id in enumerate(ids):
        row = {"sample_id": sample_id}
        for t in task_names:
            pred_all = np.concatenate(predictions[t])
            sigma_all = np.concatenate(sigmas[t])
            pred_unscaled = y_scalers[t].inverse_transform(pred_all.reshape(-1, 1)).ravel()
            sigma_unscaled = sigma_all * y_scalers[t].scale_[0] * calibration_factors[t]
            p = float(pred_unscaled[i])
            s = float(sigma_unscaled[i])
            low = p - 1.96 * s
            high = p + 1.96 * s
            row[f"{t}_prediction"] = p
            row[f"{t}_sigma"] = s
            row[f"{t}_ci_low"] = low
            row[f"{t}_ci_high"] = high
        rows.append(row)

    if output_path is not None:
        fieldnames = ["sample_id"]
        for t in task_names:
            fieldnames.extend([f"{t}_prediction", f"{t}_sigma", f"{t}_ci_low", f"{t}_ci_high"])
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved predictions to {output_path}")

    for t in task_names:
        pred_all = np.concatenate(predictions[t])
        sigma_all = np.concatenate(sigmas[t])
        pred_unscaled = y_scalers[t].inverse_transform(pred_all.reshape(-1, 1)).ravel()
        sigma_unscaled = sigma_all * y_scalers[t].scale_[0] * calibration_factors[t]

        print(f"\nTask: {t}")
        for idx, (sample_id, p, s) in enumerate(zip(ids[:10], pred_unscaled[:10], sigma_unscaled[:10])):
            low = p - 1.96 * s
            high = p + 1.96 * s
            print(f"  sample {sample_id}: pred={p:.4f}, sigma={s:.4f}, 95% CI=[{low:.4f}, {high:.4f}]")


def main(checkpoint_path: str, data_path: str, downsample_to: int = None, output_path: str = None):
    predict_on_data(checkpoint_path, data_path, downsample_to=downsample_to, output_path=output_path)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 5, 7):
        print("Usage: python predict_model.py path/to/model_state.pt path/to/your_dataset.db OR path/to/your_dataset.csv [--downsample-to N] [--output path.csv]")
        sys.exit(1)

    if len(sys.argv) == 5 and sys.argv[3] == "--downsample-to":
        main(sys.argv[1], sys.argv[2], downsample_to=int(sys.argv[4]))
    elif len(sys.argv) == 5 and sys.argv[3] == "--output":
        main(sys.argv[1], sys.argv[2], output_path=sys.argv[4])
    elif len(sys.argv) == 7 and sys.argv[3] == "--downsample-to" and sys.argv[5] == "--output":
        main(sys.argv[1], sys.argv[2], downsample_to=int(sys.argv[4]), output_path=sys.argv[6])
    else:
        main(sys.argv[1], sys.argv[2])
