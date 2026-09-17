"""Simple prediction entry point for the FTIR soil model.

This script is useful when you do not have a saved checkpoint yet. It trains
on the provided dataset in memory, calibrates the uncertainty on the validation
split, and then predicts on the full file (or a target dataset) with mean and
standard deviation outputs.

Usage:
    python predict_model.py data/features_samples.csv
"""
import sys

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from load_data import build_dataset
from model import FTIRNet
from losses import total_loss
from train_model import SoilDataset, compute_calibration_factor, train_and_calibrate


def predict_on_data(data_path: str):
    model, x_scaler, y_scalers, calibration_factors, task_names = train_and_calibrate(data_path)
    X, Y, split, sample_ids = build_dataset(data_path, target_cols=task_names, downsample_to=170)

    X_scaled = x_scaler.transform(X).astype(np.float32)
    dataset = SoilDataset(X_scaled, {t: Y[t] for t in task_names}, task_names)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = {t: [] for t in task_names}
    sigmas = {t: [] for t in task_names}

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            outputs = model(x.to(device))
            for t in task_names:
                mean, log_var = outputs[t]
                preds = mean.cpu().numpy()
                sigma = np.exp(0.5 * log_var.cpu().numpy())
                predictions[t].append(preds)
                sigmas[t].append(sigma)

    for t in task_names:
        pred_all = np.concatenate(predictions[t])
        sigma_all = np.concatenate(sigmas[t])
        pred_unscaled = y_scalers[t].inverse_transform(pred_all.reshape(-1, 1)).ravel()
        sigma_unscaled = sigma_all * y_scalers[t].scale_[0] * calibration_factors[t]

        print(f"\nTask: {t}")
        for idx, (p, s) in enumerate(zip(pred_unscaled[:10], sigma_unscaled[:10])):
            low = p - 1.96 * s
            high = p + 1.96 * s
            print(f"  sample {idx}: pred={p:.4f}, sigma={s:.4f}, 95% CI=[{low:.4f}, {high:.4f}]")


def main(data_path: str):
    predict_on_data(data_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_model.py path/to/your_dataset.db OR path/to/your_dataset.csv")
        sys.exit(1)
    main(sys.argv[1])
