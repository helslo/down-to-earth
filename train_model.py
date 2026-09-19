"""
Train FTIRNet on real OSSL data, using the ready-made guo_subset
benchmark targets and its predefined train/test split. A validation set
is carved out of the 'train' rows for per-epoch monitoring and for
uncertainty calibration; the 'test' rows are only used for final metrics.

Each run writes into its own folder, models/<timestamp>[_<run-name>]/,
containing model.pt, history.csv, test_metrics.csv and metadata.json.

Usage:
    python train_model.py path/to/your_dataset.db [--run-name no-fusion]
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
import numpy as np
import sklearn
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from model import FTIRNet
from losses import total_loss
from load_data import build_dataset, CSV_TARGET_ALIASES


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


def carve_validation_split(split: np.ndarray, val_fraction: float, seed: int) -> np.ndarray:
    """Relabel a random subset of 'train' rows as 'val'.

    val_fraction is relative to the whole dataset, so with the predefined
    ~80/20 train/test split, val_fraction=0.1 gives ~70/10/20
    train/val/test. The 'test' rows are never touched, keeping results
    comparable to the Guo et al. 2025 benchmark split.
    """
    split = split.astype(object)
    train_idx = np.flatnonzero(split == "train")
    n_val = int(round(val_fraction * len(split)))
    if not 0 < n_val < len(train_idx):
        raise ValueError(
            f"val_fraction={val_fraction} asks for {n_val} validation rows, "
            f"but only {len(train_idx)} train rows are available."
        )
    rng = np.random.default_rng(seed)
    split[rng.choice(train_idx, size=n_val, replace=False)] = "val"
    return split


def evaluate_loss(model, loader, device, task_names, epoch, n_epochs, alpha_raw) -> dict:
    """Batch-averaged loss and NLL on a held-out loader (no gradient updates)."""
    model.eval()
    sums = {"loss": 0.0, "nll": 0.0}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = {t: y[t].to(device) for t in task_names}
            loss, logs = total_loss(model(x), y, model, task_names, epoch, n_epochs, alpha_raw)
            sums["loss"] += loss.item()
            sums["nll"] += logs["nll"]
    return {k: v / len(loader) for k, v in sums.items()}


def predict_unscaled(model, loader, device, task, y_scaler):
    """Return (mean, sigma, target) in original units for one task."""
    model.eval()
    preds_all, sigma_all, targets_all = [], [], []
    with torch.no_grad():
        for x, y in loader:
            mean, log_var = model(x.to(device))[task]
            preds_all.append(mean.cpu().numpy())
            sigma_all.append(torch.exp(0.5 * log_var).cpu().numpy())
            targets_all.append(y[task].numpy())
    preds = y_scaler.inverse_transform(np.concatenate(preds_all).reshape(-1, 1)).ravel()
    targets = y_scaler.inverse_transform(np.concatenate(targets_all).reshape(-1, 1)).ravel()
    sigma = np.concatenate(sigma_all) * y_scaler.scale_[0]
    return preds, sigma, targets


def train_and_calibrate(data_path: str, epochs: int = 100, downsample_to: int = 170, max_samples: int = None,
                        val_fraction: float = 0.1, split_seed: int = 42):
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
    n_usable = len(X)
    if max_samples is not None and max_samples < len(X):
        X = X[:max_samples]
        split = split[:max_samples]
        for t in task_names:
            Y[t] = Y[t][:max_samples]
    print(f"Loaded {len(X)} samples with complete {task_names} targets.")
    split = carve_validation_split(split, val_fraction, split_seed)
    counts = {s: int(np.sum(split == s)) for s in ("train", "val", "test")}
    print("Split counts: " + ", ".join(
        f"{s}={n} ({n / len(split):.0%})" for s, n in counts.items()
    ))

    masks = {s: split == s for s in ("train", "val", "test")}

    # Scalers are fit on the training rows only, then applied everywhere.
    x_scaler = StandardScaler().fit(X[masks["train"]])
    X_scaled = x_scaler.transform(X).astype(np.float32)

    y_scalers = {}
    Y_scaled = {}
    for t in task_names:
        scaler = StandardScaler().fit(Y[t][masks["train"]].reshape(-1, 1))
        Y_scaled[t] = scaler.transform(Y[t].reshape(-1, 1)).ravel().astype(np.float32)
        y_scalers[t] = scaler

    datasets = {
        s: SoilDataset(X_scaled[m], {t: Y_scaled[t][m] for t in task_names}, task_names)
        for s, m in masks.items()
    }
    batch_size = 64
    lr = 1e-3
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size)

    # Prefer an NVIDIA GPU, then Apple Silicon's GPU (MPS), then CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model_config = {
        "in_features": int(X_scaled.shape[1]),
        "d_model": 128,
        "nhead": 8,
        "shared_layers": 2,
        "task_layers": 2,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "fixed_gamma": None,
    }
    model = FTIRNet(task_names, fusion_edges=fusion_edges, **model_config).to(device)

    alpha_init_raw = 0.5413
    grad_clip = 1.0
    alpha_raw = torch.nn.Parameter(torch.tensor(alpha_init_raw, device=device))
    optimizer = torch.optim.Adam(list(model.parameters()) + [alpha_raw], lr=lr)

    n_epochs = epochs
    history = []
    for epoch in range(n_epochs):
        model.train()
        sums = {"loss": 0.0, "nll": 0.0, "kl": 0.0, "alpha": 0.0}
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{n_epochs}", leave=False)
        for x, y in progress:
            x = x.to(device)
            y = {t: y[t].to(device) for t in task_names}
            preds = model(x)
            loss, logs = total_loss(preds, y, model, task_names, epoch, n_epochs, alpha_raw)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [alpha_raw], max_norm=grad_clip)
            optimizer.step()
            sums["loss"] += loss.item()
            for k in ("nll", "kl", "alpha"):
                sums[k] += logs[k]
            progress.set_postfix(loss=f"{loss.item():.4f}")

        # Batch-averaged training statistics for this epoch, plus the same
        # loss/NLL on the validation set for train-vs-val curves.
        record = {"epoch": epoch, **{k: v / len(train_loader) for k, v in sums.items()},
                  "ec": logs["ec"], "lambda_kl": logs["lambda_kl"]}
        val_stats = evaluate_loss(model, val_loader, device, task_names, epoch, n_epochs, alpha_raw)
        record.update({f"val_{k}": v for k, v in val_stats.items()})
        history.append(record)
        tqdm.write(
            f"epoch {epoch:3d} | loss={record['loss']:.4f} "
            f"nll={record['nll']:.4f} kl={record['kl']:.4f} ec={record['ec']:.2f} "
            f"alpha={record['alpha']:.3f} | val_loss={record['val_loss']:.4f} "
            f"val_nll={record['val_nll']:.4f}"
        )

    # Calibrate uncertainty on the validation set, then report on the
    # untouched test set so the reported numbers are not tuned on it.
    calibration_factors = {}
    test_metrics = []
    print("Test-set results (calibration fitted on validation set):")
    for t in task_names:
        val_pred, val_sigma, val_target = predict_unscaled(model, val_loader, device, t, y_scalers[t])
        calibration_factors[t] = compute_calibration_factor(val_pred, val_sigma, val_target)

        preds, sigma, targets = predict_unscaled(model, test_loader, device, t, y_scalers[t])
        sigma_cal = sigma * calibration_factors[t]
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(ss_res / len(targets))
        # ~1.0 means the calibrated sigmas match the test-set errors.
        z_rms = compute_calibration_factor(preds, sigma_cal, targets)
        test_metrics.append({
            "task": t,
            "n_test": len(targets),
            "r2": float(r2),
            "rmse": float(rmse),
            "avg_sigma": float(np.mean(sigma_cal)),
            "calibration": calibration_factors[t],
            "test_z_rms": z_rms,
        })
        print(
            f"{t}: R2={r2:.4f}  RMSE={rmse:.4f}  "
            f"avg_sigma={np.mean(sigma_cal):.4f}  calibration={calibration_factors[t]:.4f}  "
            f"test_z_rms={z_rms:.4f}"
        )

    # Details only known inside training, merged into the run's metadata.json.
    is_csv = str(data_path).lower().endswith(".csv")
    best = min(history, key=lambda r: r["val_nll"])
    run_info = {
        "data": {
            "n_samples_usable": n_usable,  # rows with complete features + targets
            "n_samples_used": len(X),      # after --max-samples
            "n_features": int(X.shape[1]),
            "split_counts": counts,
            "split_source": "predefined train/test; val carved from train",
            # task name -> column it was read from in the dataset
            "target_columns": {t: CSV_TARGET_ALIASES.get(t, t) if is_csv else t for t in task_names},
            "target_stats_train": {
                t: {"mean": float(y_scalers[t].mean_[0]), "std": float(y_scalers[t].scale_[0])}
                for t in task_names
            },
        },
        "model": {
            "task_names": task_names,
            "fusion_edges": fusion_edges,
            "model_config": model_config,
            "n_parameters": sum(p.numel() for p in model.parameters()),
        },
        "training": {
            "device": str(device),
            "epochs_completed": len(history),
            "batch_size": batch_size,
            "optimizer": "Adam",
            "lr": lr,
            "grad_clip_max_norm": grad_clip,
            "alpha_init_raw": alpha_init_raw,
        },
        "results": {
            "final_epoch": {k: history[-1][k] for k in ("epoch", "nll", "val_nll")},
            "best_val_epoch": {k: best[k] for k in ("epoch", "nll", "val_nll")},
            "calibration_factors": calibration_factors,
            "test_metrics": {m["task"]: {k: v for k, v in m.items() if k != "task"} for m in test_metrics},
        },
    }
    return model, x_scaler, y_scalers, calibration_factors, task_names, history, test_metrics, run_info


def save_csv(rows: list, path: str):
    """Write a list of same-keyed dicts (e.g. per-epoch stats) to a CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_run_dir(output_dir: str, run_name: str, started: datetime) -> str:
    """Create models/<YYYY-MM-DD_HHMMSS>[_<run_name>]/, refusing to reuse one."""
    name = started.strftime("%Y-%m-%d_%H%M%S")
    if run_name:
        # Keep folder names shell- and filesystem-friendly.
        name += "_" + re.sub(r"[^A-Za-z0-9._-]+", "-", run_name).strip("-")
    path = os.path.join(output_dir, name)
    os.makedirs(path)  # raises FileExistsError instead of overwriting a run
    return path


def git_state() -> dict:
    """Commit hash and whether there were uncommitted changes, if available."""
    repo = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                                text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True,
                                text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status.strip())}


def data_file_info(path: str) -> dict:
    """Identify the exact dataset file used: location, size, mtime and SHA-256."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    stat = os.stat(path)
    return {
        "path": path,
        "abs_path": os.path.abspath(path),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "sha256": sha.hexdigest(),
    }


def environment_info() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        **git_state(),
    }


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_checkpoint(model, x_scaler, y_scalers, calibration_factors, task_names, downsample_to: int,
                    path: str, model_config: dict = None):
    """Save the fitted model + preprocessing state for later pure prediction."""
    torch.save(
        {
            "model_state": model.state_dict(),
            "task_names": task_names,
            "fusion_edges": model.fusion_edges,
            "model_config": model_config,
            "downsample_to": int(downsample_to),
            "x_scaler_mean": x_scaler.mean_.copy(),
            "x_scaler_scale": x_scaler.scale_.copy(),
            "y_scalers": {t: {"mean": s.mean_.copy(), "scale": s.scale_.copy()} for t, s in y_scalers.items()},
            "calibration_factors": calibration_factors,
        },
        path,
    )


def main(data_path: str, epochs: int = 100, downsample_to: int = 170, max_samples: int = None,
         output_dir: str = "models", run_name: str = None,
         val_fraction: float = 0.1, split_seed: int = 42):
    started = datetime.now()
    data_file = data_file_info(data_path)  # fails early on a bad path, before any folder exists
    run_dir = make_run_dir(output_dir, run_name, started)
    print(f"Run folder: {run_dir}")

    # Written before training starts so even a crashed run records its settings;
    # "status" tells finished runs apart from failed or interrupted ones.
    metadata = {
        "run_name": run_name,
        "status": "running",
        "command": " ".join(sys.argv),
        "timing": {
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": None,
            "duration_seconds": None,
        },
        "args": {
            "data_path": data_path,
            "epochs": epochs,
            "downsample_to": downsample_to,
            "max_samples": max_samples,
            "val_fraction": val_fraction,
            "split_seed": split_seed,
        },
        "data": {"file": data_file},
        "environment": environment_info(),
    }
    metadata_path = os.path.join(run_dir, "metadata.json")
    save_json(metadata, metadata_path)

    def finish(status: str):
        finished = datetime.now()
        metadata["status"] = status
        metadata["timing"]["finished_at"] = finished.isoformat(timespec="seconds")
        metadata["timing"]["duration_seconds"] = round((finished - started).total_seconds(), 1)
        save_json(metadata, metadata_path)

    try:
        (model, x_scaler, y_scalers, calibration_factors, task_names,
         history, test_metrics, run_info) = train_and_calibrate(
            data_path,
            epochs=epochs,
            downsample_to=downsample_to,
            max_samples=max_samples,
            val_fraction=val_fraction,
            split_seed=split_seed,
        )
    except BaseException as e:
        finish("interrupted" if isinstance(e, KeyboardInterrupt) else "failed")
        raise

    save_checkpoint(model, x_scaler, y_scalers, calibration_factors, task_names, downsample_to,
                    path=os.path.join(run_dir, "model.pt"), model_config=run_info["model"]["model_config"])
    save_csv(history, os.path.join(run_dir, "history.csv"))
    save_csv(test_metrics, os.path.join(run_dir, "test_metrics.csv"))
    metadata["data"].update(run_info["data"])
    for section in ("model", "training", "results"):
        metadata[section] = run_info[section]
    finish("finished")
    print(f"Saved model.pt, history.csv, test_metrics.csv and metadata.json to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the FTIR soil-property model.")
    parser.add_argument("data_path", help="Path to a CSV or SQLite dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--downsample-to", type=int, default=170, help="Spectral downsampling target")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit rows for a quick smoke test")
    parser.add_argument("--output-dir", default="models",
                        help="Parent folder for run folders (default: models)")
    parser.add_argument("--run-name", default=None,
                        help="Optional label appended to the timestamped run folder, e.g. no-fusion")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of all samples moved from 'train' to a validation set")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Random seed for choosing the validation rows")
    args = parser.parse_args()

    main(
        args.data_path,
        epochs=args.epochs,
        downsample_to=args.downsample_to,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        run_name=args.run_name,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )
