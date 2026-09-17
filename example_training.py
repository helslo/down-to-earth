"""Backward-compatible wrapper for the canonical training entry point.

Use train_model.py as the main training script. This file remains only to
avoid breaking older commands that still call it directly.
"""
import sys

from train_model import main


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_model.py path/to/your_dataset.db OR path/to/your_dataset.csv")
        sys.exit(1)
    main(sys.argv[1])
