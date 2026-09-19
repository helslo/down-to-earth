# down-to-earth

This repository trains a multitask soil-property model based on FTIR spectra. The main code path is a single training script that can consume either:

- an SQLite dataset such as data/OSSL.db
- a CSV dataset with feature columns like z000, z001, ... and target columns like oc_pct, n_pct, ph_h2o, sand_pct, k_cmolc_kg

## Recommended workflow

1. Prepare your dataset as either a SQLite database or a CSV file.
2. Train the model. Each run gets its own folder, models/<YYYY-MM-DD_HHMMSS>[_<run-name>]/,
   containing model.pt, history.csv (per-epoch train/val stats), test_metrics.csv and
   config.json (settings, git commit, device, split counts):

   .venv/bin/python train_model.py data/features_samples.csv --run-name baseline

3. Use a saved checkpoint for inference only:

   .venv/bin/python predict_model.py models/<run-folder>/model.pt data/features_samples.csv

4. Adjust the task names, fusion edges, and model settings in train_model.py for your target properties.

## File map

- train_model.py: canonical training script; trains the model, calibrates uncertainty, and saves a checkpoint.
- predict_model.py: pure prediction script; loads a saved checkpoint and predicts on new data without training.
- load_data.py: dataset loading for SQLite and CSV inputs, including feature-only prediction inputs.
- model.py: FTIRNet model architecture and related transformer blocks.
- losses.py: multitask loss, EC schedule, and heteroscedastic regression loss.
- inspect_db.py: quick inspection of a SQLite database schema.
- inspect_db_printing.py: more detailed SQLite explorer for tables and statistics.
- example_training.py: compatibility wrapper; kept only for older commands.
- data/OSSL.db: example SQLite dataset.
- data/features_samples.csv: example CSV dataset with spectral features and labels.
- requirements.txt: project dependency list.
- pyproject.toml: packaging metadata.

## Training inputs

The model expects:

- X: a 2D array of shape (n_samples, n_features)
- Y: a dict mapping task names to 1D arrays of target values
- split: a vector with train/test labels (or a CSV split column)

The CSV loader accepts the same target names used by the training code:

- OC
- N
- pH
- Sand
- K

These are mapped internally to realistic CSV column names such as oc_pct and sand_pct.
