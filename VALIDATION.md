# Validation

This file describes how to validate the repository after installing dependencies.
The authoritative execution path for the project is `notebooks/AM01_colab_master.ipynb`.

## Local Or Colab Checks

From the repository root:

```bash
pip install -r requirements.txt
python -m compileall -q src scripts tests
pytest -q
python scripts/run_synthetic_smoke.py
```

## Real Dataset Checks

Assuming the Kuka NumPy dataset is available at `data/raw/KukaVelocityDataset`:

```bash
python scripts/audit_data.py \
  --config configs/ae_mlp.yaml \
  --data data/raw/KukaVelocityDataset \
  --output results/data_audit

python scripts/prepare_data.py \
  --config configs/ae_mlp.yaml \
  --data data/raw/KukaVelocityDataset \
  --output data/processed/kuka_default
```

Expected preparation outputs:

- `split_summary.json`;
- `dataset_summary.csv`;
- `feature_summary.csv`;
- `preprocessing_config.json`;
- `processed_train.npz`;
- `processed_val.npz`;
- `processed_test.npz`;
- `scaler.joblib`.

## Main Experiment Checks

```bash
python scripts/train.py --config configs/pca.yaml --data data/raw/KukaVelocityDataset --output results/runs/main/pca
python scripts/train.py --config configs/isolation_forest.yaml --data data/raw/KukaVelocityDataset --output results/runs/main/isolation_forest
python scripts/train.py --config configs/ae_mlp.yaml --data data/raw/KukaVelocityDataset --output results/runs/main/ae_mlp
python scripts/train.py --config configs/aae_mlp.yaml --data data/raw/KukaVelocityDataset --output results/runs/main/aae_mlp
```

Each run directory should contain:

- `config_used.json`;
- `split_summary.json`;
- processed `.npz` files;
- `scores_val.csv`;
- `scores_test.csv`;
- `metrics.json`;
- a saved model (`model.joblib` or `model.pt`).
