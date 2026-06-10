# AM01 — Detection of Anomalous Behaviour in Industrial Robot

Repository template for the **Advanced Machine Learning in Applications** project AM01.

The project implements a rigorous anomaly-detection pipeline for Kuka industrial robot time series:

- data audit and leakage-safe run-level splitting;
- row-wise scaling fitted only on normal training data;
- sliding-window generation;
- classical baselines;
- traditional autoencoders;
- adversarial autoencoders;
- threshold-free, threshold-based and event-aware evaluation.

The central research question is whether the adversarial latent-space regularization of an **Adversarial Autoencoder (AAE)** improves anomaly detection compared with a traditional reconstruction-based autoencoder.

## Recommended execution on Colab

The official project run is orchestrated by:

```text
notebooks/AM01_colab_master.ipynb
```

The notebook mounts Google Drive, installs dependencies, runs tests, audits the real
Kuka dataset, trains the selected models and writes all outputs to Drive. It does not
duplicate the implementation: reusable logic stays in `src/` and `scripts/`.

Detailed execution instructions are in `docs/colab_execution.md`.

Expected Drive layout:

```text
MyDrive/AM01/
├── data/
│   ├── KukaVelocityDataset/
│   │   ├── KukaColumnNames.npy
│   │   ├── KukaNormal.npy
│   │   └── KukaSlow.npy
│   └── processed/
└── results/
    ├── data_audit/
    ├── analysis/
    ├── figures/
    ├── phase2/
    ├── phase3_aae_diagnostics/
    ├── runs/
    └── tables/
```

Run the validation suite:

```bash
pytest -q
```

## Real KukaVelocityDataset

The real NumPy dataset is supported directly when the directory contains:

- `KukaColumnNames.npy`;
- `KukaNormal.npy`;
- `KukaSlow.npy`.

Run the data audit:

```bash
python scripts/audit_data.py \
  --config configs/ae_mlp.yaml \
  --data data/raw/KukaVelocityDataset \
  --output results/data_audit
```

Prepare leakage-safe processed windows:

```bash
python scripts/prepare_data.py \
  --config configs/ae_mlp.yaml \
  --data data/raw/KukaVelocityDataset \
  --output data/processed/kuka_default
```

Train/evaluate individual models:

```bash
python scripts/train.py --config configs/pca.yaml --data data/raw/KukaVelocityDataset --output results/runs/kuka_pca
python scripts/train.py --config configs/ae_mlp.yaml --data data/raw/KukaVelocityDataset --output results/runs/kuka_ae_mlp
python scripts/train.py --config configs/aae_mlp.yaml --data data/raw/KukaVelocityDataset --output results/runs/kuka_aae_mlp
```

Run a small comparison or ablation:

```bash
python scripts/run_experiments.py \
  --configs configs/pca.yaml configs/isolation_forest.yaml configs/ae_mlp.yaml configs/ae_conv1d.yaml configs/aae_mlp.yaml \
  --data data/raw/KukaVelocityDataset \
  --output results/runs/kuka_comparison \
  --seeds 0 1 2 \
  --window-lengths 32 64 128
```

Recompute metrics from saved scores without retraining:

```bash
python scripts/evaluate.py --run-dir results/runs/kuka_aae_mlp
```

Generate figures for analysis:

```bash
python scripts/plot_data_examples.py --config configs/ae_mlp.yaml --data data/raw/KukaVelocityDataset --output results/figures/data_examples
python scripts/plot_results.py --run-dir results/runs/kuka_aae_mlp
python scripts/plot_model_diagnostics.py --run-dir results/runs/kuka_aae_mlp --split test
```

## Expected CSV schema

A single CSV file or a directory of CSV files is supported. Each row is one timestamp.

Recommended columns:

| column | required | meaning |
|---|---:|---|
| `run_id` | yes, unless one CSV per run | trajectory/run identifier |
| `t` | optional | timestamp or sample index |
| `label` | optional | 0 normal, 1 anomalous |
| feature columns | yes | sensor values: joint positions, velocities, currents, power, etc. |

If the dataset has different column names, edit the `data` section in the YAML config.

For the bundled Kuka NumPy dataset, the loader converts `anomaly` to `label`, creates
`run_id` from contiguous `action` segments, and excludes `action`, `anomaly`,
`source_file`, `run_id`, `t` and `label` from inferred sensor features.

## Repository structure

```text
configs/                 experiment configurations
docs/                    project plan and evaluation protocol
notebooks/               Colab execution, analysis, Phase 2 and Phase 3 notebooks
src/am01/data/           loading, preprocessing, windowing, data audit
src/am01/models/         autoencoders, AAE discriminator, classical baselines
src/am01/training/       AE and AAE training loops
src/am01/evaluation/     scoring, thresholds, metrics, event-aware metrics
scripts/                 command line entry points
tests/                   correctness tests
```

## Leakage prevention

The pipeline follows three rules:

1. split by `run_id` before window generation whenever possible;
2. stratify run-level splits by label when possible;
3. fit scalers only on normal training rows.

Thresholds are selected only on validation data. The test set is used only for final reporting.

## Notes for the real AM01 dataset

Before running final experiments, adapt:

- feature column list;
- run/split definitions;
- anomaly-label interpretation;
- window length and stride based on sampling frequency;
- event-aware metric tolerance.

Do not optimize hyperparameters on the final test set.
