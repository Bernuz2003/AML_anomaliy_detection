# AM01 Kuka AAE Anomaly Detection

Minimal repository for the AM01 Advanced Machine Learning in Applications project.

The project evaluates anomaly detection on Kuka robot time series, with a central
comparison between a reconstruction-based Autoencoder and an Adversarial
Autoencoder.

## Official Entry Point

Run on Colab:

```text
notebooks/AM01_official_experiments_colab.ipynb
```

Optional appendix:

```text
notebooks/AM01_appendix_extended_ablation_colab.ipynb
```

Optional targeted AAE warm-up experiment:

```text
notebooks/AM01_aae_warmup_smoothl1_experiment_colab.ipynb
```

Expected input on Google Drive:

```text
MyDrive/AM01/data/KukaVelocityDataset/
├── KukaColumnNames.npy
├── KukaNormal.npy
└── KukaSlow.npy
```

Official outputs:

```text
MyDrive/AM01/results/official/
├── config/
├── tables/
├── figures/
├── extended_scores/
├── runs/
└── summary.md
```

The warm-up experiment writes to:

```text
MyDrive/AM01/results/official/warmup_aae_smoothl1/
```

## Repository Structure

```text
configs/      model configurations
docs/         architecture and official protocol
notebooks/    official Colab notebooks
scripts/      small CLI entry points
src/am01/     reusable project package
data/         dataset placement notes
```

## Main Scripts

```bash
python scripts/audit_data.py --config configs/ae_mlp.yaml --data data/raw/KukaVelocityDataset --output results/data_audit
python scripts/prepare_data.py --config configs/ae_mlp.yaml --data data/raw/KukaVelocityDataset --output results/preprocessed
python scripts/run_experiments.py --configs configs/ae_mlp.yaml configs/aae_mlp.yaml --data data/raw/KukaVelocityDataset --output results/runs
python scripts/evaluate.py --run-dir results/runs/ae_mlp_seed42
```

## Documentation

- `docs/ARCHITECTURE.md`
- `docs/OFFICIAL_PROTOCOL.md`
