# AM01 Repository Architecture

This repository implements the AM01 Kuka anomaly-detection pipeline used for the
Advanced Machine Learning in Applications project.

The codebase is intentionally small: reusable logic lives in `src/am01`, command
line entry points live in `scripts`, and Colab execution is handled by two notebooks.

## Data Flow

```text
KukaVelocityDataset NumPy files
        |
        v
load_timeseries_data()
        |
        v
run-level split, missing-value handling, scaler fit on normal train rows
        |
        v
sliding windows inside each run boundary
        |
        v
PCA / Isolation Forest / AE MLP / AAE MLP / AE Conv1D
        |
        v
validation threshold selection, test metrics, report tables and figures
```

Windows never cross `run_id` boundaries. This is important because Kuka runs are
derived from contiguous `action` segments, and crossing segment boundaries would
mix different robot dynamics.

## Package Layout

```text
src/am01/
├── data/
│   ├── io.py              # CSV and Kuka NumPy loading
│   ├── preprocessing.py   # missing values and feature-wise scalers
│   └── windowing.py       # run split and sliding-window generation
├── models/
│   ├── autoencoders.py    # MLP and Conv1D autoencoders
│   ├── aae.py             # AAE latent discriminator
│   └── baselines.py       # PCA and Isolation Forest detectors
├── training/
│   ├── losses.py
│   └── trainer.py         # AE and AAE training loops
├── evaluation/
│   ├── metrics.py         # threshold, binary and event-aware metrics
│   └── scoring.py         # autoencoder scoring helpers
├── pipeline.py            # prepare_data() and run_experiment()
└── reporting.py           # official report tables, figures and AAE diagnostics
```

`pipeline.py` is the core execution API. It prepares data, trains/evaluates one
configured model and writes all run artifacts. `reporting.py` builds final tables,
figures and diagnostics from saved runs; notebooks should call these functions
instead of duplicating analysis logic.

## Scripts

```text
scripts/
├── audit_data.py          # raw dataset audit
├── prepare_data.py        # split, scaling and windowing artifact generation
├── run_experiments.py     # batch training/evaluation runner
└── evaluate.py            # recompute metrics from saved score CSVs
```

`run_experiments.py` is the main runner. It supports controlled overrides for
seed, window length, latent dimension, adversarial weight, scaler, loss and
threshold method. It also supports `--skip-existing` so Colab runs can resume
without retraining completed experiments.

## Notebooks

```text
notebooks/
├── AM01_official_experiments_colab.ipynb
├── AM01_appendix_extended_ablation_colab.ipynb
└── AM01_aae_warmup_smoothl1_experiment_colab.ipynb
```

The official notebook is the primary project entry point. It produces the main
results, essential ablations, AAE diagnostics and report-ready figures.

The appendix notebook is optional. It is reserved for supporting analyses such as
multi-seed stability and artifact inspection.

The warm-up notebook is optional and targeted: it tests whether AAE improves when
Smooth L1 reconstruction is stabilized before the adversarial objective starts.

## Output Layout

The official Colab run writes to:

```text
MyDrive/AM01/results/official/
├── config/
├── tables/
├── figures/
├── extended_scores/
├── runs/
│   ├── main/
│   ├── window_sensitivity/
│   ├── aae_ablation/
│   └── preprocessing_loss/
└── summary.md
```

Generated results, checkpoints and figures are not part of the repository. They
belong on Google Drive or in local ignored output directories.
