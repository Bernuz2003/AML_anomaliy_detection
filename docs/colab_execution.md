# Colab Execution Guide

Use `notebooks/AM01_colab_master.ipynb` as the official execution notebook.
Use `notebooks/AM01_results_analysis_colab.ipynb` after experiments to inspect and
visualize the saved outputs.
Use `notebooks/AM01_phase2_colab.ipynb` for the incremental Phase 2 experiments.
Use `notebooks/AM01_phase3_aae_latent_diagnostics_colab.ipynb` for AAE-specific
diagnostics after Phase 1/Phase 2 have produced AE and AAE checkpoints.

## 1. Prepare Google Drive

Create this folder structure:

```text
MyDrive/AM01/
├── data/
│   └── KukaVelocityDataset/
│       ├── KukaColumnNames.npy
│       ├── KukaNormal.npy
│       └── KukaSlow.npy
└── results/
```

The notebook will create `processed/`, `runs/`, `figures/` and `tables/` as needed.

## 2. Open The Notebook

Open:

```text
notebooks/AM01_colab_master.ipynb
```

Set `REPO_URL` if the repository is not already present in `/content`.

## 3. Run Validation

The notebook runs:

```bash
python -m compileall -q src scripts tests
pytest -q
```

These checks validate syntax and unit/integration behavior.

## 4. Run Real Dataset Audit

Outputs:

```text
MyDrive/AM01/results/data_audit/
├── dataset_summary.csv
└── feature_summary.csv
```

Use `dataset_summary.csv` to verify:

- number of generated runs;
- normal/anomalous run balance;
- anomaly fraction per run;
- run lengths;
- missing values.

If the split or segment definition looks wrong, stop before training and revise the data configuration.

## 5. Run Preprocessing

Outputs:

```text
MyDrive/AM01/data/processed/kuka_default/
├── dataset_summary.csv
├── feature_summary.csv
├── preprocessing_config.json
├── processed_train.npz
├── processed_val.npz
├── processed_test.npz
├── scaler.joblib
└── split_summary.json
```

Use `split_summary.json` to confirm:

- train/validation/test rows and runs;
- number of windows;
- anomalous windows per split;
- normal training windows.

## 6. Run Main Experiments

The notebook trains:

- PCA;
- Isolation Forest;
- AE MLP;
- AAE MLP;
- Conv1D-AE.

Outputs are written under:

```text
MyDrive/AM01/results/runs/main/<model_name>/
```

Each model directory should contain:

```text
config_used.json
dataset_summary.csv
feature_summary.csv
history.json          # neural models only
metrics.json
model.joblib          # classical models
model.pt              # neural models
preprocessing_config.json
processed_train.npz
processed_val.npz
processed_test.npz
scaler.joblib
scores_val.csv
scores_test.csv
split_summary.json
figures/
```

## 7. Read The Results

Main table:

```text
MyDrive/AM01/results/tables/main_metrics.csv
```

Use it to compare:

- ROC-AUC;
- PR-AUC;
- F1;
- recall;
- false positive rate;
- event recall;
- mean detection delay;
- false alarms per run.

For the central research question, compare `ae_mlp` and `aae_mlp` first. PCA,
Isolation Forest and Conv1D-AE provide context.

## 8. Use Figures

Data examples:

```text
MyDrive/AM01/results/figures/data_examples/
```

Per-model figures:

```text
MyDrive/AM01/results/runs/main/<model_name>/figures/
```

Use:

- score distributions to inspect separation between normal and anomalous windows;
- ROC/PR curves for threshold-free comparison;
- timelines to inspect false alarms and detection delay;
- reconstruction plots to show qualitative model behavior;
- latent plots to compare AE and AAE representations.

## 9. Optional Ablation

Enable in the notebook:

```python
RUN_ABLATION = True
```

Outputs:

```text
MyDrive/AM01/results/runs/ablation/
└── experiment_summary.csv
```

Run ablations only after the main experiment is complete and interpretable.

## 10. Results Analysis Notebook

After the main experiment has produced outputs under:

```text
MyDrive/AM01/results/
```

open:

```text
notebooks/AM01_results_analysis_colab.ipynb
```

This notebook reads existing artifacts and creates:

```text
MyDrive/AM01/results/analysis/
├── auto_insights.md
├── figures/
└── tables/
```

Important generated tables:

- `tables/metrics_all_models.csv`;
- `tables/main_metrics_compact.csv`;
- `tables/model_ranking.csv`;
- `tables/aae_minus_ae_deltas.csv`;
- `tables/confusion_counts_by_model.csv`;
- `tables/top_false_positive_runs.csv`;
- `tables/top_false_negative_runs.csv`;
- `tables/score_spearman_correlation.csv`;
- `tables/training_history.csv`.

Important generated figures:

- metric bar plots;
- AE-vs-AAE delta plot;
- score distributions;
- combined ROC and precision-recall curves;
- confusion outcome plot;
- top false-positive runs;
- selected run timeline;
- score correlation heatmap;
- training-history curves;
- dataset run-length and label-distribution plots.

## 11. Phase 2 Notebook

After reviewing the first results, open:

```text
notebooks/AM01_phase2_colab.ipynb
```

This notebook writes only under:

```text
MyDrive/AM01/results/phase2/
├── threshold_analysis/
├── multiseed/
├── aae_ablation/
├── preprocessing_ablation/
├── figures/
├── tables/
└── phase2_auto_summary.md
```

It does not overwrite the original Phase 1 results in `MyDrive/AM01/results/runs/main`.

Recommended execution order:

1. threshold analysis, no retraining;
2. multi-seed stability;
3. AAE lambda/latent ablation;
4. preprocessing/loss ablation.

## 12. Phase 3 AAE Diagnostics Notebook

After Phase 1 and, preferably, Phase 2 have produced AE/AAE checkpoints, open:

```text
notebooks/AM01_phase3_aae_latent_diagnostics_colab.ipynb
```

This notebook does not retrain models. It selects the best available AE MLP and
AAE MLP runs by the configured PR-AUC metric already stored in `metrics.json`
(`test_pr_auc` by default, switchable to `val_pr_auc` inside the notebook), then
diagnoses whether the AAE contains useful latent/discriminator signal beyond plain
reconstruction error.

Outputs are written only under:

```text
MyDrive/AM01/results/phase3_aae_diagnostics/
├── extended_scores/
│   ├── aae_extended_scores_val.csv
│   └── aae_extended_scores_test.csv
├── figures/
├── tables/
└── phase3_aae_diagnostics_summary.md
```

Main generated tables:

- `tables/candidate_ae_aae_runs.csv`;
- `tables/aae_alternative_score_evaluation.csv`;
- `tables/ae_vs_best_aae_score.csv`;
- `tables/aae_latent_pca_coordinates.csv`;
- `tables/per_action_metrics.csv`;
- `tables/per_action_f1_deltas.csv`;
- `tables/feature_reconstruction_error_summary.csv`;
- `tables/feature_ae_aae_separation_deltas.csv`.

Main generated figures:

- AAE alternative-score PR-AUC and F1 comparisons;
- AE reconstruction vs best AAE-specific score;
- AAE score distributions for normal/anomalous windows;
- latent PCA colored by label and by TP/FP/FN/TN outcome;
- latent/discriminator distribution plots;
- per-action F1 and false-positive plots;
- per-feature reconstruction-error separation plots.

Use `phase3_aae_diagnostics_summary.md` as the first draft of the report discussion:
it states whether any AAE-specific score beats AE MLP, which actions are most
problematic, and which features lose reconstruction-error separation under AAE.
