# AM01 Official Experimental Protocol

This document defines the final, report-oriented AM01 protocol.

## Primary Research Question

Does adversarial latent-space regularization improve anomaly detection on Kuka
robot time series compared with a standard reconstruction-based autoencoder?

The primary comparison is:

```text
AE MLP vs AAE MLP
```

under identical preprocessing, windowing, architecture size and threshold
selection.

## Official Main Protocol

```text
window_length: 64
stride: 16
scaler: standard
loss: mse
latent_dim: 16
lambda_adv: 0.1
seed: 42
threshold: selected on validation only
selection_metric: val_pr_auc
```

Main models:

```text
PCA
Isolation Forest
AE MLP
AAE MLP
AE Conv1D
```

The main report table is saved as:

```text
MyDrive/AM01/results/official/tables/main_results.csv
```

## Window-Length Policy

The official protocol uses `window_length=64`.

The sensitivity analysis includes only:

```text
window_length in {32, 64}
```

`window_length=128` is intentionally excluded from the final protocol because it
leaves too few complete test windows after run-level splitting and boundary-safe
windowing. High metrics under that setting are not treated as reliable evidence.

## Essential Ablations

The official notebook includes a compact set of ablations:

```text
window_length in {32, 64}
AAE latent_dim in {16, 32}
AAE lambda_adv in {0.001, 0.01, 0.05, 0.1}
preprocessing/loss in:
  - StandardScaler + MSE
  - RobustScaler + MSE
  - StandardScaler + Huber
  - RobustScaler + Huber
```

These ablations test the main hypotheses without turning the notebook into an
unbounded grid search.

## AAE-Specific Diagnostics

AAE is first evaluated with the same reconstruction score used by AE. Then, as a
diagnostic extension, the official notebook evaluates AAE-specific signals:

```text
reconstruction score
latent norm score
latent Mahalanobis score
discriminator score
combined reconstruction + discriminator
combined reconstruction + latent Mahalanobis
```

The best AAE diagnostic run is selected using validation PR-AUC, not test PR-AUC.
The test set is used only for final reporting.

Generated AAE diagnostic artifacts include:

```text
tables/aae_specific_scores.csv
tables/per_feature_reconstruction.csv
tables/per_action_metrics.csv
figures/fig16_aae_specific_scores_pr_auc.png
figures/fig17_aae_latent_pca_label.png
figures/fig18_aae_latent_pca_action.png
figures/fig19_latent_discriminator_score_distribution.png
figures/fig20_top_feature_reconstruction_separation.png
figures/fig21_per_action_f1.png
figures/fig22_per_action_false_positives.png
```

## Colab Execution

Open:

```text
notebooks/AM01_official_experiments_colab.ipynb
```

Expected Drive input:

```text
MyDrive/AM01/data/KukaVelocityDataset/
├── KukaColumnNames.npy
├── KukaNormal.npy
└── KukaSlow.npy
```

The notebook installs dependencies, validates Python syntax with `compileall`,
runs the official experiments and writes report-ready outputs to:

```text
MyDrive/AM01/results/official/
```

For optional multi-seed stability, open:

```text
notebooks/AM01_appendix_extended_ablation_colab.ipynb
```

## Expected Final Interpretation

The final narrative should be critical and evidence-based:

1. The dataset must be evaluated as temporal windows, and the chosen window length
   changes the evaluation population.
2. Under the official `w=64` protocol, AAE does not robustly improve the AE MLP
   baseline.
3. AAE latent/discriminator scores may contain useful diagnostic signal, but they
   are supporting analysis rather than the main claim.
4. Adversarial regularization is therefore not automatically beneficial for this
   Kuka anomaly-detection setting.
