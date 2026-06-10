# Evaluation Protocol

This protocol fixes the evaluation rules for AM01 experiments before model tuning.

## Data Splitting

- Split by `run_id` before sliding-window generation.
- Use stratified run-level splitting when labels are available, so validation and test contain normal and anomalous runs whenever possible.
- Do not mix windows from the same run across train, validation and test.
- Fit scalers only on normal rows from the training split.
- For the bundled Kuka NumPy dataset, `run_id` is derived from contiguous `action` segments unless a later audit motivates a different split definition.

## Training

- PCA, Isolation Forest, AE and AAE are fitted only on normal training windows.
- Validation data is used for early stopping, threshold selection and hyperparameter selection.
- Test data is used only for final reporting.
- The main experiment is run first with a single default configuration. Ablations are run only after that baseline comparison is complete.

## Scores

- PCA score: mean squared reconstruction error.
- Isolation Forest score: negated `decision_function`, so larger values mean more anomalous.
- AE/AAE primary score: mean per-window reconstruction error.
- AE vs AAE primary comparison must use the same reconstruction score.
- Latent/discriminator-based AAE scores are optional ablations, not the main comparison.

## Thresholding

- Default threshold: validation threshold maximizing F1.
- If validation labels are missing or contain a single class, use the configured validation score percentile.
- Never choose thresholds on test scores.
- Phase 2 also reports label-free thresholds computed as the 95th/99th percentile of normal validation scores.

## Metrics

Report threshold-free metrics:

- ROC-AUC.
- PR-AUC.

Report threshold-based metrics:

- Precision.
- Recall.
- F1.
- Balanced accuracy.
- False positive rate.
- False negative rate.

Report event-aware/window-level metrics:

- Event recall.
- Event precision.
- Mean detection delay.
- Predicted events.
- False predicted events.
- False alarms per run.
- Mean false-alarm duration in windows.

## Reproducibility

- Save `config_used.json`, `preprocessing_config.json`, scaler, processed splits, scores and metrics for every run.
- Report seeds and software versions.
- Use multi-seed comparisons for final AE vs AAE claims.
- On Colab, save outputs under `MyDrive/AM01/results/` and do not rely on `/content` for final artifacts.
