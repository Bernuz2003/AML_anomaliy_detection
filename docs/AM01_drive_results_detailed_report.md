# AM01 — Direct Google Drive Results Analysis

## 1. Scope

This report is based on the files downloaded directly from `MyDrive/AM01/results` and `MyDrive/AM01/data/processed/kuka_default` after Drive access was enabled.

The analyzed artifacts include:

- `main_metrics.csv`
- per-model `metrics.json`
- per-model `scores_test.csv`
- `dataset_summary.csv`
- `feature_summary.csv`
- `split_summary.json`
- `preprocessing_config.json`
- `processed_train.npz`, `processed_val.npz`, `processed_test.npz`
- per-model `config_used.json`

## 2. Dataset and preprocessing verification

### 2.1 Dataset composition

| source_file   | run_label   |   runs |   rows |   anomalous_rows |   min_rows |   max_rows |
|:--------------|:------------|-------:|-------:|-----------------:|-----------:|-----------:|
| normal        | normal      |   2340 | 233792 |                0 |         41 |       1459 |
| slow          | anomalous   |    303 |  41538 |            41538 |         21 |        210 |

The real dataset contains **2,643 run segments**, split into:

- **2,340 normal segments**
- **303 anomalous/slow segments**
- **275,330 raw rows**
- **41,538 anomalous rows**, all from the `slow` source.

Short segments below the window length exist:

- `normal_seg_0597`: 41 rows
- `normal_seg_1400`: 42 rows
- `slow_seg_0300`: 21 rows
- `slow_seg_0301`: 25 rows

These cannot produce windows with `window_length=64`; therefore the number of unique run IDs in processed windows is slightly smaller than the raw split run count.

### 2.2 Preprocessing configuration

```json
{
  "seed": 42,
  "data": {
    "path": null,
    "format": "auto",
    "run_col": "run_id",
    "time_col": "t",
    "label_col": "label",
    "split_col": null,
    "anomaly_col": "anomaly",
    "action_col": "action",
    "kuka_run_strategy": "action_segments",
    "kuka_fixed_run_length": 512,
    "normal_label": 0,
    "feature_cols": null,
    "train_ratio": 0.6,
    "val_ratio": 0.2,
    "test_ratio": 0.2,
    "stratify_by_label": true
  },
  "preprocessing": {
    "scaler": "standard",
    "fit_only_normal": true,
    "missing_strategy": "interpolate"
  },
  "windowing": {
    "window_length": 64,
    "stride": 16,
    "anomaly_fraction": 0.1
  }
}
```

The preprocessing protocol is methodologically sound:

- run-level split prevents leakage across overlapping windows;
- `StandardScaler` is fitted only on normal training rows;
- missing values are handled by per-run interpolation;
- windowing uses `window_length=64`, `stride=16`, `anomaly_fraction=0.1`.

### 2.3 Split integrity

From `split_summary.json`:

| Split | Raw rows | Raw runs | Windows | Anomalous windows |
|---|---:|---:|---:|---:|
| train | 165495 | 1586 | 4725 | 924 |
| val | 54623 | 529 | 1537 | 315 |
| test | 55212 | 528 | 1564 | 320 |

NPZ verification confirms:

| Split | X shape | y shape | anomalous windows | unique run IDs in windows |
|---|---:|---:|---:|---:|
| train | `(4725, 64, 85)` | `(4725,)` | 924 | 1583 |
| val | `(1537, 64, 85)` | `(1537,)` | 315 | 529 |
| test | `(1564, 64, 85)` | `(1564,)` | 320 | 527 |

The minor mismatch between raw split run counts and window-level run counts is expected because four runs are shorter than 64 samples.

### 2.4 Feature audit

Number of features: **85**.

No missing values are present after loading/preprocessing.

Constant or near-constant features:

| feature         |   mean |         std |    min |    max |
|:----------------|-------:|------------:|-------:|-------:|
| sensor_id2_temp | 144.12 | 1.25766e-10 | 144.12 | 144.12 |
| sensor_id5_temp | 144.12 | 1.25766e-10 | 144.12 | 144.12 |
| sensor_id6_temp | 180.24 | 1.87186e-10 | 180.24 | 180.24 |
| sensor_id7_temp | 180.24 | 1.87186e-10 | 180.24 | 180.24 |

Main extreme-value features:

| feature                            |         mean |      std |        min |       max |
|:-----------------------------------|-------------:|---------:|-----------:|----------:|
| sensor_id1_GyroY                   |   2.25286    | 44.0863  | -1999.88   | 1968.99   |
| sensor_id1_GyroZ                   |   0.820216   | 49.9111  | -1998.9    | 1999.51   |
| sensor_id1_GyroX                   |  -0.734031   | 29.0247  | -1997.44   | 1855.9    |
| sensor_id4_GyroY                   |   0.127676   | 26.7436  | -1869.81   | 1213.5    |
| sensor_id4_GyroX                   |  -1.69923    | 20.1624  | -1859.44   |   83.7402 |
| sensor_id4_GyroZ                   |   0.18625    | 21.3219  |  -656.677  | 1286.44   |
| sensor_id3_GyroZ                   |   0.00864729 | 13.9887  |   -74.5239 |  721.558  |
| sensor_id3_GyroX                   |   0.00561183 |  7.31282 |   -42.0532 |  721.558  |
| sensor_id3_GyroY                   |   0.0127419  |  6.02363 |   -39.978  |  721.558  |
| machine_nameKuka Robot_phase_angle | 330.653      |  2.98863 |   319.273  |  338.063  |

The presence of very large gyro ranges and constant temperature channels suggests that a `RobustScaler` ablation is worth adding. Standard scaling is acceptable as a baseline, but outlier-heavy inertial channels may strongly affect reconstruction losses.

## 3. Experimental configuration

| model_name       | model_cfg                                                                                                                                    | epochs   | batch_size   | loss   | threshold   |
|:-----------------|:---------------------------------------------------------------------------------------------------------------------------------------------|:---------|:-------------|:-------|:------------|
| aae_mlp          | {'type': 'aae_mlp', 'hidden_dims': [256, 128], 'latent_dim': 16, 'dropout': 0.05, 'discriminator_hidden_dims': [128, 64], 'lambda_adv': 0.1} | 30       | 128          | mse    | best_f1     |
| ae_conv1d        | {'type': 'ae_conv1d', 'hidden_channels': 32, 'latent_dim': 16, 'kernel_size': 5, 'dropout': 0.05}                                            | 30       | 128          | mse    | best_f1     |
| ae_mlp           | {'type': 'ae_mlp', 'hidden_dims': [256, 128], 'latent_dim': 16, 'dropout': 0.05}                                                             | 30       | 128          | mse    | best_f1     |
| isolation_forest | {'type': 'isolation_forest', 'n_estimators': 200, 'contamination': 'auto', 'feature_mode': 'statistical'}                                    | -        | -            | -      | best_f1     |
| pca              | {'type': 'pca', 'n_components': 0.95}                                                                                                        | -        | -            | -      | best_f1     |

The comparison is fair for the central research question because `ae_mlp` and `aae_mlp` use the same encoder-decoder capacity and differ mainly by the adversarial latent regularization.

## 4. Main quantitative results

| run              |   test_precision |   test_recall |   test_f1 |   test_balanced_accuracy |   test_roc_auc |   test_pr_auc |   test_event_recall |   test_event_precision |   test_false_alarms_per_run |   test_mean_detection_delay |
|:-----------------|-----------------:|--------------:|----------:|-------------------------:|---------------:|--------------:|--------------------:|-----------------------:|----------------------------:|----------------------------:|
| aae_mlp          |           0.4407 |        0.8469 |    0.5797 |                   0.7852 |         0.8266 |        0.4437 |              0.8833 |                 0.235  |                      0.3397 |                      3.3208 |
| ae_conv1d        |           0.5396 |        0.7031 |    0.6106 |                   0.7744 |         0.8281 |        0.4572 |              0.6833 |                 0.3088 |                      0.1784 |                      4.2927 |
| ae_mlp           |           0.4683 |        0.8781 |    0.6109 |                   0.8108 |         0.8481 |        0.457  |              0.95   |                 0.2639 |                      0.3017 |                      3.3684 |
| isolation_forest |           0.6163 |        0.8531 |    0.7156 |                   0.8582 |         0.937  |        0.7196 |              0.9    |                 0.4357 |                      0.1499 |                      1.7778 |
| pca              |           0.3034 |        0.5594 |    0.3934 |                   0.6145 |         0.6883 |        0.2873 |              0.9    |                 0.227  |                      0.4459 |                      7.4074 |

## 5. Validation vs test consistency

| run              |   val_f1 |   val_pr_auc |   val_roc_auc |   test_f1 |   test_pr_auc |   test_roc_auc |
|:-----------------|---------:|-------------:|--------------:|----------:|--------------:|---------------:|
| isolation_forest |   0.7216 |       0.7911 |        0.9334 |    0.7156 |        0.7196 |         0.937  |
| ae_mlp           |   0.5593 |       0.4195 |        0.808  |    0.6109 |        0.457  |         0.8481 |
| ae_conv1d        |   0.4985 |       0.4016 |        0.7674 |    0.6106 |        0.4572 |         0.8281 |
| aae_mlp          |   0.5152 |       0.391  |        0.7747 |    0.5797 |        0.4437 |         0.8266 |
| pca              |   0.4456 |       0.2974 |        0.7031 |    0.3934 |        0.2873 |         0.6883 |

The model ranking is reasonably consistent between validation and test. Isolation Forest is best on validation and remains best on test.

## 6. Score distribution diagnostics

| model            |   normal_mean |   anom_mean |   normal_median |   anom_median |   normal_q95 |   anom_q05 |   threshold |
|:-----------------|--------------:|------------:|----------------:|--------------:|-------------:|-----------:|------------:|
| aae_mlp          |        0.3698 |      0.5264 |          0.2675 |        0.4899 |       0.6815 |     0.256  |      0.3306 |
| ae_conv1d        |        0.4312 |      0.5982 |          0.3354 |        0.5797 |       0.7279 |     0.3192 |      0.4859 |
| ae_mlp           |        0.3305 |      0.4899 |          0.2273 |        0.4504 |       0.6244 |     0.2409 |      0.2974 |
| isolation_forest |       -0.0601 |      0.012  |         -0.0656 |        0.0227 |      -0.0099 |    -0.0503 |     -0.0391 |
| pca              |        0.098  |      0.1069 |          0.0595 |        0.0818 |       0.203  |     0.0514 |      0.075  |

Key observations:

- Isolation Forest separates normal/anomalous windows best: normal median `-0.0656`, anomaly median `0.0227`, threshold `-0.0391`.
- AE MLP and AAE MLP detect many anomalous windows, but their normal score distributions have heavy tails, causing many false positives.
- PCA has poor score separation and high false-positive rate.
- Conv1D-AE is more conservative than AE MLP/AAE MLP, with fewer false alarms but lower recall.

## 7. Interpretation by model

### 7.1 Isolation Forest

Isolation Forest is the best model in this run:

- best F1: `0.7156`
- best balanced accuracy: `0.8582`
- best ROC-AUC: `0.9370`
- best PR-AUC: `0.7196`
- best event precision: `0.4357`
- lowest false alarms per run: `0.1499`
- lowest detection delay: `1.7778`

This suggests that the anomaly signal is strongly captured by statistical descriptors of windows, rather than requiring a deep reconstruction model.

### 7.2 AE MLP

AE MLP is the best deep model overall:

- F1: `0.6109`
- recall: `0.8781`
- ROC-AUC: `0.8481`
- event recall: `0.95`

It detects almost all anomalous events, but it produces many false alarms:

- FP: `319`
- event precision: `0.2639`
- false predicted events: `159`

This makes it useful when missing anomalies is very costly, but weak when false alarms matter.

### 7.3 AAE MLP

AAE MLP underperforms AE MLP:

| Metric | AAE - AE |
|---|---:|
| precision | -0.0277 |
| recall | -0.0313 |
| F1 | -0.0312 |
| balanced accuracy | -0.0257 |
| ROC-AUC | -0.0216 |
| PR-AUC | -0.0133 |
| event recall | -0.0667 |
| event precision | -0.0288 |
| false alarms/run | +0.0380 |

The only small advantage is slightly lower mean detection delay, but the difference is negligible and not enough to justify the adversarial component.

Conclusion: **with the current configuration, adversarial latent regularization does not improve anomaly detection**.

### 7.4 Conv1D-AE

Conv1D-AE reaches F1 comparable to AE MLP (`0.6106` vs `0.6109`) but with a different operating point:

- lower recall than AE MLP: `0.7031` vs `0.8781`
- better precision than AE MLP: `0.5396` vs `0.4683`
- lower false positives: `192` vs `319`

It is a more conservative detector, but less suitable if event recall is prioritized.

### 7.5 PCA

PCA is the weakest method:

- F1: `0.3934`
- ROC-AUC: `0.6883`
- PR-AUC: `0.2873`
- highest false alarms per run: `0.4459`

The high event recall is misleading because PCA predicts too many events: 304 predicted events against 60 true events.

## 8. Methodological correctness

Strengths:

1. Run-level split prevents direct leakage across overlapping time windows.
2. Scaler is fitted only on normal training rows.
3. Threshold is selected on validation, not on test.
4. Test is used only for final reporting.
5. Baselines are meaningful and include both reconstruction and non-reconstruction approaches.
6. Event-aware metrics are reported, not only point-wise metrics.
7. `main_metrics.csv` was verified against per-model `scores_test.csv`; recomputation matches exactly.

Limitations:

1. Only one seed appears to be reported.
2. Threshold selection uses labeled validation data with `best_f1`; this is acceptable for experimental comparison, but a deployment-oriented anomaly detector should also report a label-free threshold, e.g. 95th/99th percentile of normal validation scores.
3. No ablation results are present for `lambda_adv`, `latent_dim`, `window_length`, scaler, or reconstruction loss.
4. The AAE is evaluated only through reconstruction score; latent-space likelihood or discriminator-derived anomaly scores are not used.
5. The raw segmenting strategy `action_segments` generates many short runs. This is defensible but should be described clearly in the report.

## 9. Final answer to the research question

The central research question is whether the adversarial component improves anomaly detection compared with a traditional autoencoder.

Based on the directly inspected Drive results:

> **No. The AAE does not improve over the traditional AE MLP under the current protocol.**

The AE MLP is better than the AAE MLP on F1, recall, precision, balanced accuracy, ROC-AUC, PR-AUC, event recall and event precision. Isolation Forest is better than both, suggesting that statistical time-window features capture the current slow/anomalous behavior more effectively than the tested reconstruction-based deep models.

## 10. Recommended next steps before final report

1. Run at least 3 seeds and report mean ± std.
2. Add a label-free threshold protocol:
   - percentile 95/99 on normal validation scores;
   - compare with validation `best_f1`.
3. Add AAE ablations:
   - `lambda_adv ∈ {0.01, 0.05, 0.1, 0.5, 1.0}`;
   - `latent_dim ∈ {8, 16, 32}`.
4. Add preprocessing ablations:
   - `StandardScaler` vs `RobustScaler`;
   - MSE vs Huber loss;
   - clipping extreme gyro values.
5. For AAE, add latent diagnostics:
   - latent normality check;
   - t-SNE/PCA latent visualization;
   - optional anomaly score combining reconstruction and latent prior distance.
6. In the report, present the negative AAE result as a strength: the project evaluates the adversarial component critically instead of assuming it must help.
