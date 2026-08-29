# AM01 — KUKA Industrial Robot Anomaly Detection

> A leakage-safe pipeline for detecting anomalous slowdowns in multivariate KUKA robot time series, with a critical comparison between classical detectors, Autoencoders (AE) and Adversarial Autoencoders (AAE).

[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Project](https://img.shields.io/badge/Politecnico%20di%20Torino-2026%2FAM01-0066CC)](report_MLiA-3.pdf)

This repository contains the code, experiment notebooks and final material for the *Machine Learning in Applications* project **Detection of Anomalous Behaviour in an Industrial Robot** (project code **2026/AM01**) at Politecnico di Torino.

The central research question is:

> **Does adversarial latent-space regularization improve anomaly detection over a standard reconstruction-based Autoencoder on KUKA robot time series?**

The answer obtained in this study is negative but informative. Under the official MSE protocol, Isolation Forest is the strongest detector. Replacing MSE with **Smooth L1** makes the standard MLP Autoencoder highly competitive and produces the best seed-42 result, while the AAE remains below the corresponding AE. Hyperparameter tuning, latent/discriminator scores and adversarial warm-up do not reverse this conclusion.

## Contents

- [Main findings](#main-findings)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models](#models)
- [Experimental protocol](#experimental-protocol)
- [Results](#results)
- [Installation](#installation)
- [Running the project](#running-the-project)
- [Google Colab workflow](#google-colab-workflow)
- [Configuration and outputs](#configuration-and-outputs)
- [Repository structure](#repository-structure)
- [Reproducibility and limitations](#reproducibility-and-limitations)
- [Project material](#project-material)

## Main findings

1. **The anomaly type matters more than model complexity.** Slowdowns are well exposed by temporal summary statistics, making Isolation Forest a strong and stable baseline.
2. **Smooth L1 is the decisive improvement for neural models.** For the MLP AE, it raises test PR-AUC from **0.4570** to **0.7634** and reduces false-positive windows from **319** to **133** (−58.3%), while losing only six true positives.
3. **AAE does not improve the matched AE.** With MSE, AAE loses 0.0133 PR-AUC and 0.0312 F1 relative to AE. With Smooth L1, it still remains below AE (0.7080 versus 0.7634 PR-AUC).
4. **The conclusion is robust to additional diagnostics.** AAE latent norm, latent Mahalanobis distance, discriminator output and combined scores do not outperform its reconstruction score.
5. **Adversarial warm-up stabilizes optimization, not detection.** The best seed-42 schedule reaches 0.7028 test PR-AUC, but its three-seed mean (0.6884) is effectively unchanged from the no-warm-up AAE (0.6902).
6. **Robot behaviour is action-dependent and likely multimodal.** This offers a plausible explanation for why forcing all normal latent codes toward one isotropic Gaussian prior is too restrictive.

## Dataset

The project uses the AM01 `KukaVelocityDataset`, distributed as three NumPy arrays:

```text
data/raw/KukaVelocityDataset/
├── KukaColumnNames.npy   # 87 declared columns
├── KukaNormal.npy        # shape: (233792, 86)
└── KukaSlow.npy          # shape: (41538, 87)
```

The raw dataset is intentionally ignored by Git and must be supplied separately. The loader also supports a single CSV or a directory containing one CSV per run.

### Composition

| Source | Rows | Action segments |
|---|---:|---:|
| Normal | 233,792 | 2,340 |
| Slow/anomalous | 41,538 | 303 |
| **Total** | **275,330** | **2,643** |

After excluding metadata, the model input contains **85 numerical features**:

- 8 robot-level electrical measurements: apparent power, current, frequency, phase angle, active power, power factor, reactive power and voltage;
- 7 sensor units, each providing 3 accelerometer axes, 3 gyroscope axes, 4 orientation components and temperature (77 channels).

`action` is metadata used to define homogeneous temporal segments. `anomaly` is converted to the binary `label`; because the normal array has no anomaly column, all of its rows receive `label = 0`.

### Supported CSV format

A single CSV must contain a run identifier. When a directory of CSV files is supplied, the filename stem is used as `run_id` if the column is absent.

```text
run_id,t,label,feature_1,feature_2,...,feature_n
```

Labels are needed for stratified splitting, validation-based threshold selection and quantitative evaluation. Feature columns can be inferred automatically from numeric, non-metadata columns or specified explicitly in YAML.

## Methodology

```mermaid
flowchart LR
    A["KUKA NumPy arrays<br/>or CSV runs"] --> B["Load and align<br/>binary labels"]
    B --> C["Segment at each<br/>action change"]
    C --> D["Stratified run split<br/>60% / 20% / 20%"]
    D --> E["Per-run missing-value<br/>interpolation"]
    E --> F["Feature scaler fit on<br/>normal training rows only"]
    F --> G["Boundary-safe windows<br/>64 × 85, stride 16"]
    G --> H["Fit on normal<br/>training windows"]
    H --> I["Anomaly scores"]
    I --> J["Select threshold on<br/>validation only"]
    J --> K["Final test metrics<br/>and diagnostics"]
```

### 1. Action-based segmentation

The two original recordings are not treated as monolithic sequences. A new `run_id` is created whenever the robot action changes. This yields action-homogeneous segments and prevents windows from mixing different operating regimes.

Alternative loader strategies (`file` and `fixed_length`) are available, but `action_segments` is used in the official protocol.

### 2. Leakage-safe splitting

Complete runs—not individual rows or windows—are assigned to train, validation and test sets. The split is stratified by run label when possible and checked for disjoint `run_id` sets.

| Split | Rows | Runs | Windows | Anomalous windows | Prevalence |
|---|---:|---:|---:|---:|---:|
| Train | 165,495 | 1,586 | 4,725 | 924 | 19.56% |
| Validation | 54,623 | 529 | 1,537 | 315 | 20.49% |
| Test | 55,212 | 528 | 1,564 | 320 | 20.46% |

These counts correspond to seed 42, window length 64 and stride 16.

### 3. Missing values and scaling

Missing feature values are handled independently inside each run using linear interpolation followed by forward/backward filling. The default `StandardScaler` is fitted **only on normal training rows**, then applied unchanged to validation and test data.

Available scalers are `standard`, `robust` and `minmax`; available missing-value policies are `interpolate`, `ffill`, `zero` and `error`.

### 4. Windowing and labels

Each sample is a window

$$
X_i \in \mathbb{R}^{64 \times 85}
$$

created with stride 16 and never allowed to cross a run boundary. A window is anomalous when at least 10% of its timesteps are anomalous:

$$
y_i = \mathbb{1}\left[\frac{1}{64}\sum_{t \in X_i} y_t \ge 0.10\right].
$$

Only normal training windows are used to fit every detector. Neural models also use only normal validation windows for reconstruction-loss early stopping; the full labelled validation set is used later for threshold selection.

### 5. Scoring and threshold selection

PCA and neural autoencoders use mean per-window reconstruction error. Isolation Forest uses the negated decision function so that, for every model, a larger score means “more anomalous”.

The decision threshold is selected **on validation data only** by maximizing F1 and is then frozen for the test set. If labels are unavailable or contain only one class, the configured percentile (99th by default) is used as a fallback.

Evaluation includes:

- threshold-free ROC-AUC and PR-AUC;
- precision, recall, F1 and balanced accuracy;
- false-positive and false-negative rates;
- event precision/recall, false alarms per run, mean false-alarm duration and detection delay.

PR-AUC is the primary ranking metric because anomalous windows are the minority class. An event is a contiguous sequence of positive windows within one run; a predicted event matches a true event when the two overlap.

## Models

| Model | Input / architecture | Anomaly score |
|---|---|---|
| **PCA** | Flattened 64 × 85 window; components retaining 95% variance | Mean squared reconstruction error |
| **Isolation Forest** | 200 trees over 7 statistics per channel (595 features) | Negative `decision_function` |
| **AE MLP** | 5440 → 256 → 128 → 16 latent units; mirrored decoder | MSE, MAE or Smooth L1 reconstruction error |
| **AE Conv1D** | Two temporal Conv1D layers, 32 hidden channels, kernel 5, 16-D latent code | Reconstruction error |
| **AAE MLP** | AE MLP plus discriminator 16 → 128 → 64 → 1 | Reconstruction score; latent/discriminator signals are diagnostic extensions |

The Isolation Forest descriptors are channel-wise mean, standard deviation, minimum, maximum, energy, mean temporal difference and standard deviation of temporal differences.

AAE training alternates three phases:

1. encoder/decoder reconstruction;
2. discriminator separation of prior samples \(z \sim \mathcal{N}(0,I)\) from encoded samples;
3. adversarial encoder update that makes encoded samples resemble the prior.

Its objective is

$$
\mathcal{L}_{AAE}=\mathcal{L}_{rec}+\lambda_{adv}\mathcal{L}_{adv}.
$$

The default neural setup uses 30 epochs, batch size 128, AdamW, gradient clipping at 5, early-stopping patience 8 and automatic CUDA/CPU selection.

## Experimental protocol

### Official main protocol

| Parameter | Value |
|---|---|
| Window length / stride | 64 / 16 |
| Scaler | StandardScaler |
| Reconstruction loss | MSE |
| AE/AAE latent dimension | 16 |
| AAE adversarial weight | 0.1 |
| Seed | 42 |
| Threshold | Best validation F1 |
| Ablation selection metric | Validation PR-AUC |

The official comparison includes PCA, Isolation Forest, AE MLP, AAE MLP and AE Conv1D.

### Controlled ablations

- window length: `{32, 64}`;
- AAE latent dimension: `{16, 32}`;
- AAE adversarial weight: `{0.001, 0.01, 0.05, 0.1}`;
- scaler: `{standard, robust}`;
- reconstruction loss: `{mse, huber}`;
- multi-seed confirmation: `{0, 1, 2}`;
- AAE Smooth L1 warm-up and linear adversarial ramp.

In the CLI and configuration files, `huber` maps to PyTorch `SmoothL1Loss(beta=1.0)`. The same loss is used both for training and for the final reconstruction anomaly score.

Window length 128 is excluded from the official comparison. It leaves only 241 eligible segments, 77.18% of which are anomalous, versus about 11.4% anomalous segments at lengths 32 and 64. Its apparently high performance would therefore describe a substantially different and much smaller evaluation population.

## Results

### Official seed-42 comparison

All values below are measured on the same 1,564 test windows. “FA/run” is the number of unmatched predicted events divided by the number of test runs.

| Model | Loss | Precision | Recall | F1 | PR-AUC | ROC-AUC | FA/run |
|---|---|---:|---:|---:|---:|---:|---:|
| PCA | MSE | 0.3034 | 0.5594 | 0.3934 | 0.2873 | 0.6883 | 0.4459 |
| Isolation Forest | — | 0.6163 | 0.8531 | 0.7156 | 0.7196 | 0.9370 | 0.1499 |
| AE Conv1D | MSE | 0.5396 | 0.7031 | 0.6106 | 0.4572 | 0.8281 | 0.1784 |
| AE MLP | MSE | 0.4683 | 0.8781 | 0.6109 | 0.4570 | 0.8481 | 0.3017 |
| AAE MLP | MSE | 0.4407 | 0.8469 | 0.5797 | 0.4437 | 0.8266 | 0.3397 |
| **AE MLP** | **Smooth L1** | **0.6740** | **0.8594** | **0.7555** | **0.7634** | **0.9418** | **0.1252** |
| AAE MLP | Smooth L1 | 0.6126 | 0.7906 | 0.6903 | 0.7080 | 0.9035 | 0.1271 |

The first five rows form the official MSE protocol; the Smooth L1 rows come from the reconstruction-loss ablation under the same split and windowing setup.

Under MSE, Isolation Forest is clearly strongest. The similar PR-AUC of the MLP and convolutional AEs (0.4570 and 0.4572) indicates that merely preserving local temporal structure with this compact Conv1D architecture does not solve the score-separation problem.

Smooth L1 changes the picture: AE MLP becomes the best seed-42 model, slightly surpassing Isolation Forest in both F1 and PR-AUC. The improvement is caused mainly by fewer false alarms, not by a large recall increase.

### Direct AE–AAE comparison

| Metric | AAE MSE − AE MSE |
|---|---:|
| F1 | −0.0312 |
| PR-AUC | −0.0133 |
| ROC-AUC | −0.0216 |
| False alarms/run | +0.0380 |

With Smooth L1, window-level overlap provides an even more concrete interpretation:

- 252 anomalous windows are detected by both AE and AAE;
- 23 are detected only by AE;
- only 1 is detected only by AAE;
- at run level, both detect 47 anomalous runs, AE alone detects 9 and AAE alone detects none.

The adversarial component therefore adds almost no unique true detections while introducing additional false alarms.

### Window-length sensitivity

| Window | Model | Test windows | PR-AUC | F1 |
|---:|---|---:|---:|---:|
| 32 | AE MLP | 2,619 | 0.3787 | 0.5148 |
| 32 | AAE MLP | 2,619 | 0.3502 | 0.5058 |
| 64 | AE MLP | 1,564 | 0.4570 | 0.6109 |
| 64 | AAE MLP | 1,564 | 0.4437 | 0.5797 |

Moving from 32 to 64 samples helps both models, supporting the idea that a slowdown requires sustained temporal context. AAE remains below AE at both lengths.

### Multi-seed confirmation

Seeds 0, 1 and 2 change both model initialization and the run-level split. The standard deviations therefore describe robustness of the complete pipeline, not initialization alone.

| Experiment | Mean test PR-AUC | Std | Mean test F1 | Std | Mean FA/run |
|---|---:|---:|---:|---:|---:|
| **Isolation Forest** | **0.7929** | 0.0677 | **0.7511** | 0.0298 | 0.1246 |
| AE Smooth L1 | 0.7608 | 0.0442 | 0.7455 | 0.0164 | **0.1081** |
| AAE Smooth L1, no warm-up | 0.6902 | 0.0185 | 0.6696 | 0.0388 | 0.1714 |
| AAE Smooth L1, warm-up | 0.6884 | 0.0370 | 0.6692 | 0.0054 | 0.1771 |

The more cautious overall conclusion is therefore that AE Smooth L1 is competitive and generates fewer false alarms, while Isolation Forest retains the strongest average ranking quality.

### Diagnostic interpretation

- **Robust loss:** MSE lets a few very large residuals from legitimate normal windows dominate the score. Smooth L1 compresses these extremes and improves normal/anomaly separation.
- **Feature families:** orientation components provide the largest reconstruction-error separation, followed by electrical and gyroscope channels. Accelerometer and temperature families are weak or anti-informative on average for this anomaly.
- **Actions:** performance varies materially across robot actions, and action 7 is particularly difficult. A single global threshold therefore ignores meaningful operating-regime differences.
- **Latent space:** the AAE latent PCA projection shows a partial class shift but substantial overlap. Reconstruction remains the best AAE score (validation PR-AUC 0.6921, test PR-AUC 0.7080).
- **Warm-up:** the best seed-42 schedule uses 10 reconstruction-only epochs, a 5-epoch ramp and \(\lambda_{adv}=0.01\), reaching validation/test PR-AUC 0.7020/0.7028. This optimization improvement does not persist across seeds.

The raw experiment directories live on Google Drive or in local Git-ignored paths and are not committed. The versioned [final report](report_MLiA-3.pdf) and executed [interpretation notebook](AM01_results_interpretation_colab.ipynb) contain the evidence summarized above.

## Installation

### Requirements

- Python 3.10 or newer;
- Git;
- enough memory to load and window approximately 275k rows × 85 features;
- optional CUDA-compatible GPU for neural experiments.

Create an isolated environment and install both dependencies and the local package:

```bash
git clone https://github.com/Bernuz2003/AML_anomaliy_detection.git
cd AML_anomaliy_detection

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Running the project

Place the three KUKA arrays under `data/raw/KukaVelocityDataset/` before running the commands below.

### 1. Audit the raw data

```bash
python scripts/audit_data.py \
  --config configs/ae_mlp.yaml \
  --data data/raw/KukaVelocityDataset \
  --output results/data_audit
```

This writes a per-run dataset audit and a global feature summary.

### 2. Validate preprocessing

```bash
python scripts/prepare_data.py \
  --config configs/ae_mlp.yaml \
  --data data/raw/KukaVelocityDataset \
  --output data/processed/kuka_default
```

The command performs loading, splitting, scaling and windowing, then saves the processed arrays and split summary.

### 3. Reproduce the official model comparison

```bash
python scripts/run_experiments.py \
  --configs \
    configs/pca.yaml \
    configs/isolation_forest.yaml \
    configs/ae_mlp.yaml \
    configs/aae_mlp.yaml \
    configs/ae_conv1d.yaml \
  --data data/raw/KukaVelocityDataset \
  --output results/runs/main \
  --seeds 42 \
  --skip-existing \
  --summary-name main_results_raw.csv
```

### 4. Run the core ablations

Window length:

```bash
python scripts/run_experiments.py \
  --configs configs/ae_mlp.yaml configs/aae_mlp.yaml \
  --data data/raw/KukaVelocityDataset \
  --output results/runs/window_sensitivity \
  --seeds 42 \
  --window-lengths 32 64 \
  --skip-existing
```

AAE latent dimension and adversarial weight:

```bash
python scripts/run_experiments.py \
  --configs configs/aae_mlp.yaml \
  --data data/raw/KukaVelocityDataset \
  --output results/runs/aae_ablation \
  --seeds 42 \
  --latent-dims 16 32 \
  --lambda-advs 0.001 0.01 0.05 0.1 \
  --skip-existing
```

Scaler and reconstruction loss:

```bash
python scripts/run_experiments.py \
  --configs configs/ae_mlp.yaml configs/aae_mlp.yaml \
  --data data/raw/KukaVelocityDataset \
  --output results/runs/preprocessing_loss \
  --seeds 42 \
  --scalers standard robust \
  --losses mse huber \
  --skip-existing
```

`run_experiments.py` builds the Cartesian product of the requested overrides. It can also vary seeds, threshold methods, warm-up epochs and ramp epochs.

### 5. Re-evaluate saved scores

```bash
python scripts/evaluate.py \
  --run-dir results/runs/main/ae_mlp_seed42
```

The evaluator reads `scores_val.csv` and `scores_test.csv`, selects a threshold from validation (unless a fixed value is supplied), and writes `metrics.json`.

## Google Colab workflow

The Colab notebooks are the easiest way to reproduce the complete report-oriented workflow. They expect the dataset in:

```text
MyDrive/AM01/data/KukaVelocityDataset/
```

| Notebook | Purpose | Output root |
|---|---|---|
| [Official experiments](notebooks/AM01_official_experiments_colab.ipynb) | Audit, preprocessing, five-model comparison, core ablations, AAE diagnostics and report figures | `MyDrive/AM01/results/official/` |
| [Extended appendix](notebooks/AM01_appendix_extended_ablation_colab.ipynb) | Multi-seed stability and artifact manifest | `.../official/appendix/` |
| [AAE warm-up experiment](notebooks/AM01_aae_warmup_smoothl1_experiment_colab.ipynb) | Smooth L1 baseline, warm-up/ramp grid and three-seed confirmation | `.../official/warmup_aae_smoothl1/` |
| [Results interpretation](AM01_results_interpretation_colab.ipynb) | Deep post-hoc interpretation of the already generated official artifacts | `.../official/figures/interpretation_notebook/` |

The execution notebooks clone the repository, install `requirements.txt`, validate source syntax, record `pip freeze`, resume completed runs with `--skip-existing`, and save results on Google Drive.

## Configuration and outputs

### YAML configuration

The five default configurations live in `configs/`. Each file controls:

| Section | Main options |
|---|---|
| `data` | path format, column names, run strategy, split ratios and label semantics |
| `preprocessing` | scaler, normal-only fitting and missing-value strategy |
| `windowing` | window length, stride and anomaly fraction |
| `model` | model type and architecture/hyperparameters |
| `training` | epochs, batch size, learning rates, loss, patience and device |
| `evaluation` | threshold strategy and fallback percentile |

Command-line grid arguments override a controlled subset of YAML fields without editing the original files.

### Per-run artifacts

Each experiment directory is self-contained:

```text
<run_dir>/
├── config_used.json
├── preprocessing_config.json
├── split_summary.json
├── dataset_summary.csv
├── feature_summary.csv
├── processed_train.npz
├── processed_val.npz
├── processed_test.npz
├── scaler.joblib
├── model.joblib              # PCA / Isolation Forest
│   or model.pt               # neural models
├── history.json              # neural models
├── scores_val.csv
├── scores_test.csv
└── metrics.json
```

Neural checkpoints store the model state, configuration and feature order; AAE checkpoints also include the discriminator state. Generated runs, checkpoints, tables, figures and raw data are excluded from version control.

## Repository structure

```text
.
├── configs/                  # five official YAML configurations
├── data/raw/                 # local dataset location (Git-ignored)
├── notebooks/                # official and supporting Colab experiments
├── scripts/                  # command-line entry points
│   ├── audit_data.py
│   ├── prepare_data.py
│   ├── run_experiments.py
│   └── evaluate.py
├── src/am01/
│   ├── data/                 # loading, audit, preprocessing and windowing
│   ├── models/               # PCA, IF, AE, ConvAE and AAE
│   ├── training/             # losses and AE/AAE training loops
│   ├── evaluation/           # scoring and window/event metrics
│   ├── pipeline.py           # end-to-end experiment API
│   └── reporting.py          # tables, figures and AAE diagnostics
├── AM01_results_interpretation_colab.ipynb
├── report_MLiA-3.pdf
├── MLinAPP Presentation.pptx
├── pyproject.toml
└── requirements.txt
```

`src/am01/pipeline.py` is the main programmatic entry point. `prepare_data()` creates leakage-safe datasets, while `run_experiment()` trains, scores and serializes one configured model. `src/am01/reporting.py` reconstructs final tables, figures and diagnostic analyses from saved run artifacts.

## Reproducibility and limitations

### Reproducibility safeguards

- Python, NumPy and PyTorch random seeds are set for each run;
- deterministic cuDNN behaviour is requested and benchmarking is disabled;
- run-level split membership is deterministic for a given seed;
- preprocessing is fitted only on allowed training data;
- model and ablation selection use validation metrics, never test metrics;
- the exact configuration, split summary, scores and model state are serialized;
- Colab executions save a full environment snapshot with `pip freeze`.

### Limitations

- The study focuses on one anomaly type—slow robot behaviour—so conclusions should not be generalized to every industrial fault.
- Dependency versions are lower-bounded rather than locked; exact bitwise reproduction can vary with library versions and hardware.
- The dataset is not distributed through this Git repository and must be obtained separately.
- Window-level labels and contiguous-window events are approximations of operational incidents.
- Seeds 0, 1 and 2 alter the data split as well as neural initialization; multi-seed variance mixes both sources.
- A single global threshold does not model the action-dependent score distributions observed in the diagnostics.

Promising extensions include action-conditional thresholds, conditional or mixture-prior AAEs, feature-wise reconstruction weighting, tuning the Smooth L1 transition parameter, additional anomaly types and a richer event-level evaluation protocol.

## Project material

- [Final report](report_MLiA-3.pdf) — complete methodology, results, discussion and references.
- [Presentation](MLinAPP%20Presentation.pptx) — project slides.
- [Official experiment notebook](notebooks/AM01_official_experiments_colab.ipynb) — executable end-to-end workflow.
- [Results interpretation notebook](AM01_results_interpretation_colab.ipynb) — detailed visual analysis of the official artifacts.

Project authors: **E. Bernacchi, G. Feira, F. Marchese and S. Zare**.
