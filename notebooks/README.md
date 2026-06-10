# Notebooks

The project uses four Colab-first notebooks:

- `AM01_colab_master.ipynb`: executes tests, audit, preprocessing, training and figure generation.
- `AM01_results_analysis_colab.ipynb`: analyzes already generated results under `MyDrive/AM01/results`.
- `AM01_phase2_colab.ipynb`: runs incremental Phase 2 experiments under `MyDrive/AM01/results/phase2`.
- `AM01_phase3_aae_latent_diagnostics_colab.ipynb`: analyzes AAE-specific latent/discriminator scores, action failures and feature reconstruction errors under `MyDrive/AM01/results/phase3_aae_diagnostics`.

Reusable logic stays in `src/` and `scripts/`. The notebooks install the environment
when needed, call repository scripts, read/write Google Drive artifacts and display
tables/figures for inspection.

This replaces the earlier multi-notebook exploratory plan. The reason is practical:
the project must remain reproducible on Colab, and the notebook code must not diverge
from the production pipeline.

Expected Drive layout:

```text
MyDrive/AM01/
├── data/
│   ├── KukaVelocityDataset/
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
