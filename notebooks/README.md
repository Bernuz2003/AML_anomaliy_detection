# Notebooks

The project uses a single Colab-first orchestration notebook:

- `AM01_colab_master.ipynb`

Reusable logic stays in `src/` and `scripts/`. The notebook only installs the
environment, calls repository scripts, writes outputs to Google Drive and displays
tables/figures for inspection.

This replaces the earlier multi-notebook exploratory plan. The reason is practical:
the project must remain reproducible on Colab, and a single master notebook avoids
divergence between notebook code and the production pipeline.

Expected Drive layout:

```text
MyDrive/AM01/
├── data/
│   ├── KukaVelocityDataset/
│   └── processed/
└── results/
    ├── data_audit/
    ├── figures/
    ├── runs/
    └── tables/
```
