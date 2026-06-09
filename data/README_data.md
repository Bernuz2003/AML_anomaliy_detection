# Data Folder

Place raw datasets in `data/raw/`.

## Supported Input Formats

The pipeline supports:

1. one CSV file containing all runs;
2. a directory with one CSV file per run;
3. the AM01 `KukaVelocityDataset` NumPy directory.

Recommended CSV schema:

```text
run_id,t,label,joint_1_pos,joint_1_vel,joint_1_current,joint_1_power,...
```

Labels are optional for training but required for quantitative validation/test metrics.

## Real KukaVelocityDataset

The repository currently contains:

```text
data/raw/KukaVelocityDataset/
├── KukaColumnNames.npy  # 87 column names
├── KukaNormal.npy       # shape: (233792, 86)
└── KukaSlow.npy         # shape: (41538, 87)
```

`KukaColumnNames.npy` declares an `anomaly` column as the last field. `KukaNormal.npy`
does not contain that column, so the loader assigns `label=0` to all normal rows.
`KukaSlow.npy` contains the `anomaly` column, which is converted to binary `label`.

The `action` column is treated as metadata, not as a sensor feature. By default, the
loader creates one `run_id` per contiguous action segment, avoiding window leakage
across train/validation/test splits.

Run an audit:

```bash
python scripts/audit_data.py \
  --config configs/ae_mlp.yaml \
  --data data/raw/KukaVelocityDataset \
  --output results/data_audit
```

Prepare processed windows:

```bash
python scripts/prepare_data.py \
  --config configs/ae_mlp.yaml \
  --data data/raw/KukaVelocityDataset \
  --output data/processed/kuka_default
```
