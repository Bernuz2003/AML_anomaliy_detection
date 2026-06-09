from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_kuka_like(
    *,
    runs: int = 18,
    length: int = 220,
    joints: int = 3,
    seed: int = 42,
    anomaly_run_fraction: float = 0.35,
) -> pd.DataFrame:
    """Generate a small Kuka-like multivariate time-series dataset.

    The generator creates position, velocity, current and power channels. Some runs
    contain contiguous anomaly segments with slower movement, drift and higher current.
    This is not a substitute for the real dataset; it exists only for pipeline testing.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    anomaly_runs = set(rng.choice(np.arange(runs), size=max(1, int(runs * anomaly_run_fraction)), replace=False))

    for r in range(runs):
        run_id = f"run_{r:03d}"
        t = np.arange(length, dtype=np.float32)
        phase = rng.uniform(0, 2 * np.pi, size=joints)
        speed = rng.uniform(0.85, 1.15)
        anomaly_mask = np.zeros(length, dtype=np.int64)
        anomaly_start = None
        anomaly_end = None
        if r in anomaly_runs:
            anomaly_start = int(rng.integers(length // 4, length // 2))
            anomaly_end = min(length, anomaly_start + int(rng.integers(length // 8, length // 4)))
            anomaly_mask[anomaly_start:anomaly_end] = 1

        for i in range(length):
            local_speed = speed
            drift = 0.0
            extra_current = 0.0
            if anomaly_mask[i] == 1:
                local_speed *= 0.58
                drift = 0.012 * (i - anomaly_start)
                extra_current = 0.7 + 0.15 * rng.normal()

            row: dict[str, float | int | str] = {"run_id": run_id, "t": int(i), "label": int(anomaly_mask[i])}
            for j in range(joints):
                base = 2 * np.pi * local_speed * i / length + phase[j]
                pos = np.sin(base) + drift + 0.03 * rng.normal()
                vel = local_speed * np.cos(base) + 0.04 * rng.normal()
                current = 0.45 + 0.12 * abs(vel) + extra_current + 0.03 * rng.normal()
                power = current * (0.9 + 0.15 * abs(vel)) + 0.02 * rng.normal()
                row[f"joint_{j+1}_pos"] = float(pos)
                row[f"joint_{j+1}_vel"] = float(vel)
                row[f"joint_{j+1}_current"] = float(current)
                row[f"joint_{j+1}_power"] = float(power)
            rows.append(row)

    return pd.DataFrame(rows)


def save_synthetic_csv(output: str | Path, **kwargs) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_kuka_like(**kwargs)
    df.to_csv(output, index=False)
    return output
