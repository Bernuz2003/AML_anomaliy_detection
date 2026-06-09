from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

ScalerName = Literal["standard", "robust", "minmax"]
MissingStrategy = Literal["error", "ffill", "interpolate", "zero"]


def clean_missing_values(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    run_col: str = "run_id",
    strategy: MissingStrategy = "interpolate",
) -> pd.DataFrame:
    """Handle missing values without mixing information across robot runs."""
    out = df.copy()
    if not out[feature_cols].isna().any().any():
        return out
    if strategy == "error":
        counts = out[feature_cols].isna().sum()
        raise ValueError(f"Missing values found in features: {counts[counts > 0].to_dict()}")
    if strategy == "zero":
        out[feature_cols] = out[feature_cols].fillna(0.0)
        return out

    grouped = []
    for _, part in out.groupby(run_col, sort=False):
        part = part.copy()
        if strategy == "interpolate":
            part[feature_cols] = part[feature_cols].interpolate(method="linear", limit_direction="both")
            part[feature_cols] = part[feature_cols].ffill().bfill()
        elif strategy == "ffill":
            part[feature_cols] = part[feature_cols].ffill().bfill()
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")
        grouped.append(part)
    return pd.concat(grouped, axis=0).sort_index()


@dataclass
class TimeSeriesScaler:
    """Feature-wise scaler fitted only on the allowed training rows."""

    name: ScalerName = "standard"
    scaler: StandardScaler | RobustScaler | MinMaxScaler | None = None
    feature_cols: list[str] | None = None

    def _make(self):
        if self.name == "standard":
            return StandardScaler()
        if self.name == "robust":
            return RobustScaler()
        if self.name == "minmax":
            return MinMaxScaler()
        raise ValueError(f"Unknown scaler: {self.name}")

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "TimeSeriesScaler":
        self.feature_cols = list(feature_cols)
        self.scaler = self._make()
        values = df[self.feature_cols].to_numpy(dtype=np.float32)
        if len(values) == 0:
            raise ValueError("Cannot fit scaler on an empty dataframe.")
        self.scaler.fit(values)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None or self.feature_cols is None:
            raise RuntimeError("The scaler must be fitted before calling transform().")
        out = df.copy()
        out[self.feature_cols] = self.scaler.transform(out[self.feature_cols].to_numpy(dtype=np.float32))
        return out

    def fit_transform(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        return self.fit(df, feature_cols).transform(df)


def select_scaler_fit_rows(
    df_train: pd.DataFrame,
    *,
    label_col: str | None = "label",
    normal_label: int | float | str = 0,
    fit_only_normal: bool = True,
) -> pd.DataFrame:
    if fit_only_normal and label_col and label_col in df_train.columns:
        normal = df_train[df_train[label_col] == normal_label]
        if len(normal) > 0:
            return normal
    return df_train
