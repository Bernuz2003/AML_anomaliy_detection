from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


def flatten_windows(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError(f"Expected windows with shape [N,T,C], got {X.shape}")
    return X.reshape(X.shape[0], -1)


def statistical_window_features(X: np.ndarray) -> np.ndarray:
    """Extract simple statistical features from windows for classical baselines."""
    if X.ndim != 3:
        raise ValueError(f"Expected windows with shape [N,T,C], got {X.shape}")
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    min_ = X.min(axis=1)
    max_ = X.max(axis=1)
    energy = (X ** 2).mean(axis=1)
    diff = np.diff(X, axis=1)
    diff_mean = diff.mean(axis=1)
    diff_std = diff.std(axis=1)
    return np.concatenate([mean, std, min_, max_, energy, diff_mean, diff_std], axis=1)


@dataclass
class PCADetector:
    n_components: int | float = 0.95
    pca: PCA | None = None

    def fit(self, X: np.ndarray) -> "PCADetector":
        flat = flatten_windows(X)
        self.pca = PCA(n_components=self.n_components, random_state=0)
        self.pca.fit(flat)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("PCADetector must be fitted before scoring.")
        flat = flatten_windows(X)
        rec = self.pca.inverse_transform(self.pca.transform(flat))
        return ((flat - rec) ** 2).mean(axis=1)


@dataclass
class IsolationForestDetector:
    n_estimators: int = 200
    contamination: str | float = "auto"
    random_state: int = 42
    feature_mode: str = "statistical"
    model: IsolationForest | None = None

    def _features(self, X: np.ndarray) -> np.ndarray:
        if self.feature_mode == "statistical":
            return statistical_window_features(X)
        if self.feature_mode == "flatten":
            return flatten_windows(X)
        raise ValueError(f"Unknown feature mode: {self.feature_mode}")

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(self._features(X))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("IsolationForestDetector must be fitted before scoring.")
        # decision_function is larger for normal samples. We invert it so larger means more anomalous.
        return -self.model.decision_function(self._features(X))
