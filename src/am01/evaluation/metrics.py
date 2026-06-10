from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _has_two_classes(y_true: np.ndarray) -> bool:
    return len(np.unique(y_true)) >= 2


def select_threshold(
    y_true: np.ndarray | None,
    scores: np.ndarray,
    *,
    method: str = "best_f1",
    fallback_percentile: float = 99.0,
) -> float:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("Cannot select a threshold from empty scores.")
    if method == "percentile":
        return float(np.percentile(scores, fallback_percentile))
    if method == "normal_percentile":
        if y_true is not None:
            y_true = np.asarray(y_true).astype(int)
            normal_scores = scores[y_true == 0]
            if normal_scores.size > 0:
                return float(np.percentile(normal_scores, fallback_percentile))
        return float(np.percentile(scores, fallback_percentile))
    if y_true is None or not _has_two_classes(np.asarray(y_true)):
        return float(np.percentile(scores, fallback_percentile))
    if method != "best_f1":
        raise ValueError(f"Unknown threshold method: {method}")

    y_true = np.asarray(y_true).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return float(np.percentile(scores, fallback_percentile))
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx])


def binary_metrics(y_true: np.ndarray | None, scores: np.ndarray, threshold: float) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    if y_true is None:
        return {"threshold": float(threshold)}
    y_true = np.asarray(y_true).astype(int)
    y_pred = (scores >= threshold).astype(int)
    out: dict[str, float] = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp)})
    out["false_positive_rate"] = float(fp / max(fp + tn, 1))
    out["false_negative_rate"] = float(fn / max(fn + tp, 1))
    if _has_two_classes(y_true):
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def _segments(binary: np.ndarray) -> list[tuple[int, int]]:
    binary = np.asarray(binary).astype(int)
    segments: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(binary):
        if value == 1 and start is None:
            start = i
        elif value == 0 and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(binary) - 1))
    return segments


def event_metrics(
    y_true: np.ndarray | None,
    scores: np.ndarray,
    threshold: float,
    run_ids: np.ndarray,
    starts: np.ndarray,
) -> dict[str, float]:
    """Compute event-aware metrics at window level for each run.

    A true event is detected if at least one predicted anomalous window overlaps it.
    Detection delay is measured in samples using the window start indices.
    """
    if y_true is None:
        return {}
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    run_ids = np.asarray(run_ids)
    starts = np.asarray(starts)

    true_events = 0
    detected_true_events = 0
    predicted_events = 0
    matched_pred_events = 0
    delays: list[float] = []
    false_alarm_durations: list[float] = []
    n_runs = len(np.unique(run_ids))

    for run in np.unique(run_ids):
        idx = np.where(run_ids == run)[0]
        idx = idx[np.argsort(starts[idx])]
        yt = y_true[idx]
        yp = y_pred[idx]
        st = starts[idx]
        true_seg = _segments(yt)
        pred_seg = _segments(yp)
        true_events += len(true_seg)
        predicted_events += len(pred_seg)

        matched_pred = set()
        for ts, te in true_seg:
            overlapping = [k for k, (ps, pe) in enumerate(pred_seg) if not (pe < ts or ps > te)]
            if overlapping:
                detected_true_events += 1
                first_pred_start = min(pred_seg[k][0] for k in overlapping)
                delays.append(float(max(0, st[first_pred_start] - st[ts])))
                matched_pred.update(overlapping)
        matched_pred_events += len(matched_pred)
        for k, (ps, pe) in enumerate(pred_seg):
            if k not in matched_pred:
                false_alarm_durations.append(float(pe - ps + 1))

    false_predicted_events = max(predicted_events - matched_pred_events, 0)
    return {
        "event_recall": float(detected_true_events / max(true_events, 1)),
        "event_precision": float(matched_pred_events / max(predicted_events, 1)),
        "true_events": float(true_events),
        "predicted_events": float(predicted_events),
        "false_predicted_events": float(false_predicted_events),
        "false_alarms_per_run": float(false_predicted_events / max(n_runs, 1)),
        "mean_false_alarm_duration_windows": (
            float(np.mean(false_alarm_durations)) if false_alarm_durations else 0.0
        ),
        "mean_detection_delay": float(np.mean(delays)) if delays else float("nan"),
    }


def all_metrics(
    y_true: np.ndarray | None,
    scores: np.ndarray,
    threshold: float,
    run_ids: np.ndarray,
    starts: np.ndarray,
) -> dict[str, float]:
    out = binary_metrics(y_true, scores, threshold)
    out.update(event_metrics(y_true, scores, threshold, run_ids, starts))
    return out
