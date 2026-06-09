import numpy as np

from am01.evaluation.metrics import all_metrics, event_metrics, select_threshold


def test_threshold_best_f1_is_between_score_values():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    thr = select_threshold(y, scores, method="best_f1")
    assert scores.min() <= thr <= scores.max()


def test_binary_and_event_metrics_detect_single_event():
    y = np.array([0, 1, 1, 0, 0])
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.1])
    run_ids = np.array(["r"] * 5)
    starts = np.array([0, 10, 20, 30, 40])
    metrics = all_metrics(y, scores, 0.5, run_ids, starts)
    assert metrics["f1"] == 1.0
    assert metrics["event_recall"] == 1.0
    assert metrics["event_precision"] == 1.0
    assert metrics["mean_detection_delay"] == 0.0


def test_event_metric_counts_false_predicted_event():
    y = np.array([0, 1, 1, 0, 0, 0, 0])
    scores = np.array([0.1, 0.9, 0.8, 0.1, 0.9, 0.9, 0.1])
    run_ids = np.array(["r"] * 7)
    starts = np.arange(7)
    metrics = event_metrics(y, scores, 0.5, run_ids, starts)
    assert metrics["true_events"] == 1.0
    assert metrics["predicted_events"] == 2.0
    assert metrics["false_predicted_events"] == 1.0
