"""Metrics and evaluation utilities for group disparity analysis.

Complies with Section 7 and Section 9 of adult_plan.md:
- Confusion matrix structure: rows TRUE [0, 1], cols PREDICTED [0, 1] -> [[TN, FP], [FN, TP]].
- Accuracy = (TP + TN) / (TP + TN + FP + FN).
- False Negative Rate (FNR) = FN / (FN + TP).
- False Positive Rate (FPR) = FP / (FP + TN).
- True Positive Rate (TPR / Recall) = TP / (TP + FN).
- Missed higher-income predictions per 100 actual higher-income records = 100 * FNR.
- Zero denominators safely return None (serializes to valid JSON null, never NaN or 0).
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np


def compute_confusion_matrix_and_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
) -> Dict[str, Any]:
    """Computes binary classification confusion matrix and rates.

    Confusion matrix format:
      [[TN, FP],
       [FN, TP]]
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    if len(y_true_arr) == 0:
        return {
            "total_samples": 0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "accuracy": None,
            "fnr": None,
            "fpr": None,
            "tpr": None,
            "missed_per_100": None,
        }

    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))

    total = tn + fp + fn + tp
    actual_positives = fn + tp
    actual_negatives = tn + fp

    accuracy = float((tp + tn) / total) if total > 0 else None
    fnr = float(fn / actual_positives) if actual_positives > 0 else None
    fpr = float(fp / actual_negatives) if actual_negatives > 0 else None
    tpr = float(tp / actual_positives) if actual_positives > 0 else None
    missed_per_100 = float(100.0 * fnr) if fnr is not None else None

    return {
        "total_samples": total,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "actual_positives": actual_positives,
        "actual_negatives": actual_negatives,
        "accuracy": accuracy,
        "fnr": fnr,
        "fpr": fpr,
        "tpr": tpr,
        "missed_per_100": missed_per_100,
    }


def compute_metrics_from_cm(cm: List[List[int]]) -> Dict[str, Any]:
    """Helper for testing synthetic confusion matrices [[TN, FP], [FN, TP]]."""
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    total = tn + fp + fn + tp
    actual_positives = fn + tp
    actual_negatives = tn + fp

    accuracy = float((tp + tn) / total) if total > 0 else None
    fnr = float(fn / actual_positives) if actual_positives > 0 else None
    fpr = float(fp / actual_negatives) if actual_negatives > 0 else None
    tpr = float(tp / actual_positives) if actual_positives > 0 else None
    missed_per_100 = float(100.0 * fnr) if fnr is not None else None

    return {
        "total_samples": total,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "actual_positives": actual_positives,
        "actual_negatives": actual_negatives,
        "accuracy": accuracy,
        "fnr": fnr,
        "fpr": fpr,
        "tpr": tpr,
        "missed_per_100": missed_per_100,
    }


def compute_group_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    groups: Union[List[int], np.ndarray],
    group_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Computes overall and per-group metrics."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    groups_arr = np.asarray(groups, dtype=np.int64)

    overall = compute_confusion_matrix_and_metrics(y_true_arr, y_pred_arr)

    by_group = {}
    unique_groups = np.unique(groups_arr)
    for g in sorted(unique_groups):
        mask = groups_arr == g
        g_name = group_names.get(int(g), str(g)) if group_names else str(g)
        by_group[g_name] = compute_confusion_matrix_and_metrics(
            y_true_arr[mask], y_pred_arr[mask]
        )

    return {
        "overall": overall,
        "groups": by_group,
    }
