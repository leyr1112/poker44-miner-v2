"""Reward and scoring utilities for Poker44 poker bot detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

AP_WEIGHT = 0.35
BOT_RECALL_WEIGHT = 0.30
HUMAN_SAFETY_WEIGHT = 0.20
CALIBRATION_WEIGHT = 0.10
LATENCY_WEIGHT = 0.05


def _recall_at_fpr(
    y_score: np.ndarray,
    y_true: np.ndarray,
    *,
    max_fpr: float = 0.05,
) -> tuple[float, float]:
    """Best bot recall reachable while keeping human false-positive rate bounded."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if positive_count <= 0 or negative_count <= 0 or scores.size == 0:
        return 0.0, 0.0

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    recall = tp / max(positive_count, 1)
    fpr = fp / max(negative_count, 1)

    allowed = fpr <= float(max_fpr)
    if not np.any(allowed):
        return 0.0, 0.0

    allowed_indices = np.flatnonzero(allowed)
    best_local = int(allowed_indices[np.argmax(recall[allowed])])
    return float(recall[best_local]), float(fpr[best_local])


def _threshold_metrics(
    y_score: np.ndarray,
    y_true: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict:
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if scores.size == 0:
        return {
            "hard_bot_recall": 0.0,
            "hard_fpr": 0.0,
            "positive_prediction_rate": 0.0,
            "threshold_sanity_quality": 0.0,
        }

    hard_predictions = scores >= float(threshold)
    positive_prediction_rate = float(np.mean(hard_predictions))
    true_positives = int(np.sum(hard_predictions & (labels == 1)))
    false_positives = int(np.sum(hard_predictions & (labels == 0)))
    hard_bot_recall = (
        true_positives / max(positive_count, 1) if positive_count > 0 else 0.0
    )
    hard_fpr = (
        false_positives / max(negative_count, 1) if negative_count > 0 else 0.0
    )

    # Scores are still evaluated rank-first, but they must also be usable as a
    # risk threshold. A model that never crosses 0.5 on a mixed labeled window
    # cannot operationally flag bots, even if its relative ordering is strong.
    if positive_count <= 0 or negative_count <= 0:
        threshold_sanity_quality = 1.0
    elif true_positives <= 0:
        threshold_sanity_quality = 0.0
    elif hard_fpr <= 0.10:
        threshold_sanity_quality = 1.0
    else:
        threshold_sanity_quality = max(0.0, 1.0 - (hard_fpr - 0.10) / 0.90)

    return {
        "hard_bot_recall": float(hard_bot_recall),
        "hard_fpr": float(hard_fpr),
        "positive_prediction_rate": positive_prediction_rate,
        "threshold_sanity_quality": float(threshold_sanity_quality),
    }


def reward(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, dict]:
    """
    Compute a rank-first reward that protects humans without rewarding top-k guessing.
    """
    scores = np.asarray(y_pred, dtype=float)
    labels = np.asarray(y_true, dtype=int)
    if scores.size and np.any(labels == 1):
        ap_score = average_precision_score(labels, scores)
    else:
        ap_score = 0.0

    bot_recall, fpr = _recall_at_fpr(scores, labels, max_fpr=0.05)
    threshold_metrics = _threshold_metrics(scores, labels, threshold=0.5)
    human_safety_penalty = threshold_metrics["threshold_sanity_quality"]
    calibration_quality = human_safety_penalty
    latency_quality = 1.0

    if human_safety_penalty <= 0:
        base_score = 0.0
        rew = 0.0
    else:
        base_score = (
            AP_WEIGHT * ap_score
            + BOT_RECALL_WEIGHT * bot_recall
            + HUMAN_SAFETY_WEIGHT * human_safety_penalty
            + CALIBRATION_WEIGHT * calibration_quality
            + LATENCY_WEIGHT * latency_quality
        )
        rew = float(np.clip(base_score, 0.0, 1.0))

    res = {
        "fpr": fpr,
        "bot_recall": bot_recall,
        "ap_score": ap_score,
        "human_safety_penalty": human_safety_penalty,
        "calibration_quality": calibration_quality,
        "latency_quality": latency_quality,
        "base_score": base_score,
        "reward": rew,
        **threshold_metrics,
    }
    return rew, res
