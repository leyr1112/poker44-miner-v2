from __future__ import annotations

import inspect
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# Cosmetic noise from LightGBM<->sklearn 1.7 feature-name validation. Numpy
# rows we pass to predict_proba are correctly aligned by index; the warning
# only fires because LightGBM 4.x stores a feature signature on fit.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

from detector.features import union_features

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


class Poker44Model:
    """Small runtime wrapper for the rebuilt supervised Poker44 artifact."""

    def __init__(self, model_path: str | Path):
        if joblib is None:
            raise RuntimeError("joblib is required to load Poker44 models.")
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")

        artifact = joblib.load(self.model_path)
        self.models = list(artifact.get("models") or [])
        if not self.models and artifact.get("model") is not None:
            self.models = [artifact["model"]]
        if not self.models:
            raise RuntimeError("Model artifact contains no models.")

        self.feature_names = list(artifact.get("feature_names") or [])
        self.metadata = dict(artifact.get("metadata") or {})
        self.calibrator = artifact.get("calibrator")
        self.score_logit_bias = float(self.metadata.get("score_logit_bias", 0.0) or 0.0)
        self.score_logit_temperature = max(
            float(self.metadata.get("score_logit_temperature", 1.0) or 1.0),
            1e-6,
        )
        score_remap = self.metadata.get("score_remap")
        if isinstance(score_remap, dict) and score_remap.get("kind"):
            self.score_remap: dict[str, Any] = score_remap
        elif (
            isinstance(self.calibrator, dict)
            and self.calibrator.get("kind") == "threshold_logit_v1"
        ):
            # Legacy artifacts stored score_remap in calibrator; apply once via score_remap.
            self.score_remap = dict(self.calibrator)
            self.calibrator = None
        else:
            self.score_remap = {}
        self.model_weights = list(
            artifact.get("model_weights")
            or self.metadata.get("model_weights")
            or [1.0 for _ in self.models]
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-40.0, min(40.0, float(value)))
        return 1.0 / (1.0 + math.exp(-value))

    def _aligned_rows(self, chunks: list[list[dict[str, Any]]]) -> list[list[float]]:
        rows: list[list[float]] = []
        for chunk in chunks:
            features = union_features(chunk)
            features["hand_count"] = float(len(chunk))
            if not self.feature_names:
                self.feature_names = sorted(features)
            rows.append([float(features.get(name, 0.0)) for name in self.feature_names])
        return rows

    @staticmethod
    def _average_rank(column: np.ndarray) -> np.ndarray:
        """Average (tie-aware) ranks 1..n for one feature column.

        Pure-numpy so the serving image needs no scipy. Equal values share the
        mean of the ranks they span, matching a standard average-rank transform.
        """
        n = int(column.shape[0])
        order = np.argsort(column, kind="mergesort")
        ordered = column[order]
        ranks = np.arange(1, n + 1, dtype=np.float64)
        start = 0
        for end in range(1, n + 1):
            if end == n or ordered[end] != ordered[start]:
                if end - start > 1:
                    ranks[start:end] = (start + 1 + end) / 2.0
                start = end
        out = np.empty(n, dtype=np.float64)
        out[order] = ranks
        return out

    def _rank_normalize_rows(self, rows: list[list[float]]) -> list[list[float]]:
        """Map each feature column to ``(rank-0.5)/n`` within the request batch."""
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return rows
        n_rows, n_cols = matrix.shape
        if n_rows <= 1:
            return [[0.5 for _ in range(n_cols)] for _ in range(n_rows)]
        normalized = np.empty_like(matrix)
        for col in range(n_cols):
            normalized[:, col] = self._average_rank(matrix[:, col])
        normalized = (normalized - 0.5) / float(n_rows)
        return normalized.tolist()

    def _prepared_rows(self, chunks: list[list[dict[str, Any]]]) -> list[list[float]]:
        """Aligned feature rows, with any configured input transform applied."""
        rows = self._aligned_rows(chunks)
        if str(self.metadata.get("input_transform", "")).strip() == "per_request_rank":
            rows = self._rank_normalize_rows(rows)
        return rows

    def _model_column(
        self,
        model: Any,
        rows: list[list[float]],
        chunks: list[list[dict[str, Any]]] | None,
        apply_calibration: bool,
    ) -> list[float]:
        """Score ONE base model into a per-row column.

        ``apply_calibration`` is forwarded only to models whose
        ``predict_chunk_scores`` accepts it; the probe order degrades gracefully
        for every other signature.
        """
        if (
            chunks is not None
            and hasattr(model, "predict_chunk_scores")
            and not isinstance(model, type(self))
        ):
            for call_kwargs in (
                {"feature_rows": rows, "apply_calibration": apply_calibration},
                {"feature_rows": rows},
                {"apply_calibration": apply_calibration},
                {},
            ):
                try:
                    raw = model.predict_chunk_scores(chunks, **call_kwargs)
                    return [self._clamp01(float(value)) for value in raw]
                except TypeError:
                    continue
        if hasattr(model, "predict_proba"):
            return [self._clamp01(row[1]) for row in model.predict_proba(rows)]
        if hasattr(model, "decision_function"):
            return [self._sigmoid(value) for value in model.decision_function(rows)]
        return [self._clamp01(value) for value in model.predict(rows)]

    @staticmethod
    def _accepts_apply_calibration(fn: Any) -> bool:
        """True if ``fn`` declares an ``apply_calibration`` parameter."""
        try:
            return "apply_calibration" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False

    def _raw_model_scores(
        self,
        rows: list[list[float]],
        chunks: list[list[dict[str, Any]]] | None = None,
        *,
        apply_calibration: bool = True,
    ) -> list[float]:
        per_model = [
            self._model_column(model, rows, chunks, apply_calibration)
            for model in self.models
        ]
        return self._blend_per_model(per_model, len(rows))

    def _blend_per_model(
        self, per_model: list[list[float]], n_rows: int
    ) -> list[float]:
        """Weighted-mean blend of per-model score columns."""
        weights = [max(0.0, float(value)) for value in self.model_weights[: len(per_model)]]
        if len(weights) != len(per_model) or sum(weights) <= 0.0:
            weights = [1.0 for _ in per_model]
        total_weight = sum(weights)

        scores: list[float] = []
        for row_index in range(n_rows):
            value = sum(
                weight * model_scores[row_index]
                for weight, model_scores in zip(weights, per_model)
            ) / total_weight
            scores.append(self._clamp01(value))
        return scores

    def _apply_calibrator(self, scores: list[float]) -> list[float]:
        if not scores or self.calibrator is None:
            return [self._clamp01(value) for value in scores]
        if hasattr(self.calibrator, "predict_proba"):
            calibrated = self.calibrator.predict_proba([[float(value)] for value in scores])
            return [self._clamp01(row[1]) for row in calibrated]
        if hasattr(self.calibrator, "transform"):
            return [self._clamp01(value) for value in self.calibrator.transform(scores)]
        return [self._clamp01(value) for value in scores]

    def _apply_score_remap(self, scores: list[float]) -> list[float]:
        if not scores or not self.score_remap:
            return [self._clamp01(value) for value in scores]
        if self.score_remap.get("kind") != "threshold_logit_v1":
            return [self._clamp01(value) for value in scores]
        try:
            threshold = float(self.score_remap.get("threshold", 0.5))
            temperature = max(float(self.score_remap.get("temperature", 0.25)), 1e-6)
        except (TypeError, ValueError):
            return [self._clamp01(value) for value in scores]
        output: list[float] = []
        for value in scores:
            clipped = max(1e-6, min(1.0 - 1e-6, float(value)))
            adjusted = (clipped - threshold) / temperature
            output.append(self._clamp01(1.0 / (1.0 + math.exp(-adjusted))))
        return output

    def _apply_score_logit(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        if abs(self.score_logit_bias) < 1e-12 and abs(self.score_logit_temperature - 1.0) < 1e-12:
            return [self._clamp01(value) for value in scores]
        output: list[float] = []
        for score in scores:
            value = max(1e-6, min(1.0 - 1e-6, float(score)))
            logit = math.log(value / (1.0 - value))
            adjusted = (logit + self.score_logit_bias) / self.score_logit_temperature
            output.append(self._clamp01(1.0 / (1.0 + math.exp(-adjusted))))
        return output

    def _apply_request_score_policy(self, scores: list[float]) -> list[float]:
        config = self.metadata.get("request_score_policy")
        if not scores or not isinstance(config, dict):
            return [self._clamp01(value) for value in scores]
        if config.get("kind") != "topk_v1":
            return [self._clamp01(value) for value in scores]

        count = len(scores)
        try:
            max_positive_count = int(config.get("max_positive_count", 1))
            max_positive_fraction = float(config.get("max_positive_fraction", 0.0) or 0.0)
            positive_floor = float(config.get("positive_floor", 0.501))
            positive_ceiling = float(config.get("positive_ceiling", 0.509))
            negative_ceiling = float(config.get("negative_ceiling", 0.49))
        except (TypeError, ValueError):
            return [self._clamp01(value) for value in scores]

        if max_positive_fraction > 0.0:
            max_positive_count = min(
                max_positive_count,
                max(1, int(math.floor(count * max_positive_fraction))),
            )
        max_positive_count = max(0, min(count, max_positive_count))
        positive_floor = self._clamp01(positive_floor)
        positive_ceiling = self._clamp01(max(positive_floor, positive_ceiling))
        negative_ceiling = min(self._clamp01(negative_ceiling), positive_floor - 1e-6)

        indexed_scores = [(index, self._clamp01(value)) for index, value in enumerate(scores)]
        ranked = sorted(indexed_scores, key=lambda item: item[1], reverse=True)
        output = [0.0 for _ in scores]

        positives = ranked[:max_positive_count]
        negatives = ranked[max_positive_count:]
        if positives:
            denom = max(1, len(positives) - 1)
            for rank, (index, _score) in enumerate(positives):
                relative = 1.0 - (rank / denom if denom else 0.0)
                output[index] = positive_floor + relative * (positive_ceiling - positive_floor)

        if negatives:
            negative_values = [score for _index, score in negatives]
            min_score = min(negative_values)
            max_score = max(negative_values)
            span = max(max_score - min_score, 1e-9)
            for index, score in negatives:
                relative = (score - min_score) / span
                output[index] = max(0.0, min(negative_ceiling, relative * negative_ceiling))

        return [round(self._clamp01(value), 6) for value in output]

    def predict_chunk_scores(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        if not chunks:
            return []
        rows = self._prepared_rows(chunks)
        raw_scores = self._raw_model_scores(rows, chunks=chunks)
        calibrated_scores = self._apply_calibrator(raw_scores)
        if str(self.metadata.get("live_score_mode", "")).strip() == "raw_monotone":
            return [round(self._clamp01(value), 10) for value in calibrated_scores]
        remapped_scores = self._apply_score_remap(calibrated_scores)
        logit_scores = self._apply_score_logit(remapped_scores)
        final_scores = self._apply_request_score_policy(logit_scores)
        return [round(self._clamp01(value), 6) for value in final_scores]

    def predict_chunk_score(self, chunk: list[dict[str, Any]]) -> float:
        scores = self.predict_chunk_scores([chunk])
        return scores[0] if scores else 0.5

    def benchmark_latency(
        self,
        chunks: list[list[dict[str, Any]]],
        repeats: int = 5,
    ) -> dict[str, float]:
        if not chunks:
            return {"latency_per_chunk_ms": 0.0, "total_latency_ms": 0.0}
        repeats = max(1, int(repeats))
        started = time.perf_counter()
        for _ in range(repeats):
            self.predict_chunk_scores(chunks)
        elapsed_ms = (time.perf_counter() - started) * 1000.0 / repeats
        return {
            "latency_per_chunk_ms": elapsed_ms / max(len(chunks), 1),
            "total_latency_ms": elapsed_ms,
        }
