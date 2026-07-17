"""Poker44 subnet 126 miner neuron."""

import argparse
import asyncio
import hashlib
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

from detector.inference import Poker44Model

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = REPO_ROOT / "detector" / "artifacts" / "model.joblib"
# Every entry must be present in the published repo, so the manifest never names
# a file a reader cannot open. Weights are fingerprinted via artifact_sha256.
IMPLEMENTATION_FILES = (
    "neurons/miner.py",
    "detector/inference.py",
    "detector/features.py",
    "detector/features_ext.py",
    "detector/artifacts/meta.json",
)


def _artifact_sha256(artifact_path: Path) -> str:
    env_hash = os.getenv("POKER44_MODEL_ARTIFACT_SHA256", "").strip()
    if env_hash:
        return env_hash
    digest = hashlib.sha256()
    with artifact_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _repo_commit(repo_root: Path) -> str:
    env_commit = os.getenv("POKER44_MODEL_REPO_COMMIT", "").strip()
    if env_commit:
        return env_commit
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return ""


def _artifact_url(repo_url: str, commit: str, artifact_path: Path) -> str:
    """Permalink to the served weights, pinned to the commit rather than a branch
    so it keeps resolving to the exact bytes artifact_sha256 describes.

    Returns "" when the miner was pointed at an artifact other than the in-repo
    one, rather than advertising a file that is not what is being served.
    """
    if not repo_url or not commit:
        return ""
    try:
        if artifact_path.resolve() != DEFAULT_ARTIFACT.resolve():
            return ""
    except OSError:
        return ""
    base = repo_url.strip().rstrip("/")
    if base.endswith(".git"):
        base = base[:-4]
    return f"{base}/raw/{commit}/detector/artifacts/model.joblib"


class Miner(BaseMinerNeuron):
    """Poker44 bot-detection miner."""

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        super().add_args(parser)
        parser.add_argument(
            "--neuron.name",
            type=str,
            default="miner",
            help="Neuron name used for logging paths.",
        )

    def __init__(self, config=None):
        super().__init__(config=config)
        artifact_path = Path(os.getenv("POKER44_MODEL_PATH", str(DEFAULT_ARTIFACT)))
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {artifact_path}. "
                "Train first: python -m detector.train"
            )

        self.predictor = Poker44Model(artifact_path)
        # Latency safeguard: cap hands scored per chunk (0 disables). Live chunks
        # run ~80-100 hands, so the default 120 is effectively off for normal
        # traffic and only trims pathologically large chunks that could time out.
        self.max_hands_per_chunk_eval = max(
            0, int(os.getenv("POKER44_MAX_HANDS_PER_CHUNK_EVAL", "120"))
        )
        artifact_sha256 = _artifact_sha256(artifact_path)
        repo_url = os.getenv("POKER44_MODEL_REPO_URL", "")
        repo_commit = _repo_commit(REPO_ROOT)
        self.model_manifest = build_local_model_manifest(
            repo_root=REPO_ROOT,
            implementation_files=[REPO_ROOT / path for path in IMPLEMENTATION_FILES],
            defaults={
                "model_name": os.getenv("POKER44_MODEL_NAME", "poker44-miner-v2"),
                "model_version": os.getenv("POKER44_MODEL_VERSION", "2.0.0"),
                "framework": "scikit-learn",
                "license": "MIT",
                "repo_url": repo_url,
                "repo_commit": repo_commit,
                "artifact_sha256": artifact_sha256,
                "artifact_url": _artifact_url(repo_url, repo_commit, artifact_path),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": "Trained only on the public Poker44 benchmark.",
                "training_data_sources": ["poker44-public-benchmark"],
                "private_data_attestation": "No validator-only data is used.",
                "data_attestation": "Features use miner-visible behaviour only.",
                "notes": "Behavioural bot detector.",
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        bt.logging.info(f"Detector miner loaded artifact={artifact_path}")
        bt.logging.info(
            f"Manifest status={self.manifest_compliance['status']} "
            f"digest={self.manifest_digest}"
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _align_score_count(scores: list, n: int) -> list:
        """Force exactly one score per chunk.

        The live validator DISCARDS the entire response and scores the whole
        cycle 0 when ``len(scores) != len(chunks)``. A partial predictor failure
        that returns a short/long list would therefore zero an eval cycle, so we
        always emit exactly ``n`` scores: extras dropped, shortfalls padded with
        0.0 (treated as human -- the safe, non-false-positive value).
        """
        scores = list(scores)
        if len(scores) > n:
            return scores[:n]
        if len(scores) < n:
            return scores + [0.0] * (n - len(scores))
        return scores

    def _compress_chunk(self, chunk: list) -> list:
        """Evenly downsample a chunk's hands to bound inference latency."""
        limit = self.max_hands_per_chunk_eval
        if limit <= 0 or len(chunk) <= limit:
            return chunk
        if limit == 1:
            return [chunk[len(chunk) // 2]]
        last_index = len(chunk) - 1
        slots = limit - 1
        indices = {
            min(last_index, round(index * last_index / slots))
            for index in range(limit)
        }
        return [chunk[index] for index in sorted(indices)]

    @staticmethod
    def _fallback_score_chunk(chunk: list) -> float:
        """Model-free heuristic used only if the predictor throws.

        Emits a bounded, non-zero risk score so a transient inference failure
        degrades gracefully instead of zeroing the eval cycle. Rewards passive,
        consistent, deep-street play (bot-like); penalizes folding/aggression.
        """
        if not chunk:
            return 0.5
        hand_scores: list[float] = []
        call_ratios: list[float] = []
        aggression_ratios: list[float] = []
        for hand in chunk:
            actions = hand.get("actions") or []
            counts = Counter((a.get("action_type") or "").lower() for a in actions)
            meaningful = max(
                1,
                sum(counts.get(k, 0) for k in ("call", "check", "bet", "raise", "fold")),
            )
            aggressive = counts.get("bet", 0) + counts.get("raise", 0)
            passive = counts.get("call", 0) + counts.get("check", 0)
            call_ratio = counts.get("call", 0) / meaningful
            check_ratio = counts.get("check", 0) / meaningful
            fold_ratio = counts.get("fold", 0) / meaningful
            raise_ratio = counts.get("raise", 0) / meaningful
            aggression_ratio = aggressive / max(aggressive + passive, 1)
            street_depth = len(hand.get("streets") or []) / 4.0
            score = 0.30 * Miner._clamp01(street_depth)
            score += 0.22 * Miner._clamp01(call_ratio / 0.32)
            score += 0.12 * Miner._clamp01(check_ratio / 0.28)
            score -= 0.16 * Miner._clamp01(fold_ratio / 0.55)
            score -= 0.14 * Miner._clamp01(raise_ratio / 0.22)
            score -= 0.10 * Miner._clamp01(aggression_ratio / 0.55)
            hand_scores.append(Miner._clamp01(score))
            call_ratios.append(call_ratio)
            aggression_ratios.append(aggression_ratio)
        avg = sum(hand_scores) / len(hand_scores)
        bonus = 0.0
        if len(hand_scores) > 1:
            bonus += 0.10 * Miner._clamp01(1.0 - (max(call_ratios) - min(call_ratios)) / 0.60)
            bonus += 0.08 * Miner._clamp01(
                1.0 - (max(aggression_ratios) - min(aggression_ratios)) / 0.70
            )
        return round(Miner._clamp01(avg + bonus), 6)

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = [self._compress_chunk(list(chunk or [])) for chunk in (synapse.chunks or [])]
        backend_used = "model"
        try:
            # Offload CPU-bound inference to a worker thread so the asyncio event
            # loop stays free to verify/accept other validators' requests. Running
            # inline blocks the loop for the whole batch, aging incoming nonces
            # past the verification window ("Nonce is too old") under big snapshots.
            scores = await asyncio.to_thread(self.predictor.predict_chunk_scores, chunks)
        except Exception as err:
            bt.logging.warning(
                f"Predictor failure during chunk scoring: {err}. "
                "Falling back to heuristic backend."
            )
            backend_used = "heuristic-fallback"
            scores = [self._fallback_score_chunk(chunk) for chunk in chunks]

        # Guard against a count mismatch zeroing the entire eval cycle.
        scores = self._align_score_count(scores, len(chunks))
        scores = [self._clamp01(score) for score in scores]
        synapse.risk_scores = scores
        synapse.predictions = [score >= 0.5 for score in scores]
        synapse.model_manifest = dict(self.model_manifest)

        bt.logging.info(f"Scored {len(chunks)} chunks with backend={backend_used}.")
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(
                f"Miner UID={miner.uid} incentive={float(miner.metagraph.I[miner.uid]):.6f}"
            )
            time.sleep(300)
