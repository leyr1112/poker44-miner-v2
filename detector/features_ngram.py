"""Action-sequence n-gram features.

Encodes the ORDER of betting actions, which the aggregate/order-statistic
features do not: each hand's action stream is tokenised into
street+action+size-bucket tokens (e.g. ``pF0``, ``fK0``, ``tBp``) and the chunk's
unigram/bigram/trigram counts plus per-relative-seat position tokens are pooled,
normalised per hand so short benchmark chunks and long live chunks stay
comparable. A scripted policy replays near-identical action sequences, so its
n-gram distribution is a distinctive tell. Counts are keyed ``schema_ngram_*``;
the trained artifact's ``feature_names`` retains only the informative subset.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

_ACTION_CODES = {"fold": "F", "call": "C", "raise": "R", "check": "K",
                 "bet": "B", "action": "A", "all_in": "I"}


def _f(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        return v if v == v else d
    except (TypeError, ValueError):
        return d


def _hand_grams(hand: dict[str, Any]) -> Counter:
    actions = hand.get("actions") or []
    meta = hand.get("metadata") or {}
    button = meta.get("button_seat")
    max_seats = meta.get("max_seats") or 6
    tokens: list[str] = []
    grams: Counter = Counter()
    seats: set = set()
    for a in actions:
        street = (a.get("street") or "x")[:1]
        act = _ACTION_CODES.get(a.get("action_type") or "x", "X")
        amount = _f(a.get("amount"), 0.0)
        pot_before = _f(a.get("pot_before"), 0.0)
        if amount <= 0:
            bucket = "0"
        elif pot_before <= 0:
            bucket = "Q"
        else:
            r = amount / pot_before
            bucket = "s" if r < 0.4 else ("m" if r < 0.9 else ("p" if r < 1.5 else "o"))
        tok = street + act + bucket
        tokens.append(tok)
        grams[tok] += 1
        try:
            rel = (int(a.get("actor_seat")) - int(button)) % int(max_seats)
            grams["pos" + str(rel) + act] += 1
        except Exception:
            pass
        seats.add(a.get("actor_seat"))
    for i in range(len(tokens) - 1):
        grams[tokens[i] + "__" + tokens[i + 1]] += 1
        if i + 2 < len(tokens):
            grams[tokens[i] + "__" + tokens[i + 1] + "__" + tokens[i + 2]] += 1
    grams["len"] = len(tokens)
    grams["nseats"] = len(seats)
    return grams


def ngram_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    """Per-hand-normalised bag-of-ngram counts pooled over the chunk."""
    chunk = chunk or []
    n = max(len(chunk), 1)
    total: Counter = Counter()
    for hand in chunk:
        total.update(_hand_grams(hand))
    return {f"schema_ngram_{k.replace('?', 'Q')}": v / n for k, v in total.items()}
