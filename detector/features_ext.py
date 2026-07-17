"""Chunk -> feature views.

Two views are extracted from each chunk (a list of hand payloads for one hero):

* ``profile_features``  - broad per-hand/per-chunk statistical profile.
* ``behavior_features`` - sanitisation-invariant behavioural features. Bet sizes
  are quantised onto the validator's visible bb-bucket grid and no hero identity,
  seat or raw amount is used, so the training distribution matches what is served
  live.

Helpers prefixed ``_bv`` belong to the behaviour view. They shadow same-named
profile-view helpers with different implementations, so the two sets must stay
namespaced apart.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List

# ===========================================================================
# profile view
# ===========================================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _entropy(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(sum(counts.values()))
    if total <= 0.0 or len(counts) <= 1:
        return 0.0
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log(p + 1e-12)
    return _safe_div(ent, math.log(len(counts)))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    q = min(max(float(q), 0.0), 1.0)
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _mean(values: list[float]) -> float:
    return _safe_div(sum(values), len(values))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return math.sqrt(max(0.0, _mean([(v - m) * (v - m) for v in values])))


def _max_run_share(values: list[Any]) -> float:
    if not values:
        return 0.0
    longest = 1
    cur = 1
    for prev, cur_value in zip(values, values[1:]):
        if prev == cur_value:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    return _safe_div(longest, len(values))


def _amount_bucket(value: float) -> str:
    if value <= 0.0:
        return "z"
    if value <= 0.5:
        return "xs"
    if value <= 1.0:
        return "s"
    if value <= 2.0:
        return "m"
    if value <= 5.0:
        return "l"
        return "xl"


def _hand_features(hand: dict[str, Any]) -> dict[str, float]:
    metadata = hand.get("metadata") or {}
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    actions = hand.get("actions") or []

    max_seats = max(1, _safe_int(metadata.get("max_seats"), 6))
    hero_seat = _safe_int(metadata.get("hero_seat"), 0)
    button_seat = _safe_int(metadata.get("button_seat"), 0)
    player_count = float(len(players))
    street_count = float(len(streets))
    action_count = float(len(actions))

    action_types: list[str] = []
    actor_seats: list[int] = []
    street_names: list[str] = []
    amount_bb: list[float] = []
    pot_before: list[float] = []
    pot_after: list[float] = []
    stack_bb: list[float] = []
    raise_to_present = 0
    call_to_present = 0

    for player in players:
        if not isinstance(player, dict):
            continue
        stack_bb.append(_safe_div(_safe_float(player.get("starting_stack"), 0.0), 0.02))

    for action in actions:
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("action_type") or "").lower().strip()
        actor = _safe_int(action.get("actor_seat"), 0)
        street = str(action.get("street") or "").lower().strip()
        amt = _safe_float(action.get("normalized_amount_bb"), 0.0)
        pb = _safe_div(_safe_float(action.get("pot_before"), 0.0), 0.02)
        pa = _safe_div(_safe_float(action.get("pot_after"), 0.0), 0.02)

        action_types.append(action_type)
        if actor > 0:
            actor_seats.append(actor)
        street_names.append(street)
        amount_bb.append(max(0.0, amt))
        pot_before.append(max(0.0, pb))
        pot_after.append(max(0.0, pa))
        raise_to_present += int(action.get("raise_to") is not None)
        call_to_present += int(action.get("call_to") is not None)

    counts = Counter(action_types)
    meaningful = max(
        counts.get("call", 0)
        + counts.get("check", 0)
        + counts.get("bet", 0)
        + counts.get("raise", 0)
        + counts.get("fold", 0),
        1,
    )
    aggressive = counts.get("bet", 0) + counts.get("raise", 0)
    passive = counts.get("call", 0) + counts.get("check", 0)

    preflop_n = sum(1 for s in street_names if s == "preflop")
    postflop_n = sum(1 for s in street_names if s not in {"", "preflop"})
    nonzero_amount = sum(1 for v in amount_bb if v > 0.0)
    hero_actions = sum(1 for s in actor_seats if s == hero_seat and hero_seat > 0)
    button_actions = sum(1 for s in actor_seats if s == button_seat and button_seat > 0)

    pot_delta = [max(0.0, a - b) for a, b in zip(pot_after, pot_before)]
    monotonic = sum(
        1 for prev, cur in zip(pot_after, pot_after[1:]) if cur + 1e-9 >= prev
    )

    return {
        "schema_player_count": player_count,
        "schema_seat_utilization": _safe_div(player_count, max_seats),
        "schema_action_count": action_count,
        "schema_street_count": street_count,
        "schema_call_share": _safe_div(counts.get("call", 0), meaningful),
        "schema_check_share": _safe_div(counts.get("check", 0), meaningful),
        "schema_fold_share": _safe_div(counts.get("fold", 0), meaningful),
        "schema_bet_share": _safe_div(counts.get("bet", 0), meaningful),
        "schema_raise_share": _safe_div(counts.get("raise", 0), meaningful),
        "schema_blind_share": _safe_div(
            counts.get("small_blind", 0) + counts.get("big_blind", 0) + counts.get("ante", 0),
            max(1.0, action_count),
        ),
        "schema_allin_share": _safe_div(counts.get("all_in", 0), max(1.0, action_count)),
        "schema_aggression_share": _safe_div(aggressive, max(1.0, action_count)),
        "schema_passive_share": _safe_div(passive, max(1.0, action_count)),
        "schema_preflop_share": _safe_div(preflop_n, max(1.0, action_count)),
        "schema_postflop_share": _safe_div(postflop_n, max(1.0, action_count)),
        "schema_action_entropy": _entropy(action_types),
        "schema_actor_entropy": _entropy(actor_seats),
        "schema_street_entropy": _entropy(street_names),
        "schema_unique_actor_share": _safe_div(len(set(actor_seats)), max(1.0, player_count)),
        "schema_actor_switch_rate": _safe_div(
            sum(1 for prev, cur in zip(actor_seats, actor_seats[1:]) if prev != cur),
            max(len(actor_seats) - 1, 1),
        ),
        "schema_actor_run_max_share": _max_run_share(actor_seats),
        "schema_action_run_max_share": _max_run_share(action_types),
        "schema_amount_mean_bb": _mean(amount_bb),
        "schema_amount_std_bb": _std(amount_bb),
        "schema_amount_q90_bb": _quantile(amount_bb, 0.9),
        "schema_amount_max_bb": max(amount_bb) if amount_bb else 0.0,
        "schema_nonzero_amount_share": _safe_div(nonzero_amount, max(1.0, action_count)),
        "schema_pot_before_mean_bb": _mean(pot_before),
        "schema_pot_after_mean_bb": _mean(pot_after),
        "schema_pot_delta_mean_bb": _mean(pot_delta),
        "schema_pot_growth_bb": (
            max(pot_after) - min(pot_before) if pot_after and pot_before else 0.0
        ),
        "schema_pot_monotonic_rate": _safe_div(monotonic, max(len(pot_after) - 1, 1)),
        "schema_raise_to_share": _safe_div(raise_to_present, max(1.0, action_count)),
        "schema_call_to_share": _safe_div(call_to_present, max(1.0, action_count)),
        "schema_starting_stack_mean_bb": _mean(stack_bb),
        "schema_starting_stack_std_bb": _std(stack_bb),
        "schema_starting_stack_iqr_bb": _quantile(stack_bb, 0.75) - _quantile(stack_bb, 0.25),
        "schema_hero_action_share": _safe_div(hero_actions, max(1.0, action_count)),
        "schema_button_action_share": _safe_div(button_actions, max(1.0, action_count)),
        "schema_hero_button_same": float(hero_seat > 0 and hero_seat == button_seat),
    }


def _aggregate_feature(prefix: str, values: list[float], out: dict[str, float]) -> None:
    out[f"{prefix}_mean"] = _mean(values)
    out[f"{prefix}_std"] = _std(values)
    out[f"{prefix}_min"] = min(values) if values else 0.0
    out[f"{prefix}_max"] = max(values) if values else 0.0
    out[f"{prefix}_q10"] = _quantile(values, 0.1)
    out[f"{prefix}_q50"] = _quantile(values, 0.5)
    out[f"{prefix}_q90"] = _quantile(values, 0.9)


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    if not chunk:
        return {"hand_count": 0.0}

    out: dict[str, float] = {"hand_count": float(len(chunk))}
    per_hand = [_hand_features(hand) for hand in chunk]
    feature_names = sorted(per_hand[0].keys())

    for name in feature_names:
        series = [float(features[name]) for features in per_hand]
        _aggregate_feature(name, series, out)

    action_signatures: list[tuple[str, ...]] = []
    actor_signatures: list[tuple[int, ...]] = []
    street_signatures: list[tuple[str, ...]] = []
    amount_bucket_signatures: list[tuple[str, ...]] = []

    high_aggressive = 0
    low_action_entropy = 0
    high_actor_entropy = 0
    long_action_hand = 0

    for hand, feats in zip(chunk, per_hand):
        actions = hand.get("actions") or []
        action_types = tuple(str((a or {}).get("action_type") or "").lower().strip() for a in actions)
        actor_seq = tuple(
            _safe_int((a or {}).get("actor_seat"), 0) for a in actions if _safe_int((a or {}).get("actor_seat"), 0) > 0
        )
        street_seq = tuple(str((a or {}).get("street") or "").lower().strip() for a in actions)
        amounts = [
            max(0.0, _safe_float((a or {}).get("normalized_amount_bb"), 0.0))
            for a in actions
        ]
        amount_buckets = tuple(_amount_bucket(value) for value in amounts)

        action_signatures.append(action_types)
        actor_signatures.append(actor_seq)
        street_signatures.append(street_seq)
        amount_bucket_signatures.append(amount_buckets)

        high_aggressive += int(feats["schema_aggression_share"] >= 0.35)
        low_action_entropy += int(feats["schema_action_entropy"] <= 0.35)
        high_actor_entropy += int(feats["schema_actor_entropy"] >= 0.75)
        long_action_hand += int(feats["schema_action_count"] >= 12.0)

    n = float(len(chunk))
    out["schema_action_signature_top_share"] = _safe_div(max(Counter(action_signatures).values()), n)
    out["schema_action_signature_unique_share"] = _safe_div(len(set(action_signatures)), n)
    out["schema_actor_signature_top_share"] = _safe_div(max(Counter(actor_signatures).values()), n)
    out["schema_actor_signature_unique_share"] = _safe_div(len(set(actor_signatures)), n)
    out["schema_street_signature_top_share"] = _safe_div(max(Counter(street_signatures).values()), n)
    out["schema_street_signature_unique_share"] = _safe_div(len(set(street_signatures)), n)
    out["schema_amount_bucket_signature_top_share"] = _safe_div(
        max(Counter(amount_bucket_signatures).values()), n
    )
    out["schema_amount_bucket_signature_unique_share"] = _safe_div(
        len(set(amount_bucket_signatures)), n
    )
    out["schema_high_aggression_hand_rate"] = _safe_div(high_aggressive, n)
    out["schema_low_action_entropy_hand_rate"] = _safe_div(low_action_entropy, n)
    out["schema_high_actor_entropy_hand_rate"] = _safe_div(high_actor_entropy, n)
    out["schema_long_action_hand_rate"] = _safe_div(long_action_hand, n)
    return out

# ===========================================================================
# behaviour view (sanitisation-invariant)
# ===========================================================================

# the validator's exact visible bb-bucket grid (payload_view._VISIBLE_BB_BUCKETS)
_BUCKETS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0,
           56.0, 84.0, 126.0)
_MEANINGFUL = ("check", "call", "bet", "raise", "fold")
_AGGR = ("bet", "raise")
_PASSIVE = ("check", "call")
# per-hand scalar feature keys aggregated across the chunk
_AGG_STATS = ("mean", "std", "min", "max", "q10", "q50", "q90")


def _f(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return d
        return v
    except (TypeError, ValueError):
        return d


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _bucket_index(bb: float) -> int:
    return min(range(len(_BUCKETS)), key=lambda i: abs(_BUCKETS[i] - bb))


def _bv_entropy(counts: List[float]) -> float:
    tot = sum(counts)
    if tot <= 0:
        return 0.0
    ps = [c / tot for c in counts if c > 0]
    if len(ps) <= 1:
        return 0.0
    ent = -sum(p * math.log(p) for p in ps)
    return ent / math.log(len(ps))  # normalized to [0,1]


def _bv_max_run_share(seq: List[Any]) -> float:
    if not seq:
        return 0.0
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best / len(seq)


def _switch_rate(seq: List[Any]) -> float:
    if len(seq) < 2:
        return 0.0
    return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1]) / (len(seq) - 1)


def _bv_quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _hand_view(hand: Dict[str, Any]) -> Dict[str, Any]:
    """Per-hand summary + the signature tuples used for cross-hand regularity."""
    meta = hand.get("metadata") or {}
    hero = meta.get("hero_seat")
    actions = hand.get("actions") or []
    streets = hand.get("streets") or []

    a_types, actors, a_streets, buckets, roles = [], [], [], [], []
    amts, pots_before, pots_after = [], [], []
    for a in actions:
        at = str(a.get("action_type") or "").strip().lower()
        if not at:
            continue
        a_types.append(at)
        seat = a.get("actor_seat")
        actors.append(seat)
        roles.append("H" if (hero is not None and seat == hero) else "o")
        a_streets.append(str(a.get("street") or "").strip().lower())
        bb = _f(a.get("normalized_amount_bb"))
        buckets.append(_bucket_index(bb))
        if at in _AGGR and bb > 0:
            amts.append(bb)
        pots_before.append(_f(a.get("pot_before")))
        pots_after.append(_f(a.get("pot_after")))

    n = len(a_types)
    meaningful = [t for t in a_types if t in _MEANINGFUL]
    nm = max(len(meaningful), 1)
    cnt = Counter(a_types)

    feat: Dict[str, float] = {}
    feat["n_actions"] = float(n)
    for t in _MEANINGFUL:
        feat[f"share_{t}"] = _div(cnt.get(t, 0), nm)
    feat["share_blind"] = _div(cnt.get("small_blind", 0) + cnt.get("big_blind", 0), nm)
    feat["share_aggr"] = _div(sum(cnt.get(t, 0) for t in _AGGR), nm)
    feat["share_passive"] = _div(sum(cnt.get(t, 0) for t in _PASSIVE), nm)
    feat["aggr_ratio"] = _div(sum(cnt.get(t, 0) for t in _AGGR),
                              max(sum(cnt.get(t, 0) for t in _PASSIVE), 1))
    feat["share_preflop"] = _div(sum(1 for s in a_streets if s == "preflop"), max(n, 1))
    feat["share_postflop"] = _div(sum(1 for s in a_streets if s in ("flop", "turn", "river")), max(n, 1))

    feat["action_entropy"] = _bv_entropy([cnt.get(t, 0) for t in set(a_types)])
    feat["actor_entropy"] = _bv_entropy(list(Counter(actors).values()))
    feat["street_entropy"] = _bv_entropy(list(Counter(a_streets).values()))
    feat["action_switch_rate"] = _switch_rate(a_types)
    feat["actor_switch_rate"] = _switch_rate(actors)
    feat["action_run_max_share"] = _bv_max_run_share(a_types)
    feat["actor_run_max_share"] = _bv_max_run_share(actors)

    feat["hero_action_share"] = _div(sum(1 for r in roles if r == "H"), max(n, 1))
    feat["n_distinct_actors"] = float(len(set(actors)))
    feat["n_streets"] = float(len(set(s for s in a_streets if s)) or len(streets))

    # bet sizing in bb (quantize-aware: amts are already coarse on live)
    if amts:
        s = sorted(amts)
        feat["amt_mean"] = sum(amts) / len(amts)
        feat["amt_std"] = (sum((x - feat["amt_mean"]) ** 2 for x in amts) / len(amts)) ** 0.5
        feat["amt_max"] = s[-1]
        feat["amt_q90"] = _bv_quantile(s, 0.9)
        feat["amt_min"] = s[0]
    else:
        feat["amt_mean"] = feat["amt_std"] = feat["amt_max"] = feat["amt_q90"] = feat["amt_min"] = 0.0
    feat["nonzero_amt_share"] = _div(len(amts), max(n, 1))
    feat["bucket_entropy"] = _bv_entropy(list(Counter(buckets).values()))

    # pot dynamics
    if pots_after:
        feat["pot_after_mean"] = sum(pots_after) / len(pots_after)
        feat["pot_before_mean"] = sum(pots_before) / len(pots_before)
        deltas = [pots_after[i] - pots_before[i] for i in range(len(pots_after))]
        feat["pot_delta_mean"] = sum(deltas) / len(deltas)
        feat["pot_growth"] = _div(pots_after[-1], pots_before[0]) if pots_before and pots_before[0] else 0.0
        feat["pot_monotonic_rate"] = _div(sum(1 for d in deltas if d >= 0), max(len(deltas), 1))
    else:
        feat["pot_after_mean"] = feat["pot_before_mean"] = feat["pot_delta_mean"] = 0.0
        feat["pot_growth"] = feat["pot_monotonic_rate"] = 0.0

    sig = {
        "action_sig": tuple(a_types),
        "role_sig": tuple(roles),
        "street_sig": tuple(a_streets),
        "bucket_sig": tuple(buckets),
    }
    return {"feat": feat, "sig": sig}


def _aggregate(prefix: str, series: List[float], out: Dict[str, float]) -> None:
    if not series:
        for st in _AGG_STATS:
            out[f"{prefix}_{st}"] = 0.0
        return
    s = sorted(series)
    mean = sum(series) / len(series)
    out[f"{prefix}_mean"] = mean
    out[f"{prefix}_std"] = (sum((x - mean) ** 2 for x in series) / len(series)) ** 0.5
    out[f"{prefix}_min"] = s[0]
    out[f"{prefix}_max"] = s[-1]
    out[f"{prefix}_q10"] = _bv_quantile(s, 0.10)
    out[f"{prefix}_q50"] = _bv_quantile(s, 0.50)
    out[f"{prefix}_q90"] = _bv_quantile(s, 0.90)


def extract_behavior(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    hands = hands or []
    out["hand_count"] = float(len(hands))
    if not hands:
        return out

    views = [_hand_view(h) for h in hands]
    # aggregate every per-hand scalar to chunk-level order-stats
    keys = list(views[0]["feat"].keys())
    for k in keys:
        _aggregate(k, [v["feat"].get(k, 0.0) for v in views], out)

    # cross-hand signature regularity (the bot "replays the same hand" tell)
    n = len(views)
    for name in ("action_sig", "role_sig", "street_sig", "bucket_sig"):
        sigs = [v["sig"][name] for v in views]
        c = Counter(sigs)
        out[f"{name}_top_share"] = _div(max(c.values()), n)
        out[f"{name}_unique_share"] = _div(len(c), n)
    # rate of "extreme" hands across the chunk
    out["high_aggr_hand_rate"] = _div(sum(1 for v in views if v["feat"]["share_aggr"] > 0.5), n)
    out["low_entropy_hand_rate"] = _div(sum(1 for v in views if v["feat"]["action_entropy"] < 0.3), n)
    out["zero_hero_action_rate"] = _div(sum(1 for v in views if v["feat"]["hero_action_share"] == 0.0), n)
    return out


# ===========================================================================
# view adapters (train == serve: both trainer and miner call these)
# ===========================================================================


def profile_features(chunk) -> Dict[str, float]:
    """Profile view for one chunk, plus the chunk's hand count."""
    d = chunk_features(chunk or [])
    d["hand_count"] = float(len(chunk or []))
    return d


def behavior_features(chunk) -> Dict[str, float]:
    """Sanitisation-invariant behaviour view for one chunk."""
    return extract_behavior(chunk or [])
