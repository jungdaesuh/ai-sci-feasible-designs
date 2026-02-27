"""Shared staged-governor seed planning (focus -> bridge -> repair)."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class StagedSeedPlan:
    """Planned staged seeds for one cycle."""

    seeds: list[dict[str, Any]]
    focus_hash: str
    partner_hash: str | None
    worst_constraint: str | None


def expand_parent_group_staged_offspring(
    *,
    parent_group: Sequence[Mapping[str, Any]],
    worst_constraint: str | None,
    max_repair_candidates: int,
    bridge_blend_t: float,
    offspring_per_parent: int,
) -> list[dict[str, Any]]:
    """Expand focus/bridge/repair offspring from a selected parent group."""
    if not parent_group:
        return []

    focus = parent_group[0]
    partner = parent_group[1] if len(parent_group) > 1 else None
    focus_hash = str(focus.get("design_hash", ""))
    partner_hash = str(partner.get("design_hash", "")) if partner is not None else None
    focus_params = _coerce_params(focus.get("params"))
    if focus_params is None:
        return []

    resolved_worst = worst_constraint
    if resolved_worst is None:
        violations = _positive_violations(focus)
        resolved_worst, _ = worst_constraint_from_violations(violations)

    bridge_params = copy.deepcopy(focus_params)
    if partner is not None:
        partner_params = _coerce_params(partner.get("params"))
        if partner_params is not None:
            bridge_params = blend_boundary_params(
                focus_params,
                partner_params,
                t=bridge_blend_t,
            )

    seeds: list[dict[str, Any]] = [
        _seed_payload(
            params=focus_params,
            phase="focus",
            focus_hash=focus_hash,
            partner_hash=partner_hash,
            worst_constraint=resolved_worst,
            improvement_reason="focus_near_feasible_anchor",
        )
    ]
    if partner is not None:
        seeds.append(
            _seed_payload(
                params=bridge_params,
                phase="bridge",
                focus_hash=focus_hash,
                partner_hash=partner_hash,
                worst_constraint=resolved_worst,
                improvement_reason="bridge_blend_toward_partner",
            )
        )

    repair_limit = max(1, int(max_repair_candidates)) * max(
        1, int(offspring_per_parent)
    )
    repair_candidates = build_repair_candidates(
        params=bridge_params,
        worst_constraint=resolved_worst,
        max_candidates=repair_limit,
    )
    for repair_params in repair_candidates:
        seeds.append(
            _seed_payload(
                params=repair_params,
                phase="repair",
                focus_hash=focus_hash,
                partner_hash=partner_hash,
                worst_constraint=resolved_worst,
                improvement_reason="constraint_targeted_repair",
            )
        )

    return _dedupe_seeds(seeds)


def apply_delta_recipe(
    params: Mapping[str, Any],
    delta_recipe: Any,
) -> dict[str, Any] | None:
    """Apply sparse additive coefficient deltas onto boundary params."""
    updated = copy.deepcopy(dict(params))
    normalized_recipe = _normalize_delta_recipe(delta_recipe)
    if not normalized_recipe:
        return updated

    arrays = _matrix_fields(updated)
    if not arrays:
        return None

    for delta_entry in normalized_recipe:
        key_raw = delta_entry.get("key")
        delta_raw = delta_entry.get("delta")
        if not isinstance(key_raw, str):
            continue
        if isinstance(delta_raw, bool) or not isinstance(delta_raw, (int, float)):
            continue
        parts = key_raw.split(".")
        if len(parts) != 3:
            continue
        field, row_text, col_text = parts
        if field not in arrays:
            continue
        try:
            row = int(row_text)
            col = int(col_text)
        except ValueError:
            continue
        arr = arrays[field]
        if row < 0 or col < 0 or row >= arr.shape[0] or col >= arr.shape[1]:
            continue
        arr[row, col] += float(delta_raw)
        arrays[field] = arr

    for name, arr in arrays.items():
        updated[name] = arr.tolist()
    return updated


def _normalize_delta_recipe(recipe: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if isinstance(recipe, list):
        for entry in recipe:
            if not isinstance(entry, Mapping):
                continue
            key_raw = entry.get("key")
            delta_raw = entry.get("delta")
            if not isinstance(key_raw, str):
                continue
            if isinstance(delta_raw, bool) or not isinstance(delta_raw, (int, float)):
                continue
            delta = float(delta_raw)
            if not math.isfinite(delta):
                continue
            normalized.append({"key": key_raw, "delta": delta})
        return normalized

    if not isinstance(recipe, Mapping):
        return normalized

    key_raw = recipe.get("key")
    delta_raw = recipe.get("delta")
    if isinstance(key_raw, str) and isinstance(delta_raw, (int, float)):
        if not isinstance(delta_raw, bool):
            delta = float(delta_raw)
            if math.isfinite(delta):
                normalized.append({"key": key_raw, "delta": delta})
        return normalized

    changes_raw = recipe.get("changes")
    if not isinstance(changes_raw, list):
        return normalized
    for change in changes_raw:
        if not isinstance(change, Mapping):
            continue
        field_raw = change.get("field")
        row_raw = change.get("row")
        col_raw = change.get("col")
        delta_raw = change.get("delta")
        if not isinstance(field_raw, str):
            continue
        if isinstance(row_raw, bool) or not isinstance(row_raw, (int, float)):
            continue
        if isinstance(col_raw, bool) or not isinstance(col_raw, (int, float)):
            continue
        row_float = float(row_raw)
        col_float = float(col_raw)
        if not row_float.is_integer() or not col_float.is_integer():
            continue
        if isinstance(delta_raw, bool) or not isinstance(delta_raw, (int, float)):
            continue
        delta = float(delta_raw)
        if not math.isfinite(delta):
            continue
        row = int(row_float)
        col = int(col_float)
        normalized.append(
            {
                "key": f"{field_raw}.{row}.{col}",
                "delta": delta,
            }
        )
    return normalized


def build_delta_replay_seeds(
    *,
    focus_params: Mapping[str, Any],
    case_deltas: Sequence[Mapping[str, Any]],
    top_k: int,
    focus_hash: str,
    worst_constraint: str | None,
) -> list[dict[str, Any]]:
    """Create replay seeds from nearest-case delta recipes."""
    if top_k <= 0:
        return []

    seeds: list[dict[str, Any]] = []
    for case in case_deltas:
        recipe = _normalize_delta_recipe(case.get("delta_recipe"))
        if not recipe:
            continue
        replayed = apply_delta_recipe(focus_params, recipe)
        if replayed is None:
            continue
        source_hash = case.get("design_hash")
        source_hash_text = str(source_hash) if source_hash is not None else ""
        seeds.append(
            _seed_payload(
                params=replayed,
                phase="delta_replay",
                focus_hash=focus_hash,
                partner_hash=source_hash_text or None,
                worst_constraint=worst_constraint,
                improvement_reason="nearest_case_delta_replay",
            )
        )
        if len(seeds) >= top_k:
            break
    return _dedupe_seeds(seeds)


def worst_constraint_from_violations(
    violations: Mapping[str, float],
) -> tuple[str | None, float]:
    """Return the largest positive violation entry."""
    if not violations:
        return None, 0.0
    worst_name: str | None = None
    worst_val = -float("inf")
    for name, value in violations.items():
        if value > worst_val:
            worst_name = str(name)
            worst_val = float(value)
    return worst_name, float(worst_val)


def build_staged_seed_plan_from_snapshots(
    *,
    snapshots: Sequence[Mapping[str, Any]],
    problem: str,
    near_feasibility_threshold: float,
    max_repair_candidates: int,
    bridge_blend_t: float,
) -> StagedSeedPlan | None:
    """Create staged seed candidates from recent evaluated snapshots."""
    if not snapshots:
        return None

    focus = _select_focus_candidate(
        snapshots,
        problem=problem,
        near_feasibility_threshold=near_feasibility_threshold,
    )
    if focus is None:
        return None

    focus_hash = str(focus.get("design_hash", ""))
    violations = _positive_violations(focus)
    worst_constraint, _ = worst_constraint_from_violations(violations)
    partner = _select_partner_candidate(
        snapshots,
        focus_hash=focus_hash,
        problem=problem,
        worst_constraint=worst_constraint,
    )
    partner_hash = str(partner.get("design_hash", "")) if partner is not None else None
    group: list[Mapping[str, Any]] = [focus]
    if partner is not None:
        group.append(partner)
    deduped = expand_parent_group_staged_offspring(
        parent_group=group,
        worst_constraint=worst_constraint,
        max_repair_candidates=max_repair_candidates,
        bridge_blend_t=bridge_blend_t,
        offspring_per_parent=1,
    )
    if not deduped:
        return None
    return StagedSeedPlan(
        seeds=deduped,
        focus_hash=focus_hash,
        partner_hash=partner_hash,
        worst_constraint=worst_constraint,
    )


def build_repair_candidates(
    *,
    params: Mapping[str, Any],
    worst_constraint: str | None,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Generate bounded repair variants targeted at the dominant constraint."""
    candidates: list[dict[str, Any] | None] = []
    key = (worst_constraint or "").lower()

    if "mirror" in key:
        candidates.extend(
            [
                _scale_non_axisymmetric(params, 0.96),
                _scale_non_axisymmetric(params, 0.98),
                _scale_m_ge(params, m_min=3, factor=0.90),
            ]
        )
    elif "log10_qi" in key or key == "qi" or "qi_" in key:
        candidates.extend(
            [
                _scale_m_ge(params, m_min=3, factor=0.90),
                _scale_m_ge(params, m_min=3, factor=0.95),
                _scale_abs_n(params, abs_n=3, factor=1.04),
            ]
        )
    elif "flux" in key:
        candidates.extend(
            [
                _scale_m_ge(params, m_min=3, factor=0.80),
                _scale_m_ge(params, m_min=3, factor=0.90),
            ]
        )
    elif "vacuum" in key:
        candidates.extend(
            [
                _scale_m_ge(params, m_min=2, factor=0.90),
                _scale_m_ge(params, m_min=2, factor=0.95),
            ]
        )
    elif "iota" in key:
        candidates.extend(
            [
                _scale_abs_n(params, abs_n=1, factor=1.02),
                _scale_abs_n(params, abs_n=1, factor=1.04),
            ]
        )
    elif "aspect" in key or "elong" in key:
        candidates.extend(
            [
                _scale_non_axisymmetric(params, 0.95),
                _scale_non_axisymmetric(params, 0.98),
            ]
        )
    else:
        candidates.extend(
            [
                _scale_m_ge(params, m_min=3, factor=0.95),
                _scale_non_axisymmetric(params, 0.98),
            ]
        )

    cleaned = [candidate for candidate in candidates if isinstance(candidate, dict)]
    return cleaned[: max(0, int(max_candidates))]


def blend_boundary_params(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    t: float,
) -> dict[str, Any]:
    """Blend two boundary parameter sets with a convex factor t."""
    alpha = float(min(1.0, max(0.0, t)))
    blended = copy.deepcopy(dict(left))
    for key in ("r_cos", "z_sin", "r_sin", "z_cos"):
        lhs = left.get(key)
        rhs = right.get(key)
        if lhs is None or rhs is None:
            continue
        lhs_arr = np.asarray(lhs, dtype=float)
        rhs_arr = np.asarray(rhs, dtype=float)
        if lhs_arr.shape != rhs_arr.shape:
            continue
        mixed = ((1.0 - alpha) * lhs_arr) + (alpha * rhs_arr)
        blended[key] = mixed.tolist()
    return blended


def _dedupe_seeds(seeds: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for seed in seeds:
        params = seed.get("params")
        if not isinstance(params, Mapping):
            continue
        key = json.dumps(params, sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(seed))
    return out


def _seed_payload(
    *,
    params: Mapping[str, Any],
    phase: str,
    focus_hash: str,
    partner_hash: str | None,
    worst_constraint: str | None,
    improvement_reason: str,
) -> dict[str, Any]:
    lineage_parent_hashes = [focus_hash]
    if partner_hash:
        lineage_parent_hashes.append(partner_hash)
    return {
        "params": copy.deepcopy(dict(params)),
        "source": "staged_governor",
        "lineage_parent_hashes": lineage_parent_hashes,
        "operator_family": f"staged_{phase}",
        "staged_governor": {
            "phase": phase,
            "focus_hash": focus_hash,
            "partner_hash": partner_hash,
            "worst_constraint": worst_constraint,
        },
        "improvement_reason": improvement_reason,
    }


def _coerce_params(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    if "r_cos" not in value or "z_sin" not in value:
        return None
    return copy.deepcopy(dict(value))


def _positive_violations(snapshot: Mapping[str, Any]) -> dict[str, float]:
    raw = snapshot.get("constraint_margins")
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value_f = float(value)
        if value_f > 0.0 and math.isfinite(value_f):
            out[str(key)] = value_f
    return out


def _select_focus_candidate(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    problem: str,
    near_feasibility_threshold: float,
) -> Mapping[str, Any] | None:
    infeasible = [
        snapshot
        for snapshot in snapshots
        if not bool(snapshot.get("is_feasible", False))
        and _is_finite_positive(float(snapshot.get("feasibility", float("inf"))))
    ]
    if not infeasible:
        return None
    near = [
        snapshot
        for snapshot in infeasible
        if float(snapshot.get("feasibility", float("inf")))
        <= float(near_feasibility_threshold)
    ]
    pool = near if near else infeasible
    return max(pool, key=lambda item: _focus_score(item, problem=problem))


def _select_partner_candidate(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    focus_hash: str,
    problem: str,
    worst_constraint: str | None,
) -> Mapping[str, Any] | None:
    pool = [
        snapshot
        for snapshot in snapshots
        if str(snapshot.get("design_hash", "")) != focus_hash
    ]
    if not pool:
        return None

    metric_key, maximize = _constraint_metric_key(worst_constraint)
    if metric_key is not None:
        candidates = [
            snapshot
            for snapshot in pool
            if _metric_value(snapshot, metric_key) is not None
        ]
        if candidates:

            def _metric_for_sort(item: Mapping[str, Any]) -> float:
                value = _metric_value(item, metric_key)
                if value is None:
                    return -float("inf") if maximize else float("inf")
                return value

            if maximize:
                return max(candidates, key=_metric_for_sort)
            return min(candidates, key=_metric_for_sort)

    if (problem or "").lower().startswith("p1"):
        return min(pool, key=lambda item: _objective_for_sort(item, minimize=True))
    return max(pool, key=lambda item: _objective_for_sort(item, minimize=False))


def _focus_score(snapshot: Mapping[str, Any], *, problem: str) -> float:
    feasibility = float(snapshot.get("feasibility", float("inf")))
    objective = snapshot.get("objective")
    if objective is None or not math.isfinite(float(objective)):
        leverage = 1.0
    elif (problem or "").lower().startswith("p1"):
        leverage = 1.0 / (1.0 + max(0.0, float(objective)))
    else:
        leverage = 1.0 + max(0.0, float(objective))
    return leverage / max(feasibility, 1e-6)


def _objective_for_sort(snapshot: Mapping[str, Any], *, minimize: bool) -> float:
    objective = snapshot.get("objective")
    if objective is None:
        return float("inf") if minimize else -float("inf")
    value = float(objective)
    if not math.isfinite(value):
        return float("inf") if minimize else -float("inf")
    return value


def _constraint_metric_key(
    worst_constraint: str | None,
) -> tuple[str | None, bool]:
    name = (worst_constraint or "").lower()
    if "mirror" in name:
        return "mirror", False
    if "log10_qi" in name or name == "qi" or "qi_" in name:
        return "log10_qi", False
    if "flux" in name:
        return "flux_compression", False
    if "vacuum" in name:
        return "vacuum_well", True
    if "iota" in name:
        return "iota_edge", True
    if "aspect" in name:
        return "aspect_ratio", False
    if "elong" in name:
        return "max_elongation", False
    return None, False


def _metric_value(snapshot: Mapping[str, Any], key: str) -> float | None:
    metrics = snapshot.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    if key not in metrics:
        return None
    raw_value = metrics[key]
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        return None
    value = float(raw_value)
    if not math.isfinite(value):
        return None
    return value


def _scale_non_axisymmetric(
    params: Mapping[str, Any],
    factor: float,
) -> dict[str, Any] | None:
    scaled = copy.deepcopy(dict(params))
    arrays = _matrix_fields(scaled)
    if not arrays:
        return None
    for name, arr in arrays.items():
        n_cols = arr.shape[1]
        if n_cols % 2 == 0:
            continue
        n0 = n_cols // 2
        arr[:, :n0] *= factor
        arr[:, n0 + 1 :] *= factor
        scaled[name] = arr.tolist()
    return scaled


def _scale_m_ge(
    params: Mapping[str, Any],
    *,
    m_min: int,
    factor: float,
) -> dict[str, Any] | None:
    scaled = copy.deepcopy(dict(params))
    arrays = _matrix_fields(scaled)
    if not arrays:
        return None
    for name, arr in arrays.items():
        if arr.shape[0] <= m_min:
            continue
        arr[m_min:, :] *= factor
        scaled[name] = arr.tolist()
    return scaled


def _scale_abs_n(
    params: Mapping[str, Any],
    *,
    abs_n: int,
    factor: float,
) -> dict[str, Any] | None:
    scaled = copy.deepcopy(dict(params))
    arrays = _matrix_fields(scaled)
    if not arrays:
        return None
    for name, arr in arrays.items():
        n_cols = arr.shape[1]
        if n_cols % 2 == 0:
            continue
        ntor = n_cols // 2
        for col in range(n_cols):
            if abs(col - ntor) == abs_n:
                arr[:, col] *= factor
        scaled[name] = arr.tolist()
    return scaled


def _matrix_fields(params: Mapping[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in ("r_cos", "z_sin"):
        value = params.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 2:
            continue
        out[key] = arr.copy()
    return out


def _is_finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0
