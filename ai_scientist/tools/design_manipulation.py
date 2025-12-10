"""Design manipulation and candidate generation utilities."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, TypedDict

import numpy as np

from ai_scientist.tools.evaluation import (
    BoundaryParams,
    FlattenSchema,
    _coefficient_from_matrix,
    _derive_schema_from_params,
    _ensure_mapping,
    _quantize_float,
)

# Type for sampler results when include_distances=True
DesignParams = Mapping[str, float | Sequence[float]]


class SamplerResultWithDistance(TypedDict):
    """Result from normalized_constraint_distance_sampler with distance info."""

    params: DesignParams
    normalized_constraint_distance: float


# Union type for sampler results (with or without distance)
SamplerResult = DesignParams | SamplerResultWithDistance


def propose_boundary(
    params: Mapping[str, Any] | BoundaryParams,
    *,
    perturbation_scale: float = 0.05,
    seed: int | None = None,
) -> dict[str, Any]:
    """Perturb a given boundary parameter set with random noise."""
    params_map = _ensure_mapping(params)
    rng = np.random.default_rng(seed)
    new_params: dict[str, Any] = {}

    for key, value in params_map.items():
        if key in ("r_cos", "z_sin", "r_sin", "z_cos"):
            if value is None:
                new_params[key] = None
                continue
            arr = np.asarray(value, dtype=float)
            noise = rng.normal(scale=perturbation_scale, size=arr.shape)
            new_params[key] = (arr + noise).tolist()
        else:
            new_params[key] = value

    # Ensure symmetry constraints if flag is present
    if new_params.get("is_stellarator_symmetric"):
        if "r_cos" in new_params and new_params["r_cos"] is not None:
            r_cos = np.asarray(new_params["r_cos"])
            if r_cos.ndim > 1:
                center_idx = r_cos.shape[1] // 2
                if center_idx > 0:
                    r_cos[0, :center_idx] = 0.0
                new_params["r_cos"] = r_cos.tolist()
        if "z_sin" in new_params and new_params["z_sin"] is not None:
            z_sin = np.asarray(new_params["z_sin"])
            z_sin[0, :] = 0.0
            new_params["z_sin"] = z_sin.tolist()

    return new_params


def normalized_constraint_distance_sampler(
    base_designs: Sequence[DesignParams],
    *,
    normalized_distances: Sequence[float],
    proposal_count: int,
    jitter_scale: float = 0.01,
    rng: np.random.Generator | None = None,
    include_distances: bool = False,
) -> list[SamplerResult]:
    """Constraint-aware sampler for Task X.6 (docs/TASKS_CODEX_MINI.md:233).

    Designs with smaller normalized constraint distances are preferred so the curriculum
    nudges proposals toward near-feasible regions.
    """

    if proposal_count <= 0:
        return []

    if rng is None:
        rng = np.random.default_rng()

    total_candidates = len(base_designs)
    if total_candidates == 0:
        return []

    distances = np.asarray(normalized_distances, dtype=float)
    if distances.shape[0] != total_candidates:
        raise ValueError("normalized_distances must align with base_designs")

    clipped = np.clip(distances, 0.0, 1.0)
    weights = (1.0 - clipped) + 1e-3
    weights_sum = float(np.sum(weights))
    if weights_sum <= 0.0:
        weights = np.ones_like(weights)
        weights_sum = float(weights.size)

    probabilities = (weights / weights_sum).astype(float)
    chosen_indices = rng.choice(total_candidates, size=proposal_count, p=probabilities)
    proposals: list[SamplerResult] = []

    for idx in chosen_indices:
        candidate = base_designs[idx]
        perturbed: dict[str, float | Sequence[float]] = {}
        for key, value in candidate.items():
            array = np.asarray(value, dtype=float)
            jitter = rng.normal(scale=jitter_scale, size=array.shape)
            proposal_array = array + jitter
            if proposal_array.shape == ():
                perturbed[key] = float(proposal_array)
            else:
                perturbed[key] = proposal_array.tolist()
        if include_distances:
            proposals.append(
                {
                    "params": perturbed,
                    "normalized_constraint_distance": float(clipped[idx]),
                }
            )
        else:
            proposals.append(perturbed)

    return proposals


def recombine_designs(
    parent_a: Mapping[str, Any] | BoundaryParams,
    parent_b: Mapping[str, Any] | BoundaryParams,
    *,
    alpha: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Perform geometric crossover between two parent designs via coefficient interpolation.

    Args:
        parent_a: First parent boundary parameters.
        parent_b: Second parent boundary parameters.
        alpha: Interpolation weight (0.0 = parent_b, 1.0 = parent_a).
               If None, a random value in [0, 1] is chosen.
        seed: Random seed for alpha generation if alpha is None.

    Returns:
        A new parameter dictionary representing the interpolated boundary.
    """
    params_a = _ensure_mapping(parent_a)
    params_b = _ensure_mapping(parent_b)

    rng = np.random.default_rng(seed)
    mix_alpha = alpha if alpha is not None else rng.random()

    new_params: dict[str, Any] = {}

    # Keys to interpolate
    keys = ["r_cos", "z_sin", "r_sin", "z_cos"]

    for key in keys:
        val_a = np.asarray(params_a.get(key, []), dtype=float)
        val_b = np.asarray(params_b.get(key, []), dtype=float)

        if val_a.size == 0 and val_b.size == 0:
            continue

        # Handle potentially different shapes (padding)
        shape_a = val_a.shape
        shape_b = val_b.shape

        if shape_a != shape_b:
            # Determine max shape
            max_rows = max(
                shape_a[0] if len(shape_a) > 0 else 0,
                shape_b[0] if len(shape_b) > 0 else 0,
            )
            max_cols = max(
                shape_a[1] if len(shape_a) > 1 else 0,
                shape_b[1] if len(shape_b) > 1 else 0,
            )

            target_shape = (max_rows, max_cols)

            # Pad A
            pad_a = np.zeros(target_shape, dtype=float)
            if val_a.size > 0:
                pad_a[: shape_a[0], : shape_a[1]] = val_a
            val_a = pad_a

            # Pad B
            pad_b = np.zeros(target_shape, dtype=float)
            if val_b.size > 0:
                pad_b[: shape_b[0], : shape_b[1]] = val_b
            val_b = pad_b

        # Interpolate
        new_val = mix_alpha * val_a + (1.0 - mix_alpha) * val_b
        new_params[key] = new_val.tolist()

    # Copy metadata from parent A (or B)
    new_params["n_field_periods"] = params_a.get(
        "n_field_periods", params_b.get("n_field_periods", 1)
    )
    new_params["is_stellarator_symmetric"] = params_a.get(
        "is_stellarator_symmetric", True
    )

    return new_params


def structured_flatten(
    params: Mapping[str, Any] | BoundaryParams,
    schema: FlattenSchema | None = None,
) -> tuple[np.ndarray, FlattenSchema]:
    """Flatten Fourier coefficients with deterministic ordering and schema metadata.

    The layout is `[r_cos modes..., z_sin modes...]` with n ranging from `-ntor`
    to `ntor` for each m in `[0, mpol]`. Missing coefficients are zero-padded and
    values are rounded to the schema precision to stabilize hashes and caches.
    """

    params_map = _ensure_mapping(params)
    active_schema = schema or _derive_schema_from_params(params_map)
    rounding = active_schema.rounding

    r_cos = np.asarray(params_map.get("r_cos", []), dtype=float)
    z_sin = np.asarray(params_map.get("z_sin", []), dtype=float)

    values: list[float] = []
    for matrix in (r_cos, z_sin):
        for m in range(active_schema.mpol + 1):
            for n in range(-active_schema.ntor, active_schema.ntor + 1):
                coefficient = _coefficient_from_matrix(matrix, m, n, active_schema.ntor)
                values.append(_quantize_float(coefficient, precision=rounding))

    return np.asarray(values, dtype=float), active_schema


def structured_unflatten(
    flattened_vector: np.ndarray,
    schema: FlattenSchema,
) -> dict[str, Any]:
    """Convert a flattened array of Fourier coefficients back to a dictionary of parameters.

    Args:
        flattened_vector: A 1D numpy array of Fourier coefficients.
        schema: The FlattenSchema used to flatten the original parameters.

    Returns:
        A dictionary of parameters with 'r_cos', 'z_sin', etc.
    """
    params: dict[str, Any] = {}

    # Initialize matrices with zeros based on schema
    r_cos_matrix = np.zeros((schema.mpol + 1, 2 * schema.ntor + 1), dtype=float)
    z_sin_matrix = np.zeros((schema.mpol + 1, 2 * schema.ntor + 1), dtype=float)

    offset = 0

    # Fill r_cos
    for m in range(schema.mpol + 1):
        for n_idx in range(2 * schema.ntor + 1):  # Corresponds to n from -ntor to +ntor
            if offset < len(flattened_vector):
                r_cos_matrix[m, n_idx] = flattened_vector[offset]
                offset += 1
            else:
                break
        if offset >= len(flattened_vector):
            break

    # Fill z_sin
    for m in range(schema.mpol + 1):
        for n_idx in range(2 * schema.ntor + 1):  # Corresponds to n from -ntor to +ntor
            if offset < len(flattened_vector):
                z_sin_matrix[m, n_idx] = flattened_vector[offset]
                offset += 1
            else:
                break
        if offset >= len(flattened_vector):
            break

    params["r_cos"] = r_cos_matrix.tolist()
    params["z_sin"] = z_sin_matrix.tolist()

    return params
