"""Sampler implementations for Phase 1/2 candidate generation."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np

from ai_scientist import config as ai_config
from ai_scientist import tools
from constellaration.initial_guess import generate_nae
from constellaration.geometry import surface_rz_fourier

_LOGGER = logging.getLogger(__name__)


def _surface_to_params(surface: surface_rz_fourier.SurfaceRZFourier) -> dict[str, Any]:
    return {
        "r_cos": np.asarray(surface.r_cos).tolist(),
        "z_sin": np.asarray(surface.z_sin).tolist(),
        "n_field_periods": int(surface.n_field_periods),
        "is_stellarator_symmetric": bool(surface.is_stellarator_symmetric),
    }


class NearAxisSampler:
    """Generates candidates using Near-Axis Expansion (NAE) via Constellaration."""

    def __init__(
        self,
        template: ai_config.BoundaryTemplateConfig,
    ) -> None:
        self._template = template

    def generate(self, seeds: Sequence[int]) -> list[Mapping[str, Any]]:
        candidates: list[Mapping[str, Any]] = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            
            # Sample NAE parameters around reasonable defaults or template values
            # We assume template provides n_field_periods and mode limits.
            
            # Retry loop to handle "strictly increasing sequence" errors or geometry failures
            max_retries = 10
            for attempt in range(max_retries):
                # Harden ranges: Narrower bounds to improve stability (Phase 5.3 fix)
                # Aspect ratio: 4.0-8.0 -> 5.0-7.0
                aspect_ratio = rng.uniform(5.0, 7.0)
                
                # Elongation: 1.5-2.5 -> 1.5-2.0
                max_elongation = rng.uniform(1.5, 2.0)
                
                # Rotational transform (iota): 0.4-1.2 -> 0.4-0.8
                rotational_transform = rng.uniform(0.4, 0.8)
                
                # Mirror ratio: 1.05-1.2 -> 1.05-1.15
                mirror_ratio = rng.uniform(1.05, 1.15)

                try:
                    surface = generate_nae(
                        aspect_ratio=aspect_ratio,
                        max_elongation=max_elongation,
                        rotational_transform=rotational_transform,
                        mirror_ratio=mirror_ratio,
                        n_field_periods=self._template.n_field_periods,
                        max_poloidal_mode=self._template.n_poloidal_modes,
                        max_toroidal_mode=self._template.n_toroidal_modes,
                    )
                    params = _surface_to_params(surface)
                    candidates.append(
                        {
                            "seed": seed, # Keep original seed for traceability
                            "params": params,
                            "source": "near_axis_sampler",
                            "design_hash": tools.design_hash(params),
                            "constraint_distance": 0.0, # NAE designs are "theoretically" near feasible
                        }
                    )
                    break # Success
                except Exception as exc:
                    if attempt == max_retries - 1:
                        _LOGGER.warning(f"Near-axis generation failed for seed {seed} after {max_retries} attempts: {exc}")
                    continue
        
        return candidates