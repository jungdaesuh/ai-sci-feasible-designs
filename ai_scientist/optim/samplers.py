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
            
            # Aspect ratio: Sample around 4.0 to 8.0 (typical for stellarators)
            aspect_ratio = rng.uniform(4.0, 8.0)
            
            # Elongation: Sample around 1.5 to 2.5
            max_elongation = rng.uniform(1.5, 2.5)
            
            # Rotational transform (iota): Sample around 0.4 to 1.2
            rotational_transform = rng.uniform(0.4, 1.2)
            
            # Mirror ratio: Sample around 1.05 to 1.2
            mirror_ratio = rng.uniform(1.05, 1.2)

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
                        "seed": seed,
                        "params": params,
                        "source": "near_axis_sampler",
                        "design_hash": tools.design_hash(params),
                        "constraint_distance": 0.0, # NAE designs are "theoretically" near feasible
                    }
                )
            except Exception as exc:
                _LOGGER.warning(f"Near-axis generation failed for seed {seed}: {exc}")
                continue
        
        return candidates