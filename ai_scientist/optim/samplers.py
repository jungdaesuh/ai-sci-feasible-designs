"""Sampler implementations for Phase 1/2 candidate generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from constellaration.geometry import surface_rz_fourier
from constellaration.initial_guess import generate_nae, generate_rotating_ellipse

from ai_scientist import config as ai_config
from ai_scientist import tools

_LOGGER = logging.getLogger(__name__)


def _surface_to_params(surface: surface_rz_fourier.SurfaceRZFourier) -> dict[str, Any]:
    return {
        "r_cos": np.asarray(surface.r_cos).tolist(),
        "z_sin": np.asarray(surface.z_sin).tolist(),
        "n_field_periods": int(surface.n_field_periods),
        "is_stellarator_symmetric": bool(surface.is_stellarator_symmetric),
    }


class RotatingEllipseSampler:
    """Generates candidates using simple Rotating Ellipse via Constellaration."""

    def __init__(
        self,
        template: ai_config.BoundaryTemplateConfig,
    ) -> None:
        self._template = template

    def generate(self, seeds: Sequence[int]) -> list[Mapping[str, Any]]:
        candidates: list[Mapping[str, Any]] = []
        for seed in seeds:
            rng = np.random.default_rng(seed)

            # Defaults from runner.py
            base_surface = generate_rotating_ellipse(
                aspect_ratio=4.0,
                elongation=1.5,
                rotational_transform=1.2,
                n_field_periods=self._template.n_field_periods,
            )

            # Expand to template modes
            max_poloidal = max(1, self._template.n_poloidal_modes - 1)
            max_toroidal = max(1, (self._template.n_toroidal_modes - 1) // 2)

            expanded = surface_rz_fourier.set_max_mode_numbers(
                base_surface,
                max_poloidal_mode=max_poloidal,
                max_toroidal_mode=max_toroidal,
            )

            r_cos = np.asarray(expanded.r_cos, dtype=float)
            z_sin = np.asarray(expanded.z_sin, dtype=float)

            # Enforce template radii
            center_idx = r_cos.shape[1] // 2
            r_cos[0, center_idx] = self._template.base_major_radius
            if r_cos.shape[0] > 1:
                z_sin[1, center_idx] = self._template.base_minor_radius

            # Add perturbation
            r_cos += rng.normal(
                scale=self._template.perturbation_scale, size=r_cos.shape
            )
            z_sin += rng.normal(
                scale=self._template.perturbation_scale / 2, size=z_sin.shape
            )

            # Enforce symmetry (Runner logic)
            n_cols = r_cos.shape[1]
            center_idx = n_cols // 2
            if center_idx > 0:
                r_cos[0, :center_idx] = 0.0
            z_sin[0, :] = 0.0

            params = {
                "r_cos": r_cos.tolist(),
                "z_sin": z_sin.tolist(),
                "n_field_periods": self._template.n_field_periods,
                "is_stellarator_symmetric": True,
            }

            candidates.append(
                {
                    "seed": seed,
                    "params": params,
                    "source": "rotating_ellipse_sampler",
                    "design_hash": tools.design_hash(params),
                    "constraint_distance": 1.0,  # Higher distance as these are random
                }
            )

        return candidates


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
                            "seed": seed,  # Keep original seed for traceability
                            "params": params,
                            "source": "near_axis_sampler",
                            "design_hash": tools.design_hash(params),
                            "constraint_distance": 0.0,  # NAE designs are "theoretically" near feasible
                        }
                    )
                    break  # Success
                except Exception as exc:
                    if attempt == max_retries - 1:
                        _LOGGER.warning(
                            f"Near-axis generation failed for seed {seed} after {max_retries} attempts: {exc}"
                        )
                    continue

        return candidates


class OfflineSeedSampler:
    """Samples candidates from a pre-generated JSON file of seeds (Best-of-Failure)."""

    def __init__(
        self,
        problem: str,
        seed_file: Path | None = None,
    ) -> None:
        self._problem = problem.lower()
        if seed_file:
            self._seed_path = seed_file
        else:
            self._seed_path = Path(f"configs/seeds/{self._problem}_seeds.json")

        self._seeds: list[dict[str, Any]] = []
        self._load_seeds()

    def _load_seeds(self) -> None:
        if not self._seed_path.exists():
            _LOGGER.warning(
                f"Offline seed file {self._seed_path} not found. Sampler will be empty."
            )
            return

        try:
            with self._seed_path.open("r") as f:
                self._seeds = json.load(f)
            _LOGGER.info(f"Loaded {len(self._seeds)} seeds from {self._seed_path}")
        except Exception as exc:
            _LOGGER.error(f"Failed to load seeds from {self._seed_path}: {exc}")

    def generate(self, seeds: Sequence[int]) -> list[Mapping[str, Any]]:
        if not self._seeds:
            return []

        candidates: list[Mapping[str, Any]] = []
        rng = np.random.default_rng(seeds[0] if seeds else 0)

        for seed in seeds:
            # Pick a random seed from the loaded list
            base_params = rng.choice(self._seeds)

            # Perturb it slightly to avoid duplicates and explore vicinity
            perturbed_params = tools.propose_boundary(
                base_params,
                perturbation_scale=0.02,  # Small perturbation for stability
                seed=seed,
            )

            candidates.append(
                {
                    "seed": seed,
                    "params": perturbed_params,
                    "source": "offline_seed_sampler",
                    "design_hash": tools.design_hash(perturbed_params),
                    "constraint_distance": 0.0,  # Assumed good
                }
            )

        return candidates
