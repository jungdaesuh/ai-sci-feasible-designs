from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np

from ai_scientist import config as ai_config, memory, runner


class _FakeWorldModel:
    def recent_stage_candidates(
        self,
        experiment_id: int,
        problem: str,
        stage: str,
        *,
        limit: int = 64,
    ) -> list[tuple[dict[str, float], float]]:
        del experiment_id, problem, stage, limit
        base = [
            ({"offset": 0.05}, 0.02),
            ({"offset": 0.1}, 0.03),
            ({"offset": -0.01}, 0.01),
        ]
        return base


class _SurrogateFakeWorldModel:
    def surrogate_training_data(
        self,
        *,
        target: str = "hv",
        problem: str | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        del target, problem
        base_values = [1.0, 2.0, 3.0, 4.0]
        history: list[tuple[dict[str, Any], float]] = []
        for value in base_values:
            params = {
                "r_cos": [[value, value]],
                "z_sin": [[0.0, 0.0]],
                "n_field_periods": 1,
                "is_stellarator_symmetric": True,
            }
            metrics = {"candidate_params": params}
            entropy = float(np.sum(np.asarray(params["r_cos"], dtype=float)))
            history.append((metrics, entropy))
        return history


def _default_adaptive_config(
    budgets: ai_config.BudgetConfig,
) -> ai_config.AdaptiveBudgetConfig:
    return ai_config.AdaptiveBudgetConfig(
        enabled=False,
        hv_slope_reference=0.1,
        feasibility_target=0.5,
        cache_hit_target=0.3,
        screen_bounds=ai_config.BudgetRangeConfig(
            min=budgets.screen_evals_per_cycle,
            max=budgets.screen_evals_per_cycle,
        ),
        promote_top_k_bounds=ai_config.BudgetRangeConfig(
            min=budgets.promote_top_k,
            max=budgets.promote_top_k,
        ),
        high_fidelity_bounds=ai_config.BudgetRangeConfig(
            min=budgets.max_high_fidelity_evals_per_cycle,
            max=budgets.max_high_fidelity_evals_per_cycle,
        ),
    )


def test_propose_p3_candidates_mixes_sampler_and_random() -> None:
    """See tests/test_tools_sampler.py for the sampler baseline that the helper leverages."""

    budgets = ai_config.BudgetConfig(
        screen_evals_per_cycle=4,
        promote_top_k=2,
        max_high_fidelity_evals_per_cycle=1,
        wall_clock_minutes=1.0,
        n_workers=1,
        pool_type="thread",
    )
    cfg = ai_config.ExperimentConfig(
        problem="p3",
        cycles=1,
        random_seed=0,
        budgets=budgets,
        adaptive_budgets=_default_adaptive_config(budgets),
        fidelity_ladder=ai_config.FidelityLadder(screen="screen", promote="promote"),
        boundary_template=ai_config.BoundaryTemplateConfig(
            n_poloidal_modes=3,
            n_toroidal_modes=5,
            n_field_periods=1,
            base_major_radius=1.5,
            base_minor_radius=0.5,
            perturbation_scale=0.02,
        ),
        stage_gates=ai_config.StageGateConfig(
            s1_to_s2_feasibility_margin=0.01,
            s1_to_s2_objective_improvement=0.01,
            s1_to_s2_lookback_cycles=2,
            s2_to_s3_hv_delta=0.01,
            s2_to_s3_lookback_cycles=2,
        ),
        governance=ai_config.GovernanceConfig(
            min_feasible_for_promotion=1,
            hv_lookback=2,
        ),
        proposal_mix=ai_config.ProposalMixConfig(
            constraint_ratio=0.7,
            exploration_ratio=0.3,
            jitter_scale=0.005,
        ),
        generative=ai_config.GenerativeConfig(
            enabled=False,
            backend="vae",
            latent_dim=16,
            learning_rate=0.001,
            epochs=100,
            kl_weight=0.001,
        ),
                    reporting_dir=Path("."),
                    memory_db=Path("reports/ai_scientist.sqlite"),
                    source_config=Path("configs/experiment.example.yaml"),
                    constraint_weights=ai_config.ConstraintWeightsConfig(
                        mhd=1.0,
                        qi=1.0,
                        elongation=1.0,
                    ),
                    initialization_strategy="template",
                )
    world_model = _FakeWorldModel()
    candidates, sampler_count, random_count, vae_results_count = runner._propose_p3_candidates_for_cycle(
        cfg,
        cycle_index=0,
        world_model=cast(memory.WorldModel, world_model),
        experiment_id=1,
        screen_budget=cfg.budgets.screen_evals_per_cycle,
    )

    assert len(candidates) == cfg.budgets.screen_evals_per_cycle
    assert sampler_count > 0
    assert random_count > 0
    assert sampler_count + random_count == len(candidates)


def _sum_r_cos(params: Mapping[str, Any]) -> float:
    return float(np.sum(np.asarray(params["r_cos"], dtype=float)))


def test_surrogate_ranker_prefers_high_hv_candidates() -> None:
    budgets = ai_config.BudgetConfig(
        screen_evals_per_cycle=3,
        promote_top_k=2,
        max_high_fidelity_evals_per_cycle=1,
        wall_clock_minutes=1.0,
        n_workers=1,
        pool_type="thread",
    )
    cfg = ai_config.ExperimentConfig(
        problem="p3",
        cycles=1,
        random_seed=0,
        budgets=budgets,
        adaptive_budgets=_default_adaptive_config(budgets),
        fidelity_ladder=ai_config.FidelityLadder(screen="screen", promote="promote"),
        boundary_template=ai_config.BoundaryTemplateConfig(
            n_poloidal_modes=3,
            n_toroidal_modes=5,
            n_field_periods=1,
            base_major_radius=1.5,
            base_minor_radius=0.5,
            perturbation_scale=0.02,
        ),
        stage_gates=ai_config.StageGateConfig(
            s1_to_s2_feasibility_margin=0.01,
            s1_to_s2_objective_improvement=0.01,
            s1_to_s2_lookback_cycles=2,
            s2_to_s3_hv_delta=0.01,
            s2_to_s3_lookback_cycles=2,
        ),
        governance=ai_config.GovernanceConfig(
            min_feasible_for_promotion=1,
            hv_lookback=2,
        ),
        proposal_mix=ai_config.ProposalMixConfig(
            constraint_ratio=0.7,
            exploration_ratio=0.3,
            jitter_scale=0.005,
        ),
        generative=ai_config.GenerativeConfig(
            enabled=False,
            backend="vae",
            latent_dim=16,
            learning_rate=0.001,
            epochs=100,
            kl_weight=0.001,
        ),
        reporting_dir=Path("."),
        memory_db=Path("reports/ai_scientist.sqlite"),
        source_config=Path("configs/experiment.example.yaml"),
        constraint_weights=ai_config.ConstraintWeightsConfig(
            mhd=1.0,
            qi=1.0,
            elongation=1.0,
        ),
        initialization_strategy="template",
                )
    candidates: list[Mapping[str, Any]] = []
    for idx, value in enumerate((0.0, 0.5, 1.0, 1.5, 2.0, 2.5)):
        params = {
            "r_cos": [[value, value]],
            "z_sin": [[0.0, 0.0]],
            "n_field_periods": 1,
            "is_stellarator_symmetric": True,
        }
        candidates.append(
            {
                "seed": idx,
                "params": params,
                "design_hash": f"candidate-{idx}",
            }
        )

    world_model = _SurrogateFakeWorldModel()
    ranked = runner._surrogate_rank_screen_candidates(
        cfg,
        cfg.budgets.screen_evals_per_cycle,
        candidates,
        cast(memory.WorldModel, world_model),
        runner.SurrogateBundle(),
        verbose=False,
    )

    assert len(ranked) == cfg.budgets.screen_evals_per_cycle
    baseline = candidates[: cfg.budgets.screen_evals_per_cycle]
    baseline_hv = sum(_sum_r_cos(c["params"]) for c in baseline)
    surrogate_hv = sum(_sum_r_cos(c["params"]) for c in ranked)
    assert surrogate_hv > baseline_hv


def test_surrogate_candidate_pool_size_allows_zero() -> None:
    budgets = ai_config.BudgetConfig(
        screen_evals_per_cycle=0,
        promote_top_k=2,
        max_high_fidelity_evals_per_cycle=1,
        wall_clock_minutes=1.0,
        n_workers=1,
        pool_type="thread",
    )
    cfg = ai_config.ExperimentConfig(
        problem="p3",
        cycles=1,
        random_seed=0,
        budgets=budgets,
        adaptive_budgets=_default_adaptive_config(budgets),
        fidelity_ladder=ai_config.FidelityLadder(screen="screen", promote="promote"),
        boundary_template=ai_config.BoundaryTemplateConfig(
            n_poloidal_modes=3,
            n_toroidal_modes=5,
            n_field_periods=1,
            base_major_radius=1.5,
            base_minor_radius=0.5,
            perturbation_scale=0.02,
        ),
        stage_gates=ai_config.StageGateConfig(
            s1_to_s2_feasibility_margin=0.01,
            s1_to_s2_objective_improvement=0.01,
            s1_to_s2_lookback_cycles=2,
            s2_to_s3_hv_delta=0.01,
            s2_to_s3_lookback_cycles=2,
        ),
        governance=ai_config.GovernanceConfig(
            min_feasible_for_promotion=1,
            hv_lookback=2,
        ),
        proposal_mix=ai_config.ProposalMixConfig(
            constraint_ratio=0.0,
            exploration_ratio=0.0,
            jitter_scale=0.0,
        ),
        generative=ai_config.GenerativeConfig(
            enabled=False,
            backend="vae",
            latent_dim=16,
            learning_rate=0.001,
            epochs=100,
            kl_weight=0.001,
        ),
        reporting_dir=Path("."),
        memory_db=Path("reports/ai_scientist.sqlite"),
        source_config=Path("configs/experiment.example.yaml"),
        constraint_weights=ai_config.ConstraintWeightsConfig(
            mhd=1.0,
            qi=1.0,
            elongation=1.0,
        ),
        initialization_strategy="template",
                )
    assert (
        runner._surrogate_candidate_pool_size(
            cfg.budgets.screen_evals_per_cycle,
            cfg.proposal_mix.surrogate_pool_multiplier,
        )
        == 0
    )
