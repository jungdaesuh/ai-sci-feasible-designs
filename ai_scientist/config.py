"""Configuration helpers for the AI Scientist orchestration (Tasks 0.2 + B.*)."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping, Tuple

import yaml

DEFAULT_EXPERIMENT_CONFIG_PATH = Path("configs/experiment.example.yaml")
DEFAULT_MODEL_CONFIG_PATH = Path("configs/model.yaml")
DEFAULT_MEMORY_DB_PATH = Path("reports/ai_scientist.sqlite")


def load(path: str | Path | None = None) -> dict[str, Any]:
    """Return the raw YAML mapping for a given configuration file."""

    target = Path(path) if path is not None else DEFAULT_EXPERIMENT_CONFIG_PATH
    with target.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass(frozen=True)
class AgentGateConfig:
    model_alias: str
    allowed_tools: Tuple[str, ...]
    system_prompt: str | None
    provider_model: str | None


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    chat_path: str
    auth_env: str
    default_model: str | None
    extra_headers: Tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    instruct_model: str
    thinking_model: str
    request_timeout_seconds: int
    rate_limit_per_minute: int
    context_length: int
    dtype: str
    tensor_parallel: int
    default_provider: str
    providers: Tuple["ProviderConfig", ...]
    agent_gates: Tuple[AgentGateConfig, ...]
    role_map: Mapping[str, str]

    def get_provider(self, name: str | None = None) -> "ProviderConfig":
        alias = (name or self.default_provider).lower()
        for provider in self.providers:
            if provider.name.lower() == alias:
                return provider
        raise ValueError(f"model provider '{alias}' is not configured")


def _env_override(key: str, default: str) -> str:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return value


def load_model_config(path: str | Path | None = None) -> ModelConfig:
    payload = load(path or DEFAULT_MODEL_CONFIG_PATH)
    model_data = payload.get("model", {})
    agent_gates_payload = model_data.get("agent_gates") or {}
    agent_gates = tuple(
        _agent_gate_config_from_dict(alias, gate_data)
        for alias, gate_data in agent_gates_payload.items()
    )
    provider_data = model_data.get("providers") or {}
    providers = tuple(
        _provider_config_from_dict(name, data) for name, data in provider_data.items()
    )
    default_provider = str(
        model_data.get("provider", providers[0].name if providers else "openrouter")
    )
    default_provider = _env_override("MODEL_PROVIDER", default_provider)
    instruct_alias = str(model_data.get("instruct_model", "kimi-k2-instruct"))
    instruct_alias = _env_override("AI_SCIENTIST_INSTRUCT_MODEL", instruct_alias)
    thinking_alias = str(model_data.get("thinking_model", "kimi-k2-thinking"))
    thinking_alias = _env_override("AI_SCIENTIST_THINKING_MODEL", thinking_alias)
    role_map = model_data.get("role_map") or {}
    if not isinstance(role_map, dict):
        role_map = {}
    return ModelConfig(
        base_url=str(model_data.get("base_url", "http://localhost:8000")),
        instruct_model=instruct_alias,
        thinking_model=thinking_alias,
        request_timeout_seconds=int(model_data.get("request_timeout_seconds", 60)),
        rate_limit_per_minute=int(model_data.get("rate_limit_per_minute", 600)),
        context_length=int(model_data.get("context_length", 8192)),
        dtype=str(model_data.get("dtype", "bf16")),
        tensor_parallel=int(model_data.get("tensor_parallel", 1)),
        default_provider=default_provider,
        providers=providers,
        agent_gates=agent_gates,
        role_map=role_map,
    )


@dataclass(frozen=True)
class BudgetConfig:
    screen_evals_per_cycle: int
    promote_top_k: int
    max_high_fidelity_evals_per_cycle: int
    wall_clock_minutes: float
    n_workers: int
    pool_type: str


@dataclass(frozen=True)
class BudgetRangeConfig:
    min: int
    max: int


@dataclass(frozen=True)
class AdaptiveBudgetConfig:
    enabled: bool
    hv_slope_reference: float
    feasibility_target: float
    cache_hit_target: float
    screen_bounds: BudgetRangeConfig
    promote_top_k_bounds: BudgetRangeConfig
    high_fidelity_bounds: BudgetRangeConfig


@dataclass(frozen=True)
class FidelityLadder:
    screen: str
    promote: str


@dataclass(frozen=True)
class BoundaryTemplateConfig:
    n_poloidal_modes: int
    n_toroidal_modes: int
    n_field_periods: int
    base_major_radius: float
    base_minor_radius: float
    perturbation_scale: float
    seed_path: Path | None = None


@dataclass(frozen=True)
class StageGateConfig:
    s1_to_s2_feasibility_margin: float
    s1_to_s2_objective_improvement: float
    s1_to_s2_lookback_cycles: int
    s2_to_s3_hv_delta: float
    s2_to_s3_lookback_cycles: int


@dataclass(frozen=True)
class GovernanceConfig:
    min_feasible_for_promotion: int
    hv_lookback: int


@dataclass(frozen=True)
class ProposalMixConfig:
    constraint_ratio: float
    exploration_ratio: float
    jitter_scale: float
    surrogate_pool_multiplier: float = 2.0


@dataclass(frozen=True)
class ExperimentConfig:
    problem: str
    cycles: int
    random_seed: int
    budgets: BudgetConfig
    adaptive_budgets: AdaptiveBudgetConfig
    fidelity_ladder: FidelityLadder
    boundary_template: BoundaryTemplateConfig
    stage_gates: StageGateConfig
    governance: GovernanceConfig
    proposal_mix: ProposalMixConfig
    reporting_dir: Path
    memory_db: Path
    source_config: Path


def _boundary_template_from_dict(
    data: Mapping[str, Any] | None,
) -> BoundaryTemplateConfig:
    config = data or {}
    seed_path = config.get("seed_path")
    return BoundaryTemplateConfig(
        n_poloidal_modes=int(config.get("n_poloidal_modes", 3)),
        n_toroidal_modes=int(config.get("n_toroidal_modes", 5)),
        n_field_periods=int(config.get("n_field_periods", 1)),
        base_major_radius=float(config.get("base_major_radius", 1.5)),
        base_minor_radius=float(config.get("base_minor_radius", 0.5)),
        perturbation_scale=float(config.get("perturbation_scale", 0.05)),
        seed_path=Path(seed_path) if seed_path else None,
    )


def _stage_gate_config_from_dict(
    data: Mapping[str, Any] | None,
) -> StageGateConfig:
    config = data or {}
    return StageGateConfig(
        s1_to_s2_feasibility_margin=float(
            config.get("s1_to_s2_feasibility_margin", 0.01)
        ),
        s1_to_s2_objective_improvement=float(
            config.get("s1_to_s2_objective_improvement", 0.02)
        ),
        s1_to_s2_lookback_cycles=int(config.get("s1_to_s2_lookback_cycles", 3)),
        s2_to_s3_hv_delta=float(config.get("s2_to_s3_hv_delta", 0.01)),
        s2_to_s3_lookback_cycles=int(config.get("s2_to_s3_lookback_cycles", 3)),
    )


def _governance_config_from_dict(
    data: Mapping[str, Any] | None,
    *,
    default_hv_lookback: int,
) -> GovernanceConfig:
    config = data or {}
    min_feasible = int(config.get("min_feasible_for_promotion", 1))
    min_feasible = max(0, min_feasible)
    hv_lookback = int(config.get("hv_lookback", default_hv_lookback))
    hv_lookback = max(1, hv_lookback)
    return GovernanceConfig(
        min_feasible_for_promotion=min_feasible,
        hv_lookback=hv_lookback,
    )


def _agent_gate_config_from_dict(
    alias: str, data: Mapping[str, Any] | None
) -> AgentGateConfig:
    config = data or {}
    allowed = tuple(str(item) for item in config.get("allowed_tools", []))
    prompt = config.get("system_prompt")
    provider_model = config.get("provider_model")
    return AgentGateConfig(
        model_alias=str(alias),
        allowed_tools=allowed,
        system_prompt=str(prompt) if prompt is not None else None,
        provider_model=(
            str(provider_model) if provider_model is not None else None
        ),
    )


def _extra_headers_from_dict(
    data: Mapping[str, Any] | None,
) -> Tuple[tuple[str, str], ...]:
    config = data or {}
    return tuple((str(key), str(value)) for key, value in config.items())


def _provider_config_from_dict(
    name: str, data: Mapping[str, Any] | None
) -> ProviderConfig:
    config = data or {}
    return ProviderConfig(
        name=str(name),
        base_url=str(config.get("base_url", "")),
        chat_path=str(config.get("chat_path", "/v1/chat/completions")),
        auth_env=str(config.get("auth_env", "")),
        default_model=(
            str(config.get("default_model"))
            if config.get("default_model") is not None
            else None
        ),
        extra_headers=_extra_headers_from_dict(config.get("extra_headers")),
    )


def _proposal_mix_from_dict(
    data: Mapping[str, Any] | None,
) -> ProposalMixConfig:
    config = data or {}
    constraint_ratio = float(config.get("constraint_ratio", 0.7))
    exploration_ratio = float(config.get("exploration_ratio", 0.3))
    return ProposalMixConfig(
        constraint_ratio=constraint_ratio,
        exploration_ratio=exploration_ratio,
        jitter_scale=float(config.get("jitter_scale", 0.01)),
        surrogate_pool_multiplier=float(config.get("surrogate_pool_multiplier", 2.0)),
    )


def _budget_range_from_dict(
    data: Mapping[str, Any] | None,
    *,
    default_value: int,
) -> BudgetRangeConfig:
    config = data or {}
    min_val = int(config.get("min", default_value))
    max_val = int(config.get("max", default_value))
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    min_val = max(0, min_val)
    max_val = max(max_val, min_val)
    return BudgetRangeConfig(min=min_val, max=max_val)


def _adaptive_budget_config_from_dict(
    data: Mapping[str, Any] | None,
    *,
    base_budgets: BudgetConfig,
) -> AdaptiveBudgetConfig:
    config = data or {}
    enabled = bool(config.get("enabled", False))
    hv_reference = float(config.get("hv_slope_reference", 0.05))
    feasibility_target = float(config.get("feasibility_target", 0.5))
    cache_hit_target = float(config.get("cache_hit_target", 0.3))
    screen_bounds = _budget_range_from_dict(
        config.get("screen_evals_per_cycle"),
        default_value=base_budgets.screen_evals_per_cycle,
    )
    promote_bounds = _budget_range_from_dict(
        config.get("promote_top_k"),
        default_value=base_budgets.promote_top_k,
    )
    high_fidelity_bounds = _budget_range_from_dict(
        config.get("max_high_fidelity_evals_per_cycle"),
        default_value=base_budgets.max_high_fidelity_evals_per_cycle,
    )
    return AdaptiveBudgetConfig(
        enabled=enabled,
        hv_slope_reference=max(1e-6, hv_reference),
        feasibility_target=max(1e-6, feasibility_target),
        cache_hit_target=max(1e-6, cache_hit_target),
        screen_bounds=screen_bounds,
        promote_top_k_bounds=promote_bounds,
        high_fidelity_bounds=high_fidelity_bounds,
    )


def load_experiment_config(path: str | Path | None = None) -> ExperimentConfig:
    config_path = Path(path) if path is not None else DEFAULT_EXPERIMENT_CONFIG_PATH
    payload = load(config_path)
    budgets = payload.get("budgets", {})
    fidelity = payload.get("fidelity_ladder", {})
    stage_gates = _stage_gate_config_from_dict(payload.get("stage_gates"))
    governance = _governance_config_from_dict(
        payload.get("governance"),
        default_hv_lookback=stage_gates.s2_to_s3_lookback_cycles,
    )
    budget_config = BudgetConfig(
        screen_evals_per_cycle=int(budgets.get("screen_evals_per_cycle", 1)),
        promote_top_k=int(budgets.get("promote_top_k", 1)),
        max_high_fidelity_evals_per_cycle=int(
            budgets.get("max_high_fidelity_evals_per_cycle", 1)
        ),
        wall_clock_minutes=float(budgets.get("wall_clock_minutes", 5.0)),
        n_workers=int(budgets.get("n_workers", 1)),
        pool_type=str(budgets.get("pool_type", "process")),
    )
    return ExperimentConfig(
        problem=str(payload.get("problem", "p1")),
        cycles=int(payload.get("cycles", 1)),
        random_seed=int(payload.get("random_seed", 0)),
        budgets=budget_config,
        adaptive_budgets=_adaptive_budget_config_from_dict(
            payload.get("adaptive_budgets"),
            base_budgets=budget_config,
        ),
        fidelity_ladder=FidelityLadder(
            screen=str(fidelity.get("screen", "screen")),
            promote=str(fidelity.get("promote", "promote")),
        ),
        boundary_template=_boundary_template_from_dict(
            payload.get("boundary_template")
        ),
        stage_gates=stage_gates,
        governance=governance,
        proposal_mix=_proposal_mix_from_dict(payload.get("proposal_mix")),
        reporting_dir=Path(payload.get("reporting_dir", "reports")),
        memory_db=Path(payload.get("memory_db", DEFAULT_MEMORY_DB_PATH)),
        source_config=config_path,
    )
