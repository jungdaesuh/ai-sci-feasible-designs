"""Configuration helpers for the AI Scientist orchestration (Tasks 0.2 + B.*)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Mapping, Tuple, Literal

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
    role_map: Mapping[str, str] = field(default_factory=dict)

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
    jitter_scale: float = 0.01 # Add default to match _proposal_mix_from_dict
    exploitation_ratio: float = 0.0
    surrogate_pool_multiplier: float = 2.0
    sampler_type: str = "standard"


@dataclass(frozen=True)
class ConstraintWeightsConfig:
    mhd: float
    qi: float
    elongation: float


@dataclass(frozen=True)
class GenerativeConfig:
    enabled: bool
    backend: str = "vae"
    latent_dim: int = 16
    learning_rate: float = 1e-3
    epochs: int = 100
    kl_weight: float = 0.001


@dataclass(frozen=True)
class SurrogateConfig:
    backend: str = "random_forest"
    n_ensembles: int = 1
    learning_rate: float = 1e-3
    epochs: int = 100
    hidden_dim: int = 64
    use_offline_dataset: bool = False


@dataclass(frozen=True)
class ALMConfig:
    """ALM hyperparameters (mirrors constellaration settings)."""
    # Per-iteration settings
    penalty_parameters_increase_factor: float = 2.0
    constraint_violation_tolerance_reduction_factor: float = 0.5
    bounds_reduction_factor: float = 0.95
    penalty_parameters_max: float = 1e8
    bounds_min: float = 0.05

    # Method-level settings
    maxit: int = 25
    penalty_parameters_initial: float = 1.0
    bounds_initial: float = 2.0

    # Oracle (Nevergrad) settings
    oracle_budget_initial: int = 100
    oracle_budget_increment: int = 26
    oracle_budget_max: int = 200
    oracle_num_workers: int = 4


@dataclass(frozen=True)
class ASOConfig:
    """Configuration for Agent-Supervised Optimization loop."""
    # Control mode
    enabled: bool = False

    # Supervision frequency
    supervision_mode: Literal["every_step", "periodic", "event_triggered"] = "event_triggered"
    supervision_interval: int = 5  # Steps between LLM calls (if periodic)

    # Convergence detection
    feasibility_threshold: float = 1e-3
    stagnation_objective_threshold: float = 1e-5
    stagnation_violation_threshold: float = 0.05
    max_stagnation_steps: int = 5

    # Constraint trend detection
    violation_increase_threshold: float = 0.05
    violation_decrease_threshold: float = 0.05

    # Budget allocation
    steps_per_supervision: int = 1

    # Safety limits
    max_constraint_weight: float = 1000.0
    max_penalty_boost: float = 4.0

    # Fallback behavior
    llm_timeout_seconds: float = 10.0
    llm_max_retries: int = 2
    use_heuristic_fallback: bool = True


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
    constraint_weights: ConstraintWeightsConfig
    generative: GenerativeConfig
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    alm: ALMConfig = field(default_factory=ALMConfig)
    aso: ASOConfig = field(default_factory=ASOConfig)
    optimizer_backend: str = "nevergrad"
    experiment_tag: str = "default"
    initialization_strategy: str = "template"
    reporting_dir: Path = Path("reports")
    memory_db: Path = DEFAULT_MEMORY_DB_PATH
    source_config: Path = DEFAULT_EXPERIMENT_CONFIG_PATH
    reporting: Mapping[str, Any] = field(default_factory=dict)
    run_overrides: Mapping[str, Any] = field(default_factory=dict)
    planner: str = "deterministic"

    @staticmethod
    def p3_high_fidelity() -> "ExperimentConfig":
        """Production config for P3 with high fidelity physics."""
        # Load defaults to fill required fields
        defaults = load_experiment_config()
        return ExperimentConfig(
            problem="p3",
            cycles=10,
            random_seed=defaults.random_seed,
            budgets=BudgetConfig(
                screen_evals_per_cycle=50,
                promote_top_k=5,
                max_high_fidelity_evals_per_cycle=3,
                wall_clock_minutes=defaults.budgets.wall_clock_minutes,
                n_workers=defaults.budgets.n_workers,
                pool_type=defaults.budgets.pool_type,
            ),
            adaptive_budgets=defaults.adaptive_budgets,
            fidelity_ladder=defaults.fidelity_ladder,
            boundary_template=defaults.boundary_template,
            stage_gates=defaults.stage_gates,
            governance=defaults.governance,
            proposal_mix=defaults.proposal_mix,
            constraint_weights=defaults.constraint_weights,
            generative=defaults.generative,
            surrogate=dataclasses.replace(defaults.surrogate, backend="neural_operator"),
            aso=ASOConfig(enabled=True, supervision_mode="event_triggered"),
        )

    @staticmethod
    def p3_quick_validation() -> "ExperimentConfig":
        """Fast config for testing/CI."""
        # Load defaults to fill required fields
        defaults = load_experiment_config()
        return ExperimentConfig(
            problem="p3",
            cycles=2,
            random_seed=defaults.random_seed,
            budgets=BudgetConfig(
                screen_evals_per_cycle=5,
                promote_top_k=2,
                max_high_fidelity_evals_per_cycle=1,
                wall_clock_minutes=defaults.budgets.wall_clock_minutes,
                n_workers=defaults.budgets.n_workers,
                pool_type=defaults.budgets.pool_type,
            ),
            adaptive_budgets=defaults.adaptive_budgets,
            fidelity_ladder=defaults.fidelity_ladder,
            boundary_template=defaults.boundary_template,
            stage_gates=defaults.stage_gates,
            governance=defaults.governance,
            proposal_mix=defaults.proposal_mix,
            constraint_weights=defaults.constraint_weights,
            generative=defaults.generative,
            surrogate=defaults.surrogate,
            aso=ASOConfig(enabled=False),
        )

    @staticmethod
    def p3_aso_enabled() -> "ExperimentConfig":
        """Config with Agent-Supervised Optimization."""
        # Load defaults to fill required fields
        defaults = load_experiment_config()
        return ExperimentConfig(
            problem="p3",
            cycles=5,
            random_seed=defaults.random_seed,
            budgets=defaults.budgets,
            adaptive_budgets=defaults.adaptive_budgets,
            fidelity_ladder=defaults.fidelity_ladder,
            boundary_template=defaults.boundary_template,
            stage_gates=defaults.stage_gates,
            governance=defaults.governance,
            proposal_mix=defaults.proposal_mix,
            constraint_weights=defaults.constraint_weights,
            generative=defaults.generative,
            surrogate=defaults.surrogate,
            aso=ASOConfig(
                enabled=True,
                supervision_mode="event_triggered",
                max_stagnation_steps=5,
            ),
        )
    
    @property
    def surrogate_backend(self) -> str:
        """Backward compatibility alias."""
        return self.surrogate.backend


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
    exploitation_ratio = float(config.get("exploitation_ratio", 0.0))
    return ProposalMixConfig(
        constraint_ratio=constraint_ratio,
        exploration_ratio=exploration_ratio,
        exploitation_ratio=exploitation_ratio,
        jitter_scale=float(config.get("jitter_scale", 0.01)),
        surrogate_pool_multiplier=float(config.get("surrogate_pool_multiplier", 2.0)),
        sampler_type=str(config.get("sampler_type", "standard")),
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


def _constraint_weights_from_dict(
    data: Mapping[str, Any] | None,
) -> ConstraintWeightsConfig:
    config = data or {}
    return ConstraintWeightsConfig(
        mhd=float(config.get("mhd", 1.0)),
        qi=float(config.get("qi", 1.0)),
        elongation=float(config.get("elongation", 1.0)),
    )


def _generative_config_from_dict(
    data: Mapping[str, Any] | None,
) -> GenerativeConfig:
    config = data or {}
    return GenerativeConfig(
        enabled=bool(config.get("enabled", False)),
        backend=str(config.get("backend", "vae")),
        latent_dim=int(config.get("latent_dim", 16)),
        learning_rate=float(config.get("learning_rate", 1e-3)),
        epochs=int(config.get("epochs", 100)),
        kl_weight=float(config.get("kl_weight", 0.001)),
    )


def _surrogate_config_from_dict(
    data: Mapping[str, Any] | None,
    legacy_backend: str | None = None,
) -> SurrogateConfig:
    config = data or {}
    backend = str(config.get("backend", legacy_backend or "random_forest"))
    return SurrogateConfig(
        backend=backend,
        n_ensembles=int(config.get("n_ensembles", 1)),
        learning_rate=float(config.get("learning_rate", 1e-3)),
        epochs=int(config.get("epochs", 100)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        use_offline_dataset=bool(config.get("use_offline_dataset", False)),
    )


def _alm_config_from_dict(data: Mapping[str, Any] | None) -> ALMConfig:
    config = data or {}
    return ALMConfig(
        penalty_parameters_increase_factor=float(
            config.get("penalty_parameters_increase_factor", 2.0)
        ),
        constraint_violation_tolerance_reduction_factor=float(
            config.get("constraint_violation_tolerance_reduction_factor", 0.5)
        ),
        bounds_reduction_factor=float(config.get("bounds_reduction_factor", 0.95)),
        penalty_parameters_max=float(config.get("penalty_parameters_max", 1e8)),
        bounds_min=float(config.get("bounds_min", 0.05)),
        maxit=int(config.get("maxit", 25)),
        penalty_parameters_initial=float(config.get("penalty_parameters_initial", 1.0)),
        bounds_initial=float(config.get("bounds_initial", 2.0)),
        oracle_budget_initial=int(config.get("oracle_budget_initial", 100)),
        oracle_budget_increment=int(config.get("oracle_budget_increment", 26)),
        oracle_budget_max=int(config.get("oracle_budget_max", 200)),
        oracle_num_workers=int(config.get("oracle_num_workers", 4)),
    )


def _aso_config_from_dict(data: Mapping[str, Any] | None) -> ASOConfig:
    config = data or {}
    supervision_mode = str(config.get("supervision_mode", "event_triggered"))
    return ASOConfig(
        enabled=bool(config.get("enabled", False)),
        supervision_mode=supervision_mode,  # type: ignore
        supervision_interval=int(config.get("supervision_interval", 5)),
        feasibility_threshold=float(config.get("feasibility_threshold", 1e-3)),
        stagnation_objective_threshold=float(
            config.get("stagnation_objective_threshold", 1e-5)
        ),
        stagnation_violation_threshold=float(
            config.get("stagnation_violation_threshold", 0.05)
        ),
        max_stagnation_steps=int(config.get("max_stagnation_steps", 5)),
        violation_increase_threshold=float(
            config.get("violation_increase_threshold", 0.05)
        ),
        violation_decrease_threshold=float(
            config.get("violation_decrease_threshold", 0.05)
        ),
        steps_per_supervision=int(config.get("steps_per_supervision", 1)),
        max_constraint_weight=float(config.get("max_constraint_weight", 1000.0)),
        max_penalty_boost=float(config.get("max_penalty_boost", 4.0)),
        llm_timeout_seconds=float(config.get("llm_timeout_seconds", 10.0)),
        llm_max_retries=int(config.get("llm_max_retries", 2)),
        use_heuristic_fallback=bool(config.get("use_heuristic_fallback", True)),
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
    
    # Handle legacy surrogate_backend at top level
    legacy_surrogate_backend = payload.get("surrogate_backend")
    surrogate_config = _surrogate_config_from_dict(
        payload.get("surrogate"), 
        legacy_backend=legacy_surrogate_backend
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
        constraint_weights=_constraint_weights_from_dict(
            payload.get("constraint_weights")
        ),
        generative=_generative_config_from_dict(payload.get("generative")),
        surrogate=surrogate_config,
        alm=_alm_config_from_dict(payload.get("alm")),
        aso=_aso_config_from_dict(payload.get("aso")),
        optimizer_backend=str(payload.get("optimizer_backend", "nevergrad")),
        experiment_tag=str(payload.get("experiment_tag", "default")),
        initialization_strategy=str(payload.get("initialization_strategy", "template")),
        run_overrides=payload.get("run_overrides", {}),
        reporting_dir=Path(payload.get("reporting_dir", "reports")),
        memory_db=Path(payload.get("memory_db", DEFAULT_MEMORY_DB_PATH)),
        source_config=config_path,
        reporting=payload.get("reporting", {}),
        planner=str(payload.get("planner", "deterministic")),
    )
