# Agent-Supervised Optimization (ASO) V4 Implementation Guide

**Status:** Ready for Implementation
**Date:** 2025-11-30
**Supersedes:** `UNIFIED_PLAN.md`, `ASO_V3_PLAN.md`
**Prerequisites:** V2 Roadmap Complete (Phases 1-4)

---

## Executive Summary

This document provides a **verified implementation guide** for the Agent-Supervised Optimization (ASO) loop. Unlike previous planning documents, this guide is grounded in a thorough codebase audit and addresses critical gaps discovered in earlier plans.

### Key Findings from Codebase Audit

1. **ALM State is Already Accessible** - `AugmentedLagrangianState` in `constellaration` is a Pydantic model, directly importable without upstream modification
2. **Planner Has Multi-Turn Loop** - `planner.py:384-468` already implements a 5-turn agentic loop with tool calling
3. **Coordinator Strategy is Minimal** - EXPLOIT and HYBRID paths are nearly identical; distinction is superficial
4. **Missing Config** - `alm_settings` mentioned in plans but doesn't exist in `ExperimentConfig`
5. **Worker Contract Incomplete** - `OptimizationWorker` doesn't return ALM state for continuation

### Architecture Improvements in V4

| Feature | V3 Plan | V4 Implementation |
|---------|---------|-------------------|
| LLM Supervision | Event-triggered | Same + **Surrogate-based proxy diagnostics** |
| ALM State Access | Assumed | **Verified: Direct import from constellaration** |
| Multi-Trajectory | Proposed | **Simplified: Sequential with early termination** |
| Heuristic Fallback | Proposed | **Implemented with configurable thresholds** |
| Telemetry | SQL table | **Append-only JSONL + SQL summary** |

---

## Part 1: Current Architecture (Verified)

### 1.1 What's Actually Implemented

```
ai_scientist/
├── runner.py          # Orchestrator (3,216 lines) - cycles, budgets, promotion
├── planner.py         # PlanningAgent (478 lines) - 5-turn LLM loop, RAG, tools
├── coordinator.py     # Strategy selector (118 lines) - EXPLORE/EXPLOIT/HYBRID
├── workers.py         # OptimizationWorker, ExplorationWorker, GeometerWorker
├── memory.py          # WorldModel SQLite (1,413 lines) - experiments, candidates
├── tools.py           # Physics evaluation wrappers (1,021 lines)
└── optim/
    ├── surrogate_v2.py    # NeuralOperatorSurrogate (543 lines)
    ├── generative.py      # VAE + Diffusion (666 lines)
    ├── differentiable.py  # Gradient descent (344 lines)
    └── geometry.py        # Fourier ↔ Real-space (569 lines)

constellaration/src/constellaration/
├── optimization/
│   ├── augmented_lagrangian.py      # AugmentedLagrangianState (Pydantic!)
│   └── augmented_lagrangian_runner.py
├── forward_model.py                  # Physics evaluation
└── problems.py                       # P1, P2, P3 definitions

checkpoints/
├── surrogate_physics_v2.pt   # 32MB Neural Operator ensemble
└── v2_1/
    ├── scaler.pkl            # LogRobustScaler
    └── physics_cols.txt      # 5 physics columns
```

### 1.2 The ALM State Contract (Verified)

The `AugmentedLagrangianState` from constellaration is already a Pydantic BaseModel:

```python
# constellaration/src/constellaration/optimization/augmented_lagrangian.py
class AugmentedLagrangianState(pydantic.BaseModel):
    x: jnp.ndarray                    # Current design point
    multipliers: jnp.ndarray          # Lagrange multipliers
    penalty_parameters: jnp.ndarray   # Penalty scaling factors
    objective: jnp.ndarray            # Current objective value
    constraints: jnp.ndarray          # Current constraint violations
    bounds: jnp.ndarray               # Trust region bounds per dimension
```

**This means:**
- JSON serializable out of the box
- No modification to upstream `constellaration` needed
- Can be imported directly: `from constellaration.optimization.augmented_lagrangian import AugmentedLagrangianState`

### 1.3 ALM Settings (From constellaration)

```python
# constellaration/src/constellaration/optimization/settings.py
class AugmentedLagrangianSettings:
    constraint_violation_tolerance_reduction_factor = 0.5
    penalty_parameters_increase_factor = 2.0
    bounds_reduction_factor = 0.95
    penalty_parameters_max = 1e8
    bounds_min = 0.05
```

---

## Part 2: Gap Analysis (Plan vs Reality)

### 2.1 Gaps in UNIFIED_PLAN.md

| Status | Planned | Reality | Resolution |
|--------|---------|---------|------------|
| [ ] | `OptimizationDirective` dataclass | Not implemented | Add to `planner.py` |
| [ ] | `OptimizerDiagnostics` dataclass | Not implemented | Add to `planner.py` |
| [ ] | `analyze_optimizer_diagnostics()` | Not implemented | Add to `PlanningAgent` |
| [ ] | `generate_diagnostics()` | Not implemented | Add to `Coordinator` |
| [ ] | `apply_directive()` | Not implemented | Add to `Coordinator` |
| [ ] | `produce_candidates_aso()` | Not implemented | Add to `Coordinator` |
| [ ] | `control_mode` CLI flag | Not implemented | Add to `config.py` |
| [ ] | `alm_settings` in config | **Doesn't exist** | Add `ALMConfig` dataclass |

### 2.2 Gaps in ASO_V3_PLAN.md

| Status | Planned | Reality | Resolution |
|--------|---------|---------|------------|
| [ ] | `ASOConfig` dataclass | Not implemented | Add to `config.py` |
| [ ] | `HeuristicSupervisor` class | Not implemented | Add to `planner.py` |
| [ ] | `TrajectoryState` dataclass | Not implemented | Add to `coordinator.py` |
| [ ] | Worker returns `final_alm_state` | **Workers don't return ALM state** | Update worker contract |
| [ ] | ASO telemetry table | Not implemented | Add to `memory.py` |

### 2.3 Current Planner Capabilities (Underestimated)

The existing `PlanningAgent.plan_cycle()` already has:
- 5-turn multi-turn LLM loop (`planner.py:385-468`)
- Tool calling with JSON parsing (`planner.py:401-461`)
- RAG retrieval integration (`planner.py:276-279`)
- `suggested_params` and `config_overrides` extraction (`planner.py:421-428`)
- Failure case reflection (`planner.py:328-333`)

**This is more capable than the plans suggested.** The gap is only the ASO-specific supervision method.

---

## Part 3: Implementation Specification

### 3.1 New Config Structures

Add to `ai_scientist/config.py`:

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass(frozen=True)
class ALMConfig:
    """ALM hyperparameters (mirrors constellaration settings)."""
    penalty_parameters_increase_factor: float = 2.0
    constraint_violation_tolerance_reduction_factor: float = 0.5
    bounds_reduction_factor: float = 0.95
    penalty_parameters_max: float = 1e8
    bounds_min: float = 0.05


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
    violation_increase_threshold: float = 0.05  # 5% increase = "increasing"
    violation_decrease_threshold: float = 0.05  # 5% decrease = "decreasing"

    # Multi-trajectory (simplified)
    n_trajectories: int = 1  # Start with 1, increase after validation
    inner_budget: int = 10   # Evals per supervision check

    # Safety limits
    max_constraint_weight: float = 1000.0

    # Fallback behavior
    llm_timeout_seconds: float = 10.0
    llm_max_retries: int = 2
    use_heuristic_fallback: bool = True


# Update ExperimentConfig to include:
@dataclass
class ExperimentConfig:
    # ... existing fields ...
    alm: ALMConfig = field(default_factory=ALMConfig)
    aso: ASOConfig = field(default_factory=ASOConfig)
```

### 3.2 Core Data Structures

Add to `ai_scientist/planner.py`:

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, List, Optional


class DirectiveAction(Enum):
    """Enumerated actions for type safety."""
    CONTINUE = "CONTINUE"
    ADJUST = "ADJUST"
    STOP = "STOP"
    RESTART = "RESTART"


class DirectiveSource(Enum):
    """Source of the directive for debugging."""
    LLM = "llm"
    HEURISTIC = "heuristic"
    CONVERGENCE = "convergence"
    FALLBACK = "fallback"


@dataclass
class OptimizationDirective:
    """Structured directive from Planner to Coordinator."""
    action: DirectiveAction
    config_overrides: Optional[Mapping[str, Any]] = None
    suggested_params: Optional[Mapping[str, Any]] = None
    reasoning: str = ""
    confidence: float = 1.0
    source: DirectiveSource = DirectiveSource.HEURISTIC

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "config_overrides": dict(self.config_overrides) if self.config_overrides else None,
            "suggested_params": dict(self.suggested_params) if self.suggested_params else None,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "source": self.source.value,
        }


@dataclass
class ConstraintDiagnostic:
    """Diagnostic for a single constraint."""
    name: str
    violation: float
    penalty: float
    trend: str  # "stable", "increasing_violation", "decreasing_violation"
    delta: float = 0.0


@dataclass
class OptimizerDiagnostics:
    """Rich diagnostic report from Coordinator to Planner."""
    step: int
    trajectory_id: int
    objective: float
    objective_delta: float
    max_violation: float
    status: str  # "IN_PROGRESS", "STAGNATION", "FEASIBLE_FOUND", "DIVERGING"
    constraint_diagnostics: List[ConstraintDiagnostic]
    optimizer_health: Mapping[str, float]
    narrative: List[str]
    steps_since_improvement: int = 0

    def requires_llm_supervision(self, aso_config: "ASOConfig") -> bool:
        """Determine if this diagnostic warrants an LLM call."""
        if aso_config.supervision_mode == "every_step":
            return True
        if aso_config.supervision_mode == "periodic":
            return self.step % aso_config.supervision_interval == 0

        # Event-triggered: only on significant events
        return any([
            self.status == "STAGNATION",
            self.status == "FEASIBLE_FOUND",
            self.status == "DIVERGING",
            any(c.trend == "increasing_violation" for c in self.constraint_diagnostics),
            self.steps_since_improvement >= aso_config.max_stagnation_steps,
        ])

    def to_json(self) -> str:
        import json
        return json.dumps({
            "step": self.step,
            "trajectory_id": self.trajectory_id,
            "objective": self.objective,
            "objective_delta": self.objective_delta,
            "max_violation": self.max_violation,
            "status": self.status,
            "constraints": [
                {"name": c.name, "violation": c.violation, "trend": c.trend}
                for c in self.constraint_diagnostics
            ],
            "narrative": self.narrative,
        }, indent=2)
```

### 3.3 Heuristic Supervisor

Add to `ai_scientist/planner.py`:

```python
class HeuristicSupervisor:
    """
    Rule-based optimization supervisor.
    Handles 80%+ of cases without LLM latency.
    """

    def __init__(self, aso_config: ASOConfig):
        self.config = aso_config

    def analyze(self, diagnostics: OptimizerDiagnostics) -> OptimizationDirective:
        """
        Generate directive using heuristic rules.

        Decision tree:
        1. FEASIBLE_FOUND + stable objective -> STOP (converged)
        2. FEASIBLE_FOUND + improving -> CONTINUE
        3. STAGNATION + high violation -> ADJUST (increase penalties)
        4. STAGNATION + low violation -> RESTART (try new seed)
        5. DIVERGING -> STOP (abandon trajectory)
        6. Increasing violation on specific constraint -> ADJUST (boost weight)
        7. Otherwise -> CONTINUE
        """
        cfg = self.config

        # Case 1 & 2: Feasible region reached
        if diagnostics.status == "FEASIBLE_FOUND":
            if abs(diagnostics.objective_delta) < cfg.stagnation_objective_threshold:
                return OptimizationDirective(
                    action=DirectiveAction.STOP,
                    reasoning="Converged: feasible with stable objective",
                    source=DirectiveSource.CONVERGENCE,
                )
            return OptimizationDirective(
                action=DirectiveAction.CONTINUE,
                reasoning="Feasible and still improving",
                source=DirectiveSource.HEURISTIC,
            )

        # Case 3 & 4: Stagnation
        if diagnostics.status == "STAGNATION":
            if diagnostics.max_violation > cfg.stagnation_violation_threshold:
                return OptimizationDirective(
                    action=DirectiveAction.ADJUST,
                    config_overrides={
                        "alm": {"penalty_parameters_increase_factor": 4.0}
                    },
                    reasoning=f"Stagnation with violation={diagnostics.max_violation:.4f}, increasing penalties",
                    source=DirectiveSource.HEURISTIC,
                )
            return OptimizationDirective(
                action=DirectiveAction.RESTART,
                reasoning="Stagnation near feasibility, trying new seed",
                source=DirectiveSource.HEURISTIC,
            )

        # Case 5: Diverging
        if diagnostics.status == "DIVERGING":
            return OptimizationDirective(
                action=DirectiveAction.STOP,
                reasoning="Trajectory diverging, abandoning",
                source=DirectiveSource.HEURISTIC,
            )

        # Case 6: Specific constraint struggling
        struggling = [c for c in diagnostics.constraint_diagnostics if c.trend == "increasing_violation"]
        if struggling:
            worst = max(struggling, key=lambda c: c.violation)
            boost = min(worst.penalty * 2, cfg.max_constraint_weight)
            return OptimizationDirective(
                action=DirectiveAction.ADJUST,
                config_overrides={"constraint_weights": {worst.name: boost}},
                reasoning=f"Constraint '{worst.name}' worsening, boosting weight to {boost}",
                source=DirectiveSource.HEURISTIC,
            )

        # Case 7: Default
        return OptimizationDirective(
            action=DirectiveAction.CONTINUE,
            reasoning="Normal progress",
            source=DirectiveSource.HEURISTIC,
        )
```

### 3.4 PlanningAgent Extension

Add method to `PlanningAgent` class in `ai_scientist/planner.py`:

```python
class PlanningAgent:
    def __init__(self, ...):
        # ... existing init ...
        self.heuristic: HeuristicSupervisor | None = None

    def _ensure_heuristic(self, aso_config: ASOConfig) -> HeuristicSupervisor:
        if self.heuristic is None:
            self.heuristic = HeuristicSupervisor(aso_config)
        return self.heuristic

    def supervise(
        self,
        diagnostics: OptimizerDiagnostics,
        cycle: int,
        aso_config: ASOConfig,
    ) -> OptimizationDirective:
        """
        Tiered supervision: heuristic first, LLM on demand.
        """
        heuristic = self._ensure_heuristic(aso_config)

        # Tier 1: Check if LLM needed
        if not diagnostics.requires_llm_supervision(aso_config):
            return heuristic.analyze(diagnostics)

        # Tier 2: Try LLM with fallback
        if aso_config.use_heuristic_fallback:
            try:
                return self._llm_supervise(diagnostics, cycle, aso_config)
            except Exception as e:
                print(f"[Planner] LLM supervision failed: {e}, using heuristic")
                return heuristic.analyze(diagnostics)
        else:
            return self._llm_supervise(diagnostics, cycle, aso_config)

    def _llm_supervise(
        self,
        diagnostics: OptimizerDiagnostics,
        cycle: int,
        aso_config: ASOConfig,
    ) -> OptimizationDirective:
        """LLM-based supervision with structured output."""
        # Retrieve relevant context if stagnating
        rag_context = []
        if diagnostics.status == "STAGNATION":
            rag_context = self.retrieve_rag(
                "Strategies for escaping local minima in stellarator ALM optimization",
                k=2,
            )

        system_prompt = self._build_supervision_prompt(cycle, rag_context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Current diagnostics:\n{diagnostics.to_json()}"},
        ]

        # Use existing LLM invocation pattern from plan_cycle
        from ai_scientist import model_provider
        provider = self.config.get_provider()

        for attempt in range(aso_config.llm_max_retries):
            try:
                response = model_provider.invoke_chat_completion(
                    provider,
                    tool_call={"name": "supervise_optimization", "arguments": {}},
                    messages=messages,
                    model=self.planning_gate.provider_model,
                )

                if response.status_code != 200:
                    raise RuntimeError(f"LLM returned {response.status_code}")

                content = response.body.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                return self._parse_directive(content)

            except json.JSONDecodeError as e:
                if attempt < aso_config.llm_max_retries - 1:
                    messages.append({"role": "user", "content": f"Invalid JSON: {e}. Please output valid JSON."})
                    continue
                raise

        raise RuntimeError("LLM supervision failed after retries")

    def _build_supervision_prompt(self, cycle: int, rag_context: list) -> str:
        rag_section = ""
        if rag_context:
            rag_section = f"\n\nRelevant knowledge:\n{json.dumps(rag_context, indent=2)}"

        return f"""You are the ASO Supervisor for the AI Scientist (cycle {cycle}).

Analyze the optimization diagnostics and decide the next action.

ACTIONS:
- CONTINUE: Proceed with current settings
- ADJUST: Modify constraint weights or ALM parameters
- STOP: Terminate trajectory (converged or hopeless)
- RESTART: Abandon trajectory, try new seed

OUTPUT FORMAT (JSON):
{{
  "action": "CONTINUE | ADJUST | STOP | RESTART",
  "config_overrides": {{"constraint_weights": {{"name": value}}}},  // optional
  "reasoning": "brief explanation"
}}

ADJUSTMENT OPTIONS:
- constraint_weights: Boost weight of struggling constraints (e.g., {{"qi": 100.0}})
- alm.penalty_parameters_increase_factor: Increase ALM penalties (default 2.0, try 4.0)
{rag_section}

Respond with ONLY valid JSON."""

    def _parse_directive(self, content: str) -> OptimizationDirective:
        """Parse LLM response into OptimizationDirective."""
        # Extract JSON from potential markdown
        json_str = content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()

        data = json.loads(json_str)
        action = DirectiveAction(data.get("action", "CONTINUE"))

        return OptimizationDirective(
            action=action,
            config_overrides=data.get("config_overrides"),
            reasoning=data.get("reasoning", ""),
            source=DirectiveSource.LLM,
        )
```

### 3.5 Coordinator ASO Loop

Update `ai_scientist/coordinator.py`:

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

from constellaration.optimization.augmented_lagrangian import AugmentedLagrangianState

from ai_scientist import config as ai_config
from ai_scientist import memory
from ai_scientist.planner import (
    PlanningAgent,
    OptimizerDiagnostics,
    OptimizationDirective,
    ConstraintDiagnostic,
    DirectiveAction,
)
from ai_scientist.workers import OptimizationWorker, ExplorationWorker, GeometerWorker
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate
from ai_scientist.optim.generative import GenerativeDesignModel


@dataclass
class TrajectoryState:
    """State for a single optimization trajectory."""
    id: int
    seed: Dict[str, Any]
    alm_state: Optional[AugmentedLagrangianState] = None
    history: List[AugmentedLagrangianState] = field(default_factory=list)
    evals_used: int = 0
    steps: int = 0
    status: str = "active"  # "active", "converged", "stagnated", "abandoned"
    best_objective: float = float("inf")
    best_violation: float = float("inf")
    stagnation_count: int = 0


class Coordinator:
    """Manages Agent-Supervised Optimization with tiered supervision."""

    # Constraint names by problem type
    CONSTRAINT_NAMES = {
        "p1": ["aspect_ratio", "average_triangularity", "edge_rotational_transform"],
        "p2": ["aspect_ratio", "edge_rotational_transform", "edge_magnetic_mirror_ratio", "max_elongation", "qi"],
        "p3": ["edge_rotational_transform", "edge_magnetic_mirror_ratio", "vacuum_well", "flux_compression", "qi"],
    }

    def __init__(
        self,
        cfg: ai_config.ExperimentConfig,
        world_model: memory.WorldModel,
        planner: PlanningAgent,
        surrogate: Optional[NeuralOperatorSurrogate] = None,
        generative_model: Optional[GenerativeDesignModel] = None,
    ):
        self.cfg = cfg
        self.world_model = world_model
        self.planner = planner
        self.surrogate = surrogate
        self.generative_model = generative_model

        # Workers
        self.opt_worker = OptimizationWorker(cfg, surrogate)
        self.explore_worker = ExplorationWorker(cfg, generative_model)
        self.geo_worker = GeometerWorker(cfg)

        # Constraint names for this problem
        problem_key = (cfg.problem or "p3").lower()[:2]
        self.constraint_names = self.CONSTRAINT_NAMES.get(problem_key, self.CONSTRAINT_NAMES["p3"])

        # Telemetry buffer
        self.telemetry: List[Dict[str, Any]] = []

    def produce_candidates_aso(
        self,
        cycle: int,
        experiment_id: int,
        eval_budget: int,
        template: ai_config.BoundaryTemplateConfig,
        initial_seeds: Optional[List[Dict[str, Any]]] = None,
        initial_config: Optional[ai_config.ExperimentConfig] = None,
    ) -> List[Dict[str, Any]]:
        """
        ASO loop with tiered supervision.

        Returns list of candidate boundary parameter dicts.
        """
        config = initial_config or self.cfg
        aso = config.aso

        # 1. Prepare seeds
        seeds = self._prepare_seeds(initial_seeds, cycle, aso.n_trajectories)
        if not seeds:
            print("[Coordinator] No valid seeds, returning empty")
            return []

        # 2. Run trajectory (single for now, extensible to multi)
        traj = TrajectoryState(id=0, seed=seeds[0])
        candidates = self._run_trajectory_aso(
            traj=traj,
            budget=eval_budget,
            inner_budget=aso.inner_budget,
            cycle=cycle,
            experiment_id=experiment_id,
            config=config,
        )

        # 3. Persist telemetry
        self._persist_telemetry(experiment_id)

        return candidates

    def _prepare_seeds(
        self,
        initial_seeds: Optional[List[Dict]],
        cycle: int,
        n_needed: int,
    ) -> List[Dict]:
        """Prepare and validate seeds."""
        if initial_seeds:
            seeds = initial_seeds
        else:
            explore_ctx = {"n_samples": n_needed * 2, "cycle": cycle}
            seeds = self.explore_worker.run(explore_ctx).get("candidates", [])

        if not seeds:
            return []

        # Validate geometry
        geo_ctx = {"candidates": seeds}
        return self.geo_worker.run(geo_ctx).get("candidates", [])[:n_needed]

    def _run_trajectory_aso(
        self,
        traj: TrajectoryState,
        budget: int,
        inner_budget: int,
        cycle: int,
        experiment_id: int,
        config: ai_config.ExperimentConfig,
    ) -> List[Dict[str, Any]]:
        """Run single trajectory with ASO supervision."""
        aso = config.aso
        candidates = []

        opt_ctx = {
            "initial_guesses": [traj.seed],
            "budget": inner_budget,
            "alm_settings_overrides": {},
            "return_alm_state": True,  # Request ALM state
        }

        while traj.evals_used < budget and traj.status == "active":
            traj.steps += 1
            step_start = time.perf_counter()

            # 1. Execute optimization chunk
            res = self.opt_worker.run(opt_ctx)
            traj.evals_used += res.get("evals_used", inner_budget)
            chunk_candidates = res.get("candidates", [])
            candidates.extend(chunk_candidates)

            # 2. Get ALM state (or use surrogate proxy)
            alm_state = res.get("alm_state")
            if alm_state is None:
                # Fallback: use surrogate predictions as proxy
                diagnostics = self._generate_proxy_diagnostics(chunk_candidates, traj)
            else:
                diagnostics = self._generate_diagnostics(alm_state, traj)
                traj.history.append(alm_state)

            # 3. Update trajectory tracking
            self._update_trajectory_best(traj, diagnostics)

            # 4. Get directive (tiered supervision)
            llm_called = diagnostics.requires_llm_supervision(aso)
            directive = self.planner.supervise(diagnostics, cycle, aso)

            # 5. Log telemetry
            wall_time_ms = (time.perf_counter() - step_start) * 1000
            self._log_telemetry(
                experiment_id, cycle, traj, diagnostics, directive, wall_time_ms, llm_called
            )

            # 6. Apply directive
            if directive.action == DirectiveAction.STOP:
                traj.status = "converged" if diagnostics.status == "FEASIBLE_FOUND" else "stagnated"
                print(f"[Coordinator] STOP: {directive.reasoning}")
                break

            if directive.action == DirectiveAction.RESTART:
                new_seeds = self._prepare_seeds(None, cycle, 1)
                if new_seeds:
                    traj.seed = new_seeds[0]
                    traj.history = []
                    traj.stagnation_count = 0
                    opt_ctx["initial_guesses"] = [traj.seed]
                    opt_ctx["continue_from_state"] = None
                    print(f"[Coordinator] RESTART with new seed")
                else:
                    traj.status = "abandoned"
                    print(f"[Coordinator] RESTART failed, no seeds")
                    break
                continue

            # Apply config adjustments
            config, worker_overrides = self._apply_directive(directive, config)
            opt_ctx["alm_settings_overrides"] = worker_overrides.get("alm", {})
            opt_ctx["budget"] = min(inner_budget, budget - traj.evals_used)

            # Auto-stop on excessive stagnation
            if traj.stagnation_count >= aso.max_stagnation_steps:
                traj.status = "stagnated"
                print(f"[Coordinator] Auto-STOP (stagnation limit)")
                break

        print(f"[Coordinator] Trajectory done: {traj.status}, {traj.steps} steps, "
              f"{traj.evals_used} evals, {len(candidates)} candidates")

        return candidates

    def _generate_diagnostics(
        self,
        alm_state: AugmentedLagrangianState,
        traj: TrajectoryState,
    ) -> OptimizerDiagnostics:
        """Translate ALM state to semantic diagnostics."""
        import jax.numpy as jnp
        aso = self.cfg.aso
        prev = traj.history[-1] if traj.history else None

        # Constraint analysis
        constraint_diagnostics = []
        diverging_count = 0

        for i, name in enumerate(self.constraint_names):
            if i >= len(alm_state.constraints):
                continue

            violation = float(jnp.maximum(0.0, alm_state.constraints[i]))
            penalty = float(alm_state.penalty_parameters[i]) if i < len(alm_state.penalty_parameters) else 1.0
            trend = "stable"
            delta = 0.0

            if prev and i < len(prev.constraints):
                prev_violation = float(jnp.maximum(0.0, prev.constraints[i]))
                delta = violation - prev_violation

                if violation > prev_violation * (1 + aso.violation_increase_threshold):
                    trend = "increasing_violation"
                    diverging_count += 1
                elif violation < prev_violation * (1 - aso.violation_decrease_threshold):
                    trend = "decreasing_violation"

            constraint_diagnostics.append(ConstraintDiagnostic(
                name=name, violation=violation, penalty=penalty, trend=trend, delta=delta
            ))

        # Objective
        objective = float(alm_state.objective)
        objective_delta = objective - float(prev.objective) if prev else 0.0
        max_violation = float(jnp.max(jnp.maximum(0.0, alm_state.constraints)))

        # Status
        narrative = []
        if max_violation < aso.feasibility_threshold:
            status = "FEASIBLE_FOUND"
            narrative.append("Feasible region reached")
        elif diverging_count >= len(self.constraint_names) // 2:
            status = "DIVERGING"
            narrative.append("Multiple constraints diverging")
        elif prev and abs(objective_delta) < aso.stagnation_objective_threshold:
            if max_violation > aso.stagnation_violation_threshold:
                status = "STAGNATION"
                narrative.append("Stagnation with constraint violations")
            else:
                status = "IN_PROGRESS"
                narrative.append("Near convergence")
        else:
            status = "IN_PROGRESS"
            narrative.append("Normal progress")

        return OptimizerDiagnostics(
            step=traj.steps,
            trajectory_id=traj.id,
            objective=objective,
            objective_delta=objective_delta,
            max_violation=max_violation,
            status=status,
            constraint_diagnostics=constraint_diagnostics,
            optimizer_health={"stagnation_count": traj.stagnation_count},
            narrative=narrative,
            steps_since_improvement=traj.stagnation_count,
        )

    def _generate_proxy_diagnostics(
        self,
        candidates: List[Dict],
        traj: TrajectoryState,
    ) -> OptimizerDiagnostics:
        """Generate diagnostics from surrogate predictions when ALM state unavailable."""
        if not candidates or not self.surrogate:
            return OptimizerDiagnostics(
                step=traj.steps,
                trajectory_id=traj.id,
                objective=traj.best_objective,
                objective_delta=0.0,
                max_violation=traj.best_violation,
                status="IN_PROGRESS",
                constraint_diagnostics=[],
                optimizer_health={"stagnation_count": traj.stagnation_count},
                narrative=["Using proxy diagnostics"],
                steps_since_improvement=traj.stagnation_count,
            )

        # Use surrogate to estimate metrics
        from ai_scientist import tools
        best_candidate = candidates[0]
        try:
            flat, schema = tools.structured_flatten(best_candidate, self.cfg.boundary_template)
            pred = self.surrogate.predict([flat])
            objective = float(pred.mean[0])
            # Simplified: use objective improvement as proxy for violation
            objective_delta = objective - traj.best_objective
            status = "FEASIBLE_FOUND" if objective_delta < -0.1 else "IN_PROGRESS"
        except Exception:
            objective = traj.best_objective
            objective_delta = 0.0
            status = "IN_PROGRESS"

        return OptimizerDiagnostics(
            step=traj.steps,
            trajectory_id=traj.id,
            objective=objective,
            objective_delta=objective_delta,
            max_violation=traj.best_violation,
            status=status,
            constraint_diagnostics=[],
            optimizer_health={"stagnation_count": traj.stagnation_count},
            narrative=["Proxy diagnostics from surrogate"],
            steps_since_improvement=traj.stagnation_count,
        )

    def _update_trajectory_best(self, traj: TrajectoryState, diag: OptimizerDiagnostics):
        """Update trajectory best values and stagnation counter."""
        improved = False
        if diag.max_violation < traj.best_violation:
            traj.best_violation = diag.max_violation
            improved = True
        if diag.objective < traj.best_objective and diag.max_violation <= traj.best_violation:
            traj.best_objective = diag.objective
            improved = True

        if improved:
            traj.stagnation_count = 0
        else:
            traj.stagnation_count += 1

    def _apply_directive(
        self,
        directive: OptimizationDirective,
        config: ai_config.ExperimentConfig,
    ) -> Tuple[ai_config.ExperimentConfig, Dict[str, Any]]:
        """Apply directive config overrides with safety guards."""
        from dataclasses import replace

        if not directive.config_overrides:
            return config, {}

        aso = config.aso
        new_cfg = config
        worker_overrides: Dict[str, Any] = {}
        overrides = directive.config_overrides

        try:
            # Constraint weights
            if "constraint_weights" in overrides:
                clamped = {
                    k: min(float(v), aso.max_constraint_weight)
                    for k, v in overrides["constraint_weights"].items()
                }
                new_weights = replace(new_cfg.constraint_weights, **clamped)
                new_cfg = replace(new_cfg, constraint_weights=new_weights)

            # ALM settings (pass to worker)
            if "alm" in overrides:
                worker_overrides["alm"] = overrides["alm"]

        except Exception as e:
            print(f"[Coordinator] Failed to apply directive: {e}")

        return new_cfg, worker_overrides

    def _log_telemetry(
        self,
        experiment_id: int,
        cycle: int,
        traj: TrajectoryState,
        diag: OptimizerDiagnostics,
        directive: OptimizationDirective,
        wall_time_ms: float,
        llm_called: bool,
    ):
        """Record telemetry event."""
        from datetime import datetime, timezone
        self.telemetry.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_id": experiment_id,
            "cycle": cycle,
            "trajectory_id": traj.id,
            "step": traj.steps,
            "status": diag.status,
            "objective": diag.objective,
            "max_violation": diag.max_violation,
            "directive_action": directive.action.value,
            "directive_source": directive.source.value,
            "directive_reasoning": directive.reasoning,
            "evals_used": traj.evals_used,
            "wall_time_ms": wall_time_ms,
            "llm_called": llm_called,
        })

    def _persist_telemetry(self, experiment_id: int):
        """Persist telemetry to JSONL file."""
        if not self.telemetry:
            return

        from pathlib import Path
        telemetry_dir = Path(self.cfg.reporting_dir) / "telemetry"
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        telemetry_file = telemetry_dir / f"aso_exp{experiment_id}.jsonl"

        import json
        with open(telemetry_file, "a") as f:
            for event in self.telemetry:
                f.write(json.dumps(event) + "\n")

        self.telemetry = []

    # Keep existing methods for backward compatibility
    def decide_strategy(self, cycle: int, experiment_id: int) -> str:
        """Legacy strategy selector (unchanged)."""
        if cycle < 5:
            return "HYBRID"
        hv_delta = self.world_model.average_recent_hv_delta(experiment_id, lookback=3)
        if hv_delta is not None and hv_delta < 0.005:
            return "EXPLORE"
        return "HYBRID"

    def produce_candidates(
        self,
        cycle: int,
        experiment_id: int,
        n_candidates: int,
        template: ai_config.BoundaryTemplateConfig,
    ) -> List[Dict[str, Any]]:
        """Legacy candidate production (unchanged for backward compatibility)."""
        # ... existing implementation ...
        pass
```

### 3.6 Worker Contract Update

Update `ai_scientist/workers.py` to include ALM state return:

```python
class OptimizationWorker(Worker):
    """Worker for gradient-based optimization with ALM state return."""

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute optimization chunk.

        Args:
            context:
                initial_guesses: List of starting boundary params
                budget: Max evaluations for this chunk
                return_alm_state: If True, include ALM state in result
                alm_settings_overrides: Dict of ALM hyperparameters

        Returns:
            Dict with keys:
                candidates: List of boundary param dicts
                alm_state: AugmentedLagrangianState (if return_alm_state)
                evals_used: Actual evaluations consumed
                status: "optimized", "skipped", "failed"
        """
        initial_guesses = context.get("initial_guesses", [])
        budget = context.get("budget", 10)
        return_alm = context.get("return_alm_state", False)
        alm_overrides = context.get("alm_settings_overrides", {})

        if not initial_guesses:
            return {"candidates": [], "alm_state": None, "evals_used": 0, "status": "skipped"}

        if not self.surrogate or not self.surrogate._trained:
            return {"candidates": initial_guesses, "alm_state": None, "evals_used": 0, "status": "skipped"}

        try:
            from ai_scientist.optim import differentiable

            optimized, final_state, evals = differentiable.gradient_descent_on_inputs(
                initial_guesses,
                self.surrogate,
                self.cfg,
                budget=budget,
                alm_settings=alm_overrides,
                return_state=return_alm,
            )

            return {
                "candidates": optimized,
                "alm_state": final_state if return_alm else None,
                "evals_used": evals,
                "status": "optimized",
            }
        except Exception as e:
            print(f"[OptimizationWorker] Failed: {e}")
            return {"candidates": initial_guesses, "alm_state": None, "evals_used": 0, "status": "failed"}
```

---

## Part 4: Runner Integration

### 4.1 Control Mode Selection

Update runner to support ASO mode:

```python
# In runner.py

def _run_cycle(
    cfg: ai_config.ExperimentConfig,
    cycle_index: int,
    world_model: memory.WorldModel,
    experiment_id: int,
    planner: ai_planner.PlanningAgent,
    coordinator: Coordinator,
    budget_controller: BudgetController,
    # ... other params ...
) -> Tuple[Path | None, Dict | None, tools.P3Summary | None]:
    """Run single cycle with ASO or legacy mode."""
    cycle_number = cycle_index + 1

    # 1. High-level planning
    planning_outcome = planner.plan_cycle(
        cfg=cfg,
        cycle_index=cycle_index,
        stage_history=stage_history,
        last_summary=last_p3_summary,
        experiment_id=experiment_id,
    )

    # Apply config overrides
    active_cfg = cfg
    if planning_outcome.config_overrides:
        active_cfg = _apply_config_overrides(cfg, planning_outcome.config_overrides)

    budget_snapshot = budget_controller.snapshot()

    # 2. Branch on control mode
    if active_cfg.aso.enabled:
        # ASO V4 path
        initial_seeds = []
        if planning_outcome.suggested_params:
            initial_seeds = [planning_outcome.suggested_params]

        print(f"[runner][cycle={cycle_number}] ASO mode enabled")

        candidates = coordinator.produce_candidates_aso(
            cycle=cycle_number,
            experiment_id=experiment_id,
            eval_budget=budget_snapshot.screen_evals_per_cycle,
            template=active_cfg.boundary_template,
            initial_seeds=initial_seeds,
            initial_config=active_cfg,
        )
    else:
        # Legacy path (unchanged)
        candidates = coordinator.produce_candidates(
            cycle=cycle_number,
            experiment_id=experiment_id,
            n_candidates=budget_snapshot.screen_evals_per_cycle,
            template=active_cfg.boundary_template,
        )

    # 3. Continue with evaluation, promotion, etc. (unchanged)
    # ...
```

### 4.2 CLI Flag

Add to runner CLI:

```python
parser.add_argument(
    "--aso",
    action="store_true",
    help="Enable Agent-Supervised Optimization mode",
)

# In main():
if args.aso:
    cfg = replace(cfg, aso=replace(cfg.aso, enabled=True))
```

---

## Part 5: Implementation Checklist

### Phase 1: Foundation (Priority 0)

**File: `ai_scientist/config.py`**
- [ ] 1.1 Add `ALMConfig` dataclass (frozen, mirrors constellaration settings)
- [ ] 1.2 Add `ASOConfig` dataclass (supervision mode, thresholds, fallback settings)
- [ ] 1.3 Update `ExperimentConfig` to include `alm: ALMConfig` and `aso: ASOConfig` fields

**File: `ai_scientist/planner.py`**
- [ ] 1.4 Add `DirectiveAction` enum (`CONTINUE`, `ADJUST`, `STOP`, `RESTART`)
- [ ] 1.5 Add `DirectiveSource` enum (`LLM`, `HEURISTIC`, `CONVERGENCE`, `FALLBACK`)
- [ ] 1.6 Add `OptimizationDirective` dataclass with `to_dict()` method
- [ ] 1.7 Add `ConstraintDiagnostic` dataclass (name, violation, penalty, trend, delta)
- [ ] 1.8 Add `OptimizerDiagnostics` dataclass with `requires_llm_supervision()` and `to_json()`
- [ ] 1.9 Add `HeuristicSupervisor` class with `analyze()` method
- [ ] 1.10 Add `_ensure_heuristic()` method to `PlanningAgent`
- [ ] 1.11 Add `supervise()` method to `PlanningAgent`
- [ ] 1.12 Add `_llm_supervise()` method to `PlanningAgent`
- [ ] 1.13 Add `_build_supervision_prompt()` method to `PlanningAgent`
- [ ] 1.14 Add `_parse_directive()` method to `PlanningAgent`

### Phase 2: Coordinator (Priority 0)

**File: `ai_scientist/coordinator.py`**
- [ ] 2.1 Add import for `AugmentedLagrangianState` from constellaration
- [ ] 2.2 Add `CONSTRAINT_NAMES` class constant (mapping problem type → constraint names)
- [ ] 2.3 Add `TrajectoryState` dataclass (id, seed, alm_state, history, evals_used, status, etc.)
- [ ] 2.4 Update `__init__()` to set `constraint_names` and `telemetry` buffer
- [ ] 2.5 Add `produce_candidates_aso()` method (main ASO entry point)
- [ ] 2.6 Add `_prepare_seeds()` method (seed validation with GeometerWorker)
- [ ] 2.7 Add `_run_trajectory_aso()` method (single trajectory ASO loop)
- [ ] 2.8 Add `_generate_diagnostics()` method (ALM state → OptimizerDiagnostics)
- [ ] 2.9 Add `_generate_proxy_diagnostics()` method (surrogate-based fallback)
- [ ] 2.10 Add `_update_trajectory_best()` method (track stagnation)
- [ ] 2.11 Add `_apply_directive()` method (config overrides with safety guards)
- [ ] 2.12 Add `_log_telemetry()` method (event recording)
- [ ] 2.13 Add `_persist_telemetry()` method (JSONL file writing)

### Phase 3: Integration (Priority 1)

**File: `ai_scientist/workers.py`**
- [ ] 3.1 Update `OptimizationWorker.run()` to accept `return_alm_state` context param
- [ ] 3.2 Update `OptimizationWorker.run()` to return `alm_state` in result dict
- [ ] 3.3 Update `OptimizationWorker.run()` to accept `alm_settings_overrides` context param

**File: `ai_scientist/optim/differentiable.py`**
- [ ] 3.4 Add `return_state` parameter to `gradient_descent_on_inputs()`
- [ ] 3.5 Add `alm_settings` parameter to `gradient_descent_on_inputs()`
- [ ] 3.6 Return `(optimized, final_state, evals)` tuple when `return_state=True`

**File: `ai_scientist/runner.py`**
- [ ] 3.7 Add `--aso` CLI argument to argument parser
- [ ] 3.8 Update config initialization to set `aso.enabled` from CLI flag
- [ ] 3.9 Update `_run_cycle()` to branch on `cfg.aso.enabled`
- [ ] 3.10 Call `coordinator.produce_candidates_aso()` in ASO branch
- [ ] 3.11 Add `initialize_architecture()` helper function

### Phase 4: Testing (Priority 1)

**File: `tests/test_planner.py`**
- [ ] 4.1 Unit test: `HeuristicSupervisor.analyze()` returns STOP on FEASIBLE_FOUND + stable
- [ ] 4.2 Unit test: `HeuristicSupervisor.analyze()` returns ADJUST on STAGNATION + high violation
- [ ] 4.3 Unit test: `HeuristicSupervisor.analyze()` returns RESTART on STAGNATION + low violation
- [ ] 4.4 Unit test: `HeuristicSupervisor.analyze()` returns ADJUST on increasing_violation trend
- [ ] 4.5 Unit test: `HeuristicSupervisor.analyze()` returns CONTINUE on normal progress

**File: `tests/test_coordinator.py`**
- [ ] 4.6 Unit test: `OptimizerDiagnostics.requires_llm_supervision()` with `every_step` mode
- [ ] 4.7 Unit test: `OptimizerDiagnostics.requires_llm_supervision()` with `periodic` mode
- [ ] 4.8 Unit test: `OptimizerDiagnostics.requires_llm_supervision()` with `event_triggered` mode
- [ ] 4.9 Integration test: `produce_candidates_aso()` with mock `OptimizationWorker`
- [ ] 4.10 Integration test: Verify telemetry JSONL file is written correctly

**File: `scripts/test_aso_loop.py`**
- [ ] 4.11 Create smoke test script that runs ASO loop for 3 steps
- [ ] 4.12 Verify heuristic fallback works when LLM is disabled
- [ ] 4.13 Verify STOP directive terminates trajectory correctly

### Phase 5: Documentation & Cleanup (Priority 2)

- [ ] 5.1 Update `docs/run_protocol.md` with ASO mode instructions
- [ ] 5.2 Add example config YAML with ASO settings
- [ ] 5.3 Remove duplicate EXPLOIT/HYBRID logic in legacy `produce_candidates()`
- [ ] 5.4 Archive `UNIFIED_PLAN.md` and `ASO_V3_PLAN.md` (mark as superseded)

---

## Part 6: Success Metrics

| Metric | Baseline (Legacy) | Target (ASO V4) | Measurement |
|--------|-------------------|-----------------|-------------|
| LLM calls/cycle | 0 (no supervision) | <10 | Telemetry |
| Heuristic decisions/cycle | 0 | >40 | Telemetry |
| Wall-clock/cycle | ~5 min | <3 min | Timer |
| Feasibility rate | ~30% | >50% | WorldModel |
| Stagnation recovery | 0% | >50% | RESTART success rate |

---

## Part 7: Migration Guide

### From Legacy Mode

1. Set `aso.enabled = True` in config or use `--aso` flag
2. Legacy `produce_candidates()` still works unchanged
3. ASO telemetry written to `reports/telemetry/aso_exp{id}.jsonl`

### From ASO V3 Plan

1. `analyze_optimizer_diagnostics()` → `supervise()` + `HeuristicSupervisor`
2. Multi-trajectory → Single trajectory (simplification; extensible later)
3. SQL telemetry → JSONL telemetry (simpler, grep-friendly)
4. `alm_settings` → `alm` config section with proper dataclass

---

## Appendix A: Example Telemetry Event

```json
{
  "timestamp": "2025-11-29T14:30:22.456Z",
  "experiment_id": 42,
  "cycle": 5,
  "trajectory_id": 0,
  "step": 7,
  "status": "STAGNATION",
  "objective": 6.82,
  "max_violation": 0.12,
  "directive_action": "ADJUST",
  "directive_source": "heuristic",
  "directive_reasoning": "Constraint 'qi' worsening, boosting weight to 200.0",
  "evals_used": 70,
  "wall_time_ms": 1523.4,
  "llm_called": false
}
```

---

## Appendix B: Constraint Names by Problem

| Problem | Constraints |
|---------|-------------|
| P1 | aspect_ratio, average_triangularity, edge_rotational_transform |
| P2 | aspect_ratio, edge_rotational_transform, edge_magnetic_mirror_ratio, max_elongation, qi |
| P3 | edge_rotational_transform, edge_magnetic_mirror_ratio, vacuum_well, flux_compression, qi |

---

## Appendix C: Key File Locations

| Component | File | Lines to Modify |
|-----------|------|-----------------|
| Config | `ai_scientist/config.py` | Add ALMConfig, ASOConfig |
| Planner | `ai_scientist/planner.py` | Add dataclasses, HeuristicSupervisor, supervise() |
| Coordinator | `ai_scientist/coordinator.py` | Add TrajectoryState, produce_candidates_aso() |
| Workers | `ai_scientist/workers.py` | Update OptimizationWorker.run() |
| Differentiable | `ai_scientist/optim/differentiable.py` | Add return_state parameter |
| Runner | `ai_scientist/runner.py` | Add --aso flag, branch in _run_cycle() |
