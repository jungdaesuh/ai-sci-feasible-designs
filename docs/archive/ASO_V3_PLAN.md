> **DEPRECATED:** This document is superseded by `ASO_V4_IMPLEMENTATION_GUIDE.md`

# Agent-Supervised Optimization (ASO) V3 Plan

**Status:** Proposed
**Author:** Claude (Opus 4.5)
**Date:** 2025-11-29
**Supersedes:** `UNIFIED_PLAN.md`
**Prerequisites:** V2 Roadmap (Phases 1-4 complete)

---

## Executive Summary

This plan addresses the gaps between `UNIFIED_PLAN.md` and the current implementation, while fixing critical design issues in the original plan. The key innovations are:

1. **Tiered Supervision** - LLM calls only on significant events, not every step
2. **Heuristic Fallbacks** - Rule-based directives when LLM is unavailable/slow
3. **Multi-Trajectory Optimization** - Parallel optimization from multiple seeds
4. **Structured Telemetry** - Full observability of the ASO loop
5. **Convergence Guards** - Automatic termination on stagnation/success

---

## Part 1: Architecture Overview

### 1.1 The Problem with UNIFIED_PLAN.md

The original plan proposed calling the LLM on every optimization step:

```
while budget_remaining:
    run_optimization_chunk()
    diagnostics = generate_diagnostics()
    directive = LLM(diagnostics)  # <-- Called every iteration!
    apply_directive()
```

**Issues:**
- 10-50 LLM calls per cycle (1-5s each) = 50-250s of latency per cycle
- API rate limits become bottleneck
- Cost scales linearly with optimization steps
- No fallback when LLM fails

### 1.2 The ASO V3 Solution: Tiered Supervision

```
┌─────────────────────────────────────────────────────────────────┐
│                     ASO V3 Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Planner   │    │ Coordinator │    │   Workers   │         │
│  │ (Strategic) │◄──►│ (Tactical)  │◄──►│(Operational)│         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│        │                   │                   │                │
│        │                   │                   │                │
│        ▼                   ▼                   ▼                │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Supervision Tiers                       │       │
│  ├─────────────────────────────────────────────────────┤       │
│  │ Tier 1: Heuristic (every step)     - 0ms latency   │       │
│  │ Tier 2: LLM (on events)            - 1-5s latency  │       │
│  │ Tier 3: Human (on failures)        - async         │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Most optimization steps don't need LLM guidance. We only need the LLM when:
- Stagnation is detected
- Feasibility is achieved (to decide next objective)
- Constraint violations are increasing
- A trajectory should be abandoned

---

## Part 2: New Data Structures

### 2.1 Configuration Extensions

Add to `config.py`:

```python
@dataclass(frozen=True)
class ASOConfig:
    """Configuration for Agent-Supervised Optimization."""

    # Supervision frequency
    supervision_mode: str = "event_triggered"  # "every_step", "periodic", "event_triggered"
    supervision_interval: int = 5              # Steps between LLM calls (if periodic)

    # Convergence thresholds
    feasibility_threshold: float = 1e-3
    stagnation_objective_threshold: float = 1e-5
    stagnation_violation_threshold: float = 0.05
    max_stagnation_steps: int = 5

    # Trend detection
    violation_increase_threshold: float = 0.05  # 5% increase = "increasing"
    violation_decrease_threshold: float = 0.05  # 5% decrease = "decreasing"
    min_violation_for_trend: float = 1e-3

    # Multi-trajectory
    n_trajectories: int = 3
    trajectory_budget_split: str = "equal"  # "equal", "adaptive"

    # Safety limits
    max_constraint_weight: float = 1000.0
    max_penalty_parameter: float = 1e6

    # Fallback behavior
    llm_timeout_seconds: float = 10.0
    llm_max_retries: int = 2
    use_heuristic_fallback: bool = True


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Thresholds for diagnostic generation (extracted for testability)."""
    feasibility: float = 1e-3
    stagnation_objective: float = 1e-5
    stagnation_violation: float = 0.05
    trend_increase: float = 0.05
    trend_decrease: float = 0.05
    min_violation: float = 1e-3
```

### 2.2 Core Data Structures

Add to `planner.py`:

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Mapping, Any, List

class DirectiveAction(Enum):
    """Enumerated actions for type safety."""
    CONTINUE = "CONTINUE"
    ADJUST = "ADJUST"
    STOP = "STOP"
    RESTART = "RESTART"  # New: restart with different seed

class DirectiveSource(Enum):
    """Source of the directive for debugging."""
    LLM = "llm"
    HEURISTIC = "heuristic"
    FALLBACK = "fallback"
    CONVERGENCE = "convergence"

@dataclass
class OptimizationDirective:
    """Structured directive from Planner to Coordinator."""
    action: DirectiveAction
    config_overrides: Optional[Mapping[str, Any]] = None
    suggested_params: Optional[Mapping[str, Any]] = None
    reasoning: str = ""
    confidence: float = 1.0
    source: DirectiveSource = DirectiveSource.LLM

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "config_overrides": self.config_overrides,
            "suggested_params": self.suggested_params,
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

    # Metadata for supervision decisions
    steps_since_improvement: int = 0
    steps_in_feasible_region: int = 0

    def requires_llm_supervision(self, config: "ASOConfig") -> bool:
        """Determine if this diagnostic warrants an LLM call."""
        if config.supervision_mode == "every_step":
            return True
        if config.supervision_mode == "periodic":
            return self.step % config.supervision_interval == 0

        # Event-triggered: only call LLM on significant events
        significant_events = [
            self.status == "STAGNATION",
            self.status == "FEASIBLE_FOUND",
            self.status == "DIVERGING",
            any(c.trend == "increasing_violation" for c in self.constraint_diagnostics),
            self.steps_since_improvement >= config.max_stagnation_steps,
        ]
        return any(significant_events)
```

### 2.3 Telemetry Schema

Add to `memory.py` (new table):

```python
@dataclass
class ASOTelemetryEvent:
    """Structured telemetry for ASO loop analysis."""
    timestamp: str
    experiment_id: int
    cycle: int
    trajectory_id: int
    step: int
    diagnostics: OptimizerDiagnostics
    directive: OptimizationDirective
    evals_used: int
    wall_time_ms: float
    llm_call_made: bool

# SQL Schema addition:
"""
CREATE TABLE IF NOT EXISTS aso_telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    experiment_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    trajectory_id INTEGER NOT NULL,
    step INTEGER NOT NULL,
    status TEXT NOT NULL,
    objective REAL,
    max_violation REAL,
    directive_action TEXT NOT NULL,
    directive_source TEXT NOT NULL,
    evals_used INTEGER NOT NULL,
    wall_time_ms REAL,
    llm_call_made INTEGER NOT NULL,
    diagnostics_json TEXT,
    directive_json TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
"""
```

---

## Part 3: Heuristic Supervisor

The key innovation: a rule-based fallback that handles 80% of cases without LLM.

### 3.1 Implementation

Add to `planner.py`:

```python
class HeuristicSupervisor:
    """
    Rule-based optimization supervisor.
    Handles common scenarios without LLM latency.
    """

    def __init__(self, config: ASOConfig):
        self.config = config

    def analyze(self, diagnostics: OptimizerDiagnostics) -> OptimizationDirective:
        """
        Generate a directive using heuristic rules.

        Decision tree:
        1. FEASIBLE_FOUND + low objective delta -> STOP (converged)
        2. FEASIBLE_FOUND + improving -> CONTINUE
        3. STAGNATION + high violation -> ADJUST (increase penalties)
        4. STAGNATION + low violation -> RESTART (try new seed)
        5. DIVERGING -> STOP (give up on this trajectory)
        6. Increasing violation -> ADJUST (boost that constraint)
        7. Otherwise -> CONTINUE
        """

        # Case 1 & 2: Feasible region reached
        if diagnostics.status == "FEASIBLE_FOUND":
            if abs(diagnostics.objective_delta) < self.config.stagnation_objective_threshold:
                return OptimizationDirective(
                    action=DirectiveAction.STOP,
                    reasoning="Converged: feasible with stable objective",
                    source=DirectiveSource.HEURISTIC,
                )
            return OptimizationDirective(
                action=DirectiveAction.CONTINUE,
                reasoning="Feasible and still improving",
                source=DirectiveSource.HEURISTIC,
            )

        # Case 3 & 4: Stagnation detected
        if diagnostics.status == "STAGNATION":
            if diagnostics.max_violation > self.config.stagnation_violation_threshold:
                # High violation stagnation: increase penalty parameters
                return OptimizationDirective(
                    action=DirectiveAction.ADJUST,
                    config_overrides={
                        "alm_settings": {"penalty_parameters_increase_factor": 2.0}
                    },
                    reasoning=f"Stagnation with high violation ({diagnostics.max_violation:.4f}), increasing penalties",
                    source=DirectiveSource.HEURISTIC,
                )
            else:
                # Low violation stagnation: try a different trajectory
                return OptimizationDirective(
                    action=DirectiveAction.RESTART,
                    reasoning="Stagnation near feasibility, trying new seed",
                    source=DirectiveSource.HEURISTIC,
                )

        # Case 5: Diverging (constraints getting worse consistently)
        if diagnostics.status == "DIVERGING":
            return OptimizationDirective(
                action=DirectiveAction.STOP,
                reasoning="Trajectory diverging, abandoning",
                source=DirectiveSource.HEURISTIC,
            )

        # Case 6: Specific constraint struggling
        struggling_constraints = [
            c for c in diagnostics.constraint_diagnostics
            if c.trend == "increasing_violation"
        ]
        if struggling_constraints:
            # Boost the worst offender
            worst = max(struggling_constraints, key=lambda c: c.violation)
            weight_boost = {worst.name: min(worst.penalty * 2, self.config.max_constraint_weight)}
            return OptimizationDirective(
                action=DirectiveAction.ADJUST,
                config_overrides={"constraint_weights": weight_boost},
                reasoning=f"Constraint '{worst.name}' worsening, boosting weight",
                source=DirectiveSource.HEURISTIC,
            )

        # Case 7: Default - continue optimization
        return OptimizationDirective(
            action=DirectiveAction.CONTINUE,
            reasoning="Normal progress",
            source=DirectiveSource.HEURISTIC,
        )
```

### 3.2 Integration with LLM Supervisor

```python
class PlanningAgent:
    """Extended with tiered supervision."""

    def __init__(self, ...):
        # ... existing init ...
        self.heuristic = HeuristicSupervisor(self.config.aso)
        self._llm_failures = 0

    def supervise(
        self,
        diagnostics: OptimizerDiagnostics,
        cycle: int
    ) -> OptimizationDirective:
        """
        Tiered supervision: heuristic first, LLM on demand.
        """
        config = self.config.aso

        # Tier 1: Check if LLM supervision is needed
        if not diagnostics.requires_llm_supervision(config):
            return self.heuristic.analyze(diagnostics)

        # Tier 2: Try LLM supervision
        if config.use_heuristic_fallback:
            try:
                directive = self._llm_supervise(diagnostics, cycle)
                self._llm_failures = 0
                return directive
            except Exception as e:
                self._llm_failures += 1
                print(f"[Planner] LLM supervision failed ({self._llm_failures}x): {e}")
                return self.heuristic.analyze(diagnostics)
        else:
            return self._llm_supervise(diagnostics, cycle)

    def _llm_supervise(
        self,
        diagnostics: OptimizerDiagnostics,
        cycle: int
    ) -> OptimizationDirective:
        """
        LLM-based supervision with retry logic.
        Refactored from analyze_optimizer_diagnostics.
        """
        config = self.config.aso

        # Build context
        rag_context = []
        if diagnostics.status == "STAGNATION":
            rag_context = self.retrieve_rag(
                "Strategies for escaping local minima in ALM stellarator optimization",
                k=2,
            )

        # Prepare prompt (control schema versioned)
        control_schema = self._get_control_schema_v1()
        system_prompt = self._build_supervision_prompt(cycle, control_schema, rag_context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Diagnostics: {diagnostics.to_json()}"}
        ]

        # Invoke with retry
        for attempt in range(config.llm_max_retries):
            try:
                response = self._invoke_llm(messages, timeout=config.llm_timeout_seconds)
                directive = self._parse_directive_response(response)
                directive.source = DirectiveSource.LLM
                return directive
            except json.JSONDecodeError as e:
                if attempt < config.llm_max_retries - 1:
                    messages.append({
                        "role": "user",
                        "content": f"Invalid JSON: {e}. Please output valid JSON."
                    })
                    continue
                raise

        raise RuntimeError("LLM supervision failed after retries")

    def _get_control_schema_v1(self) -> dict:
        """Versioned control schema for LLM."""
        return {
            "version": "1.0",
            "actions": ["CONTINUE", "ADJUST", "STOP", "RESTART"],
            "config_overrides": {
                "constraint_weights": {"<name>": "<float>"},
                "alm_settings": {
                    "penalty_parameters_increase_factor": "<float>",
                    "bounds_reduction_factor": "<float>",
                },
                "proposal_mix": {"exploration_ratio": "<float>"},
            },
            "output_format": {
                "action": "CONTINUE | ADJUST | STOP | RESTART",
                "config_overrides": "optional dict",
                "reasoning": "brief explanation",
            }
        }
```

---

## Part 4: Multi-Trajectory Coordinator

### 4.1 Trajectory Manager

Add to `coordinator.py`:

```python
@dataclass
class TrajectoryState:
    """State for a single optimization trajectory."""
    id: int
    seed: Mapping[str, Any]
    alm_state: Optional[AugmentedLagrangianState] = None
    history: List[AugmentedLagrangianState] = field(default_factory=list)
    evals_used: int = 0
    steps: int = 0
    status: str = "active"  # "active", "converged", "stagnated", "abandoned"
    best_objective: float = float('inf')
    best_violation: float = float('inf')
    stagnation_count: int = 0

class Coordinator:
    """
    Manages multi-trajectory Agent-Supervised Optimization.
    """

    def __init__(
        self,
        cfg: ai_config.ExperimentConfig,
        world_model: memory.WorldModel,
        planner: PlanningAgent,
        surrogate: Optional[NeuralOperatorSurrogate] = None,
        generative_model: Optional[GenerativeDesignModel] = None
    ):
        self.cfg = cfg
        self.aso_config = cfg.aso  # New config section
        self.world_model = world_model
        self.planner = planner
        self.surrogate = surrogate
        self.generative_model = generative_model

        # Workers
        self.opt_worker = OptimizationWorker(cfg, surrogate)
        self.explore_worker = ExplorationWorker(cfg, generative_model)
        self.geo_worker = GeometerWorker(cfg)

        # Constraint names for diagnostics
        self.constraint_names = self._get_constraint_names(cfg.problem)

        # Telemetry
        self.telemetry: List[ASOTelemetryEvent] = []

    def _get_constraint_names(self, problem_type: str) -> List[str]:
        """Map constraint indices to semantic names."""
        p_key = (problem_type or "").lower()
        if p_key.startswith("p1"):
            return ["aspect_ratio", "average_triangularity", "edge_rotational_transform"]
        elif p_key.startswith("p2"):
            return ["aspect_ratio", "edge_rotational_transform", "edge_magnetic_mirror_ratio", "max_elongation", "qi_log10"]
        return ["edge_rotational_transform", "edge_magnetic_mirror_ratio", "vacuum_well", "flux_compression", "qi_log10"]

    def produce_candidates(
        self,
        cycle: int,
        experiment_id: int,
        n_candidates: int,
        template: ai_config.BoundaryTemplateConfig,
        initial_seeds: Optional[List[Dict[str, Any]]] = None,
        initial_config: Optional[ai_config.ExperimentConfig] = None
    ) -> List[Dict[str, Any]]:
        """
        Multi-trajectory ASO with tiered supervision.
        """
        config = initial_config or self.cfg
        aso = self.aso_config

        # 1. Initialize trajectories
        seeds = self._prepare_seeds(initial_seeds, cycle, aso.n_trajectories)
        if not seeds:
            print("[Coordinator] No valid seeds. Returning empty.")
            return []

        trajectories = [
            TrajectoryState(id=i, seed=seed)
            for i, seed in enumerate(seeds[:aso.n_trajectories])
        ]

        # Budget allocation
        budget_per_trajectory = n_candidates // len(trajectories)
        inner_budget = 10  # Evals between supervision checks

        all_candidates = []

        # 2. Run trajectories (can be parallelized in future)
        for traj in trajectories:
            print(f"\n[Coordinator] Starting trajectory {traj.id} with budget {budget_per_trajectory}")

            candidates = self._run_trajectory(
                traj,
                budget_per_trajectory,
                inner_budget,
                cycle,
                experiment_id,
                config
            )
            all_candidates.extend(candidates)

        # 3. Persist telemetry
        self._persist_telemetry(experiment_id)

        return all_candidates

    def _prepare_seeds(
        self,
        initial_seeds: Optional[List[Dict]],
        cycle: int,
        n_trajectories: int
    ) -> List[Dict]:
        """Prepare and validate seeds for trajectories."""
        if initial_seeds:
            seeds = initial_seeds
        else:
            explore_ctx = {"n_samples": n_trajectories * 2, "cycle": cycle}
            seeds = self.explore_worker.run(explore_ctx).get("candidates", [])

        if not seeds:
            return []

        # Validate with GeometerWorker
        geo_ctx = {"candidates": seeds}
        valid_seeds = self.geo_worker.run(geo_ctx).get("candidates", [])

        return valid_seeds

    def _run_trajectory(
        self,
        traj: TrajectoryState,
        budget: int,
        inner_budget: int,
        cycle: int,
        experiment_id: int,
        config: ai_config.ExperimentConfig
    ) -> List[Dict[str, Any]]:
        """
        Run a single optimization trajectory with tiered supervision.
        """
        aso = self.aso_config
        candidates = []

        opt_ctx = {
            "initial_guesses": [traj.seed],
            "budget": inner_budget,
            "alm_settings_overrides": {},
            "continue_from_state": None,
        }

        while traj.evals_used < budget and traj.status == "active":
            traj.steps += 1
            step_start = time.perf_counter()

            # 1. Execute optimization chunk
            res = self.opt_worker.run(opt_ctx)
            traj.evals_used += res.get("evals_used", inner_budget)

            alm_state = res.get("final_alm_state")
            if not alm_state:
                print(f"[Coordinator] Trajectory {traj.id}: Worker failed, stopping.")
                traj.status = "abandoned"
                break

            # Collect candidates from this chunk
            chunk_candidates = res.get("candidates", [])
            candidates.extend(chunk_candidates)

            # 2. Generate diagnostics
            diagnostics = self._generate_diagnostics(alm_state, traj)
            traj.history.append(alm_state)

            # Update trajectory best values
            if diagnostics.max_violation < traj.best_violation:
                traj.best_violation = diagnostics.max_violation
                traj.stagnation_count = 0
            elif diagnostics.objective < traj.best_objective and diagnostics.max_violation <= traj.best_violation:
                traj.best_objective = diagnostics.objective
                traj.stagnation_count = 0
            else:
                traj.stagnation_count += 1

            # 3. Get directive (tiered supervision)
            llm_called = diagnostics.requires_llm_supervision(aso)
            directive = self.planner.supervise(diagnostics, cycle)

            # 4. Log telemetry
            wall_time_ms = (time.perf_counter() - step_start) * 1000
            self._log_telemetry(
                experiment_id, cycle, traj.id, traj.steps,
                diagnostics, directive, traj.evals_used, wall_time_ms, llm_called
            )

            # 5. Apply directive
            if directive.action == DirectiveAction.STOP:
                traj.status = "converged" if diagnostics.status == "FEASIBLE_FOUND" else "stagnated"
                print(f"[Coordinator] Trajectory {traj.id}: STOP - {directive.reasoning}")
                break

            if directive.action == DirectiveAction.RESTART:
                # Try to get a new seed and restart
                new_seeds = self._prepare_seeds(None, cycle, 1)
                if new_seeds:
                    traj.seed = new_seeds[0]
                    traj.history = []
                    traj.stagnation_count = 0
                    opt_ctx["initial_guesses"] = [traj.seed]
                    opt_ctx["continue_from_state"] = None
                    print(f"[Coordinator] Trajectory {traj.id}: RESTART with new seed")
                else:
                    traj.status = "abandoned"
                    print(f"[Coordinator] Trajectory {traj.id}: RESTART failed, no seeds")
                    break
                continue

            # Apply config/ALM overrides
            config, worker_overrides = self._apply_directive(directive, config)

            # Prepare next iteration
            opt_ctx["continue_from_state"] = alm_state
            opt_ctx["budget"] = min(inner_budget, budget - traj.evals_used)
            if "alm_settings" in worker_overrides:
                opt_ctx["alm_settings_overrides"] = worker_overrides["alm_settings"]

            # Auto-stop on excessive stagnation
            if traj.stagnation_count >= aso.max_stagnation_steps:
                traj.status = "stagnated"
                print(f"[Coordinator] Trajectory {traj.id}: Auto-STOP (stagnation limit)")
                break

        print(f"[Coordinator] Trajectory {traj.id} finished: {traj.status}, "
              f"{traj.steps} steps, {traj.evals_used} evals, "
              f"{len(candidates)} candidates")

        return candidates

    def _generate_diagnostics(
        self,
        alm_state: AugmentedLagrangianState,
        traj: TrajectoryState
    ) -> OptimizerDiagnostics:
        """
        Translate ALM state into semantic diagnostics.
        """
        aso = self.aso_config
        prev_state = traj.history[-1] if traj.history else None

        # Constraint analysis
        constraint_diagnostics = []
        struggling = False
        diverging_count = 0

        for i, name in enumerate(self.constraint_names):
            if i >= len(alm_state.constraints):
                continue

            violation = float(jnp.maximum(0., alm_state.constraints[i]))
            penalty = float(alm_state.penalty_parameters[i])
            trend = "stable"
            delta = 0.0

            if prev_state and i < len(prev_state.constraints):
                prev_violation = float(jnp.maximum(0., prev_state.constraints[i]))
                delta = violation - prev_violation

                if violation > prev_violation * (1 + aso.violation_increase_threshold) and violation > aso.min_violation_for_trend:
                    trend = "increasing_violation"
                    struggling = True
                    diverging_count += 1
                elif violation < prev_violation * (1 - aso.violation_decrease_threshold):
                    trend = "decreasing_violation"

            constraint_diagnostics.append(ConstraintDiagnostic(
                name=name,
                violation=violation,
                penalty=penalty,
                trend=trend,
                delta=delta,
            ))

        # Objective analysis
        objective = float(alm_state.objective)
        objective_delta = 0.0
        if prev_state:
            objective_delta = objective - float(prev_state.objective)

        max_violation = float(jnp.max(jnp.maximum(0., alm_state.constraints)))

        # Status determination
        narrative = []
        if max_violation < aso.feasibility_threshold:
            status = "FEASIBLE_FOUND"
            narrative.append("Feasible region reached.")
        elif diverging_count >= len(self.constraint_names) // 2:
            status = "DIVERGING"
            narrative.append("Multiple constraints diverging.")
        elif prev_state and abs(objective_delta) < aso.stagnation_objective_threshold and max_violation > aso.stagnation_violation_threshold:
            status = "STAGNATION"
            narrative.append("Stagnation: minimal progress with constraint violations.")
        else:
            status = "IN_PROGRESS"
            if struggling:
                narrative.append("Progressing but some constraints worsening.")
            else:
                narrative.append("Normal progress.")

        return OptimizerDiagnostics(
            step=traj.steps,
            trajectory_id=traj.id,
            objective=objective,
            objective_delta=objective_delta,
            max_violation=max_violation,
            status=status,
            constraint_diagnostics=constraint_diagnostics,
            optimizer_health={
                "bounds_norm": float(jnp.linalg.norm(alm_state.bounds)),
                "stagnation_count": traj.stagnation_count,
            },
            narrative=narrative,
            steps_since_improvement=traj.stagnation_count,
        )

    def _apply_directive(
        self,
        directive: OptimizationDirective,
        config: ai_config.ExperimentConfig
    ) -> Tuple[ai_config.ExperimentConfig, Dict[str, Any]]:
        """Apply directive to config with safety guards."""
        if not directive.config_overrides:
            return config, {}

        aso = self.aso_config
        new_cfg = config
        worker_overrides = {}
        overrides = directive.config_overrides

        try:
            # Constraint weights (with clamping)
            if "constraint_weights" in overrides:
                clamped = {
                    k: min(v, aso.max_constraint_weight)
                    for k, v in overrides["constraint_weights"].items()
                }
                new_weights = replace(new_cfg.constraint_weights, **clamped)
                new_cfg = replace(new_cfg, constraint_weights=new_weights)

            # ALM settings
            if "alm_settings" in overrides:
                worker_overrides["alm_settings"] = overrides["alm_settings"]

            # Proposal mix
            if "proposal_mix" in overrides:
                new_mix = replace(new_cfg.proposal_mix, **overrides["proposal_mix"])
                new_cfg = replace(new_cfg, proposal_mix=new_mix)

        except Exception as e:
            print(f"[Coordinator] Failed to apply directive: {e}")

        return new_cfg, worker_overrides

    def _log_telemetry(
        self,
        experiment_id: int,
        cycle: int,
        trajectory_id: int,
        step: int,
        diagnostics: OptimizerDiagnostics,
        directive: OptimizationDirective,
        evals_used: int,
        wall_time_ms: float,
        llm_called: bool
    ):
        """Record telemetry event."""
        event = ASOTelemetryEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            experiment_id=experiment_id,
            cycle=cycle,
            trajectory_id=trajectory_id,
            step=step,
            diagnostics=diagnostics,
            directive=directive,
            evals_used=evals_used,
            wall_time_ms=wall_time_ms,
            llm_call_made=llm_called,
        )
        self.telemetry.append(event)

    def _persist_telemetry(self, experiment_id: int):
        """Persist telemetry to world model."""
        for event in self.telemetry:
            self.world_model.log_aso_event(experiment_id, event)
        self.telemetry = []
```

---

## Part 5: Worker Contract

### 5.1 Updated OptimizationWorker Interface

Update `workers.py`:

```python
@dataclass
class OptimizationResult:
    """Structured result from OptimizationWorker."""
    candidates: List[Dict[str, Any]]
    final_alm_state: Optional[Any]  # AugmentedLagrangianState
    evals_used: int
    converged: bool
    best_objective: float
    best_violation: float

class OptimizationWorker(Worker):
    """
    Worker for gradient-based optimization.
    Supports iterative execution with state continuation.
    """

    def __init__(self, cfg: ai_config.ExperimentConfig, surrogate: NeuralOperatorSurrogate):
        self.cfg = cfg
        self.surrogate = surrogate

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute optimization chunk.

        Args:
            context:
                initial_guesses: List of starting boundary params
                budget: Max evaluations for this chunk
                continue_from_state: Optional ALM state to resume from
                alm_settings_overrides: Dict of ALM hyperparameters

        Returns:
            Dict with keys:
                candidates: List of boundary param dicts
                final_alm_state: ALM state for continuation
                evals_used: Actual evaluations consumed
                status: "optimized", "skipped", "failed"
        """
        initial_guesses = context.get("initial_guesses", [])
        budget = context.get("budget", 10)
        continue_from = context.get("continue_from_state")
        alm_overrides = context.get("alm_settings_overrides", {})

        if not initial_guesses and not continue_from:
            return {
                "candidates": [],
                "final_alm_state": None,
                "evals_used": 0,
                "status": "skipped",
            }

        if not self.surrogate or not self.surrogate._trained:
            return {
                "candidates": initial_guesses,
                "final_alm_state": None,
                "evals_used": 0,
                "status": "skipped",
            }

        try:
            # Run differentiable optimization
            optimized, final_state, evals = differentiable.gradient_descent_on_inputs(
                initial_guesses if not continue_from else [],
                self.surrogate,
                self.cfg,
                budget=budget,
                continue_from_state=continue_from,
                alm_settings=alm_overrides,
            )

            return {
                "candidates": optimized,
                "final_alm_state": final_state,
                "evals_used": evals,
                "status": "optimized",
            }
        except Exception as e:
            print(f"[OptimizationWorker] Failed: {e}")
            return {
                "candidates": initial_guesses,
                "final_alm_state": continue_from,
                "evals_used": 0,
                "status": "failed",
            }
```

---

## Part 6: Runner Integration

### 6.1 Updated Runner Flow

Update `runner.py`:

```python
def _run_cycle(
    cfg: ai_config.ExperimentConfig,
    cycle_index: int,
    world_model: memory.WorldModel,
    experiment_id: int,
    # ... other params ...
    planner: ai_planner.PlanningAgent,  # Injected
    coordinator: Coordinator,            # Injected
    surrogate_model: BaseSurrogate,
    generative_model: GenerativeDesignModel | None = None,
    *,
    runtime: RunnerCLIConfig | None = None,
    budget_controller: BudgetController,
) -> tuple[Path | None, dict[str, Any] | None, tools.P3Summary | None]:
    """
    Run a single optimization cycle using ASO V3.
    """
    cycle_number = cycle_index + 1

    # 1. High-level planning (unchanged)
    planning_outcome = planner.plan_cycle(
        cfg=cfg,
        cycle_index=cycle_index,
        stage_history=stage_history,
        last_summary=last_p3_summary,
        experiment_id=experiment_id,
    )

    # Apply initial config overrides
    active_cfg = cfg
    if planning_outcome.config_overrides:
        active_cfg = _apply_config_overrides(cfg, planning_outcome.config_overrides)

    # Budget
    budget_snapshot = budget_controller.snapshot()

    # 2. ASO Loop (new unified path)
    initial_seeds = []
    if planning_outcome.suggested_params:
        initial_seeds = [{"params": planning_outcome.suggested_params}]

    print(f"[runner][cycle={cycle_number}] Starting ASO V3 Loop")

    candidates = coordinator.produce_candidates(
        cycle=cycle_number,
        experiment_id=experiment_id,
        n_candidates=budget_snapshot.screen_evals_per_cycle,
        template=active_cfg.boundary_template,
        initial_seeds=initial_seeds,
        initial_config=active_cfg,
    )

    # 3. Post-processing (unchanged)
    # ... evaluation, logging, promotion ...
```

### 6.2 Initialization

```python
def initialize_architecture(
    cfg: ai_config.ExperimentConfig,
    world_model: memory.WorldModel
) -> Tuple[ai_planner.PlanningAgent, Coordinator]:
    """Initialize ASO V3 architecture."""

    surrogate = _create_surrogate(cfg)
    generative = _create_generative_model(cfg)

    # Planner with heuristic supervisor
    planner = ai_planner.PlanningAgent(
        rag_index=rag.DEFAULT_INDEX_PATH,
        world_model=world_model,
        config=ai_config.load_model_config(),
    )

    # Coordinator with planner injection
    coordinator = Coordinator(
        cfg=cfg,
        world_model=world_model,
        planner=planner,  # Critical: inject planner!
        surrogate=surrogate if isinstance(surrogate, NeuralOperatorSurrogate) else None,
        generative_model=generative,
    )

    return planner, coordinator
```

---

## Part 7: Implementation Roadmap

### Phase 1: Foundation (Week 1)

| Task | File | Priority |
|------|------|----------|
| Add `ASOConfig` dataclass | `config.py` | P0 |
| Add `OptimizationDirective` with enums | `planner.py` | P0 |
| Add `OptimizerDiagnostics` dataclass | `planner.py` | P0 |
| Add `HeuristicSupervisor` class | `planner.py` | P0 |
| Add `supervise()` method to PlanningAgent | `planner.py` | P0 |

### Phase 2: Coordinator Upgrade (Week 2)

| Task | File | Priority |
|------|------|----------|
| Add `TrajectoryState` dataclass | `coordinator.py` | P0 |
| Refactor `produce_candidates()` for multi-trajectory | `coordinator.py` | P0 |
| Implement `_generate_diagnostics()` | `coordinator.py` | P0 |
| Implement `_apply_directive()` with guards | `coordinator.py` | P0 |
| Add telemetry logging | `coordinator.py` | P1 |

### Phase 3: Worker & Runner (Week 3)

| Task | File | Priority |
|------|------|----------|
| Update `OptimizationWorker` contract | `workers.py` | P0 |
| Update `gradient_descent_on_inputs()` for continuation | `optim/differentiable.py` | P0 |
| Update runner to inject Planner into Coordinator | `runner.py` | P0 |
| Add ASO telemetry table | `memory.py` | P1 |

### Phase 4: Testing & Tuning (Week 4)

| Task | File | Priority |
|------|------|----------|
| Unit tests for `HeuristicSupervisor` | `tests/test_planner.py` | P0 |
| Integration test for multi-trajectory ASO | `tests/test_coordinator.py` | P0 |
| Benchmark: LLM calls/cycle with tiered supervision | - | P1 |
| Tune `ASOConfig` defaults based on experiments | `config.py` | P1 |

---

## Part 8: Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| LLM calls per cycle | ~50 | <10 | Telemetry |
| Wall-clock time per cycle | ~5 min | <2 min | Timer |
| Convergence rate (% feasible) | ~30% | >50% | WorldModel |
| Stagnation recovery rate | 0% | >60% | Telemetry (RESTART success) |

---

## Part 9: Migration Notes

### From UNIFIED_PLAN.md

1. **`analyze_optimizer_diagnostics()`** → Split into `supervise()` + `_llm_supervise()` + `HeuristicSupervisor.analyze()`
2. **`generate_diagnostics()`** → Now `_generate_diagnostics()` with `OptimizerDiagnostics` return type
3. **`apply_directive()`** → Now `_apply_directive()` with safety guards
4. **Single-seed loop** → Multi-trajectory with `TrajectoryState`
5. **Hardcoded thresholds** → Configurable via `ASOConfig`

### Backward Compatibility

The existing `decide_strategy()` method in Coordinator is preserved for non-ASO backends. The `optimizer_backend` config flag routes to the appropriate code path.

---

## Appendix A: Example Telemetry Output

```json
{
  "timestamp": "2025-11-29T10:15:32.123Z",
  "experiment_id": 42,
  "cycle": 5,
  "trajectory_id": 0,
  "step": 7,
  "diagnostics": {
    "status": "STAGNATION",
    "objective": 6.82,
    "max_violation": 0.12,
    "constraint_diagnostics": [
      {"name": "qi_log10", "violation": 0.12, "trend": "increasing_violation"}
    ]
  },
  "directive": {
    "action": "ADJUST",
    "config_overrides": {"constraint_weights": {"qi_log10": 200.0}},
    "reasoning": "Constraint 'qi_log10' worsening, boosting weight",
    "source": "heuristic"
  },
  "evals_used": 70,
  "wall_time_ms": 1523.4,
  "llm_call_made": false
}
```

---

## Appendix B: Control Schema v1.0

```json
{
  "version": "1.0",
  "actions": {
    "CONTINUE": "Proceed with current settings",
    "ADJUST": "Modify optimization parameters",
    "STOP": "Terminate this trajectory (converged or hopeless)",
    "RESTART": "Abandon trajectory, start with new seed"
  },
  "config_overrides": {
    "constraint_weights": {
      "description": "Increase weight of struggling constraints (Soft ALM)",
      "example": {"vacuum_well": 100.0, "qi_log10": 50.0}
    },
    "alm_settings": {
      "description": "Adjust ALM hyperparameters",
      "penalty_parameters_increase_factor": "Multiply penalties (default 10)",
      "bounds_reduction_factor": "Shrink search space (default 0.95)"
    },
    "proposal_mix": {
      "description": "Adjust exploration/exploitation balance",
      "exploration_ratio": "0.0-1.0"
    }
  }
}
```
