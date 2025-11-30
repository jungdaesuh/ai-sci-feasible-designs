# Agent-Supervised Optimization (ASO) V4 Implementation Guide

**Status:** Ready for Implementation
**Date:** 2025-11-30 (Revised)
**Supersedes:** `UNIFIED_PLAN.md`, `UNIFIED_RAW.md`, `ASO_V3_PLAN.md`, `AI_SCIENTIST_ASO_MIGRATION_PLAN.md`
**Prerequisites:** V2 Roadmap Complete (Phases 1-4)

---

## Executive Summary

This document provides a **verified implementation guide** for the Agent-Supervised Optimization (ASO) loop. Unlike previous planning documents, this guide is grounded in a thorough codebase audit of both `ai_scientist/` and `constellaration/`.

### Key Findings from Codebase Audit

1. **Complete ALM Infrastructure Exists in constellaration** - The `augmented_lagrangian_runner.py` contains a full iterative ALM loop with Nevergrad oracle. We need to **wrap it**, not rebuild it.
2. **AugmentedLagrangianState is Pydantic** - Directly importable, JSON-serializable, contains `x`, `multipliers`, `penalty_parameters`, `objective`, `constraints`, `bounds`
3. **update_augmented_lagrangian_state() Exists** - Handles multiplier/penalty updates automatically
4. **Planner Has Multi-Turn Loop** - `planner.py:384-468` already implements a 5-turn agentic loop with tool calling
5. **Gap is the Bridge** - What's missing is a **steppable wrapper** that allows supervision injection between ALM outer iterations

### Architecture: Wrap, Don't Rebuild

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ASO Architecture                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ai_scientist/                           constellaration/                    │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────────────┐ │
│  │   Planner   │◄──►│  Coordinator │◄──►│  ALM Bridge (NEW)               │ │
│  │ (supervise) │    │  (ASO loop)  │    │  ┌───────────────────────────┐  │ │
│  └─────────────┘    └──────────────┘    │  │ augmented_lagrangian.py   │  │ │
│        ▲                   │            │  │ - AugmentedLagrangianState │  │ │
│        │                   │            │  │ - update_...state()        │  │ │
│        │            ┌──────▼──────┐     │  │ - augmented_lagrangian_fn  │  │ │
│        │            │ Diagnostics │     │  └───────────────────────────┘  │ │
│        └────────────┤  Translator │     │  ┌───────────────────────────┐  │ │
│                     │ (real ALM   │     │  │ augmented_lagrangian_     │  │ │
│                     │  state!)    │     │  │ runner.py                 │  │ │
│                     └─────────────┘     │  │ - run() [full loop]       │  │ │
│                                         │  │ - objective_constraints() │  │ │
│                                         │  └───────────────────────────┘  │ │
│                                         └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What's Different from Previous Plans

| Feature | Previous Plans | V4 Reality |
|---------|----------------|------------|
| ALM Infrastructure | "Needs to be built" | **Already exists in constellaration** |
| Diagnostics Source | Proxy from surrogate | **Real ALM state** (objective, constraints, multipliers, penalties) |
| Worker Contract | Returns candidates only | Wrap existing `augmented_lagrangian_runner` |
| Key Implementation | Build ALM from scratch | **Create steppable bridge layer** |
| Supervision Point | Between gradient steps | **Between ALM outer iterations** |

---

## Part 1: Current Architecture (Verified)

### 1.1 ai_scientist Module Structure

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
    ├── differentiable.py  # Gradient descent (344 lines) - has optimize_alm_inner_loop()
    └── geometry.py        # Fourier ↔ Real-space (569 lines)
```

### 1.2 constellaration ALM Infrastructure (Critical Discovery)

```
constellaration/src/constellaration/
├── optimization/
│   ├── __init__.py
│   ├── augmented_lagrangian.py       # Core ALM: State, Settings, update functions
│   ├── augmented_lagrangian_runner.py # Complete ALM loop with Nevergrad oracle
│   ├── settings.py                    # AugmentedLagrangianMethodSettings, NevergradSettings
│   └── scipy_minimize_runner.py       # Alternative optimizer
├── forward_model.py                   # Physics evaluation (ConstellarationMetrics)
├── problems.py                        # P1 (Geometrical), P2 (SimpleToBuild), P3 (MHDStable)
└── geometry/
    └── surface_rz_fourier.py          # SurfaceRZFourier, mask_and_ravel, build_mask
```

### 1.3 The ALM State Contract (Verified)

```python
# constellaration/src/constellaration/optimization/augmented_lagrangian.py

class AugmentedLagrangianState(pydantic.BaseModel, arbitrary_types_allowed=True):
    x: jnp.ndarray                    # Current design point (scaled)
    multipliers: jnp.ndarray          # Lagrange multipliers per constraint
    penalty_parameters: jnp.ndarray   # Penalty scaling factors per constraint
    objective: jnp.ndarray            # Current objective value
    constraints: jnp.ndarray          # Current constraint values (positive = violated)
    bounds: jnp.ndarray               # Trust region bounds per dimension
```

**Key Properties:**
- Pydantic BaseModel → JSON serializable
- Contains ALL information needed for diagnostics
- `multipliers` indicate constraint importance learned during optimization
- `penalty_parameters` show how aggressively constraints are being enforced
- `bounds` represent the trust region (shrinks over iterations)

### 1.4 ALM Settings Hierarchy (Verified)

```python
# constellaration/src/constellaration/optimization/augmented_lagrangian.py

class AugmentedLagrangianSettings(pydantic.BaseModel):
    constraint_violation_tolerance_reduction_factor: float = 0.5
    penalty_parameters_increase_factor: float = 2.0      # Key lever for supervision
    bounds_reduction_factor: float = 0.95                # Trust region shrinkage
    penalty_parameters_max: float = 1e8                  # Safety cap
    bounds_min: float = 0.05                             # Minimum trust region

# constellaration/src/constellaration/optimization/settings.py

class AugmentedLagrangianMethodSettings(pydantic.BaseModel):
    maxit: int                                           # Max outer iterations
    penalty_parameters_initial: float                    # Starting penalty
    bounds_initial: float                                # Starting trust region
    augmented_lagrangian_settings: AugmentedLagrangianSettings
    oracle_settings: NevergradSettings                   # Inner optimizer config

class NevergradSettings(pydantic.BaseModel):
    budget_initial: int              # Initial function evaluations
    budget_increment: int            # Added each outer iteration
    budget_max: int                  # Cap on evaluations
    max_time: float | None           # Wall-clock limit
    num_workers: int                 # Parallel evaluations
    batch_mode: bool                 # Wait for all workers
```

### 1.5 The Existing ALM Loop (augmented_lagrangian_runner.py:186-293)

This is the loop we need to make steppable:

```python
# EXISTING CODE in constellaration (simplified)
for k in range(settings.optimizer_settings.maxit):
    # 1. Setup trust region parametrization
    parametrization = nevergrad.p.Array(
        init=np.array(state.x),
        lower=np.array(state.x - state.bounds),
        upper=np.array(state.x + state.bounds),
    )

    # 2. Run Nevergrad oracle (inner optimization)
    oracle = nevergrad.optimizers.NGOpt(parametrization, budget, num_workers)
    with ProcessPoolExecutor() as executor:
        # ... parallel evaluation of candidates ...
        # Each candidate evaluated via objective_constraints()
        # Results fed to oracle.tell()

    # 3. Get recommendation and evaluate
    x = oracle.provide_recommendation().value
    (objective, constraints), metrics = objective_constraints(x, ...)

    # 4. Update ALM state (THIS IS THE KEY FUNCTION)
    state = al.update_augmented_lagrangian_state(
        x=jnp.copy(x),
        objective=objective,
        constraints=constraints,
        state=state,
        settings=settings.optimizer_settings.augmented_lagrangian_settings,
    )

    # 5. Increase budget for next iteration
    budget = min(budget_max, budget + budget_increment)

    # 6. Log (THIS IS WHERE WE INJECT SUPERVISION)
    _logging(k + 1, n_function_evals, state)
```

**Supervision Injection Point:** After step 4 (state update), before step 5 (next iteration).

### 1.6 The update_augmented_lagrangian_state Function

```python
# constellaration/src/constellaration/optimization/augmented_lagrangian.py:67-121

def update_augmented_lagrangian_state(
    x: jnp.ndarray,
    objective: jnp.ndarray,
    constraints: jnp.ndarray,
    state: AugmentedLagrangianState,
    settings: AugmentedLagrangianSettings,
    penalty_parameters: jnp.ndarray | None = None,  # Override if provided
    bounds: jnp.ndarray | None = None,              # Override if provided
) -> AugmentedLagrangianState:
    """Updates multipliers, penalties, and bounds based on constraint progress."""

    # Update Lagrange multipliers
    multipliers = jnp.maximum(
        0.0,
        state.multipliers + state.penalty_parameters * constraints,
    )

    # Increase penalties for constraints that aren't improving
    if penalty_parameters is None:
        penalty_parameters = jnp.where(
            jnp.maximum(0.0, constraints) >
                settings.constraint_violation_tolerance_reduction_factor *
                jnp.maximum(0.0, state.constraints),
            jnp.minimum(
                settings.penalty_parameters_max,
                settings.penalty_parameters_increase_factor * state.penalty_parameters,
            ),
            state.penalty_parameters,
        )

    # Shrink trust region
    if bounds is None:
        bounds = jnp.maximum(settings.bounds_min, state.bounds)

    return AugmentedLagrangianState(
        x=x, multipliers=multipliers, penalty_parameters=penalty_parameters,
        objective=objective, constraints=constraints, bounds=bounds,
    )
```

**Key Insight:** The `penalty_parameters` and `bounds` can be **overridden** by passing them explicitly. This is how supervision directives can directly modify ALM behavior.

---

## Part 2: Gap Analysis (Corrected)

### 2.1 What Previous Plans Got Wrong

| Previous Claim | Reality |
|----------------|---------|
| "ALM state not accessible" | ✅ `AugmentedLagrangianState` is Pydantic, fully accessible |
| "Need to build ALM loop" | ✅ Complete loop exists in `augmented_lagrangian_runner.py` |
| "Need proxy diagnostics" | ✅ Can use **real** ALM state for diagnostics |
| "Worker contract incomplete" | ⚠️ True, but solution is wrapping, not rebuilding |

### 2.2 Actual Gaps (What Needs Implementation)

| Gap | Description | Resolution |
|-----|-------------|------------|
| **ALM Bridge** | No steppable interface to existing ALM loop | Create `ai_scientist/optim/alm_bridge.py` |
| **Supervision Injection** | No hook between ALM outer iterations | Add callback parameter to bridge |
| **Data Structures** | `OptimizationDirective`, `OptimizerDiagnostics` not implemented | Add to `planner.py` |
| **Tiered Supervision** | `HeuristicSupervisor` not implemented | Add to `planner.py` |
| **Config** | `ASOConfig`, `ALMConfig` not in `ExperimentConfig` | Add to `config.py` |
| **Coordinator ASO Path** | `produce_candidates_aso()` not implemented | Add to `coordinator.py` |
| **CLI Flag** | `--aso` not available | Add to `runner.py` |

### 2.3 What Already Works (Leverage These)

| Component | Location | Capability |
|-----------|----------|------------|
| Multi-turn LLM loop | `planner.py:384-468` | Tool calling, JSON parsing, RAG |
| Failure reflection | `planner.py:328-333` | Learns from recent failures |
| World model | `memory.py` | HV tracking, candidate history |
| Surrogate | `optim/surrogate_v2.py` | Fast objective/constraint prediction |
| Problem definitions | `constellaration/problems.py` | P1/P2/P3 constraint specifications |

---

## Part 3: Implementation Specification

### 3.0 Phase 0: ALM Bridge Layer (NEW - Critical)

Create `ai_scientist/optim/alm_bridge.py`:

```python
"""Bridge between ai_scientist ASO loop and constellaration ALM infrastructure.

This module provides a steppable interface to the ALM optimization loop,
allowing supervision injection between outer iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
import multiprocessing
from concurrent import futures
import time

import jax.numpy as jnp
import numpy as np
import nevergrad

from constellaration.optimization.augmented_lagrangian import (
    AugmentedLagrangianState,
    AugmentedLagrangianSettings,
    augmented_lagrangian_function,
    update_augmented_lagrangian_state,
)
from constellaration.optimization.settings import (
    AugmentedLagrangianMethodSettings,
    NevergradSettings,
)
from constellaration.geometry import surface_rz_fourier as rz_fourier
from constellaration.utils import pytree
import constellaration.forward_model as forward_model
import constellaration.problems as problems


@dataclass
class ALMStepResult:
    """Result of a single ALM outer iteration."""
    state: AugmentedLagrangianState
    n_evals: int
    objective: float
    max_violation: float
    metrics: Optional[forward_model.ConstellarationMetrics]


@dataclass
class ALMContext:
    """Context for ALM optimization (created once, reused across steps)."""
    scale: jnp.ndarray
    unravel_fn: Callable[[jnp.ndarray], rz_fourier.SurfaceRZFourier]
    problem: problems.SingleObjectiveProblem | problems.MHDStableQIStellarator
    forward_settings: forward_model.ConstellarationSettings
    alm_settings: AugmentedLagrangianMethodSettings
    aspect_ratio_upper_bound: Optional[float]


def create_alm_context(
    boundary: rz_fourier.SurfaceRZFourier,
    problem: problems.SingleObjectiveProblem | problems.MHDStableQIStellarator,
    settings: "OptimizationSettings",  # from constellaration.optimization.settings
    aspect_ratio_upper_bound: Optional[float] = None,
) -> Tuple[ALMContext, AugmentedLagrangianState]:
    """
    Initialize ALM context and starting state from a boundary.

    Returns:
        Tuple of (context for reuse, initial ALM state)
    """
    # Apply mode limits
    boundary = rz_fourier.set_max_mode_numbers(
        surface=boundary,
        max_poloidal_mode=settings.max_poloidal_mode,
        max_toroidal_mode=settings.max_toroidal_mode,
    )

    # Build mask for optimization
    mask = rz_fourier.build_mask(
        boundary,
        max_poloidal_mode=settings.max_poloidal_mode,
        max_toroidal_mode=settings.max_toroidal_mode,
    )

    # Flatten to optimization vector
    initial_guess, unravel_fn = pytree.mask_and_ravel(boundary, mask)

    # Compute scaling
    scale = rz_fourier.compute_infinity_norm_spectrum_scaling_fun(
        poloidal_modes=boundary.poloidal_modes.flatten(),
        toroidal_modes=boundary.toroidal_modes.flatten(),
        alpha=settings.infinity_norm_spectrum_scaling,
    ).reshape(boundary.poloidal_modes.shape)
    scale = jnp.array(np.concatenate([scale[mask.r_cos], scale[mask.z_sin]]))

    x0 = jnp.array(initial_guess) / scale

    # Evaluate initial point
    from constellaration.optimization.augmented_lagrangian_runner import objective_constraints
    (objective, constraints), _ = objective_constraints(
        x0, scale, problem, unravel_fn,
        settings.forward_model_settings, aspect_ratio_upper_bound,
    )

    # Create initial state
    alm_settings = settings.optimizer_settings
    state = AugmentedLagrangianState(
        x=jnp.copy(x0),
        multipliers=jnp.zeros_like(constraints),
        penalty_parameters=alm_settings.penalty_parameters_initial * jnp.ones_like(constraints),
        objective=objective,
        constraints=constraints,
        bounds=jnp.ones_like(x0) * alm_settings.bounds_initial,
    )

    context = ALMContext(
        scale=scale,
        unravel_fn=unravel_fn,
        problem=problem,
        forward_settings=settings.forward_model_settings,
        alm_settings=alm_settings,
        aspect_ratio_upper_bound=aspect_ratio_upper_bound,
    )

    return context, state


def step_alm(
    context: ALMContext,
    state: AugmentedLagrangianState,
    budget: int,
    *,
    penalty_override: Optional[jnp.ndarray] = None,
    bounds_override: Optional[jnp.ndarray] = None,
    num_workers: int = 4,
) -> ALMStepResult:
    """
    Execute ONE outer ALM iteration.

    This is the steppable interface that allows supervision between iterations.

    Args:
        context: ALM context (created once via create_alm_context)
        state: Current ALM state
        budget: Number of function evaluations for this step
        penalty_override: Optional direct override of penalty parameters
        bounds_override: Optional direct override of trust region bounds
        num_workers: Parallel workers for evaluation

    Returns:
        ALMStepResult with updated state and metrics
    """
    from constellaration.optimization.augmented_lagrangian_runner import objective_constraints

    # Apply overrides to state before optimization
    if penalty_override is not None:
        state = state.copy(update={"penalty_parameters": penalty_override})
    if bounds_override is not None:
        state = state.copy(update={"bounds": bounds_override})

    # Setup trust region parametrization
    parametrization = nevergrad.p.Array(
        init=np.array(state.x),
        lower=np.array(state.x - state.bounds),
        upper=np.array(state.x + state.bounds),
    )

    oracle = nevergrad.optimizers.NGOpt(
        parametrization=parametrization,
        budget=budget,
        num_workers=num_workers,
    )
    oracle.suggest(np.array(state.x))

    n_evals = 0
    last_metrics = None

    mp_context = multiprocessing.get_context("forkserver")

    with futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as executor:
        running: list[Tuple[futures.Future, Any]] = []
        rest_budget = budget

        while rest_budget or running:
            # Submit new evaluations
            while len(running) < min(num_workers, rest_budget):
                candidate = oracle.ask()
                future = executor.submit(
                    objective_constraints,
                    jnp.array(candidate.value),
                    context.scale,
                    context.problem,
                    context.unravel_fn,
                    context.forward_settings,
                    context.aspect_ratio_upper_bound,
                )
                running.append((future, candidate))
                rest_budget -= 1

            # Wait for completions
            completed, _ = futures.wait(
                [f for f, _ in running],
                return_when=futures.FIRST_COMPLETED,
            )

            for future, candidate in running:
                if future in completed:
                    n_evals += 1
                    (obj, cons), metrics = future.result()
                    last_metrics = metrics

                    oracle.tell(
                        candidate,
                        augmented_lagrangian_function(obj, cons, state).item(),
                    )

            running = [(f, c) for f, c in running if f not in completed]

    # Get final recommendation
    recommendation = oracle.provide_recommendation()
    x = recommendation.value

    # Evaluate final point
    (objective, constraints), final_metrics = objective_constraints(
        x, context.scale, context.problem, context.unravel_fn,
        context.forward_settings, context.aspect_ratio_upper_bound,
    )

    # Update ALM state
    new_state = update_augmented_lagrangian_state(
        x=jnp.copy(x),
        objective=objective,
        constraints=constraints,
        state=state,
        settings=context.alm_settings.augmented_lagrangian_settings,
    )

    max_violation = float(jnp.max(jnp.maximum(0.0, constraints)))

    return ALMStepResult(
        state=new_state,
        n_evals=n_evals,
        objective=float(objective),
        max_violation=max_violation,
        metrics=final_metrics,
    )


def state_to_boundary_params(
    context: ALMContext,
    state: AugmentedLagrangianState,
) -> Dict[str, Any]:
    """Convert ALM state back to boundary parameter dict."""
    boundary = context.unravel_fn(state.x * context.scale)
    return {
        "r_cos": np.asarray(boundary.r_cos).tolist(),
        "z_sin": np.asarray(boundary.z_sin).tolist(),
        "n_field_periods": boundary.n_field_periods,
        "is_stellarator_symmetric": boundary.is_stellarator_symmetric,
    }
```

### 3.1 Config Structures

Add to `ai_scientist/config.py`:

```python
from dataclasses import dataclass, field
from typing import Literal

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
    violation_increase_threshold: float = 0.05  # 5% increase = "increasing"
    violation_decrease_threshold: float = 0.05  # 5% decrease = "decreasing"

    # Budget allocation
    steps_per_supervision: int = 1  # ALM outer iterations between supervision checks

    # Safety limits
    max_constraint_weight: float = 1000.0
    max_penalty_boost: float = 4.0  # Max multiplier for penalty override

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
import json


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
    alm_overrides: Optional[Mapping[str, Any]] = None  # Direct ALM state manipulation
    suggested_params: Optional[Mapping[str, Any]] = None
    reasoning: str = ""
    confidence: float = 1.0
    source: DirectiveSource = DirectiveSource.HEURISTIC

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "config_overrides": dict(self.config_overrides) if self.config_overrides else None,
            "alm_overrides": dict(self.alm_overrides) if self.alm_overrides else None,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "source": self.source.value,
        }


@dataclass
class ConstraintDiagnostic:
    """Diagnostic for a single constraint (from real ALM state)."""
    name: str
    violation: float           # max(0, constraint_value)
    penalty: float             # Current penalty parameter
    multiplier: float          # Lagrange multiplier (learned importance)
    trend: str                 # "stable", "increasing_violation", "decreasing_violation"
    delta: float = 0.0         # Change from previous step


@dataclass
class OptimizerDiagnostics:
    """Rich diagnostic report from real ALM state."""
    step: int
    trajectory_id: int

    # From AugmentedLagrangianState
    objective: float
    objective_delta: float
    max_violation: float
    constraints_raw: List[float]        # Raw constraint values
    multipliers: List[float]            # Lagrange multipliers
    penalty_parameters: List[float]     # Current penalties
    bounds_norm: float                  # Trust region size

    # Derived analysis
    status: str  # "IN_PROGRESS", "STAGNATION", "FEASIBLE_FOUND", "DIVERGING"
    constraint_diagnostics: List[ConstraintDiagnostic]
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
        return json.dumps({
            "step": self.step,
            "objective": round(self.objective, 4),
            "objective_delta": round(self.objective_delta, 6),
            "max_violation": round(self.max_violation, 4),
            "status": self.status,
            "bounds_norm": round(self.bounds_norm, 4),
            "constraints": [
                {
                    "name": c.name,
                    "violation": round(c.violation, 4),
                    "penalty": round(c.penalty, 2),
                    "multiplier": round(c.multiplier, 4),
                    "trend": c.trend,
                }
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

    Key insight: We can now use REAL ALM state (multipliers, penalties, bounds)
    to make informed decisions.
    """

    def __init__(self, aso_config: "ASOConfig"):
        self.config = aso_config

    def analyze(self, diagnostics: OptimizerDiagnostics) -> OptimizationDirective:
        """
        Generate directive using heuristic rules based on real ALM state.

        Decision tree:
        1. FEASIBLE_FOUND + stable objective -> STOP (converged)
        2. FEASIBLE_FOUND + improving -> CONTINUE
        3. STAGNATION + high violation -> ADJUST (boost penalties directly)
        4. STAGNATION + low violation + small bounds -> STOP (local minimum)
        5. DIVERGING -> STOP (abandon trajectory)
        6. Specific constraint worsening + low multiplier -> ADJUST (boost that penalty)
        7. Otherwise -> CONTINUE
        """
        cfg = self.config

        # Case 1 & 2: Feasible region reached
        if diagnostics.status == "FEASIBLE_FOUND":
            if abs(diagnostics.objective_delta) < cfg.stagnation_objective_threshold:
                return OptimizationDirective(
                    action=DirectiveAction.STOP,
                    reasoning=f"Converged: feasible (violation={diagnostics.max_violation:.4f}) with stable objective",
                    source=DirectiveSource.CONVERGENCE,
                )
            return OptimizationDirective(
                action=DirectiveAction.CONTINUE,
                reasoning="Feasible and still improving objective",
                source=DirectiveSource.HEURISTIC,
            )

        # Case 3 & 4: Stagnation
        if diagnostics.status == "STAGNATION":
            if diagnostics.max_violation > cfg.stagnation_violation_threshold:
                # High violation stagnation: boost penalties
                # Find constraints with highest violation relative to their penalty
                worst_idx = max(
                    range(len(diagnostics.constraint_diagnostics)),
                    key=lambda i: diagnostics.constraint_diagnostics[i].violation /
                                  (diagnostics.constraint_diagnostics[i].penalty + 1e-6)
                )
                worst = diagnostics.constraint_diagnostics[worst_idx]

                # Create penalty override: boost worst constraint's penalty
                new_penalties = diagnostics.penalty_parameters.copy()
                new_penalties[worst_idx] = min(
                    worst.penalty * cfg.max_penalty_boost,
                    cfg.max_constraint_weight
                )

                return OptimizationDirective(
                    action=DirectiveAction.ADJUST,
                    alm_overrides={"penalty_parameters": new_penalties},
                    reasoning=f"Stagnation with violation={diagnostics.max_violation:.4f}, "
                              f"boosting penalty for '{worst.name}' from {worst.penalty:.1f} to {new_penalties[worst_idx]:.1f}",
                    source=DirectiveSource.HEURISTIC,
                )
            else:
                # Low violation stagnation: check if bounds are too small (stuck)
                if diagnostics.bounds_norm < 0.1:
                    return OptimizationDirective(
                        action=DirectiveAction.STOP,
                        reasoning="Stagnation with small trust region, likely local minimum",
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
                reasoning="Multiple constraints diverging, abandoning trajectory",
                source=DirectiveSource.HEURISTIC,
            )

        # Case 6: Specific constraint struggling
        struggling = [c for c in diagnostics.constraint_diagnostics if c.trend == "increasing_violation"]
        if struggling:
            worst = max(struggling, key=lambda c: c.violation)
            worst_idx = next(i for i, c in enumerate(diagnostics.constraint_diagnostics) if c.name == worst.name)

            new_penalties = diagnostics.penalty_parameters.copy()
            new_penalties[worst_idx] = min(worst.penalty * 2, cfg.max_constraint_weight)

            return OptimizationDirective(
                action=DirectiveAction.ADJUST,
                alm_overrides={"penalty_parameters": new_penalties},
                reasoning=f"Constraint '{worst.name}' worsening (violation={worst.violation:.4f}), "
                          f"boosting penalty to {new_penalties[worst_idx]:.1f}",
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

Add methods to `PlanningAgent` class in `ai_scientist/planner.py`:

```python
class PlanningAgent:
    def __init__(self, ...):
        # ... existing init ...
        self.heuristic: HeuristicSupervisor | None = None

    def _ensure_heuristic(self, aso_config: "ASOConfig") -> HeuristicSupervisor:
        if self.heuristic is None:
            self.heuristic = HeuristicSupervisor(aso_config)
        return self.heuristic

    def supervise(
        self,
        diagnostics: OptimizerDiagnostics,
        cycle: int,
        aso_config: "ASOConfig",
    ) -> OptimizationDirective:
        """
        Tiered supervision: heuristic first, LLM on demand.

        The key insight is that we now have REAL ALM state, so heuristics
        can make much better decisions than with proxy diagnostics.
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
        aso_config: "ASOConfig",
    ) -> OptimizationDirective:
        """LLM-based supervision with real ALM state context."""
        # Retrieve relevant context if stagnating
        rag_context = []
        if diagnostics.status == "STAGNATION":
            rag_context = self.retrieve_rag(
                "Strategies for escaping local minima in stellarator ALM optimization",
                k=2,
            )

        system_prompt = self._build_supervision_prompt(cycle, rag_context, diagnostics)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Current ALM diagnostics:\n{diagnostics.to_json()}"},
        ]

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
                return self._parse_directive(content, diagnostics)

            except json.JSONDecodeError as e:
                if attempt < aso_config.llm_max_retries - 1:
                    messages.append({"role": "user", "content": f"Invalid JSON: {e}. Please output valid JSON."})
                    continue
                raise

        raise RuntimeError("LLM supervision failed after retries")

    def _build_supervision_prompt(self, cycle: int, rag_context: list, diagnostics: OptimizerDiagnostics) -> str:
        rag_section = ""
        if rag_context:
            rag_section = f"\n\nRelevant knowledge:\n{json.dumps(rag_context, indent=2)}"

        constraint_names = [c.name for c in diagnostics.constraint_diagnostics]

        return f"""You are the ASO Supervisor for the AI Scientist (cycle {cycle}).

You have access to REAL Augmented Lagrangian Method (ALM) state:
- objective: Current objective function value
- constraints: {constraint_names}
- penalty_parameters: How strongly each constraint is being enforced
- multipliers: Lagrange multipliers (learned constraint importance)
- bounds_norm: Size of trust region (smaller = more focused search)

ACTIONS:
- CONTINUE: Proceed with current settings
- ADJUST: Modify penalty_parameters to steer optimization
- STOP: Terminate (converged or hopeless)
- RESTART: Abandon trajectory, try new seed

OUTPUT FORMAT (JSON):
{{
  "action": "CONTINUE | ADJUST | STOP | RESTART",
  "alm_overrides": {{
    "penalty_parameters": [p1, p2, ...]  // Optional: new penalties per constraint
  }},
  "reasoning": "brief explanation"
}}

ADJUSTMENT STRATEGY:
- If a constraint has high violation but low penalty: increase that penalty
- If stuck (small bounds_norm) with violations: try RESTART
- If multiplier is high but violation persists: constraint may be infeasible
{rag_section}

Respond with ONLY valid JSON."""

    def _parse_directive(self, content: str, diagnostics: OptimizerDiagnostics) -> OptimizationDirective:
        """Parse LLM response into OptimizationDirective."""
        json_str = content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()

        data = json.loads(json_str)
        action = DirectiveAction(data.get("action", "CONTINUE"))

        return OptimizationDirective(
            action=action,
            alm_overrides=data.get("alm_overrides"),
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
import jax.numpy as jnp

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
from ai_scientist.optim.alm_bridge import (
    ALMContext,
    ALMStepResult,
    create_alm_context,
    step_alm,
    state_to_boundary_params,
)


@dataclass
class TrajectoryState:
    """State for a single optimization trajectory."""
    id: int
    seed: Dict[str, Any]
    alm_context: Optional[ALMContext] = None
    alm_state: Optional[AugmentedLagrangianState] = None
    history: List[AugmentedLagrangianState] = field(default_factory=list)
    evals_used: int = 0
    steps: int = 0
    status: str = "active"
    best_objective: float = float("inf")
    best_violation: float = float("inf")
    stagnation_count: int = 0
    budget_used: int = 0


class Coordinator:
    """Manages Agent-Supervised Optimization with real ALM state."""

    CONSTRAINT_NAMES = {
        "p1": ["aspect_ratio", "average_triangularity", "edge_rotational_transform"],
        "p2": ["aspect_ratio", "edge_rotational_transform", "edge_magnetic_mirror_ratio",
               "max_elongation", "qi"],
        "p3": ["edge_rotational_transform", "edge_magnetic_mirror_ratio",
               "vacuum_well", "flux_compression", "qi"],
    }

    def __init__(
        self,
        cfg: ai_config.ExperimentConfig,
        world_model: memory.WorldModel,
        planner: PlanningAgent,
        # ... other params ...
    ):
        self.cfg = cfg
        self.world_model = world_model
        self.planner = planner

        problem_key = (cfg.problem or "p3").lower()[:2]
        self.constraint_names = self.CONSTRAINT_NAMES.get(problem_key, self.CONSTRAINT_NAMES["p3"])
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
        ASO loop with real ALM state supervision.
        """
        config = initial_config or self.cfg
        aso = config.aso
        alm = config.alm

        # 1. Prepare seeds
        seeds = self._prepare_seeds(initial_seeds, cycle, 1)
        if not seeds:
            print("[Coordinator] No valid seeds, returning empty")
            return []

        # 2. Run trajectory with real ALM
        traj = TrajectoryState(id=0, seed=seeds[0])
        candidates = self._run_trajectory_aso(
            traj=traj,
            eval_budget=eval_budget,
            cycle=cycle,
            experiment_id=experiment_id,
            config=config,
        )

        # 3. Persist telemetry
        self._persist_telemetry(experiment_id)

        return candidates

    def _run_trajectory_aso(
        self,
        traj: TrajectoryState,
        eval_budget: int,
        cycle: int,
        experiment_id: int,
        config: ai_config.ExperimentConfig,
    ) -> List[Dict[str, Any]]:
        """Run trajectory with real ALM state and supervision."""
        aso = config.aso
        alm = config.alm
        candidates = []

        # Initialize ALM context and state
        boundary = self._seed_to_boundary(traj.seed)
        problem = self._get_problem(config)
        settings = self._build_optimization_settings(config)

        traj.alm_context, traj.alm_state = create_alm_context(
            boundary=boundary,
            problem=problem,
            settings=settings,
        )

        oracle_budget = alm.oracle_budget_initial

        while traj.budget_used < eval_budget and traj.status == "active":
            traj.steps += 1
            step_start = time.perf_counter()

            # 1. Execute ALM step
            result = step_alm(
                context=traj.alm_context,
                state=traj.alm_state,
                budget=min(oracle_budget, eval_budget - traj.budget_used),
                num_workers=alm.oracle_num_workers,
            )

            traj.alm_state = result.state
            traj.budget_used += result.n_evals
            traj.history.append(result.state)

            # 2. Generate diagnostics from REAL ALM state
            diagnostics = self._generate_diagnostics(result.state, traj)

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
                # Extract final candidate
                candidates.append({
                    "params": state_to_boundary_params(traj.alm_context, traj.alm_state),
                    "objective": result.objective,
                    "max_violation": result.max_violation,
                    "source": "aso",
                })
                break

            if directive.action == DirectiveAction.RESTART:
                # Try new seed
                new_seeds = self._prepare_seeds(None, cycle, 1)
                if new_seeds:
                    # Save current best before restart
                    candidates.append({
                        "params": state_to_boundary_params(traj.alm_context, traj.alm_state),
                        "objective": result.objective,
                        "max_violation": result.max_violation,
                        "source": "aso_pre_restart",
                    })

                    traj.seed = new_seeds[0]
                    traj.history = []
                    traj.stagnation_count = 0
                    boundary = self._seed_to_boundary(traj.seed)
                    traj.alm_context, traj.alm_state = create_alm_context(
                        boundary=boundary,
                        problem=problem,
                        settings=settings,
                    )
                    oracle_budget = alm.oracle_budget_initial
                    print(f"[Coordinator] RESTART with new seed")
                else:
                    traj.status = "abandoned"
                    print(f"[Coordinator] RESTART failed, no seeds")
                    break
                continue

            if directive.action == DirectiveAction.ADJUST:
                # Apply ALM overrides directly to state
                if directive.alm_overrides and "penalty_parameters" in directive.alm_overrides:
                    new_penalties = jnp.array(directive.alm_overrides["penalty_parameters"])
                    traj.alm_state = traj.alm_state.copy(
                        update={"penalty_parameters": new_penalties}
                    )
                    print(f"[Coordinator] ADJUST penalties: {directive.reasoning}")

            # Increase oracle budget for next iteration
            oracle_budget = min(alm.oracle_budget_max, oracle_budget + alm.oracle_budget_increment)

            # Auto-stop on excessive stagnation
            if traj.stagnation_count >= aso.max_stagnation_steps:
                traj.status = "stagnated"
                print(f"[Coordinator] Auto-STOP (stagnation limit)")
                candidates.append({
                    "params": state_to_boundary_params(traj.alm_context, traj.alm_state),
                    "objective": result.objective,
                    "max_violation": result.max_violation,
                    "source": "aso_stagnation",
                })
                break

        print(f"[Coordinator] Trajectory done: {traj.status}, {traj.steps} steps, "
              f"{traj.budget_used} evals, {len(candidates)} candidates")

        return candidates

    def _generate_diagnostics(
        self,
        alm_state: AugmentedLagrangianState,
        traj: TrajectoryState,
    ) -> OptimizerDiagnostics:
        """Generate rich diagnostics from REAL ALM state."""
        aso = self.cfg.aso
        prev = traj.history[-2] if len(traj.history) >= 2 else None

        # Extract all fields from ALM state
        objective = float(alm_state.objective)
        constraints = [float(c) for c in alm_state.constraints]
        multipliers = [float(m) for m in alm_state.multipliers]
        penalties = [float(p) for p in alm_state.penalty_parameters]
        bounds_norm = float(jnp.linalg.norm(alm_state.bounds))

        objective_delta = objective - float(prev.objective) if prev else 0.0
        max_violation = max(0.0, max(constraints)) if constraints else 0.0

        # Constraint analysis
        constraint_diagnostics = []
        diverging_count = 0

        for i, name in enumerate(self.constraint_names):
            if i >= len(constraints):
                continue

            violation = max(0.0, constraints[i])
            penalty = penalties[i] if i < len(penalties) else 1.0
            multiplier = multipliers[i] if i < len(multipliers) else 0.0
            trend = "stable"
            delta = 0.0

            if prev and i < len(prev.constraints):
                prev_violation = max(0.0, float(prev.constraints[i]))
                delta = violation - prev_violation

                if violation > prev_violation * (1 + aso.violation_increase_threshold) and violation > 1e-4:
                    trend = "increasing_violation"
                    diverging_count += 1
                elif violation < prev_violation * (1 - aso.violation_decrease_threshold):
                    trend = "decreasing_violation"

            constraint_diagnostics.append(ConstraintDiagnostic(
                name=name,
                violation=violation,
                penalty=penalty,
                multiplier=multiplier,
                trend=trend,
                delta=delta,
            ))

        # Status determination
        narrative = []
        if max_violation < aso.feasibility_threshold:
            status = "FEASIBLE_FOUND"
            narrative.append(f"Feasible region reached (max_violation={max_violation:.4f})")
        elif diverging_count >= len(self.constraint_names) // 2:
            status = "DIVERGING"
            narrative.append(f"{diverging_count} constraints diverging")
        elif prev and abs(objective_delta) < aso.stagnation_objective_threshold:
            if max_violation > aso.stagnation_violation_threshold:
                status = "STAGNATION"
                narrative.append(f"Stagnation: obj_delta={objective_delta:.6f}, violation={max_violation:.4f}")
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
            constraints_raw=constraints,
            multipliers=multipliers,
            penalty_parameters=penalties,
            bounds_norm=bounds_norm,
            status=status,
            constraint_diagnostics=constraint_diagnostics,
            narrative=narrative,
            steps_since_improvement=traj.stagnation_count,
        )

    def _update_trajectory_best(self, traj: TrajectoryState, diag: OptimizerDiagnostics):
        """Update best values and stagnation counter."""
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
        """Record telemetry event with full ALM state."""
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
            "bounds_norm": diag.bounds_norm,
            "penalties": diag.penalty_parameters,
            "multipliers": diag.multipliers,
            "directive_action": directive.action.value,
            "directive_source": directive.source.value,
            "directive_reasoning": directive.reasoning,
            "evals_used": traj.budget_used,
            "wall_time_ms": wall_time_ms,
            "llm_called": llm_called,
        })

    def _persist_telemetry(self, experiment_id: int):
        """Persist telemetry to JSONL file."""
        if not self.telemetry:
            return

        from pathlib import Path
        import json

        telemetry_dir = Path(self.cfg.reporting_dir) / "telemetry"
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        telemetry_file = telemetry_dir / f"aso_exp{experiment_id}.jsonl"

        with open(telemetry_file, "a") as f:
            for event in self.telemetry:
                f.write(json.dumps(event) + "\n")

        self.telemetry = []

    # Helper methods (implement based on existing code)
    def _prepare_seeds(self, initial_seeds, cycle, n_needed):
        """Prepare and validate seeds using ExplorationWorker + GeometerWorker."""
        # ... implementation ...
        pass

    def _seed_to_boundary(self, seed):
        """Convert seed dict to SurfaceRZFourier."""
        # ... implementation ...
        pass

    def _get_problem(self, config):
        """Get problem instance from config."""
        # ... implementation ...
        pass

    def _build_optimization_settings(self, config):
        """Build OptimizationSettings from config."""
        # ... implementation ...
        pass
```

---

## Part 4: Runner Integration

### 4.1 Control Mode Selection

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
    # ...
):
    cycle_number = cycle_index + 1

    # 1. High-level planning
    planning_outcome = planner.plan_cycle(
        cfg=cfg,
        cycle_index=cycle_index,
        stage_history=stage_history,
        last_summary=last_p3_summary,
        experiment_id=experiment_id,
    )

    active_cfg = cfg
    if planning_outcome.config_overrides:
        active_cfg = _apply_config_overrides(cfg, planning_outcome.config_overrides)

    budget_snapshot = budget_controller.snapshot()

    # 2. Branch on control mode
    if active_cfg.aso.enabled:
        print(f"[runner][cycle={cycle_number}] ASO mode with real ALM state")

        initial_seeds = []
        if planning_outcome.suggested_params:
            initial_seeds = [planning_outcome.suggested_params]

        candidates = coordinator.produce_candidates_aso(
            cycle=cycle_number,
            experiment_id=experiment_id,
            eval_budget=budget_snapshot.screen_evals_per_cycle,
            template=active_cfg.boundary_template,
            initial_seeds=initial_seeds,
            initial_config=active_cfg,
        )
    else:
        # Legacy path
        candidates = coordinator.produce_candidates(...)

    # 3. Continue with evaluation, promotion, etc.
    # ...
```

### 4.2 CLI Flag

```python
parser.add_argument(
    "--aso",
    action="store_true",
    help="Enable Agent-Supervised Optimization with real ALM state",
)

# In main():
if args.aso:
    from dataclasses import replace
    cfg = replace(cfg, aso=replace(cfg.aso, enabled=True))
```

---

## Part 5: Implementation Checklist (Reordered by Priority)

### Phase 0: ALM Bridge (Priority 0 - Critical Path)

**File: `ai_scientist/optim/alm_bridge.py` (NEW)**
- [x] 0.1 Create `ALMStepResult` dataclass
- [x] 0.2 Create `ALMContext` dataclass
- [x] 0.3 Implement `create_alm_context()` - wraps constellaration setup
- [x] 0.4 Implement `step_alm()` - single outer ALM iteration
- [x] 0.5 Implement `state_to_boundary_params()` - convert back to dict
- [ ] 0.6 Unit test: `create_alm_context()` returns valid state
- [ ] 0.7 Unit test: `step_alm()` reduces violation over iterations

### Phase 1: Data Structures (Priority 0)

**File: `ai_scientist/config.py`**
- [x] 1.1 Add `ALMConfig` dataclass (mirrors constellaration)
- [x] 1.2 Add `ASOConfig` dataclass
- [x] 1.3 Update `ExperimentConfig` with `alm` and `aso` fields
- [x] 1.4 Add `_alm_config_from_dict()` loader
- [x] 1.5 Add `_aso_config_from_dict()` loader

**File: `ai_scientist/planner.py`**
- [x] 1.6 Add `DirectiveAction` enum
- [x] 1.7 Add `DirectiveSource` enum
- [x] 1.8 Add `OptimizationDirective` dataclass with `alm_overrides` field
- [x] 1.9 Add `ConstraintDiagnostic` dataclass with `multiplier` field
- [x] 1.10 Add `OptimizerDiagnostics` dataclass with full ALM state fields

### Phase 2: Supervision (Priority 0)

**File: `ai_scientist/planner.py`**
- [ ] 2.1 Add `HeuristicSupervisor` class
- [ ] 2.2 Implement decision tree with penalty boost logic
- [ ] 2.3 Add `_ensure_heuristic()` to `PlanningAgent`
- [ ] 2.4 Add `supervise()` to `PlanningAgent`
- [ ] 2.5 Add `_llm_supervise()` with ALM-aware prompt
- [ ] 2.6 Add `_build_supervision_prompt()` with penalty guidance
- [ ] 2.7 Add `_parse_directive()` supporting `alm_overrides`

### Phase 3: Coordinator (Priority 1)

**File: `ai_scientist/coordinator.py`**
- [ ] 3.1 Add `TrajectoryState` dataclass with `alm_context`, `alm_state`
- [ ] 3.2 Add `CONSTRAINT_NAMES` mapping
- [ ] 3.3 Add `produce_candidates_aso()` entry point
- [ ] 3.4 Implement `_run_trajectory_aso()` using `step_alm()`
- [ ] 3.5 Implement `_generate_diagnostics()` from real ALM state
- [ ] 3.6 Implement `_update_trajectory_best()`
- [ ] 3.7 Implement `_log_telemetry()` with full ALM state
- [ ] 3.8 Implement `_persist_telemetry()` to JSONL
- [ ] 3.9 Implement helper methods (`_seed_to_boundary`, etc.)

### Phase 4: Runner Integration (Priority 1)

**File: `ai_scientist/runner.py`**
- [ ] 4.1 Add `--aso` CLI argument
- [ ] 4.2 Update config initialization for ASO flag
- [ ] 4.3 Update `_run_cycle()` to branch on `cfg.aso.enabled`
- [ ] 4.4 Call `produce_candidates_aso()` in ASO path

### Phase 5: Testing (Priority 1)

**File: `tests/test_alm_bridge.py` (NEW)**
- [ ] 5.1 Test `create_alm_context()` with P3 problem
- [ ] 5.2 Test `step_alm()` executes and returns valid state
- [ ] 5.3 Test penalty override is applied correctly

**File: `tests/test_planner.py`**
- [ ] 5.4 Test `HeuristicSupervisor` STOP on FEASIBLE_FOUND
- [ ] 5.5 Test `HeuristicSupervisor` ADJUST on STAGNATION
- [ ] 5.6 Test `HeuristicSupervisor` penalty boost calculation

**File: `tests/test_coordinator.py`**
- [ ] 5.7 Test `_generate_diagnostics()` extracts all ALM fields
- [ ] 5.8 Integration test: 3-step ASO loop with mock problem

### Phase 6: Documentation (Priority 2)

- [ ] 6.1 Update `docs/run_protocol.md` with ASO instructions
- [ ] 6.2 Add example config YAML with ASO/ALM settings
- [ ] 6.3 Move superseded docs to `docs/archive/`

---

## Part 6: Success Metrics

| Metric | Baseline (Legacy) | Target (ASO V4) | Measurement |
|--------|-------------------|-----------------|-------------|
| LLM calls/cycle | 0 | <10 | Telemetry |
| Heuristic decisions/cycle | 0 | >40 | Telemetry |
| Wall-clock/cycle | ~5 min | ~5 min (same, different work) | Timer |
| Feasibility rate | ~30% | >50% | WorldModel |
| Stagnation recovery | 0% | >50% | RESTART success |
| Constraint-specific adjustments | 0 | >20 | Telemetry (ADJUST count) |

---

## Part 7: Migration Guide

### From Legacy Mode

1. Set `aso.enabled = True` in config or use `--aso` flag
2. Optionally tune `alm.*` settings (defaults match constellaration)
3. ASO telemetry written to `reports/telemetry/aso_exp{id}.jsonl`
4. Legacy `produce_candidates()` still works unchanged

### From Previous Planning Documents

| Old Concept | New Implementation |
|-------------|-------------------|
| "Build ALM loop" | Use `alm_bridge.py` wrapping constellaration |
| "Proxy diagnostics" | Real diagnostics from `AugmentedLagrangianState` |
| "config_overrides" | `alm_overrides` with direct penalty manipulation |
| "surrogate-based supervision" | Real ALM state + heuristics |

---

## Part 8: Risks and Mitigations

### 8.1 Risk: Multiprocessing Boilerplate Duplication

**Problem:** The `step_alm()` function in `alm_bridge.py` replicates the `concurrent.futures` logic from `augmented_lagrangian_runner.py`. This code is complex (forkserver context, running evaluations tracking, `FIRST_COMPLETED` vs `ALL_COMPLETED`). Copy-paste errors or subtle divergence will cause hard-to-debug failures.

**Mitigations:**

1. **Option A (Recommended): Extract shared inner loop upstream**

   Submit a PR to constellaration to extract the inner loop:

   ```python
   # constellaration/src/constellaration/optimization/augmented_lagrangian_runner.py

   def run_single_outer_iteration(
       state: al.AugmentedLagrangianState,
       budget: int,
       scale: jnp.ndarray,
       problem: problems.SingleObjectiveProblem | problems.MHDStableQIStellarator,
       unravel_fn: Callable,
       forward_settings: forward_model.ConstellarationSettings,
       aspect_ratio_upper_bound: float | None,
       alm_settings: al.AugmentedLagrangianSettings,
       oracle_settings: NevergradSettings,
   ) -> tuple[al.AugmentedLagrangianState, int, forward_model.ConstellarationMetrics | None]:
       """Execute ONE outer ALM iteration. Extracted for reuse by alm_bridge."""
       mp_context = multiprocessing.get_context("forkserver")
       # ... existing ProcessPoolExecutor logic from lines 204-264 ...
       return new_state, n_evals, metrics


   def run(boundary, settings, problem, aspect_ratio_upper_bound=None):
       # ... setup code ...
       for k in range(settings.optimizer_settings.maxit):
           state, n_evals, metrics = run_single_outer_iteration(
               state, budget, scale, problem, unravel_fn,
               settings.forward_model_settings, aspect_ratio_upper_bound,
               settings.optimizer_settings.augmented_lagrangian_settings,
               settings.optimizer_settings.oracle_settings,
           )
           # ... budget update, logging ...
   ```

   Then `alm_bridge.py` simply imports and calls `run_single_outer_iteration()`.

2. **Option B (Fallback): Literal copy-paste with hash verification**

   If upstream refactoring is not feasible:

   ```python
   # alm_bridge.py

   # SYNC MARKER: This block is copied from augmented_lagrangian_runner.py:204-264
   # Source hash: sha256(lines 204-264) = "abc123..."
   # Last synced: 2025-11-30
   # If upstream changes, CI will fail (see test_alm_bridge.py::test_source_sync)

   def _run_inner_loop(state, budget, context):
       # ... copied logic ...
   ```

   With corresponding test:

   ```python
   # tests/test_alm_bridge.py

   import hashlib
   from pathlib import Path

   EXPECTED_HASH = "abc123..."  # Update when intentionally syncing

   def test_source_sync():
       """Fail if upstream ProcessPoolExecutor logic has changed."""
       runner_path = Path("constellaration/src/constellaration/optimization/augmented_lagrangian_runner.py")
       lines = runner_path.read_text().splitlines()[203:264]  # 0-indexed
       actual_hash = hashlib.sha256("\n".join(lines).encode()).hexdigest()[:12]
       assert actual_hash == EXPECTED_HASH, (
           f"augmented_lagrangian_runner.py:204-264 has changed (hash {actual_hash}). "
           f"Review changes and update alm_bridge.py accordingly, then update EXPECTED_HASH."
       )
   ```

**Recommendation:** Pursue Option A. The upstream refactoring is a clean separation of concerns and benefits constellaration independently.

---

### 8.2 Risk: API Drift / Upstream Breakage

**Problem:** `alm_bridge.py` imports from `constellaration.optimization`. Any changes to function signatures (e.g., `objective_constraints`, `update_augmented_lagrangian_state`) will break the bridge at runtime.

**Mitigations:**

1. **Explicit version coupling with type checking**

   ```python
   # alm_bridge.py

   # =============================================================================
   # CONSTELLARATION API CONTRACT
   # =============================================================================
   # This module depends on specific constellaration function signatures.
   # If these imports fail or type-check errors occur, constellaration has changed.
   # Last verified: 2025-11-30
   # Minimum constellaration version: (pin in pyproject.toml)
   # =============================================================================

   from constellaration.optimization.augmented_lagrangian import (
       AugmentedLagrangianState,
       AugmentedLagrangianSettings,
       augmented_lagrangian_function,
       update_augmented_lagrangian_state,
   )
   from constellaration.optimization.augmented_lagrangian_runner import (
       objective_constraints,
   )
   from constellaration.optimization.settings import (
       AugmentedLagrangianMethodSettings,
       NevergradSettings,
       OptimizationSettings,
   )

   # Type assertions to catch signature changes at import time
   from typing import TYPE_CHECKING
   if TYPE_CHECKING:
       from typing import Callable
       _: Callable[
           [jnp.ndarray, jnp.ndarray, "SingleObjectiveProblem", Callable, "ConstellarationSettings", float | None],
           tuple[tuple[jnp.ndarray, jnp.ndarray], "ConstellarationMetrics | None"]
       ] = objective_constraints
   ```

2. **Rigorous integration test in CI**

   ```python
   # tests/test_alm_bridge.py

   import pytest
   from ai_scientist.optim.alm_bridge import (
       ALMContext,
       ALMStepResult,
       create_alm_context,
       step_alm,
       state_to_boundary_params,
   )
   from constellaration.optimization.augmented_lagrangian import AugmentedLagrangianState
   import constellaration.problems as problems


   @pytest.fixture
   def minimal_p3_problem():
       """Minimal P3 problem for testing."""
       return problems.MHDStableQIStellarator(
           edge_rotational_transform_over_n_field_periods_lower_bound=0.1,
           log10_qi_upper_bound=-1.0,
           edge_magnetic_mirror_ratio_upper_bound=0.3,
           flux_compression_in_regions_of_bad_curvature_upper_bound=1.0,
           vacuum_well_lower_bound=0.0,
       )


   @pytest.fixture
   def minimal_boundary():
       """Minimal SurfaceRZFourier for testing."""
       from constellaration.geometry.surface_rz_fourier import SurfaceRZFourier
       import jax.numpy as jnp
       # ... create minimal valid boundary ...
       return boundary


   class TestALMBridgeAPIContract:
       """Tests that verify the constellaration API contract is stable."""

       def test_augmented_lagrangian_state_fields(self):
           """Verify AugmentedLagrangianState has expected fields."""
           import jax.numpy as jnp
           state = AugmentedLagrangianState(
               x=jnp.zeros(2),
               multipliers=jnp.zeros(3),
               penalty_parameters=jnp.ones(3),
               objective=jnp.array(1.0),
               constraints=jnp.zeros(3),
               bounds=jnp.ones(2),
           )
           # These will fail if fields are renamed/removed
           assert hasattr(state, 'x')
           assert hasattr(state, 'multipliers')
           assert hasattr(state, 'penalty_parameters')
           assert hasattr(state, 'objective')
           assert hasattr(state, 'constraints')
           assert hasattr(state, 'bounds')

       def test_objective_constraints_signature(self):
           """Verify objective_constraints accepts expected arguments."""
           import inspect
           from constellaration.optimization.augmented_lagrangian_runner import objective_constraints
           sig = inspect.signature(objective_constraints)
           params = list(sig.parameters.keys())
           # Must have these positional params in order
           assert params[:6] == ['x', 'scale', 'problem', 'unravel_and_unmask_fn', 'settings', 'aspect_ratio_upper_bound']

       def test_update_state_signature(self):
           """Verify update_augmented_lagrangian_state accepts overrides."""
           import inspect
           from constellaration.optimization.augmented_lagrangian import update_augmented_lagrangian_state
           sig = inspect.signature(update_augmented_lagrangian_state)
           params = sig.parameters
           # Must accept penalty_parameters and bounds overrides
           assert 'penalty_parameters' in params
           assert 'bounds' in params


   class TestALMBridgeFunctionality:
       """Tests that verify alm_bridge works end-to-end."""

       @pytest.mark.slow
       def test_create_context_and_step(self, minimal_boundary, minimal_p3_problem):
           """Integration test: create context and run one step."""
           from constellaration.optimization.settings import OptimizationSettings
           # ... create settings ...

           context, initial_state = create_alm_context(
               minimal_boundary, minimal_p3_problem, settings
           )

           assert isinstance(context, ALMContext)
           assert isinstance(initial_state, AugmentedLagrangianState)

           result = step_alm(context, initial_state, budget=10, num_workers=2)

           assert isinstance(result, ALMStepResult)
           assert isinstance(result.state, AugmentedLagrangianState)
           assert result.n_evals > 0

       @pytest.mark.slow
       def test_penalty_override_applied(self, minimal_boundary, minimal_p3_problem):
           """Test that penalty_override actually modifies behavior."""
           # ... setup ...
           import jax.numpy as jnp

           context, state = create_alm_context(...)

           # Run with default penalties
           result_default = step_alm(context, state, budget=10)

           # Run with boosted penalties
           boosted = state.penalty_parameters * 10.0
           result_boosted = step_alm(context, state, budget=10, penalty_override=boosted)

           # Results should differ (optimizer sees different loss landscape)
           # This is a weak test but catches "override is ignored" bugs
           assert not jnp.allclose(result_default.state.x, result_boosted.state.x)
   ```

3. **Pin constellaration version in pyproject.toml**

   ```toml
   [project.dependencies]
   constellaration = ">=0.5.0,<0.6.0"  # Pin to minor version
   ```

   With CI job that tests against `constellaration@main` to catch breakages early:

   ```yaml
   # .github/workflows/ci.yml
   jobs:
     test-constellaration-head:
       name: Test against constellaration main
       runs-on: ubuntu-latest
       continue-on-error: true  # Don't block PRs, but notify
       steps:
         - uses: actions/checkout@v4
         - run: pip install git+https://github.com/proxima-fusion/constellaration@main
         - run: pytest tests/test_alm_bridge.py -v
   ```

---

### 8.3 Risk: Divergent Behavior Between Bridge and Original Runner

**Problem:** Subtle differences between `step_alm()` and the original `run()` loop could cause different optimization trajectories, making debugging difficult.

**Mitigations:**

1. **Behavioral equivalence test**

   ```python
   # tests/test_alm_bridge.py

   @pytest.mark.slow
   def test_bridge_matches_original_runner():
       """Verify step_alm produces same trajectory as original run()."""
       import numpy as np
       from constellaration.optimization.augmented_lagrangian_runner import run
       from constellaration.utils.seed_util import seed_everything

       # Setup identical conditions
       seed_everything(42)
       boundary = ...
       problem = ...
       settings = ...

       # Run original for 3 iterations
       # (would need to modify run() to return intermediate states, or patch _logging)

       # Run bridge for 3 steps
       seed_everything(42)
       context, state = create_alm_context(boundary, problem, settings)
       states_bridge = [state]
       for _ in range(3):
           result = step_alm(context, states_bridge[-1], budget=settings.oracle_settings.budget_initial)
           states_bridge.append(result.state)

       # Compare (objective, constraint violation) trajectories
       # Allow small numerical tolerance
       for i, (orig, bridge) in enumerate(zip(states_orig, states_bridge)):
           np.testing.assert_allclose(orig.objective, bridge.objective, rtol=1e-5,
               err_msg=f"Objective diverged at step {i}")
           np.testing.assert_allclose(orig.constraints, bridge.constraints, rtol=1e-5,
               err_msg=f"Constraints diverged at step {i}")
   ```

2. **Deterministic seeding in bridge**

   ```python
   # alm_bridge.py

   def step_alm(
       context: ALMContext,
       state: AugmentedLagrangianState,
       budget: int,
       *,
       seed: int | None = None,  # For reproducibility testing
       # ... other params ...
   ) -> ALMStepResult:
       if seed is not None:
           import numpy as np
           np.random.seed(seed)
       # ... rest of function ...
   ```

---

### 8.4 Risk: Silent Failures in Multiprocessing

**Problem:** `ProcessPoolExecutor` can silently swallow exceptions in child processes, or fail to propagate them clearly. This makes debugging physics evaluation failures difficult.

**Mitigations:**

1. **Explicit exception handling with context**

   ```python
   # alm_bridge.py

   class ALMEvaluationError(Exception):
       """Raised when objective_constraints fails in worker."""
       def __init__(self, candidate_value, original_error):
           self.candidate_value = candidate_value
           self.original_error = original_error
           super().__init__(f"Evaluation failed for x={candidate_value[:3]}...: {original_error}")


   def _safe_objective_constraints(x, scale, problem, unravel_fn, settings, ar_bound):
       """Wrapper that captures exceptions with context."""
       try:
           return objective_constraints(x, scale, problem, unravel_fn, settings, ar_bound)
       except Exception as e:
           # Return sentinel that step_alm can detect
           return (("ERROR", str(e), x.tolist()[:5]), None)


   def step_alm(...):
       # ...
       for future, candidate in running:
           if future in completed:
               result = future.result()
               if isinstance(result[0][0], str) and result[0][0] == "ERROR":
                   _, error_msg, x_sample = result[0]
                   logger.warning(f"Evaluation failed: {error_msg} at x={x_sample}")
                   # Skip this candidate, don't tell oracle
                   continue
               # ... normal processing ...
   ```

2. **Timeout per evaluation**

   ```python
   # alm_bridge.py

   EVAL_TIMEOUT_SECONDS = 300  # 5 minutes per evaluation

   def step_alm(...):
       # ...
       for future, candidate in running:
           if future in completed:
               try:
                   result = future.result(timeout=EVAL_TIMEOUT_SECONDS)
               except futures.TimeoutError:
                   logger.warning(f"Evaluation timed out for candidate")
                   continue
               # ...
   ```

---

### 8.5 Risk: Memory Leaks in Long-Running ASO Loops

**Problem:** Repeated `create_alm_context()` calls or JAX array accumulation could cause memory growth over long optimization runs.

**Mitigations:**

1. **Reuse context across steps** (already in design)

   ```python
   # Coordinator.produce_candidates_aso()
   context, state = create_alm_context(...)  # Once
   for step in range(max_steps):
       result = step_alm(context, state, ...)  # Reuse context
       state = result.state
   ```

2. **Explicit garbage collection between trajectories**

   ```python
   # coordinator.py

   def produce_candidates_aso(self, ...):
       try:
           # ... optimization loop ...
       finally:
           # Force cleanup between trajectories
           import gc
           gc.collect()
   ```

3. **Monitor memory in telemetry**

   ```python
   # coordinator.py

   import psutil

   def _emit_telemetry(self, event_type, state, ...):
       process = psutil.Process()
       memory_mb = process.memory_info().rss / 1024 / 1024
       event = {
           # ... existing fields ...
           "memory_mb": memory_mb,
       }
   ```

---

### 8.6 Summary: Implementation Checklist for Risks

| Risk | Mitigation | Owner | Priority |
|------|------------|-------|----------|
| Multiprocessing duplication | Extract `run_single_outer_iteration()` upstream | constellaration PR | P0 |
| API drift | Integration tests + version pin | ai-sci-feasible-designs CI | P0 |
| Behavioral divergence | Equivalence test with seeding | tests/test_alm_bridge.py | P1 |
| Silent failures | `_safe_objective_constraints` wrapper | alm_bridge.py | P1 |
| Memory leaks | Context reuse + gc.collect() | coordinator.py | P2 |

---

## Appendix A: constellaration ALM API Reference

### AugmentedLagrangianState

```python
class AugmentedLagrangianState(pydantic.BaseModel):
    x: jnp.ndarray                    # Design vector (scaled)
    multipliers: jnp.ndarray          # λ_i: Lagrange multipliers
    penalty_parameters: jnp.ndarray   # ρ_i: Penalty scaling
    objective: jnp.ndarray            # f(x)
    constraints: jnp.ndarray          # g_i(x), positive = violated
    bounds: jnp.ndarray               # Trust region per dimension
```

### Key Functions

```python
# Compute augmented Lagrangian value
augmented_lagrangian_function(objective, constraints, state) -> jnp.ndarray

# Update state after optimization step
update_augmented_lagrangian_state(
    x, objective, constraints, state, settings,
    penalty_parameters=None,  # Override if provided
    bounds=None,              # Override if provided
) -> AugmentedLagrangianState
```

### Problem Types

```python
# P1: Geometrical (3 constraints)
GeometricalProblem: aspect_ratio, average_triangularity, edge_rotational_transform

# P2: SimpleToBuildQIStellarator (5 constraints)
SimpleToBuildQIStellarator: aspect_ratio, edge_rotational_transform,
                            edge_magnetic_mirror_ratio, max_elongation, qi

# P3: MHDStableQIStellarator (6 constraints)
MHDStableQIStellarator: aspect_ratio, edge_rotational_transform,
                        edge_magnetic_mirror_ratio, flux_compression,
                        vacuum_well, qi
```

---

## Appendix B: Example Telemetry Event (Enhanced)

```json
{
  "timestamp": "2025-11-30T14:30:22.456Z",
  "experiment_id": 42,
  "cycle": 5,
  "trajectory_id": 0,
  "step": 7,
  "status": "STAGNATION",
  "objective": 6.82,
  "max_violation": 0.12,
  "bounds_norm": 0.34,
  "penalties": [10.0, 20.0, 40.0, 10.0, 80.0],
  "multipliers": [0.5, 1.2, 3.4, 0.1, 5.6],
  "directive_action": "ADJUST",
  "directive_source": "heuristic",
  "directive_reasoning": "Constraint 'qi' worsening (violation=0.12), boosting penalty to 160.0",
  "evals_used": 700,
  "wall_time_ms": 45234.5,
  "llm_called": false
}
```

---

## Appendix C: Key File Locations

| Component | File | Key Functions/Classes |
|-----------|------|----------------------|
| ALM Core | `constellaration/.../augmented_lagrangian.py` | `AugmentedLagrangianState`, `update_...state()` |
| ALM Runner | `constellaration/.../augmented_lagrangian_runner.py` | `run()`, `objective_constraints()` |
| ALM Settings | `constellaration/.../settings.py` | `AugmentedLagrangianMethodSettings` |
| **ALM Bridge** | `ai_scientist/optim/alm_bridge.py` | `create_alm_context()`, `step_alm()` |
| Config | `ai_scientist/config.py` | `ALMConfig`, `ASOConfig` |
| Planner | `ai_scientist/planner.py` | `HeuristicSupervisor`, `supervise()` |
| Coordinator | `ai_scientist/coordinator.py` | `produce_candidates_aso()` |
| Runner | `ai_scientist/runner.py` | `--aso` flag, `_run_cycle()` |
