# ASO Implementation Prompts for Coding Agents

**Purpose:** Copy-paste prompts for coding agents to implement ASO V4 tasks.
**Reference:** See `ASO_V4_IMPLEMENTATION_GUIDE.md` for full specifications.

---

## Progress Tracker

### Phase 0: ALM Bridge (Priority 0 - Critical Path)
- [ ] Task 0: Create `ai_scientist/optim/alm_bridge.py`

### Phase 1: Config (Priority 0)
- [ ] Task 1: Add config structures to `ai_scientist/config.py`

### Phase 2: Planner Data Structures (Priority 0)
- [ ] Task 2: Add data structures to `ai_scientist/planner.py`

### Phase 3: Heuristic Supervisor (Priority 0)
- [ ] Task 3: Implement `HeuristicSupervisor` in `ai_scientist/planner.py`

### Phase 4: PlanningAgent Extension (Priority 0)
- [ ] Task 4: Add supervision methods to `PlanningAgent`

### Phase 5: Coordinator ASO Loop (Priority 0)
- [ ] Task 5: Implement `produce_candidates_aso()` in `ai_scientist/coordinator.py`

### Phase 6: Runner Integration (Priority 1)
- [ ] Task 6: Add ASO mode to `ai_scientist/runner.py`

### Phase 7: Testing (Priority 1)
- [ ] Task 7: Create `tests/test_alm_bridge.py`
- [ ] Task 8: Create `tests/test_planner_aso.py`
- [ ] Task 9: Create integration test

### Phase 8: Documentation (Priority 2)
- [ ] Task 10: Update run protocol and archive old docs

---

## Task 0: Create ALM Bridge Layer

**File to create:** `ai_scientist/optim/alm_bridge.py`

### Prompt

```
## Task: Implement ALM Bridge Layer

**Goal:** Create `ai_scientist/optim/alm_bridge.py` - a steppable interface that wraps the existing constellaration ALM infrastructure to enable supervision injection between outer iterations.

### Context

The constellaration library already has a complete Augmented Lagrangian Method (ALM) implementation. We need a **bridge layer** that:
1. Exposes a steppable interface (one outer iteration at a time)
2. Allows overriding penalty parameters and bounds between steps
3. Returns rich state for diagnostics

### Files to Read First

Before implementing, read these files to understand the existing infrastructure:

1. `constellaration/src/constellaration/optimization/augmented_lagrangian.py` - Contains `AugmentedLagrangianState`, `AugmentedLagrangianSettings`, `update_augmented_lagrangian_state()`, `augmented_lagrangian_function()`

2. `constellaration/src/constellaration/optimization/augmented_lagrangian_runner.py` - Contains `run()` loop (lines 186-293) and `objective_constraints()` function. The ProcessPoolExecutor pattern (lines 204-264) should be replicated carefully.

3. `constellaration/src/constellaration/optimization/settings.py` - Contains `AugmentedLagrangianMethodSettings`, `NevergradSettings`, `OptimizationSettings`

4. `constellaration/optimization_examples/augmented_lagrangian_toy_example.py` - Shows how the ALM loop works in practice

5. `docs/ASO_V4_IMPLEMENTATION_GUIDE.md` - Section 3.0 contains the specification and example implementation

### Implementation Requirements

#### 1. ALMStepResult dataclass
```python
@dataclass
class ALMStepResult:
    """Result of a single ALM outer iteration."""
    state: AugmentedLagrangianState
    n_evals: int
    objective: float
    max_violation: float
    metrics: Optional[forward_model.ConstellarationMetrics]
```

#### 2. ALMContext dataclass
```python
@dataclass
class ALMContext:
    """Context for ALM optimization (created once, reused across steps)."""
    scale: jnp.ndarray
    unravel_fn: Callable[[jnp.ndarray], rz_fourier.SurfaceRZFourier]
    problem: problems.SingleObjectiveProblem | problems.MHDStableQIStellarator
    forward_settings: forward_model.ConstellarationSettings
    alm_settings: AugmentedLagrangianMethodSettings
    aspect_ratio_upper_bound: Optional[float]
```

#### 3. create_alm_context() function
- Takes: `boundary`, `problem`, `settings`, optional `aspect_ratio_upper_bound`
- Returns: `Tuple[ALMContext, AugmentedLagrangianState]`
- Should:
  - Apply mode limits via `rz_fourier.set_max_mode_numbers()`
  - Build mask via `rz_fourier.build_mask()`
  - Flatten to optimization vector via `pytree.mask_and_ravel()`
  - Compute scaling via `rz_fourier.compute_infinity_norm_spectrum_scaling_fun()`
  - Evaluate initial point via `objective_constraints()`
  - Create initial `AugmentedLagrangianState`

#### 4. step_alm() function
- Takes: `context`, `state`, `budget`, optional `penalty_override`, `bounds_override`, `num_workers`
- Returns: `ALMStepResult`
- Should:
  - Apply overrides to state if provided (use `state.model_copy(update={...})`)
  - Setup Nevergrad parametrization with trust region bounds
  - Run ProcessPoolExecutor loop (replicate pattern from augmented_lagrangian_runner.py:204-264)
  - Use `multiprocessing.get_context("forkserver")`
  - Submit evaluations via `objective_constraints()`
  - Tell oracle with `augmented_lagrangian_function()` values
  - Get recommendation and evaluate final point
  - Update state via `update_augmented_lagrangian_state()`
  - Return result with metrics

#### 5. state_to_boundary_params() function
- Takes: `context`, `state`
- Returns: `Dict[str, Any]` with boundary parameters
- Should unravel state back to boundary and extract `r_cos`, `z_sin`, `n_field_periods`, `is_stellarator_symmetric`

### Critical Implementation Notes

1. **API Contract Header:** Add this comment block at the top of imports:
```python
# =============================================================================
# CONSTELLARATION API CONTRACT
# =============================================================================
# This module depends on specific constellaration function signatures.
# If these imports fail or type-check errors occur, constellaration has changed.
# Last verified: 2025-11-30
# =============================================================================
```

2. **ProcessPoolExecutor Pattern:** The multiprocessing logic is complex. Copy the pattern carefully:
   - Use `forkserver` context
   - Track `running_evaluations` list of `(Future, candidate)` tuples
   - Use `futures.FIRST_COMPLETED` (not batch mode)
   - Remove completed futures from running list after processing

3. **Error Handling:** Wrap `objective_constraints` calls:
```python
def _safe_objective_constraints(x, scale, problem, unravel_fn, settings, ar_bound):
    try:
        return objective_constraints(x, scale, problem, unravel_fn, settings, ar_bound)
    except Exception as e:
        return (("ERROR", str(e), x.tolist()[:5]), None)
```

### Verification

After implementation:
1. Verify imports work: `python -c "from ai_scientist.optim.alm_bridge import ALMContext, step_alm"`
2. Check types align with constellaration's `AugmentedLagrangianState`
3. The file should be ~200-250 lines

### Do NOT
- Do not modify any constellaration files
- Do not create test files (separate task)
- Do not add CLI integration (separate task)
```

---

## Task 1: Add Config Structures

**File to modify:** `ai_scientist/config.py`

### Prompt

```
## Task: Add ALM and ASO Config Structures

**Goal:** Add `ALMConfig` and `ASOConfig` dataclasses to `ai_scientist/config.py` and integrate them into `ExperimentConfig`.

### Files to Read First

1. `ai_scientist/config.py` - Understand existing config structure and patterns
2. `constellaration/src/constellaration/optimization/settings.py` - See ALM settings to mirror
3. `docs/ASO_V4_IMPLEMENTATION_GUIDE.md` - Section 3.1 has the exact specifications

### Implementation Requirements

#### 1. Add ALMConfig dataclass

```python
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
```

#### 2. Add ASOConfig dataclass

```python
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
```

#### 3. Update ExperimentConfig

Add these fields to `ExperimentConfig`:
```python
alm: ALMConfig = field(default_factory=ALMConfig)
aso: ASOConfig = field(default_factory=ASOConfig)
```

#### 4. Add loader functions

Add `_alm_config_from_dict()` and `_aso_config_from_dict()` following the existing pattern for other config sections (e.g., `_boundary_template_from_dict()`).

### Verification

1. `python -c "from ai_scientist.config import ALMConfig, ASOConfig, ExperimentConfig"`
2. Verify defaults: `ExperimentConfig().aso.enabled` should be `False`
3. Verify YAML loading works if you have example configs

### Do NOT
- Do not change any existing config fields
- Do not add CLI argument handling (separate task)
- Do not add validation logic beyond type hints
```

---

## Task 2: Add Planner Data Structures

**File to modify:** `ai_scientist/planner.py`

### Prompt

```
## Task: Add ASO Data Structures to Planner

**Goal:** Add enums, dataclasses for optimization directives and diagnostics to `ai_scientist/planner.py`.

### Files to Read First

1. `ai_scientist/planner.py` - Understand existing structure
2. `docs/ASO_V4_IMPLEMENTATION_GUIDE.md` - Section 3.2 has exact specifications

### Implementation Requirements

Add these at module level (after imports, before class definitions):

#### 1. DirectiveAction enum
```python
class DirectiveAction(Enum):
    """Enumerated actions for type safety."""
    CONTINUE = "CONTINUE"
    ADJUST = "ADJUST"
    STOP = "STOP"
    RESTART = "RESTART"
```

#### 2. DirectiveSource enum
```python
class DirectiveSource(Enum):
    """Source of the directive for debugging."""
    LLM = "llm"
    HEURISTIC = "heuristic"
    CONVERGENCE = "convergence"
    FALLBACK = "fallback"
```

#### 3. OptimizationDirective dataclass
```python
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
```

#### 4. ConstraintDiagnostic dataclass
```python
@dataclass
class ConstraintDiagnostic:
    """Diagnostic for a single constraint (from real ALM state)."""
    name: str
    violation: float           # max(0, constraint_value)
    penalty: float             # Current penalty parameter
    multiplier: float          # Lagrange multiplier (learned importance)
    trend: str                 # "stable", "increasing_violation", "decreasing_violation"
    delta: float = 0.0         # Change from previous step
```

#### 5. OptimizerDiagnostics dataclass
```python
@dataclass
class OptimizerDiagnostics:
    """Rich diagnostic report from real ALM state."""
    step: int
    trajectory_id: int

    # From AugmentedLagrangianState
    objective: float
    objective_delta: float
    max_violation: float
    constraints_raw: List[float]
    multipliers: List[float]
    penalty_parameters: List[float]
    bounds_norm: float

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
        # Event-triggered
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

### Required Imports

Add these imports at the top of the file:
```python
from enum import Enum
from typing import Any, Mapping, List, Optional
import json
```

### Verification

1. `python -c "from ai_scientist.planner import DirectiveAction, OptimizationDirective, OptimizerDiagnostics"`
2. Test serialization: `OptimizationDirective(action=DirectiveAction.CONTINUE).to_dict()`

### Do NOT
- Do not modify any existing code in the file
- Do not add the HeuristicSupervisor class (separate task)
- Do not add methods to PlanningAgent (separate task)
```

---

## Task 3: Implement HeuristicSupervisor

**File to modify:** `ai_scientist/planner.py`

### Prompt

```
## Task: Implement HeuristicSupervisor Class

**Goal:** Add the `HeuristicSupervisor` class to `ai_scientist/planner.py` that handles 80%+ of supervision decisions without LLM calls.

### Prerequisites
- Task 2 must be completed (DirectiveAction, OptimizationDirective, etc. must exist)

### Files to Read First

1. `ai_scientist/planner.py` - Verify Task 2 data structures exist
2. `ai_scientist/config.py` - Verify ASOConfig exists (from Task 1)
3. `docs/ASO_V4_IMPLEMENTATION_GUIDE.md` - Section 3.3 has exact specification

### Implementation Requirements

Add this class after the data structure definitions:

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
                worst_idx = max(
                    range(len(diagnostics.constraint_diagnostics)),
                    key=lambda i: diagnostics.constraint_diagnostics[i].violation /
                                  (diagnostics.constraint_diagnostics[i].penalty + 1e-6)
                )
                worst = diagnostics.constraint_diagnostics[worst_idx]

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

### Verification

1. `python -c "from ai_scientist.planner import HeuristicSupervisor"`
2. Test with mock diagnostics (create a simple test case)

### Do NOT
- Do not modify any existing methods in PlanningAgent
- Do not add the `supervise()` method to PlanningAgent (separate task)
```

---

## Task 4: Add Supervision Methods to PlanningAgent

**File to modify:** `ai_scientist/planner.py`

### Prompt

```
## Task: Add Supervision Methods to PlanningAgent

**Goal:** Extend the existing `PlanningAgent` class with `supervise()`, `_llm_supervise()`, and related methods for ASO.

### Prerequisites
- Task 2: Data structures must exist
- Task 3: HeuristicSupervisor must exist

### Files to Read First

1. `ai_scientist/planner.py` - Understand existing PlanningAgent structure
2. `docs/ASO_V4_IMPLEMENTATION_GUIDE.md` - Section 3.4 has exact specification

### Implementation Requirements

Add these methods to the existing `PlanningAgent` class:

#### 1. Add instance variable in __init__
```python
def __init__(self, ...):
    # ... existing init code ...
    self.heuristic: HeuristicSupervisor | None = None
```

#### 2. Add _ensure_heuristic method
```python
def _ensure_heuristic(self, aso_config: "ASOConfig") -> HeuristicSupervisor:
    if self.heuristic is None:
        self.heuristic = HeuristicSupervisor(aso_config)
    return self.heuristic
```

#### 3. Add supervise method (main entry point)
```python
def supervise(
    self,
    diagnostics: OptimizerDiagnostics,
    cycle: int,
    aso_config: "ASOConfig",
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
```

#### 4. Add _llm_supervise method
```python
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
```

#### 5. Add _build_supervision_prompt method
```python
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
```

#### 6. Add _parse_directive method
```python
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

### Verification

1. `python -c "from ai_scientist.planner import PlanningAgent; p = PlanningAgent(...); print(hasattr(p, 'supervise'))"`
2. Verify method signatures match the specification

### Do NOT
- Do not modify any existing methods (plan_cycle, etc.)
- Do not change the constructor signature
```

---

## Task 5: Implement Coordinator ASO Loop

**File to modify:** `ai_scientist/coordinator.py`

### Prompt

```
## Task: Implement ASO Loop in Coordinator

**Goal:** Add `produce_candidates_aso()` and supporting methods to `ai_scientist/coordinator.py`.

### Prerequisites
- Task 0: alm_bridge.py must exist
- Task 1: ALMConfig, ASOConfig must exist
- Task 2-4: Planner data structures and methods must exist

### Files to Read First

1. `ai_scientist/coordinator.py` - Understand existing structure
2. `ai_scientist/optim/alm_bridge.py` - Know the bridge API (from Task 0)
3. `ai_scientist/planner.py` - Know the supervision API (from Task 4)
4. `docs/ASO_V4_IMPLEMENTATION_GUIDE.md` - Section 3.5 has exact specification

### Implementation Requirements

#### 1. Add imports at top of file
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
```

#### 2. Add TrajectoryState dataclass
```python
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
```

#### 3. Add CONSTRAINT_NAMES class variable to Coordinator
```python
CONSTRAINT_NAMES = {
    "p1": ["aspect_ratio", "average_triangularity", "edge_rotational_transform"],
    "p2": ["aspect_ratio", "edge_rotational_transform", "edge_magnetic_mirror_ratio",
           "max_elongation", "qi"],
    "p3": ["aspect_ratio", "edge_rotational_transform", "edge_magnetic_mirror_ratio",
           "vacuum_well", "flux_compression", "qi"],
}
```

#### 4. Update __init__ to add telemetry list and constraint_names
```python
def __init__(self, ...):
    # ... existing init ...
    problem_key = (cfg.problem or "p3").lower()[:2]
    self.constraint_names = self.CONSTRAINT_NAMES.get(problem_key, self.CONSTRAINT_NAMES["p3"])
    self.telemetry: List[Dict[str, Any]] = []
```

#### 5. Add produce_candidates_aso method
See `docs/ASO_V4_IMPLEMENTATION_GUIDE.md` Section 3.5 for the complete implementation (~200 lines including helper methods).

Key methods to implement:
- `produce_candidates_aso()` - Main entry point
- `_run_trajectory_aso()` - Single trajectory execution
- `_generate_diagnostics()` - Create OptimizerDiagnostics from ALM state
- `_update_trajectory_best()` - Track best values and stagnation
- `_log_telemetry()` - Record events
- `_persist_telemetry()` - Write JSONL file

Helper methods (implement stubs or use existing code):
- `_prepare_seeds()` - Get seeds from ExplorationWorker
- `_seed_to_boundary()` - Convert dict to SurfaceRZFourier
- `_get_problem()` - Get problem instance from config
- `_build_optimization_settings()` - Build OptimizationSettings

### Verification

1. `python -c "from ai_scientist.coordinator import Coordinator, TrajectoryState"`
2. Verify `produce_candidates_aso` method exists

### Do NOT
- Do not modify existing `produce_candidates()` method
- Do not change the constructor signature in a breaking way
```

---

## Task 6: Runner Integration

**File to modify:** `ai_scientist/runner.py`

### Prompt

```
## Task: Add ASO Mode to Runner

**Goal:** Add CLI flag and control flow branching for ASO mode in `ai_scientist/runner.py`.

### Prerequisites
- Tasks 0-5 must be completed

### Files to Read First

1. `ai_scientist/runner.py` - Understand existing structure
2. `docs/ASO_V4_IMPLEMENTATION_GUIDE.md` - Section 4 (Part 4) has specification

### Implementation Requirements

#### 1. Add CLI argument
```python
parser.add_argument(
    "--aso",
    action="store_true",
    help="Enable Agent-Supervised Optimization with real ALM state",
)
```

#### 2. In main(), apply flag to config
```python
if args.aso:
    from dataclasses import replace
    cfg = replace(cfg, aso=replace(cfg.aso, enabled=True))
```

#### 3. In _run_cycle(), add branching logic
```python
# After planning_outcome and active_cfg setup...

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
    # Legacy path (existing code)
    candidates = coordinator.produce_candidates(...)
```

### Verification

1. `python -m ai_scientist.runner --help` should show `--aso` flag
2. Test with: `python -m ai_scientist.runner --aso --dry-run` (if dry-run exists)

### Do NOT
- Do not change the existing produce_candidates() call path
- Do not modify budget calculation logic
- Do not add new dependencies
```

---

## Task 7: Create ALM Bridge Tests

**File to create:** `tests/test_alm_bridge.py`

### Prompt

```
## Task: Create ALM Bridge Tests

**Goal:** Create `tests/test_alm_bridge.py` with unit and integration tests.

### Prerequisites
- Task 0 must be completed

### Files to Read First

1. `ai_scientist/optim/alm_bridge.py` - The module to test
2. `docs/ASO_V4_IMPLEMENTATION_GUIDE.md` - Part 8 has test specifications
3. Existing test files in `tests/` for patterns

### Implementation Requirements

#### 1. API Contract Tests
```python
class TestALMBridgeAPIContract:
    """Tests that verify the constellaration API contract is stable."""

    def test_augmented_lagrangian_state_fields(self):
        """Verify AugmentedLagrangianState has expected fields."""
        import jax.numpy as jnp
        from constellaration.optimization.augmented_lagrangian import AugmentedLagrangianState
        state = AugmentedLagrangianState(
            x=jnp.zeros(2),
            multipliers=jnp.zeros(3),
            penalty_parameters=jnp.ones(3),
            objective=jnp.array(1.0),
            constraints=jnp.zeros(3),
            bounds=jnp.ones(2),
        )
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
        assert params[:6] == ['x', 'scale', 'problem', 'unravel_and_unmask_fn', 'settings', 'aspect_ratio_upper_bound']

    def test_update_state_signature(self):
        """Verify update_augmented_lagrangian_state accepts overrides."""
        import inspect
        from constellaration.optimization.augmented_lagrangian import update_augmented_lagrangian_state
        sig = inspect.signature(update_augmented_lagrangian_state)
        params = sig.parameters
        assert 'penalty_parameters' in params
        assert 'bounds' in params
```

#### 2. Import Tests
```python
def test_alm_bridge_imports():
    """Verify all exports are importable."""
    from ai_scientist.optim.alm_bridge import (
        ALMContext,
        ALMStepResult,
        create_alm_context,
        step_alm,
        state_to_boundary_params,
    )
```

#### 3. Integration Tests (mark as slow)
```python
@pytest.mark.slow
class TestALMBridgeFunctionality:
    """Integration tests that run actual optimization."""

    @pytest.fixture
    def minimal_p3_problem(self):
        from constellaration import problems
        return problems.MHDStableQIStellarator(...)

    def test_create_context_returns_valid_state(self, minimal_p3_problem):
        # ... test create_alm_context returns ALMContext and AugmentedLagrangianState

    def test_step_alm_executes(self, minimal_p3_problem):
        # ... test step_alm runs and returns ALMStepResult
```

### Verification

1. `pytest tests/test_alm_bridge.py -v`
2. `pytest tests/test_alm_bridge.py -v -m "not slow"` for fast tests only

### Do NOT
- Do not test private functions
- Do not create mock-heavy tests that don't test real behavior
```

---

## Task 8: Create Planner ASO Tests

**File to create:** `tests/test_planner_aso.py`

### Prompt

```
## Task: Create Planner ASO Tests

**Goal:** Create `tests/test_planner_aso.py` testing HeuristicSupervisor and supervision methods.

### Prerequisites
- Tasks 2-4 must be completed

### Implementation Requirements

#### 1. HeuristicSupervisor Tests
```python
class TestHeuristicSupervisor:
    @pytest.fixture
    def aso_config(self):
        from ai_scientist.config import ASOConfig
        return ASOConfig()

    @pytest.fixture
    def supervisor(self, aso_config):
        from ai_scientist.planner import HeuristicSupervisor
        return HeuristicSupervisor(aso_config)

    def test_stop_on_feasible_found(self, supervisor):
        """FEASIBLE_FOUND + stable objective -> STOP"""
        diagnostics = create_mock_diagnostics(
            status="FEASIBLE_FOUND",
            objective_delta=1e-7,
            max_violation=1e-4,
        )
        directive = supervisor.analyze(diagnostics)
        assert directive.action == DirectiveAction.STOP

    def test_continue_on_feasible_improving(self, supervisor):
        """FEASIBLE_FOUND + improving -> CONTINUE"""
        diagnostics = create_mock_diagnostics(
            status="FEASIBLE_FOUND",
            objective_delta=-0.1,
            max_violation=1e-4,
        )
        directive = supervisor.analyze(diagnostics)
        assert directive.action == DirectiveAction.CONTINUE

    def test_adjust_on_stagnation_high_violation(self, supervisor):
        """STAGNATION + high violation -> ADJUST"""
        diagnostics = create_mock_diagnostics(
            status="STAGNATION",
            max_violation=0.5,
        )
        directive = supervisor.analyze(diagnostics)
        assert directive.action == DirectiveAction.ADJUST
        assert directive.alm_overrides is not None

    def test_stop_on_diverging(self, supervisor):
        """DIVERGING -> STOP"""
        diagnostics = create_mock_diagnostics(status="DIVERGING")
        directive = supervisor.analyze(diagnostics)
        assert directive.action == DirectiveAction.STOP
```

#### 2. Helper function
```python
def create_mock_diagnostics(**kwargs) -> OptimizerDiagnostics:
    """Create mock diagnostics for testing."""
    defaults = {
        "step": 1,
        "trajectory_id": 0,
        "objective": 1.0,
        "objective_delta": 0.0,
        "max_violation": 0.1,
        "constraints_raw": [0.1, 0.0, 0.0],
        "multipliers": [1.0, 1.0, 1.0],
        "penalty_parameters": [1.0, 1.0, 1.0],
        "bounds_norm": 1.0,
        "status": "IN_PROGRESS",
        "constraint_diagnostics": [
            ConstraintDiagnostic(name="c1", violation=0.1, penalty=1.0, multiplier=1.0, trend="stable"),
            ConstraintDiagnostic(name="c2", violation=0.0, penalty=1.0, multiplier=1.0, trend="stable"),
            ConstraintDiagnostic(name="c3", violation=0.0, penalty=1.0, multiplier=1.0, trend="stable"),
        ],
        "narrative": ["test"],
        "steps_since_improvement": 0,
    }
    defaults.update(kwargs)
    return OptimizerDiagnostics(**defaults)
```

### Verification

1. `pytest tests/test_planner_aso.py -v`

### Do NOT
- Do not test LLM supervision (requires mocking API calls)
- Do not test the full PlanningAgent constructor
```

---

## Task 9: Create Integration Test

**File to create:** `tests/test_aso_integration.py`

### Prompt

```
## Task: Create ASO Integration Test

**Goal:** Create end-to-end integration test that runs a short ASO loop.

### Prerequisites
- All previous tasks (0-8) must be completed

### Implementation Requirements

```python
@pytest.mark.slow
@pytest.mark.integration
class TestASOIntegration:
    """End-to-end integration tests for ASO."""

    def test_three_step_aso_loop(self):
        """Run 3-step ASO loop with toy problem."""
        # 1. Create minimal config
        # 2. Create mock/minimal boundary
        # 3. Create Coordinator with ASO enabled
        # 4. Call produce_candidates_aso with small budget
        # 5. Verify candidates returned
        # 6. Verify telemetry file created
        pass

    def test_aso_respects_stop_directive(self):
        """Verify ASO stops when supervisor says STOP."""
        # ... test early termination
        pass
```

### Verification

1. `pytest tests/test_aso_integration.py -v -m integration`
```

---

## Task 10: Documentation Updates

**Files to modify:** Various docs

### Prompt

```
## Task: Update Documentation

**Goal:** Update run protocol docs and archive superseded planning documents.

### Implementation Requirements

#### 1. Update `docs/run_protocol.md` (if exists)
Add section:
```markdown
## ASO Mode (Agent-Supervised Optimization)

Enable with `--aso` flag:
```bash
python -m ai_scientist.runner --aso --experiment-id 1
```

ASO mode uses real ALM state for supervision decisions. See `ASO_V4_IMPLEMENTATION_GUIDE.md` for details.
```

#### 2. Create `docs/archive/` directory and move:
- `UNIFIED_PLAN.md`
- `UNIFIED_RAW.md`
- `ASO_V3_PLAN.md`
- `AI_SCIENTIST_ASO_MIGRATION_PLAN.md`

#### 3. Add deprecation notice to archived files
Add at top of each:
```markdown
> **DEPRECATED:** This document is superseded by `ASO_V4_IMPLEMENTATION_GUIDE.md`
```

### Verification

1. Verify archive directory exists with moved files
2. Verify deprecation notices added
```

---

## Quick Reference: Task Dependencies

```
Task 0 (ALM Bridge) ─────────────────────────────────┐
                                                      │
Task 1 (Config) ──────────────────────────────────────┤
                                                      │
Task 2 (Data Structures) ────────┐                    │
                                 │                    │
Task 3 (HeuristicSupervisor) ────┤                    │
                                 │                    │
Task 4 (PlanningAgent) ──────────┘                    │
                                                      │
Task 5 (Coordinator) ─────────────────────────────────┘
         │
         v
Task 6 (Runner) ──────────────────────────────────────
         │
         v
Tasks 7-9 (Tests) ────────────────────────────────────
         │
         v
Task 10 (Docs) ───────────────────────────────────────
```

**Parallel execution possible:**
- Tasks 0, 1, 2 can run in parallel
- Tasks 3, 4 depend on Task 2
- Task 5 depends on Tasks 0, 1, 3, 4
- Tasks 7, 8, 9 can run after their respective implementation tasks
