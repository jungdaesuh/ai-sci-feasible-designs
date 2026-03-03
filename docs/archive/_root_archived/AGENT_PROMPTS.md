# AI Scientist Improvement — Coding Agent Prompts

**Reference:** `docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md`
**Purpose:** Self-contained prompts for a coding agent to execute each improvement task.

---

## How to Use This Document

Each prompt below is designed to be copy-pasted directly to a coding agent. Prompts include:
- Task ID matching `TODO_AI_SCIENTIST_IMPROVEMENTS.md`
- Full context needed to complete the task
- Acceptance criteria
- Files to read/modify
- Reference patterns from `constellaration/`

---

## 🔴 HIGH PRIORITY — Architecture & Maintainability

---

### Task 1.1: Extract `budget_manager.py` from `runner.py`

```
TASK: Extract budget management logic from runner.py into a dedicated module.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 1.1

CONTEXT:
The file `ai_scientist/runner.py` is 129KB (~3000+ LOC) which makes it difficult to navigate, test, and review. We need to extract budget-related logic into a separate module.

INSTRUCTIONS:

1. Read `ai_scientist/runner.py` and identify all budget-related code:
   - Look for `BudgetConfig`, `AdaptiveBudgetConfig` dataclasses (if defined there)
   - Find `BudgetController` class and `BudgetSnapshot` if they exist
   - Locate budget calculation logic in `_run_cycle` and related functions
   - Find any adaptive budget adjustment code (HV slope, feasibility rate tracking)

2. Read `ai_scientist/config.py` to understand existing budget config structures.

3. Create new file `ai_scientist/budget_manager.py` containing:
   - Move or create `BudgetController` class with methods:
     - `__init__(self, config: ExperimentConfig)`
     - `snapshot(self) -> BudgetSnapshot` - returns current budget state
     - `adjust_for_cycle(self, hv_delta: float, feasibility_rate: float)` - adaptive adjustment
     - `consume(self, n_evals: int)` - track budget usage
   - Create `BudgetSnapshot` dataclass with fields:
     - `screen_evals_per_cycle: int`
     - `promote_top_k: int`
     - `max_high_fidelity_evals_per_cycle: int`
     - `remaining_budget: int`
   - Include any budget-related helper functions

4. Update `ai_scientist/runner.py`:
   - Add import: `from ai_scientist.budget_manager import BudgetController, BudgetSnapshot`
   - Replace inline budget logic with `BudgetController` calls
   - Remove moved code (do not leave duplicates)

5. Update any other files that import budget-related items from runner.py

ACCEPTANCE CRITERIA:
- `budget_manager.py` exists with `BudgetController` and `BudgetSnapshot`
- `runner.py` imports from `budget_manager` and uses its API
- All existing tests pass (run: `pytest tests/`)
- No duplicate code between files
- Type hints preserved on all functions/classes

FILES TO MODIFY:
- CREATE: `ai_scientist/budget_manager.py`
- MODIFY: `ai_scientist/runner.py`
- MODIFY: Any files importing budget items from runner
```

---

### Task 1.2: Extract `fidelity_controller.py` from `runner.py`

```
TASK: Extract fidelity ladder and promotion logic from runner.py into a dedicated module.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 1.2

CONTEXT:
Continuing the runner.py decomposition. Fidelity management (screening, promotion, high-fidelity evaluation) should be a separate concern.

INSTRUCTIONS:

1. Read `ai_scientist/runner.py` and identify fidelity-related code:
   - `_evaluate_stage()` function or similar stage evaluation logic
   - Promotion gating logic (deciding which candidates get promoted)
   - Fidelity ladder configuration handling
   - Stage transition logic

2. Read `ai_scientist/config.py` for `StageGateConfig` or similar.

3. Create new file `ai_scientist/fidelity_controller.py` containing:
   - `FidelityController` class with methods:
     - `__init__(self, config: ExperimentConfig)`
     - `evaluate_stage(self, stage: int, candidates: list, executor) -> list[EvalResult]`
     - `should_promote(self, results: list[EvalResult]) -> list[EvalResult]`
     - `get_promotion_candidates(self, results: list, top_k: int) -> list`
   - Move `_evaluate_stage()` function (rename to method or keep as module function)
   - Include stage gating logic

4. Update `ai_scientist/runner.py`:
   - Add import: `from ai_scientist.fidelity_controller import FidelityController`
   - Replace inline fidelity logic with `FidelityController` calls
   - Remove moved code

ACCEPTANCE CRITERIA:
- `fidelity_controller.py` exists with `FidelityController` class
- `runner.py` uses `FidelityController` for all stage/promotion decisions
- All existing tests pass
- ProcessPoolExecutor/ThreadPoolExecutor logic preserved correctly

FILES TO MODIFY:
- CREATE: `ai_scientist/fidelity_controller.py`
- MODIFY: `ai_scientist/runner.py`
```

---

### Task 1.3: Extract `cycle_executor.py` from `runner.py`

```
TASK: Extract cycle execution logic from runner.py into a dedicated module.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 1.3

CONTEXT:
The `_run_cycle()` function is the core orchestration unit. It should be in its own module for testability.

PREREQUISITES: Complete tasks 1.1 and 1.2 first (budget_manager and fidelity_controller).

INSTRUCTIONS:

1. Read `ai_scientist/runner.py` and locate `_run_cycle()` function:
   - Understand its inputs (config, cycle_index, world_model, etc.)
   - Understand its outputs (cycle results, updated state)
   - Note dependencies on BudgetController, FidelityController (from previous tasks)

2. Create new file `ai_scientist/cycle_executor.py` containing:
   - `CycleExecutor` class with methods:
     - `__init__(self, config, world_model, planner, coordinator, budget_controller, fidelity_controller)`
     - `run_cycle(self, cycle_index: int) -> CycleResult`
   - Or alternatively, a `run_cycle()` function with explicit dependencies
   - Move all cycle-level orchestration logic here
   - Include candidate generation, evaluation, promotion flow

3. Create `CycleResult` dataclass:
   - `cycle_index: int`
   - `candidates_evaluated: int`
   - `candidates_promoted: int`
   - `best_objective: float`
   - `hypervolume: float`
   - `feasibility_rate: float`

4. Update `ai_scientist/runner.py`:
   - Import `CycleExecutor` or `run_cycle`
   - Replace inline `_run_cycle` with calls to the new module

ACCEPTANCE CRITERIA:
- `cycle_executor.py` exists with cycle orchestration logic
- `runner.py` is significantly smaller (target: <500 LOC for core runner)
- All existing tests pass
- Cycle execution behavior unchanged

FILES TO MODIFY:
- CREATE: `ai_scientist/cycle_executor.py`
- MODIFY: `ai_scientist/runner.py`
```

---

### Task 1.4: Create `experiment_runner.py` as Thin Orchestrator

```
TASK: Refactor runner.py to be a thin orchestrator that composes extracted modules.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 1.4

CONTEXT:
After extracting budget_manager, fidelity_controller, and cycle_executor, the main runner should only contain high-level experiment orchestration.

PREREQUISITES: Complete tasks 1.1, 1.2, 1.3 first.

INSTRUCTIONS:

1. Read current `ai_scientist/runner.py` after previous extractions.

2. Create new file `ai_scientist/experiment_runner.py` (or rename existing runner.py):
   - Keep only `run_experiment()` as the main entry point
   - Keep CLI argument parsing if present
   - Keep experiment-level setup (config loading, world model init)
   - Compose: BudgetController, FidelityController, CycleExecutor, Coordinator, Planner

3. Structure should be:
   ```python
   def run_experiment(config: ExperimentConfig) -> ExperimentResult:
       # 1. Initialize components
       world_model = WorldModel(...)
       budget_controller = BudgetController(config)
       fidelity_controller = FidelityController(config)
       planner = PlanningAgent(config, world_model)
       coordinator = Coordinator(config, world_model, planner)
       cycle_executor = CycleExecutor(...)

       # 2. Run cycles
       for cycle in range(config.cycles):
           result = cycle_executor.run_cycle(cycle)
           # ... logging, checkpointing ...

       # 3. Finalize
       return ExperimentResult(...)
   ```

4. Ensure `__main__` entry point works: `python -m ai_scientist.runner`

5. Update `ai_scientist/__init__.py` to export key classes.

ACCEPTANCE CRITERIA:
- Main runner file is <300 LOC
- Clear composition of extracted modules
- CLI still works: `python -m ai_scientist.runner --config ...`
- All tests pass

FILES TO MODIFY:
- MODIFY/RENAME: `ai_scientist/runner.py` → `ai_scientist/experiment_runner.py`
- MODIFY: `ai_scientist/__init__.py`
```

---

### Task 1.5: Split `memory.py` into Submodule

```
TASK: Split memory.py (46KB) into a memory/ submodule with focused files.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 1.5

CONTEXT:
`ai_scientist/memory.py` contains SQLite schema, CRUD operations, and PropertyGraph class. These should be separated for maintainability.

INSTRUCTIONS:

1. Read `ai_scientist/memory.py` completely to understand its structure.

2. Create directory `ai_scientist/memory/` with:

   a. `ai_scientist/memory/__init__.py`:
      - Re-export public API: `WorldModel`, `PropertyGraph`, key functions
      - Maintain backward compatibility for existing imports

   b. `ai_scientist/memory/schema.py`:
      - SQLite table definitions (CREATE TABLE statements)
      - Migration logic if any
      - Schema version constants

   c. `ai_scientist/memory/repository.py`:
      - `WorldModel` class with all CRUD operations
      - Database connection management
      - Query methods

   d. `ai_scientist/memory/graph.py`:
      - `PropertyGraph` class
      - Graph-related utilities

3. Update imports throughout codebase:
   - `from ai_scientist.memory import WorldModel` should still work
   - Internal imports use specific submodules

4. Delete original `ai_scientist/memory.py` after migration.

ACCEPTANCE CRITERIA:
- `ai_scientist/memory/` directory exists with 4 files
- `from ai_scientist.memory import WorldModel` works (backward compatible)
- All tests pass
- No circular imports

FILES TO MODIFY:
- DELETE: `ai_scientist/memory.py`
- CREATE: `ai_scientist/memory/__init__.py`
- CREATE: `ai_scientist/memory/schema.py`
- CREATE: `ai_scientist/memory/repository.py`
- CREATE: `ai_scientist/memory/graph.py`
- MODIFY: Files importing from memory.py
```

---

### Task 1.6: Split `tools.py` into Submodule

```
TASK: Split tools.py (36KB) into a tools/ submodule with focused files.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 1.6

CONTEXT:
`ai_scientist/tools.py` contains physics evaluation, design manipulation, and hypervolume utilities. These are distinct concerns.

INSTRUCTIONS:

1. Read `ai_scientist/tools.py` completely.

2. Create directory `ai_scientist/tools/` with:

   a. `ai_scientist/tools/__init__.py`:
      - Re-export public API for backward compatibility
      - `from ai_scientist.tools import evaluate_p3_set, propose_boundary, ...`

   b. `ai_scientist/tools/evaluation.py`:
      - `evaluate_p3_set()`, `evaluate_p2_set()`, `evaluate_p1_set()`
      - Physics evaluation wrappers
      - Caching logic (`_EVALUATION_CACHE`, `get_cache_stats`)
      - `design_hash()` function

   c. `ai_scientist/tools/design_manipulation.py`:
      - `propose_boundary()` - perturbation-based candidate generation
      - `recombine_designs()` - geometric crossover
      - `normalized_constraint_distance_sampler()` if present
      - Boundary parameter utilities

   d. `ai_scientist/tools/hypervolume.py`:
      - Hypervolume calculation (pymoo integration)
      - Pareto front utilities
      - Multi-objective metrics

3. Update imports throughout codebase.

4. Delete original `ai_scientist/tools.py` after migration.

ACCEPTANCE CRITERIA:
- `ai_scientist/tools/` directory exists with 4 files
- Backward compatible imports work
- All tests pass
- Evaluation caching still works correctly

FILES TO MODIFY:
- DELETE: `ai_scientist/tools.py`
- CREATE: `ai_scientist/tools/__init__.py`
- CREATE: `ai_scientist/tools/evaluation.py`
- CREATE: `ai_scientist/tools/design_manipulation.py`
- CREATE: `ai_scientist/tools/hypervolume.py`
- MODIFY: Files importing from tools.py
```

---

### Task 2.1: Convert `TrajectoryState` to Pydantic

```
TASK: Convert TrajectoryState dataclass to Pydantic BaseModel.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 2.1

CONTEXT:
`constellaration` uses Pydantic for state objects, enabling JSON serialization, validation, and JAX compatibility. We should adopt this pattern.

REFERENCE PATTERN (from constellaration):
```python
# constellaration/optimization/augmented_lagrangian.py
class AugmentedLagrangianState(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    x: jnp.ndarray
    multipliers: jnp.ndarray
    ...
```

INSTRUCTIONS:

1. Read `ai_scientist/coordinator.py` and find `TrajectoryState` class.

2. Read `constellaration/src/constellaration/optimization/augmented_lagrangian.py` for the Pydantic pattern.

3. Convert `TrajectoryState` from dataclass to Pydantic:
   ```python
   import pydantic
   from typing import Any, Optional

   class TrajectoryState(pydantic.BaseModel):
       model_config = pydantic.ConfigDict(
           arbitrary_types_allowed=True,
           frozen=True,  # Immutable like frozen dataclass
       )

       id: int
       seed: dict[str, Any]
       alm_context: Optional["ALMContext"] = None
       alm_state: Optional["AugmentedLagrangianState"] = None
       history: list = pydantic.Field(default_factory=list)
       evals_used: int = 0
       steps: int = 0
       status: str = "active"
       best_objective: float = float("inf")
       best_violation: float = float("inf")
       stagnation_count: int = 0
       budget_used: int = 0
   ```

4. Replace `dataclass` usages:
   - Replace `replace(state, field=value)` with `state.model_copy(update={"field": value})`
   - Replace manual `to_dict()` with `state.model_dump()`

5. Update any code that constructs TrajectoryState.

ACCEPTANCE CRITERIA:
- TrajectoryState is a Pydantic BaseModel
- Immutability preserved (frozen=True or use model_copy)
- All coordinator tests pass
- JSON serialization works via model_dump()

FILES TO MODIFY:
- MODIFY: `ai_scientist/coordinator.py`
```

---

### Task 2.2: Convert `OptimizerDiagnostics` to Pydantic

```
TASK: Convert OptimizerDiagnostics dataclass to Pydantic BaseModel.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 2.2

CONTEXT:
`OptimizerDiagnostics` already has a `to_json()` method. Pydantic provides this automatically via `model_dump_json()`.

INSTRUCTIONS:

1. Read `ai_scientist/planner.py` and find `OptimizerDiagnostics` class.

2. Convert to Pydantic:
   ```python
   class OptimizerDiagnostics(pydantic.BaseModel):
       model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

       step: int
       trajectory_id: int
       objective: float
       objective_delta: float
       max_violation: float
       constraints_raw: list[float]
       multipliers: list[float]
       penalty_parameters: list[float]
       bounds_norm: float
       status: str
       constraint_diagnostics: list["ConstraintDiagnostic"]
       narrative: list[str]
       steps_since_improvement: int = 0

       def requires_llm_supervision(self, aso_config: "ASOConfig") -> bool:
           # Keep existing logic
           ...
   ```

3. Remove manual `to_json()` method - use `model_dump_json()` instead.

4. Also convert `ConstraintDiagnostic` to Pydantic if it's a dataclass.

5. Update callers that use `to_json()` to use `model_dump_json()`.

ACCEPTANCE CRITERIA:
- OptimizerDiagnostics is a Pydantic BaseModel
- ConstraintDiagnostic is a Pydantic BaseModel
- `to_json()` calls replaced with `model_dump_json()`
- All planner tests pass

FILES TO MODIFY:
- MODIFY: `ai_scientist/planner.py`
```

---

### Task 2.3: Convert `OptimizationDirective` to Pydantic

```
TASK: Convert OptimizationDirective dataclass to Pydantic BaseModel.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 2.3

INSTRUCTIONS:

1. Read `ai_scientist/planner.py` and find `OptimizationDirective` class.

2. Convert to Pydantic:
   ```python
   class OptimizationDirective(pydantic.BaseModel):
       action: DirectiveAction
       config_overrides: Optional[dict[str, Any]] = None
       alm_overrides: Optional[dict[str, Any]] = None
       suggested_params: Optional[dict[str, Any]] = None
       reasoning: str = ""
       confidence: float = 1.0
       source: DirectiveSource = DirectiveSource.HEURISTIC
   ```

3. Remove manual `to_dict()` method - use `model_dump()` instead.

4. Ensure `DirectiveAction` and `DirectiveSource` enums work with Pydantic.

5. Update callers to use `model_dump()`.

ACCEPTANCE CRITERIA:
- OptimizationDirective is a Pydantic BaseModel
- Enum fields serialize correctly
- All tests pass

FILES TO MODIFY:
- MODIFY: `ai_scientist/planner.py`
```

---

### Task 2.4: Add Pydantic Pytree Registration

```
TASK: Port pytree registration utilities from constellaration for JAX-Pydantic interop.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 2.4

CONTEXT:
`constellaration/utils/pytree.py` enables Pydantic models to work with JAX transformations (grad, vmap, etc.).

INSTRUCTIONS:

1. Read `constellaration/src/constellaration/utils/pytree.py` completely.

2. Create `ai_scientist/utils/` directory if it doesn't exist.

3. Create `ai_scientist/utils/__init__.py` with exports.

4. Create `ai_scientist/utils/pytree.py` with:
   - `register_pydantic_data` decorator
   - `mask_and_ravel()` function
   - Helper functions for pytree flattening/unflattening
   - Type definitions

5. The key functionality:
   ```python
   from typing import Callable, Tuple
   import jax
   import pydantic

   def register_pydantic_data(cls: type) -> type:
       """Register a Pydantic model as a JAX pytree."""
       # Implementation from constellaration
       ...

   def mask_and_ravel(
       pytree: Any,
       mask: Any,
   ) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], Any]]:
       """Selectively flatten a pytree based on a mask."""
       # Implementation from constellaration
       ...
   ```

6. Add unit tests in `tests/utils/test_pytree.py`.

ACCEPTANCE CRITERIA:
- `ai_scientist/utils/pytree.py` exists with key functions
- Pydantic models can be registered as pytrees
- Unit tests pass
- Can use `mask_and_ravel` on boundary parameters

FILES TO MODIFY:
- CREATE: `ai_scientist/utils/__init__.py`
- CREATE: `ai_scientist/utils/pytree.py`
- CREATE: `tests/utils/test_pytree.py`

REFERENCE FILES TO READ:
- `constellaration/src/constellaration/utils/pytree.py`
```

---

### Task 3.1: Create Centralized Forward Model

```
TASK: Create a centralized forward model orchestrator for physics evaluations.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 3.1

CONTEXT:
`constellaration/forward_model.py` has a clean single entry point for all evaluations. `ai_scientist` evaluation logic is scattered in `tools.py`.

REFERENCE PATTERN:
```python
# constellaration/forward_model.py
def forward_model(boundary, ideal_mhd_parameters, settings)
    -> tuple[ConstellarationMetrics, VmecppWOut]:
    # Single orchestration point
```

INSTRUCTIONS:

1. Read `ai_scientist/tools.py` (or `ai_scientist/tools/evaluation.py` after task 1.6).

2. Read `constellaration/src/constellaration/forward_model.py` for pattern.

3. Create `ai_scientist/forward_model.py`:
   ```python
   from typing import Optional
   import pydantic
   from constellaration.forward_model import forward_model as constellation_forward
   from constellaration.forward_model import ConstellarationMetrics

   class EvaluationResult(pydantic.BaseModel):
       """Result of a physics evaluation."""
       model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

       metrics: ConstellarationMetrics
       objective: float
       constraints: list[float]
       feasibility: float
       is_feasible: bool
       cache_hit: bool = False
       design_hash: str
       evaluation_time_sec: float

   def forward_model(
       boundary: dict,
       settings: "ForwardModelSettings",
       *,
       use_cache: bool = True,
   ) -> EvaluationResult:
       """
       Single entry point for all physics evaluations.

       Handles:
       - Cache lookup
       - Boundary validation
       - Calling constellaration forward_model
       - Result packaging
       """
       # Implementation
       ...

   def forward_model_batch(
       boundaries: list[dict],
       settings: "ForwardModelSettings",
       *,
       n_workers: int = 4,
   ) -> list[EvaluationResult]:
       """Parallel batch evaluation."""
       ...
   ```

4. Include caching logic from existing tools.py.

5. Include design_hash computation.

ACCEPTANCE CRITERIA:
- `ai_scientist/forward_model.py` exists
- `EvaluationResult` Pydantic model defined
- Single entry point `forward_model()` works
- Caching integrated
- Unit tests added

FILES TO MODIFY:
- CREATE: `ai_scientist/forward_model.py`
- CREATE: `tests/test_forward_model.py`
```

---

### Task 3.2: Create `EvaluationResult` Dataclass

```
TASK: Define EvaluationResult Pydantic model for forward model outputs.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 3.2

NOTE: This may be combined with Task 3.1. If 3.1 already created EvaluationResult, this task verifies and enhances it.

INSTRUCTIONS:

1. If Task 3.1 is complete, read `ai_scientist/forward_model.py` and verify `EvaluationResult`.

2. Ensure EvaluationResult has all fields needed by downstream consumers:
   ```python
   class EvaluationResult(pydantic.BaseModel):
       model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

       # Core metrics from constellaration
       metrics: ConstellarationMetrics

       # Optimization-relevant fields
       objective: float
       constraints: list[float]
       constraint_names: list[str]
       feasibility: float  # max(0, max_constraint_violation)
       is_feasible: bool

       # Caching
       cache_hit: bool = False
       design_hash: str

       # Telemetry
       evaluation_time_sec: float
       fidelity: str  # "low", "medium", "high"

       # Optional equilibrium data
       equilibrium_converged: bool = True
       error_message: Optional[str] = None
   ```

3. Add methods if useful:
   - `to_pareto_point() -> tuple[float, float]` for P3
   - `dominates(other: EvaluationResult) -> bool`

4. Add to `ai_scientist/__init__.py` exports.

ACCEPTANCE CRITERIA:
- EvaluationResult is complete and documented
- All fields have type hints
- Serialization works (model_dump, model_dump_json)
- Integrated with forward_model()

FILES TO MODIFY:
- MODIFY: `ai_scientist/forward_model.py`
```

---

### Task 3.3: Migrate Existing Callers to Forward Model

```
TASK: Update runner.py and coordinator.py to use the new centralized forward model.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 3.3

PREREQUISITES: Tasks 3.1 and 3.2 complete.

INSTRUCTIONS:

1. Read `ai_scientist/forward_model.py` to understand the new API.

2. Find all places in `ai_scientist/runner.py` that call:
   - `tools.evaluate_p3_set()`
   - `tools.evaluate_p2_set()`
   - `tools.evaluate_p1_set()`
   - Direct calls to `constellaration.forward_model`

3. Replace with new forward model:
   ```python
   # Before
   from ai_scientist import tools
   results = tools.evaluate_p3_set(candidates, config, use_cache=True)

   # After
   from ai_scientist.forward_model import forward_model_batch
   results = forward_model_batch(candidates, settings, n_workers=4)
   ```

4. Do the same for `ai_scientist/coordinator.py`.

5. Add backward-compatible wrappers in `tools.py` if needed:
   ```python
   # ai_scientist/tools/evaluation.py
   def evaluate_p3_set(*args, **kwargs):
       """Deprecated: Use forward_model_batch instead."""
       import warnings
       warnings.warn("Use forward_model_batch instead", DeprecationWarning)
       from ai_scientist.forward_model import forward_model_batch
       return forward_model_batch(*args, **kwargs)
   ```

ACCEPTANCE CRITERIA:
- runner.py uses forward_model API
- coordinator.py uses forward_model API
- Old tools functions emit deprecation warnings
- All tests pass
- No performance regression

FILES TO MODIFY:
- MODIFY: `ai_scientist/runner.py` (or experiment_runner.py)
- MODIFY: `ai_scientist/coordinator.py`
- MODIFY: `ai_scientist/tools/evaluation.py` (add deprecation wrappers)
```

---

## 🟡 MEDIUM PRIORITY — Developer Experience

---

### Task 4.1: Add Factory Methods to `ExperimentConfig`

```
TASK: Add factory methods to ExperimentConfig for common presets.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 4.1

CONTEXT:
`constellaration` uses factory methods like `default_high_fidelity()` for preset configurations.

REFERENCE PATTERN:
```python
# constellaration pattern
class ConstellarationSettings:
    @staticmethod
    def default_high_fidelity() -> "ConstellarationSettings":
        return ConstellarationSettings(
            vmec_preset_settings=VmecPresetSettings(fidelity="high_fidelity")
        )
```

INSTRUCTIONS:

1. Read `ai_scientist/config.py` and find `ExperimentConfig` class.

2. Add static factory methods:
   ```python
   @dataclass
   class ExperimentConfig:
       # ... existing fields ...

       @staticmethod
       def p3_high_fidelity() -> "ExperimentConfig":
           """Production config for P3 with high fidelity physics."""
           return ExperimentConfig(
               problem="p3",
               cycles=10,
               aso=ASOConfig(enabled=True, supervision_mode="event_triggered"),
               surrogate=SurrogateConfig(backend="neural_operator"),
               budgets=BudgetConfig(
                   screen_evals_per_cycle=50,
                   promote_top_k=5,
                   max_high_fidelity_evals_per_cycle=3,
               ),
           )

       @staticmethod
       def p3_quick_validation() -> "ExperimentConfig":
           """Fast config for testing/CI."""
           return ExperimentConfig(
               problem="p3",
               cycles=2,
               aso=ASOConfig(enabled=False),
               budgets=BudgetConfig(
                   screen_evals_per_cycle=5,
                   promote_top_k=2,
                   max_high_fidelity_evals_per_cycle=1,
               ),
           )

       @staticmethod
       def p3_aso_enabled() -> "ExperimentConfig":
           """Config with Agent-Supervised Optimization."""
           return ExperimentConfig(
               problem="p3",
               cycles=5,
               aso=ASOConfig(
                   enabled=True,
                   supervision_mode="event_triggered",
                   max_stagnation_steps=5,
               ),
           )
   ```

3. Add docstrings explaining when to use each preset.

4. Add unit tests for factory methods.

ACCEPTANCE CRITERIA:
- At least 3 factory methods on ExperimentConfig
- Methods return valid, complete configs
- Unit tests verify factory output
- Docstrings explain use cases

FILES TO MODIFY:
- MODIFY: `ai_scientist/config.py`
- CREATE/MODIFY: `tests/test_config.py`
```

---

### Task 4.2: Add Factory Methods to `ASOConfig`

```
TASK: Add factory methods to ASOConfig for common presets.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 4.2

INSTRUCTIONS:

1. Read `ai_scientist/config.py` and find `ASOConfig` class.

2. Add static factory methods:
   ```python
   @dataclass(frozen=True)
   class ASOConfig:
       # ... existing fields ...

       @staticmethod
       def default_event_triggered() -> "ASOConfig":
           """ASO with event-triggered supervision (recommended)."""
           return ASOConfig(
               enabled=True,
               supervision_mode="event_triggered",
               max_stagnation_steps=5,
               use_heuristic_fallback=True,
           )

       @staticmethod
       def default_periodic(interval: int = 5) -> "ASOConfig":
           """ASO with periodic LLM supervision."""
           return ASOConfig(
               enabled=True,
               supervision_mode="periodic",
               supervision_interval=interval,
           )

       @staticmethod
       def disabled() -> "ASOConfig":
           """ASO disabled (legacy mode)."""
           return ASOConfig(enabled=False)
   ```

ACCEPTANCE CRITERIA:
- 3 factory methods on ASOConfig
- Unit tests for each factory
- Docstrings explain differences

FILES TO MODIFY:
- MODIFY: `ai_scientist/config.py`
```

---

### Task 4.3: Add Factory Methods to `ALMConfig`

```
TASK: Add factory methods to ALMConfig for common presets.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 4.3

INSTRUCTIONS:

1. Read `ai_scientist/config.py` and find `ALMConfig` class.

2. Add static factory methods:
   ```python
   @dataclass(frozen=True)
   class ALMConfig:
       # ... existing fields ...

       @staticmethod
       def default() -> "ALMConfig":
           """Standard ALM settings."""
           return ALMConfig()  # Uses default values

       @staticmethod
       def aggressive_penalties() -> "ALMConfig":
           """Faster constraint satisfaction, may sacrifice objective."""
           return ALMConfig(
               penalty_parameters_increase_factor=4.0,
               penalty_parameters_initial=10.0,
           )

       @staticmethod
       def conservative() -> "ALMConfig":
           """Slower convergence, better objective quality."""
           return ALMConfig(
               penalty_parameters_increase_factor=1.5,
               bounds_reduction_factor=0.98,
               maxit=50,
           )
   ```

ACCEPTANCE CRITERIA:
- 3 factory methods on ALMConfig
- Unit tests verify settings
- Docstrings explain trade-offs

FILES TO MODIFY:
- MODIFY: `ai_scientist/config.py`
```

---

### Task 4.4: Document Factory Methods in CLI Help

```
TASK: Add --preset flag to runner CLI that maps to factory methods.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 4.4

INSTRUCTIONS:

1. Read the CLI argument parsing in `ai_scientist/runner.py`.

2. Add `--preset` argument:
   ```python
   parser.add_argument(
       "--preset",
       choices=["p3-high-fidelity", "p3-quick", "p3-aso"],
       help="Use a predefined configuration preset. Overrides --config.",
   )
   ```

3. Implement preset loading:
   ```python
   PRESET_MAP = {
       "p3-high-fidelity": ExperimentConfig.p3_high_fidelity,
       "p3-quick": ExperimentConfig.p3_quick_validation,
       "p3-aso": ExperimentConfig.p3_aso_enabled,
   }

   if args.preset:
       config = PRESET_MAP[args.preset]()
   elif args.config:
       config = ExperimentConfig.from_yaml(args.config)
   ```

4. Update `--help` output to list available presets with descriptions.

ACCEPTANCE CRITERIA:
- `--preset` flag works
- `python -m ai_scientist.runner --preset p3-quick` runs
- `--help` shows preset descriptions
- Presets override YAML if both specified

FILES TO MODIFY:
- MODIFY: `ai_scientist/runner.py`
```

---

### Task 5.1: Port `mask_and_ravel()` Utility

```
TASK: Port mask_and_ravel() from constellaration for selective parameter flattening.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 5.1

NOTE: May overlap with Task 2.4. If 2.4 is complete, verify and enhance.

INSTRUCTIONS:

1. Read `constellaration/src/constellaration/utils/pytree.py` focusing on `mask_and_ravel`.

2. Create/update `ai_scientist/utils/pytree.py` with:
   ```python
   from typing import Any, Callable, Tuple
   import jax.numpy as jnp
   import numpy as np

   NpOrJaxArray = np.ndarray | jnp.ndarray

   def mask_and_ravel(
       pytree: Any,
       mask: Any,
   ) -> Tuple[NpOrJaxArray, Callable[[NpOrJaxArray], Any]]:
       """
       Selectively flatten a pytree based on a mask.

       Args:
           pytree: Nested structure (dict, Pydantic model, etc.)
           mask: Boolean mask with same structure as pytree

       Returns:
           Tuple of (flattened_array, unravel_function)
           The unravel function reconstructs the original structure.
       """
       # Port implementation from constellaration
       ...
   ```

3. Add comprehensive unit tests.

ACCEPTANCE CRITERIA:
- `mask_and_ravel` works on dicts and Pydantic models
- Unravel function correctly reconstructs original
- Works with JAX arrays and NumPy arrays
- Unit tests cover edge cases

FILES TO MODIFY:
- MODIFY: `ai_scientist/utils/pytree.py`
- MODIFY: `tests/utils/test_pytree.py`

REFERENCE:
- `constellaration/src/constellaration/utils/pytree.py`
```

---

### Task 5.2: Port `register_pydantic_data` Decorator

```
TASK: Port register_pydantic_data decorator for JAX pytree compatibility.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 5.2

NOTE: May overlap with Task 2.4. Verify and enhance.

INSTRUCTIONS:

1. Read `constellaration/src/constellaration/utils/pytree.py` focusing on `register_pydantic_data`.

2. Implement in `ai_scientist/utils/pytree.py`:
   ```python
   import jax
   import pydantic

   def register_pydantic_data(cls: type) -> type:
       """
       Register a Pydantic model as a JAX pytree node.

       This enables using Pydantic models with JAX transformations
       like jax.grad, jax.vmap, etc.

       Usage:
           @register_pydantic_data
           class MyState(pydantic.BaseModel):
               x: jnp.ndarray
               y: float
       """
       def flatten(obj):
           # Extract children (array fields) and aux data (non-array fields)
           ...

       def unflatten(aux_data, children):
           # Reconstruct object
           ...

       jax.tree_util.register_pytree_node(cls, flatten, unflatten)
       return cls
   ```

3. Test with the converted Pydantic state classes (TrajectoryState, etc.)

ACCEPTANCE CRITERIA:
- Decorator works on Pydantic models
- Registered models work with jax.grad
- Works with model_copy() for updates
- Unit tests with simple JAX operations

FILES TO MODIFY:
- MODIFY: `ai_scientist/utils/pytree.py`
- MODIFY: `tests/utils/test_pytree.py`
```

---

### Task 5.3: Update `optim/differentiable.py` to Use Pytree Utils

```
TASK: Refactor differentiable.py to use mask_and_ravel instead of manual flattening.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 5.3

PREREQUISITES: Tasks 5.1 and 5.2 complete.

INSTRUCTIONS:

1. Read `ai_scientist/optim/differentiable.py`.

2. Find manual parameter flattening/unflattening code.

3. Replace with pytree utilities:
   ```python
   from ai_scientist.utils.pytree import mask_and_ravel

   def gradient_descent_on_inputs(candidates, surrogate, config):
       # Before: manual flattening
       # flat_params = np.concatenate([c['r_cos'].ravel(), c['z_sin'].ravel()])

       # After: use mask_and_ravel
       mask = build_optimization_mask(candidates[0])
       flat_params, unravel = mask_and_ravel(candidates[0], mask)

       # Optimize
       optimized_flat = optimizer.run(flat_params)

       # Reconstruct
       optimized_candidate = unravel(optimized_flat)
   ```

4. Ensure JAX gradients still work correctly.

5. Update tests.

ACCEPTANCE CRITERIA:
- Manual flattening replaced with mask_and_ravel
- Gradient computation still works
- Tests pass
- Code is cleaner and more maintainable

FILES TO MODIFY:
- MODIFY: `ai_scientist/optim/differentiable.py`
```

---

### Task 6.1: Create Problem ABC in ai_scientist

```
TASK: Create Problem abstract base class following constellaration pattern.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 6.1

CONTEXT:
`constellaration/problems.py` has a clean ABC pattern with template methods for feasibility.

REFERENCE PATTERN:
```python
# constellaration/problems.py
class _Problem(abc.ABC):
    @abc.abstractmethod
    def _normalized_constraint_violations(self, metrics) -> np.ndarray:
        pass

    def is_feasible(self, metrics) -> bool:
        violations = self._normalized_constraint_violations(metrics)
        return np.all(violations <= 0)
```

INSTRUCTIONS:

1. Read `constellaration/src/constellaration/problems.py` for the pattern.

2. Read `ai_scientist/prompts.py` to understand existing `ProblemSpec`.

3. Create `ai_scientist/problems.py`:
   ```python
   from abc import ABC, abstractmethod
   from typing import Any
   import numpy as np

   class Problem(ABC):
       """Abstract base class for optimization problems."""

       @property
       @abstractmethod
       def name(self) -> str:
           """Problem identifier (p1, p2, p3)."""
           ...

       @property
       @abstractmethod
       def constraint_names(self) -> list[str]:
           """Names of constraints for this problem."""
           ...

       @abstractmethod
       def _normalized_constraint_violations(
           self, metrics: dict[str, Any]
       ) -> np.ndarray:
           """
           Compute normalized constraint violations.

           Returns array where positive values indicate violation.
           """
           ...

       @abstractmethod
       def get_objective(self, metrics: dict[str, Any]) -> float:
           """Extract objective value from metrics."""
           ...

       # Template methods
       def is_feasible(self, metrics: dict[str, Any]) -> bool:
           """Check if metrics satisfy all constraints."""
           violations = self._normalized_constraint_violations(metrics)
           return bool(np.all(violations <= 0))

       def compute_feasibility(self, metrics: dict[str, Any]) -> float:
           """Compute continuous feasibility metric (0 = feasible)."""
           violations = self._normalized_constraint_violations(metrics)
           return float(np.sum(np.maximum(violations, 0)))

       def max_violation(self, metrics: dict[str, Any]) -> float:
           """Get maximum constraint violation."""
           violations = self._normalized_constraint_violations(metrics)
           return float(np.max(np.maximum(violations, 0)))
   ```

4. Add to `ai_scientist/__init__.py` exports.

ACCEPTANCE CRITERIA:
- Problem ABC exists with abstract and template methods
- Matches constellaration pattern
- Well-documented
- Unit tests for template methods

FILES TO MODIFY:
- CREATE: `ai_scientist/problems.py`
- MODIFY: `ai_scientist/__init__.py`
- CREATE: `tests/test_problems.py`
```

---

### Task 6.2: Implement P1Problem, P2Problem, P3Problem

```
TASK: Implement concrete Problem classes for each problem type.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 6.2

PREREQUISITES: Task 6.1 complete.

INSTRUCTIONS:

1. Read `constellaration/src/constellaration/problems.py` for constraint specifications.

2. Read `ai_scientist/prompts.py` for existing constraint info.

3. Add to `ai_scientist/problems.py`:
   ```python
   class P1Problem(Problem):
       """Geometrical Problem - minimize elongation."""

       @property
       def name(self) -> str:
           return "p1"

       @property
       def constraint_names(self) -> list[str]:
           return ["aspect_ratio", "average_triangularity", "edge_rotational_transform"]

       def _normalized_constraint_violations(self, metrics: dict) -> np.ndarray:
           # Constraint: aspect_ratio == 4.0 (tolerance)
           # Constraint: average_triangularity == -0.6 (tolerance)
           # Constraint: edge_rotational_transform_over_nfp == 0.3 (tolerance)
           violations = np.array([
               abs(metrics.get("aspect_ratio", 0) - 4.0) - 0.1,  # tolerance 0.1
               abs(metrics.get("average_triangularity", 0) + 0.6) - 0.1,
               abs(metrics.get("edge_rotational_transform_over_nfp", 0) - 0.3) - 0.05,
           ])
           return violations

       def get_objective(self, metrics: dict) -> float:
           return metrics.get("max_elongation", float("inf"))


   class P2Problem(Problem):
       """Simple-to-Build QI Stellarator."""
       # Similar implementation for P2 constraints
       ...


   class P3Problem(Problem):
       """MHD-Stable QI Stellarator (multi-objective)."""
       # Similar implementation for P3 constraints
       ...


   # Factory function
   def get_problem(name: str) -> Problem:
       """Get problem instance by name."""
       problems = {
           "p1": P1Problem(),
           "p2": P2Problem(),
           "p3": P3Problem(),
       }
       return problems[name.lower()]
   ```

4. Match constraint thresholds with constellaration definitions.

ACCEPTANCE CRITERIA:
- All three problem classes implemented
- Constraints match constellaration definitions
- `get_problem()` factory works
- Unit tests for each problem

FILES TO MODIFY:
- MODIFY: `ai_scientist/problems.py`
- MODIFY: `tests/test_problems.py`
```

---

### Task 6.3: Integrate Problems with `prompts.ProblemSpec`

```
TASK: Connect new Problem classes with existing ProblemSpec and coordinator.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 6.3

PREREQUISITES: Tasks 6.1 and 6.2 complete.

INSTRUCTIONS:

1. Read `ai_scientist/prompts.py` to understand `ProblemSpec`.

2. Read `ai_scientist/coordinator.py` for `_generate_diagnostics()`.

3. Update coordinator to use Problem classes:
   ```python
   from ai_scientist.problems import get_problem, Problem

   class Coordinator:
       def __init__(self, cfg, ...):
           self.problem: Problem = get_problem(cfg.problem or "p3")

       def _generate_diagnostics(self, alm_state, traj):
           # Use self.problem.constraint_names instead of hardcoded
           constraint_names = self.problem.constraint_names

           # Use self.problem for feasibility checks
           is_feasible = self.problem.is_feasible(metrics)
   ```

4. Update `prompts.py` if needed to align with Problem classes.

5. Ensure backward compatibility - existing code still works.

ACCEPTANCE CRITERIA:
- Coordinator uses Problem classes
- constraint_names comes from Problem
- Feasibility computed via Problem.is_feasible()
- All tests pass

FILES TO MODIFY:
- MODIFY: `ai_scientist/coordinator.py`
- MODIFY: `ai_scientist/prompts.py` (if needed)
```

---

### Task 7.1: Add jaxtyping to `optim/surrogate_v2.py`

```
TASK: Add jaxtyping shape annotations to surrogate module.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 7.1

CONTEXT:
`constellaration` uses explicit tensor shape annotations for readability.

INSTRUCTIONS:

1. Add jaxtyping import to `ai_scientist/optim/surrogate_v2.py`:
   ```python
   from jaxtyping import Float, Int
   import numpy as np
   import torch
   ```

2. Add shape annotations to key functions:
   ```python
   def forward(
       self,
       spectral_input: Float[torch.Tensor, "batch n_modes 2"],
       geometric_input: Float[torch.Tensor, "batch n_points 3"],
   ) -> Float[torch.Tensor, "batch n_outputs"]:
       ...

   def predict(
       self,
       boundaries: list[dict],
   ) -> tuple[
       Float[np.ndarray, "n_boundaries n_objectives"],
       Float[np.ndarray, "n_boundaries n_objectives"],  # uncertainty
   ]:
       ...
   ```

3. Document shape semantics in docstrings.

ACCEPTANCE CRITERIA:
- All public functions have shape annotations
- Shapes documented in docstrings
- Code still runs (jaxtyping is runtime-checked)

FILES TO MODIFY:
- MODIFY: `ai_scientist/optim/surrogate_v2.py`
```

---

### Task 7.2: Add jaxtyping to `optim/geometry.py`

```
TASK: Add jaxtyping shape annotations to geometry module.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 7.2

INSTRUCTIONS:

1. Add jaxtyping to `ai_scientist/optim/geometry.py`:
   ```python
   from jaxtyping import Float
   import numpy as np

   def fourier_to_real_space(
       r_cos: Float[np.ndarray, "n_poloidal n_toroidal"],
       z_sin: Float[np.ndarray, "n_poloidal n_toroidal"],
       theta: Float[np.ndarray, "n_theta"],
       phi: Float[np.ndarray, "n_phi"],
   ) -> tuple[
       Float[np.ndarray, "n_theta n_phi"],  # R
       Float[np.ndarray, "n_theta n_phi"],  # Z
   ]:
       ...

   def aspect_ratio(
       R: Float[np.ndarray, "n_theta n_phi"],
       Z: Float[np.ndarray, "n_theta n_phi"],
   ) -> float:
       ...
   ```

ACCEPTANCE CRITERIA:
- Shape annotations on all geometry functions
- Docstrings explain coordinate systems
- Tests still pass

FILES TO MODIFY:
- MODIFY: `ai_scientist/optim/geometry.py`
```

---

### Task 7.3: Add jaxtyping to `optim/alm_bridge.py`

```
TASK: Add jaxtyping shape annotations to ALM bridge module.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 7.3

INSTRUCTIONS:

1. Add jaxtyping to `ai_scientist/optim/alm_bridge.py`:
   ```python
   from jaxtyping import Float
   import jax.numpy as jnp

   @dataclass
   class ALMStepResult:
       state: AugmentedLagrangianState
       n_evals: int
       objective: float
       max_violation: float
       metrics: Optional[ConstellarationMetrics]

   def step_alm(
       context: ALMContext,
       state: AugmentedLagrangianState,
       budget: int,
       *,
       penalty_override: Optional[Float[jnp.ndarray, "n_constraints"]] = None,
       bounds_override: Optional[Float[jnp.ndarray, "n_params"]] = None,
   ) -> ALMStepResult:
       ...
   ```

ACCEPTANCE CRITERIA:
- Shape annotations on ALM functions
- Constraint and parameter dimensions clear
- Tests pass

FILES TO MODIFY:
- MODIFY: `ai_scientist/optim/alm_bridge.py`
```

---

### Task 7.4: Add jaxtyping to `coordinator.py`

```
TASK: Add jaxtyping shape annotations to coordinator module.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 7.4

INSTRUCTIONS:

1. Add jaxtyping to `ai_scientist/coordinator.py`:
   ```python
   from jaxtyping import Float

   def _generate_diagnostics(
       self,
       alm_state: AugmentedLagrangianState,
       traj: TrajectoryState,
   ) -> OptimizerDiagnostics:
       # Document that:
       # alm_state.constraints: Float[jnp.ndarray, "n_constraints"]
       # alm_state.multipliers: Float[jnp.ndarray, "n_constraints"]
       ...
   ```

ACCEPTANCE CRITERIA:
- Diagnostic arrays have shape annotations
- Code remains readable
- Tests pass

FILES TO MODIFY:
- MODIFY: `ai_scientist/coordinator.py`
```

---

## 🟢 LOW PRIORITY — Polish & Tooling

---

### Task 8.1: Create `.pre-commit-config.yaml`

```
TASK: Create pre-commit configuration with Ruff, Black, isort, Pyright.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 8.1

CONTEXT:
`constellaration` has comprehensive pre-commit hooks for code quality.

INSTRUCTIONS:

1. Read `constellaration/.pre-commit-config.yaml` for reference.

2. Create `.pre-commit-config.yaml` in repo root:
   ```yaml
   repos:
     - repo: https://github.com/pre-commit/pre-commit-hooks
       rev: v4.5.0
       hooks:
         - id: trailing-whitespace
         - id: end-of-file-fixer
         - id: check-yaml
         - id: check-json
         - id: check-added-large-files

     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.1.9
       hooks:
         - id: ruff
           args: [--fix, --line-length=88]
         - id: ruff-format

     - repo: https://github.com/pycqa/isort
       rev: 5.13.2
       hooks:
         - id: isort
           args: [--profile=black, --line-length=88]

     - repo: https://github.com/RobertCraiworthy/pyright-action
       rev: v1.1.344
       hooks:
         - id: pyright
           additional_dependencies: [pyright]
   ```

3. Create/update `pyproject.toml` with tool configs:
   ```toml
   [tool.ruff]
   line-length = 88
   target-version = "py310"

   [tool.isort]
   profile = "black"
   line_length = 88

   [tool.pyright]
   pythonVersion = "3.10"
   typeCheckingMode = "basic"
   ```

4. Test locally: `pre-commit run --all-files`

ACCEPTANCE CRITERIA:
- `.pre-commit-config.yaml` exists
- All hooks configured
- `pre-commit run --all-files` succeeds (may have findings)

FILES TO MODIFY:
- CREATE: `.pre-commit-config.yaml`
- MODIFY: `pyproject.toml`
```

---

### Task 8.2: Run Initial Formatting Pass

```
TASK: Apply pre-commit hooks to format all ai_scientist/ files.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 8.2

PREREQUISITES: Task 8.1 complete.

INSTRUCTIONS:

1. Install pre-commit: `pip install pre-commit`

2. Install hooks: `pre-commit install`

3. Run on all files: `pre-commit run --all-files`

4. Fix any issues that auto-fix doesn't handle.

5. Commit all formatting changes as a single commit:
   ```
   git add -A
   git commit -m "style: Apply pre-commit formatting to ai_scientist/"
   ```

6. Do NOT mix formatting with other changes.

ACCEPTANCE CRITERIA:
- All files pass pre-commit checks
- Single formatting commit
- No functional changes in this commit

FILES TO MODIFY:
- All files in `ai_scientist/`
```

---

### Task 8.3: Add Pre-commit to CI Workflow

```
TASK: Add pre-commit checks to GitHub Actions CI.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 8.3

INSTRUCTIONS:

1. Read existing `.github/workflows/` files.

2. Add or update CI workflow:
   ```yaml
   # .github/workflows/ci.yml
   name: CI

   on: [push, pull_request]

   jobs:
     lint:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: "3.10"
         - name: Install pre-commit
           run: pip install pre-commit
         - name: Run pre-commit
           run: pre-commit run --all-files

     test:
       runs-on: ubuntu-latest
       needs: lint
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: "3.10"
         - name: Install dependencies
           run: pip install -e ".[test]"
         - name: Run tests
           run: pytest tests/
   ```

ACCEPTANCE CRITERIA:
- CI runs pre-commit checks
- PRs fail if formatting issues
- Tests run after lint passes

FILES TO MODIFY:
- CREATE/MODIFY: `.github/workflows/ci.yml`
```

---

### Task 9.1: Reorganize Test Directory Structure

```
TASK: Reorganize tests/ to mirror ai_scientist/ source structure.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 9.1

INSTRUCTIONS:

1. Current structure:
   ```
   tests/
   ├── test_coordinator_aso.py
   ├── test_planner_aso.py
   ├── test_tools_*.py
   ├── test_alm_bridge.py
   └── ...
   ```

2. Target structure:
   ```
   tests/
   ├── optim/
   │   ├── __init__.py
   │   ├── test_surrogate_v2.py
   │   ├── test_alm_bridge.py
   │   ├── test_generative.py
   │   ├── test_geometry.py
   │   └── test_differentiable.py
   ├── tools/
   │   ├── __init__.py
   │   ├── test_evaluation.py
   │   └── test_design_manipulation.py
   ├── utils/
   │   ├── __init__.py
   │   └── test_pytree.py
   ├── test_coordinator.py
   ├── test_planner.py
   ├── test_runner.py
   ├── test_config.py
   └── conftest.py
   ```

3. Move files to appropriate subdirectories.

4. Update imports in moved tests.

5. Verify pytest discovers all tests: `pytest tests/ --collect-only`

ACCEPTANCE CRITERIA:
- Test structure mirrors source structure
- All tests discoverable by pytest
- All tests pass

FILES TO MODIFY:
- MOVE: Various test files to subdirectories
- CREATE: `tests/optim/__init__.py`, `tests/tools/__init__.py`, etc.
```

---

### Task 9.2: Update Pytest Configuration

```
TASK: Ensure pytest configuration works with new test structure.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 9.2

INSTRUCTIONS:

1. Update `pyproject.toml` pytest configuration:
   ```toml
   [tool.pytest.ini_options]
   testpaths = ["tests"]
   python_files = ["test_*.py"]
   python_classes = ["Test*"]
   python_functions = ["test_*"]
   addopts = "-v --tb=short"
   markers = [
       "slow: marks tests as slow (deselect with '-m \"not slow\"')",
       "integration: marks integration tests",
   ]
   ```

2. Ensure `conftest.py` fixtures are discovered.

3. Run full test suite: `pytest tests/ -v`

ACCEPTANCE CRITERIA:
- `pytest tests/` runs all tests
- `pytest tests/optim/` runs only optim tests
- Markers work for filtering

FILES TO MODIFY:
- MODIFY: `pyproject.toml`
- MODIFY: `tests/conftest.py` (if needed)
```

---

### Task 10.1: Add Hypothesis Tests for ALM Invariants

```
TASK: Add property-based tests for ALM mathematical invariants.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 10.1

INSTRUCTIONS:

1. Install hypothesis: `pip install hypothesis`

2. Add to `tests/test_alm_bridge.py` (or `tests/optim/test_alm_bridge.py`):
   ```python
   import hypothesis
   from hypothesis import strategies as st
   import numpy as np

   @hypothesis.given(
       multipliers=st.lists(
           st.floats(0.0, 100.0, allow_nan=False),
           min_size=3, max_size=5
       ),
       violations=st.lists(
           st.floats(-1.0, 1.0, allow_nan=False),
           min_size=3, max_size=5
       ),
   )
   def test_multiplier_update_non_negative(multipliers, violations):
       """Lagrange multipliers should always be non-negative."""
       # Ensure same length
       min_len = min(len(multipliers), len(violations))
       multipliers = np.array(multipliers[:min_len])
       violations = np.array(violations[:min_len])

       # Simulated update: λ ← max(0, λ + ρ * g)
       rho = 1.0
       new_multipliers = np.maximum(0, multipliers + rho * violations)

       assert np.all(new_multipliers >= 0)

   @hypothesis.given(
       penalties=st.lists(
           st.floats(1.0, 1e6, allow_nan=False),
           min_size=3, max_size=5
       ),
   )
   def test_penalties_bounded(penalties):
       """Penalties should be bounded by max."""
       PENALTY_MAX = 1e8
       penalties = np.array(penalties)
       bounded = np.minimum(penalties * 2.0, PENALTY_MAX)
       assert np.all(bounded <= PENALTY_MAX)
   ```

ACCEPTANCE CRITERIA:
- Hypothesis tests added
- Tests pass with multiple seeds
- Edge cases covered

FILES TO MODIFY:
- MODIFY: `tests/test_alm_bridge.py` or `tests/optim/test_alm_bridge.py`
```

---

### Task 10.2: Add Hypothesis Tests for Geometry Functions

```
TASK: Add property-based tests for geometry mathematical invariants.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 10.2

INSTRUCTIONS:

1. Add to `tests/optim/test_geometry.py`:
   ```python
   import hypothesis
   from hypothesis import strategies as st
   import numpy as np

   @hypothesis.given(
       R_major=st.floats(0.5, 10.0),
       R_minor=st.floats(0.1, 2.0),
   )
   def test_aspect_ratio_positive(R_major, R_minor):
       """Aspect ratio should always be positive."""
       hypothesis.assume(R_minor < R_major)
       aspect = R_major / R_minor
       assert aspect > 0

   @hypothesis.given(
       coeffs=st.lists(
           st.floats(-1.0, 1.0, allow_nan=False),
           min_size=4, max_size=16
       ),
   )
   @hypothesis.settings(max_examples=50)
   def test_fourier_to_real_smooth(coeffs):
       """Real-space representation should be smooth (no NaN/inf)."""
       # Create small coefficient array
       n = int(np.sqrt(len(coeffs)))
       r_cos = np.array(coeffs[:n*n]).reshape(n, n)
       z_sin = np.array(coeffs[:n*n]).reshape(n, n)

       theta = np.linspace(0, 2*np.pi, 32)
       phi = np.linspace(0, 2*np.pi, 32)

       # Would call fourier_to_real_space here
       # R, Z = fourier_to_real_space(r_cos, z_sin, theta, phi)
       # assert np.all(np.isfinite(R))
       # assert np.all(np.isfinite(Z))
   ```

ACCEPTANCE CRITERIA:
- Property tests for geometry functions
- Tests verify mathematical properties
- No numerical instabilities

FILES TO MODIFY:
- CREATE/MODIFY: `tests/optim/test_geometry.py`
```

---

### Task 10.3: Add Hypothesis Tests for Surrogate Predictions

```
TASK: Add property-based tests for surrogate ensemble properties.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Task 10.3

INSTRUCTIONS:

1. Add to `tests/optim/test_surrogate.py`:
   ```python
   import hypothesis
   from hypothesis import strategies as st
   import numpy as np

   @hypothesis.given(
       predictions=st.lists(
           st.lists(st.floats(-10, 10, allow_nan=False), min_size=3, max_size=3),
           min_size=2, max_size=5
       ),
   )
   def test_ensemble_uncertainty_positive(predictions):
       """Ensemble uncertainty should be non-negative."""
       predictions = np.array(predictions)  # shape: (n_models, n_outputs)
       mean = predictions.mean(axis=0)
       variance = predictions.var(axis=0)
       std = np.sqrt(variance)

       assert np.all(std >= 0)

   @hypothesis.given(
       predictions=st.lists(
           st.lists(st.floats(-10, 10, allow_nan=False), min_size=3, max_size=3),
           min_size=3, max_size=10
       ),
   )
   def test_ensemble_variance_decreases_with_agreement(predictions):
       """When models agree, variance should be low."""
       predictions = np.array(predictions)
       variance = predictions.var(axis=0)

       # If all predictions are similar (within 0.1), variance should be small
       range_per_output = predictions.ptp(axis=0)
       for i, (rng, var) in enumerate(zip(range_per_output, variance)):
           if rng < 0.1:
               assert var < 0.01, f"Output {i}: range={rng}, variance={var}"
   ```

ACCEPTANCE CRITERIA:
- Ensemble property tests added
- Tests verify uncertainty semantics
- Edge cases with agreement/disagreement

FILES TO MODIFY:
- CREATE/MODIFY: `tests/optim/test_surrogate.py`
```

---

## 📋 Outstanding Roadmap Tasks

---

### Task R.1: HuggingFace Dataset Loading

```
TASK: Implement HuggingFace dataset loading with filtering for ai_scientist.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Outstanding Items - Dataset Tools

INSTRUCTIONS:

1. Create `ai_scientist/datasets/` directory.

2. Create `ai_scientist/datasets/sampler.py`:
   ```python
   from datasets import load_dataset
   from typing import Optional
   import numpy as np

   def load_constellaration_dataset(
       split: str = "train",
       problem: Optional[str] = None,
   ):
       """Load HuggingFace constellaration dataset."""
       ds = load_dataset("proxima-fusion/constellaration", split=split)

       if problem == "p1":
           ds = ds.filter(lambda ex: (
               abs(ex['aspect_ratio'] - 4.0) < 0.1 and
               abs(ex['average_triangularity'] + 0.6) < 0.1
           ))
       elif problem == "p2":
           # P2 filtering
           ...

       return ds
   ```

3. Add unit tests.

ACCEPTANCE CRITERIA:
- Dataset loading works
- Filtering by problem works
- Tests pass

FILES TO MODIFY:
- CREATE: `ai_scientist/datasets/__init__.py`
- CREATE: `ai_scientist/datasets/sampler.py`
- CREATE: `tests/datasets/test_sampler.py`
```

---

### Task R.2: P2 Orchestration Pipeline

```
TASK: Implement P2 (Simple-to-Build QI) orchestration pipeline.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Outstanding Items - PLAN_CHECKLIST.md

INSTRUCTIONS:

1. Read existing P3 orchestration in runner.py/coordinator.py.

2. Read `constellaration/src/constellaration/problems.py` for `SimpleToBuildQIStellarator`.

3. Ensure P2Problem class exists (from Task 6.2).

4. Update runner to support `--problem p2`:
   - Use SimpleToBuildQIStellarator constraints
   - Same JSONL logging as P3
   - Same promotion pipeline

5. Add P2-specific tests.

ACCEPTANCE CRITERIA:
- `python -m ai_scientist.runner --problem p2` works
- P2 constraints correctly evaluated
- JSONL logging matches P3 format

FILES TO MODIFY:
- MODIFY: `ai_scientist/runner.py`
- MODIFY: `ai_scientist/problems.py`
- CREATE: `tests/test_p2_pipeline.py`
```

---

### Task R.3: Feasibility Prefilter Classifier

```
TASK: Train quick classifier to reject obviously infeasible candidates.

REFERENCE: docs/TODO_AI_SCIENTIST_IMPROVEMENTS.md - Outstanding Items - Feasibility Prefilter

INSTRUCTIONS:

1. Create `ai_scientist/prefilter.py`:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   import numpy as np

   class FeasibilityPrefilter:
       """Quick classifier to reject obviously infeasible candidates."""

       def __init__(self):
           self.model: Optional[RandomForestClassifier] = None

       def train(self, X: np.ndarray, y_feasible: np.ndarray):
           """Train on historical evaluations."""
           self.model = RandomForestClassifier(n_estimators=100)
           self.model.fit(X, y_feasible)

       def predict_feasible(
           self, X: np.ndarray, threshold: float = 0.5
       ) -> np.ndarray:
           """Return boolean mask of likely-feasible candidates."""
           if self.model is None:
               return np.ones(len(X), dtype=bool)  # Pass all if not trained
           probs = self.model.predict_proba(X)[:, 1]
           return probs >= threshold

       def filter_candidates(
           self, candidates: list[dict], threshold: float = 0.5
       ) -> list[dict]:
           """Filter candidates, keeping only likely-feasible ones."""
           # Convert to features, predict, filter
           ...
   ```

2. Integrate into runner before VMEC calls.

3. Log prefilter decisions.

ACCEPTANCE CRITERIA:
- Prefilter trains on historical data
- Integration before physics evaluation
- Logging of filter decisions
- Tests verify filtering behavior

FILES TO MODIFY:
- CREATE: `ai_scientist/prefilter.py`
- MODIFY: `ai_scientist/runner.py`
- CREATE: `tests/test_prefilter.py`
```

---

## Execution Notes

1. **Dependencies**: Some tasks depend on others (noted in PREREQUISITES)
2. **Testing**: Every task should include or update tests
3. **Backward Compatibility**: Preserve existing APIs where noted
4. **Reference Files**: Read indicated files before making changes
5. **Commit Granularity**: One task = one commit/PR

---

## Version History

- 2025-11-30: Initial version with 42 task prompts
