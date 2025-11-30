# AI Scientist Improvement TODO

**Date:** 2025-11-30
**Status:** Active
**Purpose:** Consolidated improvement checklist based on codebase analysis comparing `ai_scientist/` with `constellaration/` patterns, plus outstanding items from existing roadmaps.

---

## Executive Summary

This document consolidates all improvement opportunities for the `ai_scientist` module into a single actionable checklist. Items are organized by priority (🔴 High, 🟡 Medium, 🟢 Low) and include full context for implementation.

**Key Finding:** The `constellaration` codebase demonstrates superior patterns for:
- Type safety (Pydantic + jaxtyping)
- JAX integration (pytree registration)
- Configuration management (factory methods)
- Problem abstraction (template method pattern)
- Forward model orchestration (centralized pipeline)

---

## 🔴 High Priority — Architecture & Maintainability

### 1. Break Up Large Files

**Context:** `runner.py` at 129KB is the most pressing maintainability issue. Large files make navigation, testing, and code review difficult.

| File | Current Size | Recommended Split |
|------|-------------|-------------------|
| `runner.py` (129KB) | ~3000+ LOC | See sub-tasks below |
| `memory.py` (46KB) | ~1200 LOC | `memory/schema.py`, `memory/repository.py`, `memory/graph.py` |
| `tools.py` (36KB) | ~1000 LOC | `tools/evaluation.py`, `tools/design_manipulation.py`, `tools/hypervolume.py` |
| `planner.py` (35KB) | ~900 LOC | Already well-structured; consider extracting `heuristic_supervisor.py` |

- [x] **1.1 Extract `budget_manager.py` from `runner.py`**
  - Move `BudgetConfig`, `AdaptiveBudgetConfig`, budget calculation logic
  - File: `ai_scientist/budget_manager.py`
  - Classes: `BudgetController`, `BudgetSnapshot`
  - References: `runner.py:_run_cycle` budget handling

- [x] **1.2 Extract `fidelity_controller.py` from `runner.py`**
  - Move fidelity ladder logic, promotion decisions
  - File: `ai_scientist/fidelity_controller.py`
  - Functions: `_evaluate_stage`, promotion gating

- [x] **1.3 Extract `cycle_executor.py` from `runner.py`**
  - Move `_run_cycle` and cycle-level orchestration
  - File: `ai_scientist/cycle_executor.py`

- [x] **1.4 Create `experiment_runner.py` as thin orchestrator**
  - Keep only `run_experiment()` entry point that composes the above
  - File: `ai_scientist/experiment_runner.py`

- [x] **1.5 Split `memory.py` into submodule**
  - Create `ai_scientist/memory/` directory
  - `schema.py` - Table definitions, migrations
  - `repository.py` - CRUD operations
  - `graph.py` - `PropertyGraph` class

- [ ] **1.6 Split `tools.py` into submodule**
  - Create `ai_scientist/tools/` directory
  - `evaluation.py` - `evaluate_p3_set()`, physics wrappers
  - `design_manipulation.py` - `propose_boundary()`, `recombine_designs()`
  - `hypervolume.py` - HV calculations, Pareto utilities

---

### 2. Adopt Pydantic for State Management

**Context:** `constellaration` uses Pydantic BaseModel for all state objects, enabling JSON serialization, validation, and JAX pytree compatibility. `ai_scientist` uses frozen dataclasses which lack these features.

**Pattern from constellaration:**
```python
# constellaration/optimization/augmented_lagrangian.py
class AugmentedLagrangianState(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    x: jnp.ndarray
    multipliers: jnp.ndarray
    penalty_parameters: jnp.ndarray
    objective: jnp.ndarray
    constraints: jnp.ndarray
    bounds: jnp.ndarray
```

- [ ] **2.1 Convert `TrajectoryState` to Pydantic**
  - File: `ai_scientist/coordinator.py`
  - Add jaxtyping annotations for array shapes
  - Enable `model_copy()` for immutable updates

- [ ] **2.2 Convert `OptimizerDiagnostics` to Pydantic**
  - File: `ai_scientist/planner.py`
  - Already has `to_json()` - Pydantic provides this automatically

- [ ] **2.3 Convert `OptimizationDirective` to Pydantic**
  - File: `ai_scientist/planner.py`
  - Leverage Pydantic's `model_dump()` for serialization

- [ ] **2.4 Add Pydantic pytree registration**
  - Port `constellaration/utils/pytree.py` patterns
  - Implement `register_pydantic_data` decorator
  - File: `ai_scientist/utils/pytree.py` (new)

---

### 3. Centralized Forward Model

**Context:** `constellaration` has a clean `forward_model()` function that orchestrates the entire evaluation pipeline. `ai_scientist` evaluation logic is scattered across `tools.py`.

**Pattern from constellaration:**
```python
# constellaration/forward_model.py
def forward_model(boundary, ideal_mhd_parameters, settings)
    -> tuple[ConstellarationMetrics, VmecppWOut]:
    # Single orchestration point for all evaluations
```

- [ ] **3.1 Create centralized forward model**
  - File: `ai_scientist/forward_model.py` (new)
  - Consolidate `tools.evaluate_p3_set()`, `design_hash()`, caching
  - Single entry point: `forward_model(boundary, settings) -> EvaluationResult`

- [ ] **3.2 Create `EvaluationResult` dataclass**
  - Include: `metrics`, `objective`, `constraints`, `feasibility`, `cache_hit`
  - Use Pydantic for serialization

- [ ] **3.3 Migrate existing callers**
  - Update `runner.py`, `coordinator.py` to use new forward model
  - Preserve backward compatibility via `tools.py` wrappers

---

## 🟡 Medium Priority — Developer Experience

### 4. Factory Methods for Config

**Context:** `constellaration` uses factory methods for preset configurations, making common use cases discoverable and reducing YAML hunting.

**Pattern from constellaration:**
```python
class ConstellarationSettings:
    @staticmethod
    def default_high_fidelity() -> "ConstellarationSettings":
        return ConstellarationSettings(
            vmec_preset_settings=VmecPresetSettings(fidelity="high_fidelity")
        )
```

- [ ] **4.1 Add factory methods to `ExperimentConfig`**
  - File: `ai_scientist/config.py`
  - Methods: `p3_high_fidelity()`, `p3_quick_validation()`, `p3_aso_enabled()`

- [ ] **4.2 Add factory methods to `ASOConfig`**
  - Methods: `default_event_triggered()`, `default_periodic()`, `disabled()`

- [ ] **4.3 Add factory methods to `ALMConfig`**
  - Methods: `default()`, `aggressive_penalties()`, `conservative()`

- [ ] **4.4 Document factory methods in CLI help**
  - Add `--preset` flag to runner that maps to factory methods

---

### 5. Pytree Registration Pattern

**Context:** `constellaration/utils/pytree.py` provides `mask_and_ravel()` for selective parameter extraction, enabling JAX optimization on subsets of Pydantic models.

- [ ] **5.1 Port `mask_and_ravel()` utility**
  - File: `ai_scientist/utils/pytree.py`
  - Enable selective flattening of boundary params for optimization

- [ ] **5.2 Port `register_pydantic_data` decorator**
  - Enable JAX pytree operations on Pydantic models

- [ ] **5.3 Update `optim/differentiable.py` to use pytree utils**
  - Replace manual flattening with `mask_and_ravel()`

---

### 6. Problem Abstraction Alignment

**Context:** `ai_scientist/prompts.py` has `ProblemSpec` but it's disconnected from evaluation logic. `constellaration/problems.py` has a clean ABC pattern.

**Pattern from constellaration:**
```python
class _Problem(abc.ABC):
    @abc.abstractmethod
    def _normalized_constraint_violations(self, metrics) -> np.ndarray:
        pass

    def is_feasible(self, metrics) -> bool:
        violations = self._normalized_constraint_violations(metrics)
        return np.all(violations <= 0)
```

- [ ] **6.1 Create Problem ABC in ai_scientist**
  - File: `ai_scientist/problems.py` (new)
  - Abstract methods: `_normalized_constraint_violations()`, `get_objective()`
  - Template methods: `is_feasible()`, `compute_feasibility()`

- [ ] **6.2 Implement P1Problem, P2Problem, P3Problem**
  - Inherit from ABC
  - Define constraint specifications per problem

- [ ] **6.3 Integrate with existing `prompts.ProblemSpec`**
  - Map constraint names to Problem classes
  - Use in `coordinator._generate_diagnostics()`

---

### 7. jaxtyping Annotations

**Context:** `constellaration` uses explicit tensor shape annotations for readability and static checking.

**Pattern:**
```python
from jaxtyping import Float

def gradient_descent_on_inputs(
    candidates: list[BoundaryParams],
    surrogate: StellaratorNeuralOp,
) -> Float[np.ndarray, "n_candidates n_params"]:
    ...
```

- [ ] **7.1 Add jaxtyping to `optim/surrogate_v2.py`**
  - Annotate model inputs/outputs with shapes

- [ ] **7.2 Add jaxtyping to `optim/geometry.py`**
  - Annotate Fourier ↔ real-space conversions

- [ ] **7.3 Add jaxtyping to `optim/alm_bridge.py`**
  - Annotate ALM state arrays

- [ ] **7.4 Add jaxtyping to `coordinator.py`**
  - Annotate diagnostic arrays

---

## 🟢 Lower Priority — Polish & Tooling

### 8. Pre-commit Hooks

**Context:** `constellaration` has comprehensive `.pre-commit-config.yaml` with Ruff, Black, isort, Pyright.

- [ ] **8.1 Create `.pre-commit-config.yaml`**
  - Add Ruff with 88-char line length
  - Add Black formatter
  - Add isort with Black profile
  - Add Pyright type checker

- [ ] **8.2 Run initial formatting pass**
  - Apply to all `ai_scientist/` files
  - Commit as single formatting PR

- [ ] **8.3 Add to CI workflow**
  - Run pre-commit checks in GitHub Actions

---

### 9. Domain-Mirrored Test Organization

**Context:** `constellaration` mirrors source structure in tests for discoverability.

**Current:**
```
tests/
├── test_coordinator_aso.py
├── test_planner_aso.py
├── test_tools_*.py
```

**Recommended:**
```
tests/
├── optim/
│   ├── test_surrogate_v2.py
│   ├── test_alm_bridge.py
│   └── test_generative.py
├── test_coordinator.py
├── test_planner.py
└── test_runner.py
```

- [ ] **9.1 Reorganize test directory structure**
  - Create `tests/optim/` subdirectory
  - Move optimization-related tests

- [ ] **9.2 Update pytest configuration**
  - Ensure test discovery works with new structure

---

### 10. Property-Based Testing

**Context:** `constellaration` uses property-based testing for mathematical invariants. `ai_scientist` relies heavily on mocks.

- [ ] **10.1 Add hypothesis for ALM invariants**
  - Test: multipliers grow when constraints violated
  - Test: penalties bounded by max
  - File: `tests/test_alm_bridge.py`

- [ ] **10.2 Add hypothesis for geometry functions**
  - Test: `fourier_to_real_space` invertible
  - Test: aspect ratio always positive
  - File: `tests/optim/test_geometry.py`

- [ ] **10.3 Add hypothesis for surrogate predictions**
  - Test: ensemble uncertainty ≥ individual variance
  - File: `tests/optim/test_surrogate.py`

---

## 📋 Outstanding Items from Existing Roadmaps

### From `SAS_TODO.md` (Stellarator AI Scientist)

- [ ] **Dataset Tools** `datasets/sampler.py`
  - [ ] HuggingFace dataset loading with filtering
  - [ ] PCA + GMM for latent space sampling
  - [ ] MCMC posterior sampling

- [ ] **Docker** `Dockerfile.sas`
  - [ ] Production container with all dependencies

### From `roadmap.md` (AI Scientist Implementation)

All phases marked complete ✅ - no outstanding items.

### From `PLAN_CHECKLIST.md` (Orchestration)

- [ ] **P2 Orchestration** (Simple-to-Build QI)
  - [ ] Runner wrapping `SimpleToBuildQIStellarator`
  - [ ] Same JSONL logging/promotion pipeline

- [ ] **P3 Orchestration** (MHD-Stable QI, Multi-Objective)
  - [ ] Multi-objective search (L_grad, A)
  - [ ] Hypervolume computation and logging

- [ ] **Feasibility Prefilter** (Data-Driven)
  - [ ] Train quick classifier to reject obvious infeasible candidates

### From `ASO_V4_IMPLEMENTATION_GUIDE.md`

All phases marked complete ✅ - ASO V4 fully implemented.

---

## 📊 Summary Table

| Category | High Priority | Medium Priority | Low Priority | Total |
|----------|--------------|-----------------|--------------|-------|
| Architecture | 6 tasks | - | - | 6 |
| State Management | 4 tasks | - | - | 4 |
| Forward Model | 3 tasks | - | - | 3 |
| Config | - | 4 tasks | - | 4 |
| Pytree | - | 3 tasks | - | 3 |
| Problem Abstraction | - | 3 tasks | - | 3 |
| Type Annotations | - | 4 tasks | - | 4 |
| Pre-commit | - | - | 3 tasks | 3 |
| Test Organization | - | - | 2 tasks | 2 |
| Property Tests | - | - | 3 tasks | 3 |
| **Subtotal** | **13** | **14** | **8** | **35** |
| Outstanding Roadmaps | 7 tasks | | | **7** |
| **Grand Total** | | | | **42** |

---

## 🚀 Recommended Execution Order

1. **Week 1:** Break up `runner.py` (tasks 1.1-1.4) - biggest maintainability win
2. **Week 2:** Create centralized `forward_model.py` (tasks 3.1-3.3)
3. **Week 3:** Add pre-commit hooks (tasks 8.1-8.3) + initial formatting
4. **Week 4:** Convert key dataclasses to Pydantic (tasks 2.1-2.3)
5. **Ongoing:** Add jaxtyping as files are touched (tasks 7.1-7.4)

---

## References

- `docs/ASO_V4_IMPLEMENTATION_GUIDE.md` - ASO architecture (complete)
- `docs/roadmap.md` - Phase implementation guide (complete)
- `docs/improvement-plan.md` - Narrative rationale for improvements
- `docs/PLAN_CHECKLIST.md` - Orchestration checklist
- `constellaration/` - Reference implementation for patterns
