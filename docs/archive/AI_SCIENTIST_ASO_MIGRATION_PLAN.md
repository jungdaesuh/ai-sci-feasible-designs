> **DEPRECATED:** This document is superseded by `ASO_V4_IMPLEMENTATION_GUIDE.md`

# AI Scientist: ASO Migration & Refactoring Plan

**Status:** Draft
**Date:** November 29, 2025
**Based on:** `docs/UNIFIED_PLAN.md` and Codebase Analysis

This document outlines the concrete engineering steps required to migrate the AI Scientist from its current monolithic `runner.py` architecture to the target **Agent-Supervised Optimization (ASO)** loop.

## 1. Architecture Overview

The core goal is to break the "God Object" pattern in `runner.py` and enable a "Steppable" optimization loop where the Planner agent can intervene between optimization chunks.

### Key Dependencies
- **New Module:** `ai_scientist/structs.py` (Avoids circular imports between Planner and Coordinator).
- **Config Update:** `control_mode` flag (Legacy vs. ASO).
- **Refactor:** `OptimizationWorker` becomes the owner of ALM state and stepping logic.

---

## 2. Phase 0: Foundations (Safe & Non-Breaking)

**Goal:** Establish shared data structures and configuration flags without changing logic.

### 2.1 Shared Data Structures
Create `ai_scientist/structs.py` to hold the communication protocols.
- `OptimizerDiagnostics`: Data transfer object (DTO) for ALM state (objective, violation, trends).
- `OptimizationDirective`: DTO for Planner commands (STOP, ADJUST, CONTINUE) and config overrides.

### 2.2 Configuration Updates
Update `ai_scientist/config.py`:
- Add `control_mode: str = "legacy"` to `ExperimentConfig`.
  - `legacy`: Keeps current `runner.py` behavior.
  - `aso`: Will enable the new loop (implemented in Phase 5).

---

## 3. Phase 1: ALM Logic Extraction

**Goal:** Move physics/optimization setup code out of `runner.py`.

### 3.1 Extract Setup Logic
Currently, `runner.py` (lines 2800+) manually handles:
- `SurfaceRZFourier` masking.
- Infinity norm scaling calculation.
- `AugmentedLagrangianSettings` instantiation.

**Action:** Move this logic into a factory or static method, likely within `ai_scientist/workers.py` or a new `ai_scientist/optim/setup.py`.
- `prepare_alm_state(template, params, ...) -> (initial_guess, scale, unravel_fn, mask)`

### 3.2 Extract Step Logic
Current `optim/differentiable.py` and `runner.py` have mixed optimization loops.
**Action:** ensure `optimize_alm_inner_loop` in `differentiable.py` is self-contained and accepts a standard state dict, returning a standard state dict.

---

## 4. Phase 2: Steppable Worker Interface

**Goal:** Make `OptimizationWorker` capable of pausing and resuming.

### 4.1 Refactor `OptimizationWorker`
Update `ai_scientist/workers.py`:
- **Current:** `run(context)` runs to completion.
- **New:** Add `init_state(context)` and `step(state, n_steps=10)`.
- The state object must persist:
  - Current parameters (`x`).
  - ALM multipliers and penalty parameters.
  - Optimizer momentum (if using Adam/SGD).

---

## 5. Phase 3: Coordinator Intelligence

**Goal:** Implement the translation layer between Numerical State and Semantic State.

### 5.1 Diagnostics Generation
Implement `Coordinator.generate_diagnostics(alm_state, history)`:
- Calculate `objective_delta`.
- Detect `STAGNATION` (tiny delta, high violation).
- Map constraints to readable trends (e.g., "MHD violation increasing").

### 5.2 Directive Application
Implement `Coordinator.apply_directive(directive, active_config)`:
- **ADJUST:** Update `active_config` (e.g., increase weights) or ALM params (e.g., `penalty_increase_factor`).
- **STOP:** Signal the loop to terminate early.

---

## 6. Phase 4: Planner Integration

**Goal:** Enable the LLM to reason about the optimization process.

### 6.1 Implement `analyze_optimizer_diagnostics`
In `ai_scientist/planner.py`:
- Add the method signature defined in Phase 0.
- Construct a prompt that presents `OptimizerDiagnostics` to the LLM.
- Parse LLM output into `OptimizationDirective`.

---

## 7. Phase 5: The ASO Loop

**Goal:** Wire components into the new control loop.

### 7.1 Implement `produce_candidates_aso`
In `ai_scientist/coordinator.py`:
```python
def produce_candidates_aso(self, ...):
    # 1. Initialize State
    state = worker.init_state(initial_seeds)
    
    while budget_remaining:
        # 2. Run Chunk
        state = worker.step(state, steps=10)
        
        # 3. Diagnostics
        diag = self.generate_diagnostics(state)
        
        # 4. Agent Reasoning
        directive = planner.analyze_optimizer_diagnostics(diag)
        
        # 5. Apply Control
        if directive.action == "STOP": break
        self.apply_directive(directive)
        
    return state.candidates
```

---

## 8. Phase 6: Switchover

**Goal:** Expose the new capability to users.

### 8.1 Update `runner.py`
- In `_run_cycle`, read `cfg.control_mode`.
- If `aso`:
  - Delegate candidate generation entirely to `coordinator.produce_candidates_aso`.
  - Skip the legacy "ALM Block Placeholder" code.
- Verify feature parity with `p3` benchmarks.

---

## Success Criteria
1.  **Modularity:** `runner.py` LoC reduced by moving optimization logic to Workers.
2.  **Intervention:** Planner logs show successful "ADJUST" commands responding to stagnation.
3.  **Performance:** ASO loop achieves comparable or better Feasibility/HV scores than the monolithic baseline.
