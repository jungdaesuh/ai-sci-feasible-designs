# Comprehensive To-Do List for AI Scientist Autonomous Feasibility Search

**Goal:** Transform the AI Scientist from a passive "wrapper" into an active, strategic solver capable of discovering feasible stellarator designs in a sparse, high-dimensional landscape.

**Context:**
The current system has a robust infrastructure (World Model, RAG, Tool Definitions) and a newly activated ReAct planning loop (`planner.py`). However, the underlying optimization logic (`runner.py`) is a naive filter that fails when the initial dataset contains zero feasible examples ("Cold Start" problem). To succeed, the system must actively *hunt* for feasibility using gradient-based signals, adaptive objectives, and strategic meta-control.

---

## Phase 1: Strategic Control & Feasibility Restoration (High Priority)

**Objective:** Empower the Agent to dynamically switch strategies (e.g., from "Optimize Aspect Ratio" to "Minimize MHD Violation") when feasibility is lost.

- [ ] **1.1 Implement "Feasibility Restoration Mode" in `runner.py`**
    - [ ] Modify `_run_cycle` to detect when `feasibility_rate` is effectively zero (or below a threshold) for $N$ consecutive cycles.
    - [ ] Introduce a `mode` flag in the cycle state ("optimization" vs. "restoration").
    - [ ] **Update:** When in "restoration" mode, the runner should switch to the **Augmented Lagrangian Method (ALM)** solver (see 1.3) rather than just changing the ranking sort order.
    - [ ] Log the active `mode` in the World Model (`budgets` or `cycles` table) for observability.

- [ ] **1.2 Adaptive Promotion Gates (Fix the "Stall" Bug)**
    - [ ] Refactor the promotion logic in `_run_cycle`.
    - [ ] **Logic Change:** If `screen_summary.feasible_count < min_feasible_for_promotion`:
        - [ ] Do NOT skip promotion.
        - [ ] Instead, select the top $K$ candidates with the *lowest constraint violations* (`min(max_violation)`).
        - [ ] Mark these promoted candidates with a `reason="exploration"` or `reason="restoration"` tag for analysis.

- [ ] **1.3 Native ALM Integration (Solver Upgrade)**
    - [ ] **Context:** The `constellaration` library already implements ALM in `constellaration.optimization.augmented_lagrangian`. We should use this instead of "guessing" weights.
    - [ ] **Implementation:**
        - [ ] Import `AugmentedLagrangianState` and `update_augmented_lagrangian_state` in `runner.py`.
        - [ ] Create a new runner path (or tool) that initializes an ALM state from the current best candidate.
        - [ ] Execute a batch of ALM updates to actively drive the design toward feasibility using the physics gradients (or simulated gradients).
    - [ ] **Agent Meta-Control:**
        - [ ] Empower the Agent to configure the ALM solver via `config_overrides` (e.g., setting `penalty_parameters_initial`, `bounds_initial`).
        - [ ] Allow the Agent to explicitly trigger this mode: "Switching to ALM to resolve constraints."

---

## Phase 2: Granular Surrogate Modeling (Intelligence Upgrade)

**Objective:** Provide the optimizer with a gradient. Instead of "Infeasible (0.0)", tell it "MHD Violation = 0.5".

- [ ] **2.1 Refactor `SurrogateBundle` for Regression**
    - [ ] Modify `ai_scientist/optim/surrogate.py`.
    - [ ] Instead of a single classifier (`prob_feasible`), train separate regression models (e.g., Random Forests or GPs) for critical constraints:
        - [ ] `predict_mhd_well`
        - [ ] `predict_qi_residual`
        - [ ] `predict_elongation`
    - [ ] Use `world_model.surrogate_training_data` to fetch the raw scalar values for these metrics (already available in `metrics` table).

- [ ] **2.2 Update Ranking Logic (`runner.py`)**
    - [ ] Modify `_surrogate_rank_screen_candidates`.
    - [ ] Use the new regressors to predict the *vector* of constraints for candidate pool.
    - [ ] Calculate a predicted `max_violation` for each candidate.
    - [ ] Rank candidates by this predicted violation when in "Restoration Mode" (see 1.1).

---

## Phase 3: Advanced Exploration & Recombination (Breaking Local Minima)

**Objective:** Generate better candidates by combining successful traits of partial solutions, rather than just random perturbation.

- [ ] **3.1 Implement Genetic Crossover (Recombination)**
    - [ ] Add a new helper `tools.crossover_boundaries(parent_a, parent_b)`.
    - [ ] Implement a geometric-aware crossover (e.g., interpolate coefficients) rather than naive array slicing to ensure the resulting surface is closed and continuous.
    - [ ] Expose this as a tool `recombine_designs` in `tools_api.py` for the Agent to use.

- [ ] **3.2 Agent-Directed Exploration**
    - [ ] Update `planner.py` system prompt.
    - [ ] Instruct the agent: "If you see two designs where Design A has good MHD but bad QI, and Design B has good QI but bad MHD, use the `recombine_designs` tool to merge them."
    - [ ] Add logic in `runner.py`'s `_propose_p3_candidates_for_cycle` to accept "recombined" candidates from the agent (which is already partially supported by `suggested_params` but needs explicit tool support).

---

## Phase 4: Documentation & Benchmarking

**Objective:** Verify that the "Active Agent" actually solves the "Cold Start" problem defined in ConStellaration.

- [ ] **4.1 "Infeasible Start" Benchmark**
    - [ ] Create a test config (`configs/benchmark_cold_start.yaml`) that seeds the experiment with a known *infeasible* design (e.g., a plain tokamak-like torus that fails stellarator stability).
    - [ ] Assert that the system (via Agent or Restoration Mode) recovers feasibility within $N$ cycles.

- [ ] **4.2 Update Docs**
    - [ ] Update `docs/AI_SCIENTIST_AUTONOMY_PLAN.md` to reflect the move from "Filter" to "Solver."
    - [ ] Document the "Feasibility Restoration" logic and the "Soft ALM" agent capabilities.
