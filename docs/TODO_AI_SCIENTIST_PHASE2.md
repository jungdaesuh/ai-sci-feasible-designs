# Comprehensive To-Do List for AI Scientist Autonomous Feasibility Search

**Goal:** Transform the AI Scientist from a passive "wrapper" into an active, strategic solver capable of discovering feasible stellarator designs in a sparse, high-dimensional landscape.

**Context:**
The current system has a robust infrastructure (World Model, RAG, Tool Definitions) and a newly activated ReAct planning loop (`planner.py`). However, the underlying optimization logic (`runner.py`) is a naive filter that fails when the initial dataset contains zero feasible examples ("Cold Start" problem). To succeed, the system must actively *hunt* for feasibility using gradient-based signals (via Native ALM) and strategic meta-control.

---

## Phase 1: Strategic Control & Feasibility Restoration (High Priority)

**Objective:** Empower the Agent to dynamically switch strategies from passive sampling to active solving using the Native Augmented Lagrangian Method (ALM).

- [x] **1.1 Implement "Feasibility Restoration Mode" in `runner.py`**
    - [x] Modify `_run_cycle` to detect when `feasibility_rate` is effectively zero (or below a threshold) for $N$ consecutive cycles.
    - [x] Introduce a `mode` flag in the cycle state ("optimization" vs. "restoration").
    - [x] **Update:** When in "restoration" mode (triggered by Agent or heuristics), the runner switches to the **Native ALM** solver (see 1.3).
    - [x] Log the active `mode` in the World Model (`budgets` or `cycles` table) for observability.

- [x] **1.2 Adaptive Promotion Gates (Fix the "Stall" Bug)**
    - [x] Refactor the promotion logic in `_run_cycle`.
    - [x] **Logic Change:** If `screen_summary.feasible_count < min_feasible_for_promotion`:
        - [x] Do NOT skip promotion.
        - [x] Instead, select the top $K$ candidates with the *lowest constraint violations* (`min(max_violation)`).
        - [x] Mark these promoted candidates with a `reason="exploration"` or `reason="restoration"` tag for analysis.

- [x] **1.3 Native ALM Integration (Solver Upgrade)**
    - [x] **Context:** The `constellaration` library already implements ALM in `constellaration.optimization.augmented_lagrangian`. We use this instead of "Soft ALM".
    - [x] **Hybrid Workflow Implementation:**
        - [x] **Agent Role:** Acts as a **Meta-Optimizer**. It monitors feasibility and outputs `config_overrides={"optimizer": "alm", "alm_settings": {...}}` to configure the solver (e.g., `penalty_parameters_initial`, `bounds_initial`, `maxit`).
        - [x] **Runner Role:** Detects `optimizer="alm"`. Instead of standard sampling (`_propose_p3_candidates`), it:
            1. [x] Instantiates `AugmentedLagrangianState` from the current best candidate.
            2. [x] Runs a batch of updates using `update_augmented_lagrangian_state`.
            3. [x] Returns the optimized result as the new candidate.
    - [x] **Code Action:** Import `AugmentedLagrangianState` and `update_augmented_lagrangian_state` in `runner.py` and wire up the "alm" branch in `_run_cycle`.

---

## Phase 2: Granular Surrogate Modeling (Intelligence Upgrade)

**Objective:** Provide the optimizer with a gradient. Instead of "Infeasible (0.0)", tell it "MHD Violation = 0.5".

- [x] **2.1 Refactor `SurrogateBundle` for Regression**
    - [x] Modify `ai_scientist/optim/surrogate.py`.
    - [x] Instead of a single classifier (`prob_feasible`), train separate regression models (e.g., Random Forests or GPs) for critical constraints:
        - [x] `predict_mhd_well`
        - [x] `predict_qi_residual`
        - [x] `predict_elongation`
    - [x] Use `world_model.surrogate_training_data` to fetch the raw scalar values for these metrics (already available in `metrics` table).

- [x] **2.2 Update Ranking Logic (`runner.py`)**
    - [x] Modify `_surrogate_rank_screen_candidates`.
    - [x] Use the new regressors to predict the *vector` of constraints for candidate pool.
    - [x] Calculate a predicted `max_violation` for each candidate.
    - [x] Rank candidates by this predicted violation when in "Restoration Mode" (if ALM isn't active or for initial screening).

---

## Phase 3: Advanced Exploration & Recombination (Breaking Local Minima)

**Objective:** Generate better candidates by combining successful traits of partial solutions, rather than just random perturbation.

- [x] **3.1 Implement Genetic Crossover (Recombination)**
    - [x] Add a new helper `tools.crossover_boundaries(parent_a, parent_b)`.
    - [x] Implement a geometric-aware crossover (e.g., interpolate coefficients) rather than naive array slicing to ensure the resulting surface is closed and continuous.
    - [x] Expose this as a tool `recombine_designs` in `tools_api.py` for the Agent to use.

- [x] **3.2 Agent-Directed Exploration**
    - [x] Update `planner.py` system prompt.
    - [x] Instruct the agent: "If you see two designs where Design A has good MHD but bad QI, and Design B has good QI but bad MHD, use the `recombine_designs` tool to merge them."
    - [x] Add logic in `runner.py`'s `_propose_p3_candidates_for_cycle` to accept "recombined" candidates from the agent.

---

## Phase 4: Documentation & Benchmarking

**Objective:** Verify that the "Active Agent" actually solves the "Cold Start" problem.

- [x] **4.1 "Infeasible Start" Benchmark**
    - [x] Create a test config (`configs/benchmark_cold_start.yaml`) that seeds the experiment with a known *infeasible* design.
    - [x] Assert that the system (via Agent -> ALM) recovers feasibility within $N$ cycles.

- [x] **4.2 Update Docs**
    - [x] Update `docs/AI_SCIENTIST_AUTONOMY_PLAN.md` to reflect the move from "Filter" to "Solver" (Native ALM).
    - [x] Document the "Feasibility Restoration" logic and the Agent's "Meta-Optimizer" role.

---

## Phase 5: Advanced Solver Refinements (Hybrid & Initialization)

**Objective:** Enhance the ALM solver with Surrogate Acceleration and Physics-Informed Initialization.

- [x] **5.1 Hybrid Surrogate-Assisted ALM (SA-ALM)**
    - [x] **Concept:** Accelerate the inner loop of ALM by using the `SurrogateBundle` (trained in Phase 2) to predict objectives and constraints, rather than calling the expensive VMEC++ forward model every step.
    - [x] **Implementation:**
        - [x] Modify `_objective_constraints` in `runner.py` to accept a `predictor` callable.
        - [x] In `_run_cycle`, if `optimizer_mode == "sa-alm"`, pass a predictor wrapping `_SURROGATE_BUNDLE.rank_candidates` (or direct regression predict) to the inner loop.
        - [x] Periodically verify the surrogate's "best" result with a real VMEC call to update the trust region/surrogate data.

- [x] **5.2 Advanced Physics Initialization (NAE)**
    - [x] **Concept:** Use Near-Axis Expansion (NAE) to generate high-quality initial guesses that are closer to QI, rather than random perturbations.
    - [x] **Implementation:**
        - [x] Import `generate_nae` from `constellaration.initial_guess`.
        - [x] Create a new initialization helper `_generate_nae_candidate_params` in `runner.py`.
        - [x] Update `_build_template_params_for_alm` (or creating a new builder) to use `generate_nae` when `boundary_template.seed_path` is missing or when requested by the agent.
        - [x] Expose an `initialization_strategy` config option ("template", "nae", "random") that the Agent can tune.
