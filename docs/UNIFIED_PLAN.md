This document is a living roadmap for unifying the semantic reasoning layer  
(`ai_scientist` agents) with the numerical optimization layer (ALM + VMEC++)  
into a single Neuro‑Symbolic feedback loop.

It has been updated to reflect the **current codebase reality** and to define a
**concrete, incremental migration plan** instead of pretending the refactor is
already complete.

---

## 1. Scope and Goals

- Unify the AI Scientist agent stack (`planner`, `coordinator`, `workers`, `runner`)
  with the ConStellaration optimization loops without modifying upstream
  `constellaration` or `vmecpp` code.
- Replace the coarse EXPLORE / EXPLOIT toggling with a tighter
  **Agent‑Supervised Optimization (ASO)** loop where:
  - The **Planner** supervises optimization via structured directives.
  - The **Coordinator** owns ALM state and applies those directives.
  - The **Workers** execute numerical steps and return ALM state snapshots.
- Keep the existing runner and experiment configs usable throughout the
  transition (no hard breaking changes).

Non‑goals:

- Do not redesign physics or metrics; use `constellaration` as the single source
  of truth for objectives/constraints.
- Do not over‑generalize into a meta‑framework; keep the design specific to
  the ConStellaration problems (P1–P3).

---

## 2. Current Architecture (Reality Check)

This summarizes what is actually implemented today, not what we want eventually.

### 2.1 Planner (`ai_scientist/planner.py`)

- Provides `PlanningOutcome` which bundles:
  - `context`: budgets, constraint weights, stage history, failure cases, etc.
  - `evaluation_summary` / `boundary_summary`.
  - Optional `suggested_params` and `config_overrides`.
- `PlanningAgent`:
  - Gates LLM access via roles: `planning`, `literature`, `analysis`.
  - Implements:
    - `_build_template_params` → basic boundary from `BoundaryTemplateConfig`.
    - `retrieve_rag`, `write_note`, `evaluate_p3`, `make_boundary`,
      `propose_boundary`, `recombine_designs`.
    - `plan_cycle`:
      - Evaluates a template boundary at the screen fidelity.
      - Builds a rich JSON context (including recent failures).
      - Runs a multi‑turn LLM loop that can call tools and eventually emit:
        - `suggested_params`: a candidate boundary seed.
        - `config_overrides`: tweaks such as constraint weights or proposal mix.

Important: **Planner currently only acts at cycle boundaries.**  
It does not yet supervise the inner ALM loop.

### 2.2 Coordinator (`ai_scientist/coordinator.py`)

- Owns three workers:
  - `OptimizationWorker`: performs ALM‑based optimization over candidates.
  - `ExplorationWorker`: generative / sampler based exploration.
  - `GeometerWorker`: geometric filtering of candidates.
- Strategy:
  - `decide_strategy(cycle, experiment_id)` returns `"EXPLORE"`, `"EXPLOIT"`,
    or `"HYBRID"` based on:
    - Early cycles → `"HYBRID"`.
    - Later cycles → checks HV delta from world model for stagnation → `"EXPLORE"`.
- `produce_candidates(cycle, experiment_id, n_candidates, template)`:
  - EXPLORE: just exploration worker.
  - EXPLOIT/HYBRID: exploration → geometer → optimization worker.
  - No explicit ALM state diagnostics, no calls back into the Planner.

### 2.3 Runner (`ai_scientist/runner.py`)

- Orchestrates:
  - Experiment configs and CLI flags.
  - Budget control (`BudgetController`).
  - Surrogate models (neural operator vs. bundle).
  - Generative models (VAE / diffusion).
  - Boundary seeding (rotating ellipse, NAE).
  - Problem evaluators (`evaluate_p1/p2/p3`).
  - Candidate proposal, surrogate ranking, and promotion.
- Physics/optimization glue:
  - `_objective_constraints` maps metrics from `forward_model` to:
    - P1: objective = `max_elongation`.
    - P3: objective = `aspect_ratio`.
    - Constraints via `tools.compute_constraint_margins`.
- Planner integration:
  - `plan_cycle` is used for P3 planning and to inject `suggested_params`
    into the candidate pool, but not to supervise the inner ALM iteration.
- Coordinator integration:
  - The current runner uses the Coordinator in its original EXPLORE/EXPLOIT
    role; the ASO loop described in the earlier version of this document is
    **not implemented**.

---

## 3. Target Architecture: Neuro‑Symbolic ASO Loop

The target architecture keeps the current modules but refines their roles.

### 3.1 Conceptual Loop

1. **Math step (Workers)**  
   `OptimizationWorker` runs a short ALM chunk, starting from a given
   `AugmentedLagrangianState` and returning:
   - Updated ALM state.
   - Number of ALM evaluations used.

2. **Translate (Coordinator)**  
   The Coordinator converts ALM state into a semantic diagnostics object:
   - `objective`, `max_violation`.
   - Per‑constraint violation + trend.
   - Simple health stats (e.g., bounds norm).
   - Status: `IN_PROGRESS`, `STAGNATION`, `FEASIBLE_FOUND`.

3. **Reason (Planner)**  
   `PlanningAgent.analyze_optimizer_diagnostics(diagnostics, cycle)` uses an
   LLM to:
   - Decide whether to continue, adjust hyperparameters, re‑seed, or stop.
   - Emit a structured `OptimizationDirective`.

4. **Direct (Coordinator)**  
   Coordinator applies the directive:
   - Update `cfg.constraint_weights`, `cfg.proposal_mix`, etc.
   - Pass ALM‑specific overrides (e.g. `penalty_parameters_increase_factor`)
     into the `OptimizationWorker` for the next chunk.

5. **Repeat** until:
   - Budget exhausted, or
   - Planner issues `STOP`, or
   - Convergence is detected.

### 3.2 Key Interfaces (Target, Not Yet Implemented)

These are **design contracts**, not promises that the current code already has
them. They are the end‑state for the refactor.

- Diagnostics (Coordinator → Planner):

```python
@dataclass
class OptimizerDiagnostics:
    step: int
    objective: float
    max_violation: float
    constraint_trends: dict[str, dict[str, float | str]]
    objective_delta: float
    status: str              # "IN_PROGRESS" | "STAGNATION" | "FEASIBLE_FOUND"
    narrative: list[str]
```

- Directive (Planner → Coordinator):

```python
@dataclass
class OptimizationDirective:
    action: str  # "CONTINUE" | "ADJUST" | "STOP"
    config_overrides: dict[str, Any] | None = None
    suggested_params: dict[str, Any] | None = None
    reasoning: str | None = None
```

- New Planner method:

```python
def analyze_optimizer_diagnostics(
    self,
    diagnostics: OptimizerDiagnostics,
    cycle_index: int,
) -> OptimizationDirective: ...
```

- New Coordinator entry point (ASO path):

```python
def produce_candidates_aso(
    self,
    cycle: int,
    experiment_id: int,
    eval_budget: int,
    template: ai_config.BoundaryTemplateConfig,
    initial_seeds: list[dict[str, Any]] | None,
    initial_config: ai_config.ExperimentConfig | None,
) -> list[dict[str, Any]]: ...
```

---

## 4. Gap Analysis (Planned vs Implemented)

This section explicitly calls out where the old version of this document was
misleading.

- **Planner**
  - Planned: `OptimizationDirective` + `analyze_optimizer_diagnostics` +
    dedicated orchestration gate.
  - Current: Only `plan_cycle` is implemented; there is no diagnostics‑driven
    supervision method and no `OptimizationDirective` type.

- **Coordinator**
  - Planned: ALM‑state‑driven ASO loop with `generate_diagnostics`,
    `apply_directive`, and `produce_candidates` running micro‑cycles.
  - Current: Strategy selector (`decide_strategy`) and single‑pass
    EXPLORE/EXPLOIT/HYBRID logic; no diagnostics translation and no link
    back into Planner.

- **Runner**
  - Planned: `initialize_architecture` wiring Planner and Coordinator plus a
    refactored `_run_cycle` that uses the ASO loop for screening.
  - Current: Rich but largely legacy flow that:
    - Uses Planner at the cycle level for P3 seed/config suggestions.
    - Uses Coordinator in its EXPLORE/EXPLOIT role.
    - Does not expose an ASO mode.

---

## 5. Incremental Migration Plan

The refactor should be done in small, testable steps. The high‑level phases:

### Phase 0 – Feature Flag and Safety Rails

- Add a config/CLI switch, e.g.:
  - `cfg.control_mode` or `--control-mode {legacy,aso}`.
- Default to `legacy`:
  - Existing behavior is unchanged.
  - ASO behavior is opt‑in for experiments.

### Phase 1 – Planner Extensions (Non‑Disruptive)

Goal: extend `PlanningAgent` without breaking `plan_cycle`.

- Add the `OptimizationDirective` dataclass to `ai_scientist/planner.py`.
- Add a new method:

  - `analyze_optimizer_diagnostics(self, diagnostics: Mapping[str, Any], cycle_index: int) -> OptimizationDirective`

- Internals:
  - Use `self.config.agent_gates` / provider selection like `plan_cycle`.
  - Use a concise system prompt defining:
    - Possible actions (`CONTINUE`, `ADJUST`, `STOP`).
    - Legal `config_overrides` keys (`constraint_weights`, `alm_settings`, `proposal_mix`).
  - Return `CONTINUE` with a simple reason if:
    - Agent gates are disabled, or
    - LLM call fails or produces invalid JSON.

This phase does **not** change any call sites; it just provides the interface.

### Phase 2 – Coordinator Diagnostics Layer

Goal: teach the Coordinator to translate ALM state into diagnostics and apply
directives, while still being callable in legacy mode.

- Add a small diagnostics helper in `ai_scientist/coordinator.py`:
  - `generate_diagnostics(alm_state, previous_state) -> OptimizerDiagnostics`.
  - Use only quantities already available from `AugmentedLagrangianState`:
    - `objective`, `constraints`, `penalty_parameters`, `bounds`.
  - Implement simple trend heuristics:
    - `increasing_violation` vs `decreasing_violation` using a fixed factor.
    - `STAGNATION` when `objective_delta` is tiny and `max_violation` is large.
- Add `apply_directive(directive, cfg) -> (cfg, worker_overrides)`:
  - Mutate config via `dataclasses.replace`; do not directly mutate fields.
  - Pass ALM overrides to the optimization worker via a small dict.
- Keep the existing `decide_strategy` and `produce_candidates` untouched.

### Phase 3 – ASO Candidate Production Path

Goal: implement a new ASO path alongside the existing one.

- Implement `produce_candidates_aso` in Coordinator:
  - Inputs:
    - `cycle`, `experiment_id`, `eval_budget`.
    - `template` and optional `initial_seeds`/`initial_config`.
  - Flow:
    1. If `initial_seeds` is empty:
       - Use `ExplorationWorker` and `GeometerWorker` to obtain at least one
         valid seed.
    2. Initialize ALM via `OptimizationWorker` with a small inner budget,
       e.g. 5–10 evaluations per inner step.
    3. After each inner step:
       - Call `generate_diagnostics`.
       - Call `planner.analyze_optimizer_diagnostics`.
       - Apply directive:
         - Update config and worker overrides.
         - Optionally reset ALM state if a new seed is injected.
    4. Stop when:
       - `eval_budget` is exhausted.
       - Planner returns `STOP`.
  - Return:
    - A list of candidate boundary parameter dicts, reusing the
      `OptimizationWorker`’s existing history structure.

No caller uses this yet; it is wired in the next phase.

### Phase 4 – Runner Integration (ASO Mode)

Goal: let experiments opt into ASO while preserving legacy behavior.

- In `runner.py`:
  - Add `initialize_architecture(cfg, world_model)` that:
    - Creates the surrogate + generative models.
    - Instantiates `PlanningAgent`.
    - Instantiates `Coordinator` with the planner and models.
  - In `_run_cycle`, branch on `control_mode` / CLI flag:
    - `legacy`:
      - Keep using the existing path:
        - Candidate proposal via `_propose_p3_candidates_for_cycle` etc.
        - Standard screening and promotion.
    - `aso`:
      - Use `planner.plan_cycle` to get:
        - Optional `suggested_params` and `config_overrides`.
      - Apply config overrides to a local `active_cfg`.
      - Derive an evaluation budget from `BudgetController.snapshot()`.
      - Call:

        ```python
        candidates = coordinator.produce_candidates_aso(
            cycle=cycle_number,
            experiment_id=experiment_id,
            eval_budget=active_budgets.screen_evals_per_cycle,
            template=active_cfg.boundary_template,
            initial_seeds=formatted_suggested_params,
            initial_config=active_cfg,
        )
        ```

      - Then feed `candidates` into the existing screening + reporting code.

### Phase 5 – Cleanup and Promotion

- Once ASO is validated:
  - Consider making `aso` the default control mode for P3 (and, if desired,
    P2), leaving `legacy` as a fallback.
  - Optionally:
    - Deprecate or simplify `decide_strategy`.
    - Remove duplicate exploration logic between `runner` and `Coordinator`.
  - Keep documentation and configs updated so new experiments start with ASO.

---

## 6. Success Criteria

We consider the ASO refactor successful when:

- For P3:
  - The ASO loop reaches feasibility and matches or exceeds the legacy
    hypervolume and best points using comparable wall‑clock budgets.
  - High‑level agent interventions (e.g. boosting MHD constraint weights under
    repeated violations) improve robustness in difficult regions.
- For P1/P2:
  - The same machinery can be reused with minimal problem‑specific code:
    only the objective/constraint mapping differs.
- Operationally:
  - `control_mode=legacy` reproduces current behavior within noise.
  - `control_mode=aso` is deterministic enough across runs to be usable in
    CI/regression tests.

---

## 7. Notes and Constraints

- **Do not modify** original code in the upstream `constellaration` or
  `vmecpp` repositories. All behavior changes should live in
  `ai_scientist` and surrounding orchestration.
- Favor:
  - Pure, typed data structures (`dataclass`, immutable configs).
  - Simple, linear control flow in the ASO loop; avoid deep callback chains.
  - Small, testable units:
    - Unit tests for `generate_diagnostics` trend logic.
    - Unit tests for `apply_directive` config merging.
    - Smoke tests for the ASO path with a stub `OptimizationWorker`.

This updated plan should now match the current state of the codebase while
providing a clear, stepwise path to the full Neuro‑Symbolic ASO architecture.
