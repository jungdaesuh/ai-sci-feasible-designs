# Physics Correctness vs Benchmark Definitions (End-to-End)

This document specifies the **requirements and checklist** for a focused “benchmark alignment” audit.

The goal is **not** to prove that VMEC++/Boozer physics is correct. The goal is to ensure that this repo’s
code uses **the same constraint/objective names, sign conventions, ordering, and transformations**
as the official `constellaration` benchmark definitions, end-to-end.

---

## 1) Scope and Goal

### Goal
Confirm that **every constraint/objective name + sign** is consistent across:

1. `ai_scientist/problems.py` (problem wrappers and objective extraction)
2. `ai_scientist/forward_model.py:compute_constraint_margins` (normalized violation convention + stage gating)
3. ALM wiring (constraint vector ordering + objective direction), including:
   - `ai_scientist/optim/alm_bridge.py`
   - `ai_scientist/optim/differentiable.py:optimize_alm_inner_loop`
   - `ai_scientist/cycle_executor.py` (any predictor/adapter logic)
4. Surrogate training targets and storage columns:
   - `ai_scientist/objective_types.py`
   - `ai_scientist/memory/schema.py` + `ai_scientist/memory/repository.py`
5. Reporting and telemetry:
   - `ai_scientist/reporting.py`
   - any JSONL/DB writes under `ai_scientist/memory/*`

### Non-goals
- Validating VMEC++ numerics, Boozer transform accuracy, or real-world “physics truth”.
- Tuning thresholds/weights for better performance.
- Changing benchmark definitions.

---

## 2) Canonical Benchmarks and Conventions

### Benchmark reference implementation
The canonical definitions live in the bundled `constellaration` package, especially:
- `constellaration/src/constellaration/problems.py`
- `constellaration/src/constellaration/forward_model.py`

### Constraint violation sign convention
Across the codebase, **constraint margins/violations must satisfy**:
- **Positive** value means **violation** (infeasible).
- **≤ tolerance** means **feasible**.

Official relative tolerance used by the benchmark:
- `_DEFAULT_RELATIVE_TOLERANCE = 1e-2` in `constellaration/src/constellaration/problems.py`

### Constraint normalization convention (benchmark style)
The benchmark normalizes per-constraint violations by a denominator that is typically `abs(bound)`:

- Upper bound constraint `x <= ub`:
  - violation = `x - ub`
  - normalized = `(x - ub) / abs(ub)`
- Lower bound constraint `x >= lb`:
  - violation = `lb - x`
  - normalized = `(lb - x) / abs(lb)`

Special case:
- `vacuum_well >= 0.0` uses denominator `max(1e-1, abs(lb))` to avoid divide-by-zero.

### Objective vocabulary (must not drift)
This repo uses **three different “objective-like” scalars**. They must not be mixed:

1. **Physics objective** (benchmark goal)
   - Used to describe the problem goal (min/max depends on problem).
   - Implemented in `ai_scientist/forward_model.py:compute_objective`.
2. **Ranking score** (always “higher is better”)
   - Used for candidate ranking and some surrogate training.
   - Implemented in `ai_scientist/objective_types.py:compute_ranking_score`.
3. **ALM objective** (always minimized)
   - Used inside constellaration’s Augmented Lagrangian solver.
   - For P2/P3 commonly uses `20 - gradient`.

---

## 3) Problem-by-Problem Alignment Matrix (Expected Semantics)

### P1 (GeometricalProblem)
**Benchmark objective**
- Minimize: `max_elongation`

**Benchmark constraints**
- `aspect_ratio <= 4.0`
- `average_triangularity <= -0.5`
- `edge_rotational_transform_over_n_field_periods >= 0.3`

**Notes**
- P1 has **no QI**, **no vacuum well**, **no flux compression** constraints.
- If any component penalizes QI for P1, that is a benchmark misalignment.

### P2 (SimpleToBuildQIStellarator)
**Benchmark objective**
- Maximize: `minimum_normalized_magnetic_gradient_scale_length`

**Benchmark constraints**
- `aspect_ratio <= 10.0`
- `edge_rotational_transform_over_n_field_periods >= 0.25`
- `log10(qi) <= -4.0` (QI residual is constrained in log space)
- `edge_magnetic_mirror_ratio <= 0.2`
- `max_elongation <= 5.0`

**Notes**
- P2 does **not** include `vacuum_well >= 0.0` as a constraint in the official benchmark.

### P3 (MHDStableQIStellarator)
**Benchmark objectives (multi-objective)**
- Minimize: `aspect_ratio`
- Maximize: `minimum_normalized_magnetic_gradient_scale_length`

**Benchmark constraints**
- `edge_rotational_transform_over_n_field_periods >= 0.25`
- `log10(qi) <= -3.5`
- `edge_magnetic_mirror_ratio <= 0.25`
- `flux_compression_in_regions_of_bad_curvature <= 0.9`
- `vacuum_well >= 0.0`

**Notes**
- Any single-scalar “P3 objective” is a **proxy** or a scalarization. It must be labeled clearly.
- Hypervolume calculations must use the minimization transform `(-gradient, aspect_ratio)`.

---

## 4) Hard Requirements (What Must Match)

### R1: Constraint names and ordering must be stable
The following must agree (including ordering):
- `ai_scientist/constraints.py:get_constraint_names(problem)`
- `ai_scientist/problems.py:<Problem>.constraint_names`
- `ai_scientist/forward_model.py:compute_constraint_margins(...).keys()`
- `ai_scientist/forward_model.py:EvaluationResult.constraint_names`

Why this matters:
- Any ALM penalty vector or logging that assumes a fixed order will become silently wrong if order drifts.

### R2: “Positive means violation” everywhere
For any module that computes constraints:
- If a constraint is violated, its reported value must become **more positive**.
- Feasibility must be computed via `max(0, max(constraint_margins))` (or equivalent infinity norm of positive parts).

### R3: Stage gating must never claim “full feasibility” with missing constraints
If a stage intentionally skips physics constraints (e.g., `promote` skipping QI/Boozer), it must be treated as:
- **screening feasibility only**, not benchmark feasibility
- explicitly labeled and/or warned (e.g., “QI skipped; feasibility incomplete”)

### R4: Objective sign/direction must match the caller’s expectation
Every place that uses a scalar objective must consistently specify:
- objective value meaning (what metric it represents)
- direction (minimize vs maximize)
- any transforms (negation, `20 - x`, ratio scalarizations)

This is especially critical at these integration points:
- Coordinator training (`ai_scientist/coordinator.py`)
- Surrogate ranking (`ai_scientist/optim/surrogate_v2.py:rank_candidates`)
- ALM inner loop objective (`ai_scientist/optim/differentiable.py:optimize_alm_inner_loop`)
- Cycle reporting (`ai_scientist/reporting.py`)

### R5: Surrogate training targets must match how they are consumed
If a surrogate is trained with `minimize_objective=True`, then:
- downstream rankers must treat it as a minimization quantity (do not manually negate targets)

If a surrogate is trained on a “higher is better” score proxy (e.g., `gradient/aspect`), then:
- downstream rankers must treat it as a maximization quantity.

Additionally:
- If QI feasibility is defined on `log10(qi)` in the benchmark, any surrogate QI head or feasibility proxy must either:
  - predict `log10(qi)` directly, or
  - predict raw `qi` and convert consistently before applying the threshold.

---

## 5) Audit Procedure (Step-by-Step)

### Step A: Establish the canonical benchmark definitions
Confirm benchmark constraints/objectives directly in:
- `constellaration/src/constellaration/problems.py`

Extract (for each problem):
- objective value and direction
- constraint list, bounds, and normalization denominators
- feasibility tolerance

### Step B: Confirm local wrappers match the benchmark
Check:
- `ai_scientist/problems.py`:
  - constants match benchmark bounds
  - normalized violation formulas match benchmark normalization
  - objective extraction uses correct metric and direction

### Step C: Confirm centralized forward model is benchmark-compatible
Check:
- `ai_scientist/forward_model.py:compute_constraint_margins`
  - keys match `constraints.py`
  - stage gating is explicit (screen vs promote vs p2/p3 vs high)
  - `max_violation` correctly treats NaN/inf as infeasible

### Step D: Confirm ALM bridge ordering and sign
Check:
- `ai_scientist/optim/differentiable.py:optimize_alm_inner_loop`
  - constraint vector assembled in the exact order the ALM state expects
  - constraint magnitudes have the same sign convention (“positive = violation”)
  - objective term is consistent with the selected `TargetKind`

If a predictor builds “predicted ALM constraints” (e.g. in `ai_scientist/cycle_executor.py`), verify:
- it is in the same canonical order and uses the same threshold transforms.

### Step E: Confirm surrogate training columns and semantics
Check:
- DB schema: `ai_scientist/memory/schema.py` (`metrics.hv`, `metrics.objective`, `metrics.feasibility`)
- Data loading: `ai_scientist/memory/repository.py:surrogate_training_data`
  - correct mapping of `gradient_proxy` ↔ DB `hv` column
- Training call sites (Coordinator and CycleExecutor):
  - use correct `target` column for each problem
  - pass consistent `minimize_objective` flag

### Step F: Confirm reporting matches semantics
Check:
- any field named `hv` is either:
  - true hypervolume (set-level statistic), or
  - explicitly a proxy (e.g., “gradient_proxy”)

Ensure P2/P3 objectives aren’t accidentally reported as minimized if they are maximized (or vice versa).

---

## 6) Deliverables and Acceptance Criteria

### Deliverables
- A short report listing:
  - any mismatches found (name/order/sign/transform)
  - impacted files and why it matters
  - minimal patch proposals
- Regression tests for any high-impact mismatch that could silently reappear.

### Acceptance criteria
- No mismatches in constraint list names or ordering across SSOT modules.
- No mismatches in objective direction between producer and consumer.
- No code path labels an evaluation as “benchmark feasible” if required constraints were skipped.
- P3 hypervolume always uses minimization form `(-gradient, aspect_ratio)` with the same reference point as the benchmark.

---

## 7) Suggested Commands (Local)

These commands help mechanically confirm alignment without manual browsing:

- Grep for constraint name list usage:
  - `rg -n "get_constraint_names\\(|constraint_names" ai_scientist`
- Grep for log-space QI usage:
  - `rg -n "log10\\(qi\\)|qi_log10|log10_qi" ai_scientist`
- Run constraint/unit tests:
  - `pytest -q tests/test_problems.py`
  - `pytest -q tests/optim/test_differentiable_constraints.py`
  - `pytest -q tests/optim`
