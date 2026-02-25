# Integrated Evolution Plan for P1/P2/P3

## 1) Goal

Build one robust optimization stack that integrates ideas from AlphaEvolve, ShinkaEvolve, GEA, DGM, and ALMA while preserving deterministic physics evaluation and reproducibility.

---

## 2) Codebase Reality (Verified)

### Runtime anchors

- `ai_scientist/experiment_runner.py` orchestrates cycles and stage gates.
- `ai_scientist/cycle_executor.py` is still the central runtime path.
- `scripts/p3_governor.py` is deterministic and recipe-driven (`_select_recipe`), not LLM- or bandit-driven.
- `scripts/p3_worker.py` is the deterministic evaluator and must remain SSOT.

### Hard gaps before adaptive evolution works

1. **No adaptive control plane yet**
   - No novelty service, no model-router bandit, no parent-group policy in runtime.
2. **DB schema cannot support novelty/group evolution signals**
   - Current `candidates`/`metrics` tables store params, objective/HV/feasibility only.
   - No parent lineage, novelty score, operator family score, trace embedding, or model-route decision fields.
3. **Deprecated wrappers still sit on the active path**
   - Main cycle execution already routes through `forward_model_batch`.
   - Deprecated `tools.evaluate_*` wrappers still persist in bootstrap/tooling/experimental paths and block clean removal.

---

## 3) Integrated Architecture (Target)

### 3.1 Outer adaptive loop (mainly P3)

Per batch:

1. Select parent group using performance + child-count penalty + novelty distance.
2. Aggregate traces (success/failure deltas, operator metadata, constraints hit patterns).
3. Reflect (LLM agents) to generate structured proposals (not direct evaluator calls).
4. Apply novelty gate and hard validity checks.
5. Enqueue candidates with full lineage metadata.
6. Evaluate deterministically (screen → promoted → high fidelity).
7. Update archive, operator bandit, and model-router bandit.

### 3.2 Inner problem loops

- **P1/P2:** keep ALM/NGOpt loop as optimizer backbone; add only lightweight adaptive seed selection + novelty gating.
- **P3:** make governor adaptive and archive-driven; deterministic evaluator remains unchanged.

---

## 4) Problem-Specific Integration

## 4.1 P1 (lightweight)

- Keep `scripts/p1_alm_ngopt_multifidelity.py` unchanged as core optimizer.
- Add adaptive restart seed selector (performance + diversity + child penalty).
- Add novelty rejector for restart perturbations to avoid replaying equivalent neighborhoods.

## 4.2 P2 (lightweight + feasibility-aware)

- Keep `scripts/p2_alm_ngopt_multifidelity.py` unchanged as core optimizer.
- Reuse P1 seed selector with stricter feasibility neighborhood checks (QI + vacuum well focused).
- Add constrained novelty gate to reduce expensive infeasible repeats.

## 4.3 P3 (full adaptive governor)

- Replace static `_select_recipe` dominance with:
  - parent-group selector
  - operator-family bandit
  - model-router bandit for reflection/proposal generation
  - novelty gate pre-enqueue
- Keep `scripts/p3_worker.py` deterministic and immutable.
- Persist move-family and lineage metadata in DB (not only artifact JSON).

---

## 5) Dead Code Analysis and Retirement

## 5.1 Dead/unintegrated now

1. `ai_scientist/curriculum.py`
   - Marked as implemented but not integrated into cycle execution.
2. `ai_scientist/optim/search.py` (`P3SearchWrapper`)
   - Appears test/doc oriented; not on main runtime path.

## 5.2 Becomes dead if new plan lands

1. `scripts/p3_governor.py` static recipe tree (`_select_recipe`, static branch logic).
2. Duplicate enqueue internals in both:
   - `scripts/p3_propose.py`
   - `scripts/p3_enqueue_submission.py`
3. Legacy fallback/legacy ALM branch in `ai_scientist/cycle_executor.py`.
4. Deprecated wrappers in `ai_scientist/tools/evaluation.py` once non-runtime callers also move to `forward_model_batch`.

## 5.3 Mandatory removal order

1. Add replacement path + feature flags.
2. Add telemetry counters for old path usage.
3. Switch defaults after canary success.
4. Remove dead path only when usage is zero for release window and tests are migrated.

---

## 6) Migration Prerequisites (Must-Have)

1. **Schema migration**
   - Add fields/tables for: `parent_candidate_ids`, `operator_family`, `novelty_score`, `model_route`, `trace_summary`, `constraint_signature`.
2. **Shared enqueue module**
   - Create one enqueue library used by both propose and submission CLIs.
3. **Evaluation API migration**
   - Move remaining bootstrap/tooling/experimental callers from deprecated wrappers to `forward_model_batch`.
4. **Governor contract**
   - Define strict IO contract for adaptive governor decisions and rollback fallback.

---

## 7) Execution Phases

## Phase 0 — Baseline + Telemetry Lock

- Freeze comparable P1/P2/P3 baselines.
- Add counters for operator usage, novelty reject rate, feasible yield, and fallback-path usage.

## Phase 1 — Data Plane + Shared Primitives

- Implement schema migration and shared enqueue module first.
- Add parent selector, novelty gate, trace aggregator, and model router (library only; no default behavior switch).

## Phase 2 — P3 Adaptive Governor (Flagged)

- Integrate adaptive governor behind feature flag.
- Keep static recipe path as rollback.
- Run A/B against current governor using same budgets.

## Phase 3 — P1/P2 Lightweight Adoption

- Add adaptive seed selector + novelty gating around restarts only.
- Keep ALM/NGOpt loop unchanged.

## Phase 4 — Retirement

- Remove static P3 recipe path after sustained superiority.
- Consolidate duplicate enqueue logic.
- Resolve curriculum (`integrate` or `delete`).
- Remove deprecated evaluation wrappers last.

---

## 8) Acceptance Criteria

## P1

- No regression in best feasible objective and convergence budget.

## P2

- Match or improve feasible hit rate and best `L_gradB` at fixed budget.

## P3

- Improve feasible yield per 100 HF evaluations.
- Improve hypervolume slope and final HV.
- Shorten recovery after stagnation windows.

## Global

- Deterministic evaluator unchanged.
- Reproducible lineage from DB metadata alone.
- Dead-code removals validated by full tests and runbook smoke checks.

---

## 9) Immediate Next Actions

1. Implement schema migration and shared enqueue library.
2. Add telemetry for currently deprecated/legacy paths.
3. Integrate adaptive governor behind flag and run A/B.
4. Migrate remaining bootstrap/tooling/experimental evaluation calls to `forward_model_batch`.
5. Start dead-code retirement only after objective gates pass.

---

## 10) Reviewer-Verified Open Gaps (Current State)

1. Schema still lacks lineage/novelty/operator persistence fields required by adaptive selection.
2. Duplicate enqueue implementations still exist in `p3_propose` and `p3_enqueue_submission`.
3. P3 governor still uses static `_select_recipe` flow.
4. Deprecated wrappers are no longer primary runtime path but are still used in bootstrap/tooling/experimental paths.
