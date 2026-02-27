# Integrated Evolution Plan for P1/P2/P3

## 1) Goal

Build one robust optimization stack that integrates ideas from AlphaEvolve, ShinkaEvolve, GEA, DGM, and ALMA while preserving deterministic physics evaluation and reproducibility.

---

## 2) Codebase Reality (Verified)

### Runtime anchors

- `ai_scientist/experiment_runner.py` orchestrates cycles and stage gates.
- `ai_scientist/cycle_executor.py` is still the central runtime path.
- `scripts/p3_governor.py` default path is deterministic/recipe-driven (`_select_recipe`), while `--adaptive` now provides parent-group selection + operator-family ranking + novelty gating with static rollback.
- `scripts/p3_worker.py` is the deterministic evaluator and must remain SSOT.

### Hard gaps before full adaptive-evolution closure

1. **P3 adaptive path and performance-evidence gate are landed**
   - `--adaptive` path (parent group + operator-family bandit + novelty gate + fallback) is implemented.
   - Meaningful fixed-budget non-regression evidence gate is closed (`M2.4`, 2026-02-25; `exp13` vs `exp12`, budget `20`).
2. **P1/P2 lightweight adaptive layer is landed**
   - Adaptive restart seed selector is landed and hardening-closed (`M3.1`), including runtime + AST wiring coverage and reviewer PASS.
   - Constrained novelty gating is now landed (`M3.2`, 2026-02-25).
   - M3.3 fixed-budget evidence gate is closed (`M3.3`, 2026-02-26) via budget-20 P1/P2 validator passes in legacy metadata mode (`artifacts/m3/m33_probe/p1_report_b20_legacy.json`, `artifacts/m3/m33_probe/p2_report_b20_legacy.json`).
3. **Cross-problem policy hardening is closed**
   - Shared two-stage novelty gate interface and model-router reward contract are implemented (`M3.4` + `M3.5`, 2026-02-25).
   - M3.6 validator tooling and paired-run evidence are both landed (`scripts/m3_policy_hardening_validate.py`, 2026-02-26; strict pass artifact: `artifacts/m3/m36_probe/m36_policy_report_b21.json`).
4. **Deprecated wrappers still sit on non-primary paths**
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

- Keep the runner-based P2 path (`python -m ai_scientist.runner --problem p2`) as the core optimizer.
- Keep `experiments/p1_p2/p2_alm_ngopt_multifidelity.py` as an optional research harness, not production SSOT.
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
2. Legacy fallback/legacy ALM branch in `ai_scientist/cycle_executor.py`.
3. Deprecated wrappers in `ai_scientist/tools/evaluation.py` once non-runtime callers also move to `forward_model_batch`.

Shared enqueue internals are already consolidated in `ai_scientist/p3_enqueue.py` (done 2026-02-25).

## 5.3 Mandatory removal order

1. Add replacement path + feature flags.
2. Add telemetry counters for old path usage.
3. Switch defaults after canary success.
4. Remove dead path only when usage is zero for release window and tests are migrated.

---

## 6) Migration Prerequisites (Must-Have)

1. **Schema migration**
   - Add fields/tables for: `parent_candidate_ids`, `operator_family`, `novelty_score`, `model_route`, `trace_summary`, `constraint_signature`.
   - Status: completed (2026-02-25).
2. **Shared enqueue module**
   - Create one enqueue library used by both propose and submission CLIs.
   - Status: completed (2026-02-25).
3. **Evaluation API migration**
   - Move remaining bootstrap/tooling/experimental callers from deprecated wrappers to `forward_model_batch`.
4. **Governor contract**
   - Define strict IO contract for adaptive governor decisions and rollback fallback.
   - Status: contract gate completed (M2.3, 2026-02-25); performance-evidence gate completed (M2.4, 2026-02-25).

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

1. Run strict-metadata P1/P2 paired fixed-budget comparisons (without `--allow-legacy-restart-metadata`) so adaptive arms carry restart-label evidence directly in-budget.
2. Execute codex-native expansion validation gates (`M4.5`-`M4.6`) after the pinned canary-default rollout (`M4.4`).
3. Migrate remaining bootstrap/tooling/experimental evaluation callers away from deprecated wrappers.
4. Run a fresh-proposal P3 paired A/B (non-seeded) for adaptive-vs-static performance characterization beyond gate closure.

---

## 10) Reviewer-Verified Open Gaps (Current State)

1. P3 adaptive contract and performance-evidence gates are closed (`M2.3` + `M2.4`).
   - Latest passing evidence: budget `20`, static `exp13` vs adaptive `exp12`, `m24_performance_evidence_pass=true`.
2. P1/P2 adaptive restart seed selector + constrained novelty gate + fixed-budget evidence are closed (`M3.1` + `M3.2` + `M3.3`), with a follow-up hardening task to refresh strict restart-label metadata evidence.
3. Cross-problem policy hardening is closed through fixed-budget validation: two-stage novelty gate and router reward contract are landed (`M3.4` + `M3.5`), and strict M3.6 evidence gate passed (`M3.6`, artifact: `artifacts/m3/m36_probe/m36_policy_report_b21.json`).
4. Native codex subscription integration is still partial: canary defaults are pinned, but local adapter server + OAuth/profile management remain pending.
5. Deprecated wrappers are no longer primary runtime path but are still used in bootstrap/tooling/experimental paths.
