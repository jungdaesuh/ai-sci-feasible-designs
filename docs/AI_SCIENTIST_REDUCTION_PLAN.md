# AI Scientist Reduction Plan

Date: 2026-02-26  
Owner: AI Scientist maintainers  
Status: In Progress (PR-1 and PR-2 complete, PR-3/PR-4 pending)

## Goal

Reduce `ai_scientist` file/LOC surface without changing behavior.

Primary target:
- [ ] Remove dead or near-dead modules with no runtime role.
- [ ] Merge overlapping logic between `forward_model` and `tools/evaluation`.
- [x] Deduplicate repeated candidate-production pipeline logic in `coordinator`.

Expected outcome (first pass):
- [ ] `~820-1,118` LOC reduction (broad low/medium path).
- [ ] `3-4` file reduction minimum.

## Baseline (Current Snapshot)

- `ai_scientist` Python files: `72`
- `ai_scientist` LOC: `26,126`
- Largest files:
  - `ai_scientist/cycle_executor.py` (`2722`)
  - `ai_scientist/optim/generative.py` (`1488`)
  - `ai_scientist/memory/repository.py` (`1355`)
  - `ai_scientist/optim/surrogate_v2.py` (`1284`)
  - `ai_scientist/optim/geometry.py` (`1182`)
  - `ai_scientist/planner.py` (`1092`)
  - `ai_scientist/coordinator.py` (`1074`)
  - `ai_scientist/forward_model.py` (`994`)

## Scope

In scope:
- `ai_scientist/model_endpoint.py`
- `ai_scientist/guards.py`
- `ai_scientist/optim/validation.py`
- `ai_scientist/coordinator.py`
- `ai_scientist/tools/evaluation.py`
- `ai_scientist/tools/design_manipulation.py` (only if needed for import-safe cleanup)

Out of scope (this plan iteration):
- Large behavioral rewrites in `cycle_executor.py`
- ALM backend strategy redesign
- Changes in `constellaration/` or `vmecpp/`

## Guardrails

- [ ] No dynamic imports.
- [ ] No `any` casting.
- [ ] No defensive try/catch additions beyond existing behavior.
- [ ] No changes to upstream external repos.
- [ ] Preserve current public tool API names (`evaluate_p1`, `evaluate_p2`, `evaluate_p3`).

## Pinned Engineering Principles

- [ ] KISS: Prefer the simplest design that works.
- [ ] YAGNI: Do not introduce abstractions/features not needed for this plan.
- [ ] DRY: Eliminate duplicated logic once behavior parity is verified.
- [ ] SSOT: Keep one canonical implementation per concern.
- [ ] SRP: Each module/function should have one clear responsibility.
- [ ] SOLID: Keep interfaces stable and extension-friendly.
- [ ] Composition over inheritance: Reuse behavior via small composable helpers.
- [ ] Functional/immutable style: Prefer pure transformations and avoid mutation-heavy flows.
- [ ] Readability first: Optimize for maintainability over cleverness.
- [ ] Separation of concerns: Keep orchestration, evaluation, and config parsing clearly separated.

## Workstream A: Orphan Module Cleanup (Low Risk)

Hypothesis: these modules have no runtime/test call sites outside themselves/docs.

Targets:
- [ ] `ai_scientist/model_endpoint.py` (`199 LOC`)
- [ ] `ai_scientist/guards.py` (`37 LOC`)
- [ ] `ai_scientist/optim/validation.py` (`162 LOC`)

Tasks:
- [ ] Confirm no runtime imports with `rg` over `ai_scientist tests scripts`.
- [ ] Remove module files.
- [ ] Remove stale doc references to deleted modules (if any).
- [ ] Run targeted tests for impacted areas.

Exit criteria:
- [ ] No import errors in test collection for touched modules.
- [ ] No broken references in docs for deleted paths.

Estimated reduction:
- [ ] `398 LOC`
- [ ] `3 files`

## Workstream B: Coordinator Pipeline Dedup (Medium Risk)

Problem:
- `EXPLOIT` and `HYBRID` branches in `Coordinator.produce_candidates` are near-identical.

Tasks:
- [x] Extract shared stage runner helper for stages 1-6.
- [x] Keep `EXPLORE` branch behavior unchanged.
- [x] Keep log semantics equivalent (same stage reporting intent).
- [x] Keep retraining trigger path unchanged.

Suggested implementation shape:
- [ ] `_run_standard_pipeline(cycle, n_candidates, allow_rl: bool) -> list[dict]`
- [ ] Strategy block only decides mode flags and context knobs.

Validation:
- [x] Existing coordinator-related tests pass.
- [ ] No change in returned candidate schema.

Estimated reduction:
- [ ] `~120-180 LOC`

## Workstream C: forward_model vs tools/evaluation Consolidation (Medium Risk)

Problem:
- `tools/evaluation` duplicates helper patterns already in `forward_model`.
- Some helpers are dead; some are shared by `tools/design_manipulation`.

Tasks (safe order):
- [ ] Remove dead internal helpers in `tools/evaluation`:
  - [ ] `_hash_params`
  - [ ] `_evaluate_cached_stage`
  - [ ] `_log10_or_large` (if not referenced after cleanup)
- [ ] Keep shared helpers still consumed by `tools/design_manipulation`:
  - [ ] `_derive_schema_from_params`
  - [ ] `_coefficient_from_matrix`
  - [ ] `_quantize_float`
- [ ] Consolidate repeated `evaluate_p1/p2/p3` scaffolding through a shared helper while preserving outputs.
- [ ] Preserve public exports and tool-call compatibility in `ai_scientist/tools/__init__.py`.

Validation:
- [ ] `tests/tools/test_tools_p1.py`
- [ ] `tests/tools/test_tools_p2.py`
- [ ] `tests/tools/test_tools_p3.py`
- [ ] `tests/tools/test_tools_reliability.py`
- [ ] `tests/test_planner_problem_routing.py`

Estimated reduction:
- [ ] Conservative: `~120-220 LOC`
- [ ] Broader: `~260-420 LOC`

## Workstream D: Small Type/Config Duplication Cleanup (Low-Med Risk)

Opportunities:
- Duplicate `ProblemEvaluator` protocol in:
  - [ ] `ai_scientist/adapter.py`
  - [ ] `ai_scientist/fidelity_controller.py`
  - [ ] `ai_scientist/cycle_executor.py`
- Duplicate `CurriculumConfig` declarations in:
  - [ ] `ai_scientist/config.py`
  - [ ] `ai_scientist/curriculum.py`

Tasks:
- [ ] Create one SSOT `ProblemEvaluator` protocol location and import it.
- [ ] Decide SSOT for curriculum config type and align both files.
- [ ] Keep current config serialization behavior unchanged.

Estimated reduction:
- [ ] `~40-120 LOC`

## Execution Plan (PR Sequence)

PR-1 Safe cleanup:
- [ ] Workstream A only
- [ ] Validate targeted tests
- [ ] Merge

PR-2 Structural dedupe:
- [x] Workstream B
- [x] Run coordinator/planner/cycle tests
- [ ] Merge

PR-3 Evaluation consolidation:
- [ ] Workstream C
- [ ] Run tools + planner + selected cycle tests
- [ ] Merge

PR-4 Optional finishing pass:
- [ ] Workstream D
- [ ] Run full targeted suite
- [ ] Merge

## TDD Workflow (Required)

Apply Red-Green-Refactor for every workstream PR.

Global TDD rules:
- [ ] Red: add/adjust failing tests that capture current contract or intended no-regression behavior.
- [ ] Green: implement the minimum code change to pass tests.
- [ ] Refactor: simplify/cleanup only after green, while keeping tests green.
- [ ] Commit order per PR: tests-first commit, implementation commit, optional cleanup commit.

Workstream A TDD (orphan module cleanup):
- [ ] Red: add import-collection safety test to assert no runtime path depends on removed modules.
- [ ] Green: delete orphan modules and update references.
- [ ] Refactor: remove stale docs/comments and keep imports minimal.

Workstream B TDD (coordinator dedupe):
- [ ] Red: add characterization tests for `produce_candidates` flow by strategy (`EXPLORE`, `EXPLOIT`, `HYBRID`) with deterministic mocks.
- [ ] Red: assert candidate count/order/schema invariants and stage invocation order.
- [ ] Green: extract shared pipeline helper and wire strategy branches.
- [ ] Refactor: simplify duplicated context assembly/log formatting without behavior drift.

Workstream C TDD (evaluation consolidation):
- [ ] Red: add contract tests for `evaluate_p1/p2/p3` response keys/types and feasibility/objective semantics.
- [ ] Red: add regression test for `tools/design_manipulation` imports that rely on shared helper functions.
- [ ] Green: remove dead helpers and consolidate repeated scaffolding.
- [ ] Refactor: centralize helper ownership while preserving public exports.

Workstream D TDD (optional type/config dedupe):
- [ ] Red: add tests for protocol/config import paths used by call sites.
- [ ] Green: move to SSOT definitions and update imports.
- [ ] Refactor: tighten type/docs alignment.

## Validation Checklist

Core quality gates:
- [ ] `ruff check .`
- [ ] `python3 -m py_compile ai_scientist/*.py`

Targeted functional tests:
- [ ] `python3 -m pytest -q tests/test_planner_problem_routing.py`
- [ ] `python3 -m pytest -q tests/tools/test_tools_p1.py tests/tools/test_tools_p2.py tests/tools/test_tools_p3.py tests/tools/test_tools_reliability.py`
- [ ] `python3 -m pytest -q tests/test_p3_governor_wiring.py tests/test_p3_governor_ab.py`

Known environment caveats:
- [ ] `jax`/`pymoo` dependent tests may be skipped or blocked locally.

## Risk Register

- [ ] Risk: Removing modules referenced only by docs but expected by users.
  - Mitigation: mention removals in changelog/docs and provide migration note.
- [ ] Risk: Coordinator dedupe alters candidate ordering subtly.
  - Mitigation: preserve flow order and key selection logic exactly.
- [ ] Risk: Evaluation refactor changes schema fields.
  - Mitigation: snapshot test output keys for `evaluate_p1/p2/p3`.

## Done Definition

- [ ] File count reduced by at least `3`.
- [ ] LOC reduced by at least `500` (minimum acceptable) or within target band.
- [ ] No regression in targeted test suite.
- [ ] Docs updated for any removed or renamed module.
- [ ] Follow-up backlog items created for high-risk deferrals.

## Progress Tracker

### PR-1: Orphans
- [ ] Analysis complete
- [ ] Code changes complete
- [ ] Tests pass
- [ ] Merged

### PR-2: Coordinator
- [x] Analysis complete
- [x] Code changes complete
- [x] Tests pass
- [ ] Merged

### PR-3: Evaluation
- [ ] Analysis complete
- [ ] Code changes complete
- [ ] Tests pass
- [ ] Merged

### PR-4: Optional Dedupe
- [ ] Analysis complete
- [ ] Code changes complete
- [ ] Tests pass
- [ ] Merged
