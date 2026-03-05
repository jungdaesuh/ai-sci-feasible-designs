# Harness Cleanup Execution Plan (General P1/P2/P3)

Date: 2026-03-04
Document Role: Execution plan
Status: **Superseded** by `docs/harness/HARNESS_CODEGEN_PLAN.md` (2026-03-04)
Owner: Harness maintainers
Last Updated: 2026-03-04
Applies To: `chore/docs-cleanup`

Related docs:
- `docs/harness/AUTONOMOUS_HARNESS_PLAN.md` (strategy)
- `docs/harness/HARNESS_PORTING_BLUEPRINT.md` (architecture)
- `docs/harness/HARNESS_DOC_INDEX.md` (index + status)

## Objective
Convert current mixed/P3-biased runtime into one clean P1/P2/P3 harness while preserving working behavior.

## Global Rule
- Governor is sole stop authority.
- Persistent decision client/gateway + fresh DB-derived observation each cycle.

## Harness Path Contract (Strangler Toggle)
- Runtime selector: `--harness-path legacy|new`.
- Optional env override: `HARNESS_PATH=legacy|new`.
- Default behavior until parity is proven: `legacy`.
- Governor must persist `harness_path` in each cycle record for attribution and replay.

## Phase Tracker
- Prerequisite 0: In progress
- Phase 0: In progress
- Phase 1-7, 7.5, 8: Pending

## Prerequisite 0: Compatibility Contract (Blocker)
Tasks:
1. Restore or retarget missing legacy entrypoints required by tests (`scripts/p3_governor.py`, `scripts/p3_enqueue_submission.py`).
2. Ensure compatibility test collection is stable before migration changes.
3. Record compatibility decision (wrapper restore vs test retarget) in docs.

Deliverables:
- Passing compatibility test collection for legacy contract tests.
- Short compatibility note in docs with chosen path.

Validation:
- `pytest -q tests/test_p3_governor_contract.py tests/test_p3_enqueue_submission.py`

Acceptance criteria:
- Compatibility contract is green and reproducible.
- Migration can compare `legacy` vs `new` from a known-good baseline.

## Phase 0: Baseline and Safety
Tasks:
1. Freeze reproducible baseline run config and DB snapshot.
2. Capture current stop/fallback behavior in tests.
3. Define rollback checkpoint/tag.

Deliverables:
- Baseline artifact bundle under `runs/` or `artifacts/`.
- Baseline behavior note in docs.

Validation:
- `pytest -q` (or existing CI-equivalent subset).

Acceptance criteria:
- Baseline snapshot saved.
- Existing tests green before refactor.

## Phase 1: Naming and Surface Cleanup
Tasks:
1. Introduce neutral runtime names:
- `scripts/propose.py` (with temporary wrapper from `scripts/p3_propose.py`).
- `scripts/worker.py` (with temporary wrapper from `scripts/p3_worker.py`).
- `ai_scientist/enqueue.py` (with temporary wrapper from `ai_scientist/p3_enqueue.py`).
2. Add `--harness-path legacy|new` selector and `HARNESS_PATH` env override, defaulting to `legacy`.
3. Persist selected `harness_path` in per-cycle records.
4. Update governor defaults to neutral paths.

File targets:
- `scripts/governor.py`
- `scripts/p3_propose.py`
- `scripts/p3_worker.py`
- `ai_scientist/p3_enqueue.py`

Acceptance criteria:
- Governor runs with neutral defaults.
- Compatibility wrappers preserve old entrypoints.
- `legacy` remains default path until parity gate passes.

## Phase 2: Remove P3-Gated Control Paths
Tasks:
1. Remove `if profile.problem == "p3"` control gates in core loop.
2. Route problem-specific behavior through adapter interface.

File targets:
- `scripts/governor.py`

Acceptance criteria:
- No P3-only gate in core action-selection path.

## Phase 3: Problem Adapter Extraction
Tasks:
1. Add explicit adapter interface and P1/P2/P3 implementations.
2. Wire governor to adapter methods for scoring/frontier/target/focus-partner.

File targets:
- `ai_scientist/problem_profiles.py`
- new adapter module (for example `ai_scientist/problem_adapters.py`)
- `scripts/governor.py`

Acceptance criteria:
- Shared control path works for P1/P2/P3.

## Phase 4: Frontier and Snapshot Generalization
Tasks:
1. Use adapter-selected frontier metric (P1/P2 objective delta, P3 HV+delta).
2. Ensure snapshot fields remain coherent across all problems.

File targets:
- `scripts/governor.py`
- `ai_scientist/memory/schema.py` (only if schema normalization is required)

Acceptance criteria:
- P1/P2 do not depend on P3-specific HV path.
- P3 frontier behavior preserved.

## Phase 5: LLM Reliability Transport Upgrade
Tasks:
1. Preserve governor decision call seam.
2. Replace transport internals with clawdbot-style reliability:
- auth profiles
- cooldown windows
- rotation/failover
- bounded retries
3. Keep schema-validated decision contract.
4. Keep early-phase transport minimal: before Phase 5, use simple synchronous decision call + bounded retry + deterministic fallback only.

File targets:
- `ai_scientist/llm_controller.py`
- optional new transport module (for example `ai_scientist/decision_client.py`)

Acceptance criteria:
- Decision continuity under transient auth/rate-limit failures.
- Override/fallback reason always persisted.

## Phase 6: Stop Controller Hardening
Tasks:
1. Centralize manual/target/stall/runtime/queue-health stop checks.
2. Add explicit per-problem target stop logic.

File targets:
- `scripts/governor.py`

Acceptance criteria:
- Stop reasons are explicit and logged.
- Workers and LLM do not own termination decisions.

## Phase 7: Tests and Evidence
Tasks:
1. Expand test matrix to P1/P2/P3 for adapters and loop behavior.
2. Add transport fallback tests and queue-state parsing tests.

File targets:
- `tests/test_problem_profiles.py`
- `tests/test_p3_worker.py` (or generalized worker tests)
- `tests/test_p3_enqueue.py` (or generalized enqueue tests)
- new loop/transport tests as needed

Validation commands:
- `pytest -q tests/test_problem_profiles.py`
- `pytest -q tests/test_p3_worker.py`
- `pytest -q tests/test_p3_enqueue.py`
- broader `pytest -q` before merge

Acceptance criteria:
- Matrix tests pass for P1/P2/P3.
- End-to-end loop test validates bounded repeated cycles.

## Phase 7.5: Parity Gate (Required Before Cutover/Prune)
Tasks:
1. Run parity campaigns per problem (`p1`, `p2`, `p3`) with matched initial conditions for both paths (`legacy`, `new`).
2. Execute at least 20 completed cycles per path for each problem campaign.
3. Compare behavior and outcomes with relative tolerances.
4. Record parity artifact and sign-off decision.

Acceptance criteria:
- Minimum cycles: for each of `p1`, `p2`, `p3`, `>=20` completed cycles on `legacy` and `>=20` on `new`.
- Frontier movement parity: relative delta difference within `5%`.
- Feasibility yield parity: relative difference within `10%`.
- Stop behavior parity: no unexplained stop-reason class drift.
- No schema/queue-state regressions under `new`.

## Phase 8: Compatibility Prune and Final Surface Cleanup
Tasks:
1. Remove deprecated P3-only compatibility branches after migration window.
2. Keep neutral canonical entrypoints as SSOT for ongoing operations.

File targets:
- `scripts/p3_propose.py` (if converted to wrapper-only or removed per migration decision)
- `scripts/p3_worker.py` (if converted to wrapper-only or removed per migration decision)
- `ai_scientist/p3_enqueue.py` (if converted to wrapper-only or removed per migration decision)
- affected tests/docs references

Acceptance criteria:
- Deprecated compatibility paths removed or reduced to explicit wrappers only.
- Canonical loop tests pass without reliance on deprecated branches.

## Keep vs Prune Rule
Keep:
- DB schema SSOT and artifacts.
- Core governor loop and deterministic workers.
- Problem profile definitions.

Prune:
- P3-only naming and dead compatibility branches after migration window.
- Redundant silent override paths.
- Unused scripts/tests tied to removed code paths.

Do not prune:
- `docs/`, `artifacts/`, historical run evidence.

## Risks
1. Objective-direction mismatch for P1/P2 when converting utility logic.
2. Hidden P3 assumptions in partner/frontier selection.
3. Behavioral drift during rename compatibility period.

## Rollback
1. Commit phase-by-phase.
2. Tag baseline and each completed phase.
3. Revert only current phase on acceptance failure.

## Exit Criteria
- One harness executes P1/P2/P3 by configuration only.
- No P3-specialized core control gate remains.
- LLM transport has profile/cooldown/failover reliability.
- Stop controller is deterministic, auditable, governor-owned.
- `new` path has passed the parity gate and is safe to promote as default.
