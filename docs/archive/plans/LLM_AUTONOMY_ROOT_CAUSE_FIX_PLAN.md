# LLM Autonomy Root-Cause Fix Plan (Invariant-Driven)

## Goal
Make the governor truly autonomous in runtime behavior, not just LLM-enabled.

Definition of success:
- The loop cannot stall in repeated zero-yield cycles.
- LLM decisions are grounded on correct violation telemetry.
- Restart paths always produce novel executable candidates or trigger deterministic escalation.

## Scope
- In scope:
  - `scripts/governor.py`
  - `ai_scientist/llm_controller.py` (only if needed for contract alignment)
  - `ai_scientist/p3_enqueue.py` (no schema changes)
  - tests under `tests/test_p3_governor_contract.py`, `tests/test_llm_controller.py`, and new e2e regression tests
- Out of scope:
  - new DB schema
  - multi-agent runtime architecture
  - new optimizer stack

## Root Causes (Confirmed)
1. Violation signal loss in candidate fetch path can produce `dominant_violation=none`.
2. Hard restart policy can repeatedly lock into `global_restart`.
3. `global_restart` execution can replay identical command/parents/t-grid and produce `inserted=0`.
4. Batch progression can freeze when `_next_batch_id` depends on meta files and no inserts occur.
5. Anti-repeat and reflection learning can fail to activate when event continuity is incomplete.
6. Command execution path does not treat zero-yield as a hard control failure.
7. Tests mostly validate contracts, not autonomy-progress invariants.

## Design Principles
- KISS: one governor loop, one controller API, one deterministic evaluator path.
- YAGNI: no new services, no extra orchestrators.
- SSOT:
  - decision schema in `llm_controller.py`
  - policy constants in `problem_profiles.py`
  - queue integrity in `p3_enqueue.py`
- DRY: one post-plan control gate stack used every cycle.

## Mandatory Runtime Invariants
1. **Observation truth invariant**
   - `dominant_violation` must derive from real violation data if present in metric payload.
   - Never emit `none` when raw metric rows contain positive violations.

2. **Cycle productivity invariant**
   - Every executed cycle must satisfy one of:
     - inserted candidates > 0, or
     - forced operator/parent/grid mutation for next attempt, or
     - deterministic circuit-break/stop.

3. **Batch monotonicity invariant**
   - Cycle identity (`batch_id`, `seed_base`) must advance even when insert count is zero.
   - Must not depend on side-effect file creation.

4. **Restart diversity invariant**
   - `global_restart` may not emit the same parent pair + identical t-grid more than once without forced jitter.

5. **Learning continuity invariant**
   - Every proposal cycle must write step-0 and step-1 compatible events for anti-repeat/reflection consumption.

## Implementation Plan

### Step 1: Fix violation extraction correctness (truth source)
- File: `scripts/governor.py` (`_fetch_candidates`).
- Change:
  - Use `constraint_margins` only when it is a non-empty dict.
  - Else fallback to `violations` when present and dict.
  - Preserve current behavior for already-normalized payloads.
- Acceptance:
  - For payloads with only `violations`, candidate rows expose non-empty `row.violations`.
  - `dominant_violation` is not `none` in that case.

### Step 2: Add zero-yield enforcement in execution loop
- File: `scripts/governor.py` (`_run_cmds` call path).
- Change:
  - Capture command stdout for each `p3_propose` call.
  - Parse `inserted`/`skipped` summary.
  - If total `inserted == 0` for cycle:
    - mark control state as zero-yield failure,
    - force next-cycle escalation path (`bridge/jump/restart-parent-jitter`),
    - record explicit override reason.
- Acceptance:
  - Zero-yield cannot silently continue with identical plan.

### Step 3: Make batch progression independent of meta files
- File: `scripts/governor.py`.
- Change:
  - Replace `_next_batch_id(run_dir)` usage in loop decisions with manifest-based monotonic next cycle id.
  - Keep file scanning only as backward-compatible fallback.
- Acceptance:
  - Batch id increments even when `inserted=0`.
  - No repeated `batch_id` reuse in steady loop unless explicit replay mode.

### Step 4: Diversify `global_restart` command generation
- File: `scripts/governor.py` (`global_restart` command branch).
- Change:
  - Introduce deterministic jitter derived from `(run_seed, batch_id)`:
    - parent pair rotation from valid archive pool,
    - bounded t-grid shift or step schedule variation under profile caps.
  - Maintain deterministic reproducibility.
- Acceptance:
  - Consecutive global restarts do not produce identical command fingerprint unless explicitly replaying.

### Step 5: Ensure reflection/anti-repeat continuity for all cycles
- File: `scripts/governor.py`.
- Change:
  - Guarantee step-0 event logging for bootstrap and non-bootstrap cycles.
  - Guarantee step-1 reflection eligibility for each emitted cycle.
- Acceptance:
  - `anti_repeat` query has usable reflection rows after each cycle.
  - Repeated no-progress action tuples trigger forced action transition.

### Step 6: Keep hard policy deterministic but not no-op
- File: `scripts/governor.py`.
- Change:
  - Keep `circuit_break` deterministic-only.
  - For hard `global_restart`, allow only deterministic diversified restart path (not static replay).
  - Do not downgrade hard restart to softer actions.
- Acceptance:
  - Hard policy remains enforced and still changes search state.

### Step 7: Add autonomy regression tests (required gate)
- Files:
  - `tests/test_p3_governor_contract.py` (extend)
  - `tests/test_governor_autonomy_invariants.py` (new)
- Required tests:
  1. violation fallback correctness (`violations` payload path)
  2. zero-yield triggers forced escalation or stop
  3. batch id monotonic progression under zero insertions
  4. restart fingerprint diversity across consecutive restart cycles
  5. reflection continuity enables anti-repeat trigger
  6. replay determinism remains stable with new batch-id logic

## Delivery Sequence (No Bloat)
1. `fix: restore violation truth source in candidate fetch`
2. `fix: enforce zero-yield cycle guard and monotonic batch progression`
3. `fix: deterministic diversified global restart path`
4. `fix: complete reflection continuity for anti-repeat`
5. `test: add autonomy invariant regression suite`

## Validation Gate (Must Pass Before Merge)
- `ruff check scripts/governor.py ai_scientist/llm_controller.py ai_scientist/p3_enqueue.py tests/`
- `python -m py_compile scripts/governor.py ai_scientist/llm_controller.py ai_scientist/p3_enqueue.py`
- `pytest -q tests/test_llm_controller.py tests/test_p3_governor_contract.py tests/test_governor_autonomy_invariants.py`
- One live dry-run:
  - `--llm-enabled --llm-fallback --execute --loop` with short cap,
  - demonstrate no repeated zero-yield same-fingerprint cycle.

## Exit Criteria
- On a live loop from hard starting basin:
  - no repeated identical restart command with zero insertions over more than 1 cycle,
  - `dominant_violation` remains informative,
  - anti-repeat or deterministic restart diversity changes operator/parent/grid automatically,
  - measurable movement in either feasibility best-so-far or frontier candidate count within configured cycle budget.
