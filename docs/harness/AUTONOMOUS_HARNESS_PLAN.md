# Autonomous Frontier Harness Plan (P1/P2/P3, LLM-In-The-Loop)

Date: 2026-03-04
Document Role: Strategy (north star)
Status: Active
Owner: Harness maintainers
Last Updated: 2026-03-04
Applies To: `chore/docs-cleanup`

Related docs:
- `docs/harness/HARNESS_CODEGEN_PLAN.md` (active implementation plan — code-generation decision interface)
- `docs/harness/CODEGEN_IDEAS_FROM_LITERATURE.md` (research synthesis feeding into codegen plan)
- `docs/harness/HARNESS_DOC_INDEX.md` (index + status)
- `docs/harness/archive/HARNESS_PORTING_BLUEPRINT.md` (superseded — schema-bounded architecture)
- `docs/harness/archive/HARNESS_CLEANUP_EXECUTION_PLAN.md` (superseded — schema-bounded migration plan)

## Objective
Build one simple autonomous harness that optimizes P1, P2, and P3 by reusing one closed loop and swapping only problem adapters.

## Problem Summary
Failures were caused by control complexity, not by using an LLM:
- Too many override paths hide real control flow.
- LLM decisions are frequently superseded by policy side paths.
- Queue observability can be misread (`running:*` vs literal `running`).
- Runtime and code bloat slow interventions and debugging.

## Design Principles (Hard Rules)
- One explicit governor loop owns cycle boundaries.
- Governor is sole stop authority.
- LLM is strategist; workers are deterministic executors.
- SQLite + artifacts are SSOT; LLM memory is advisory only.
- Bounded actions only; no unbounded enqueue behavior.
- Persistent LLM client/gateway for reliability, fresh DB-derived context every cycle.

## Minimal Target Architecture
1. `governor_loop`
- One control loop.
- No hidden side loops.

2. `state_reader`
- Reads queue health, candidate outcomes, frontier deltas.
- Treats `status LIKE 'running:%'` as active running.

3. `decision_client` (code-generation interface)
- LLM generates a bounded Python proposal script each cycle.
- Script writes candidate JSONs to a sandboxed staging directory.
- Replaces the original schema-bounded `llm_planner` — see `HARNESS_CODEGEN_PLAN.md`.

4. `sandbox`
- Validates and executes the LLM-generated script in an isolated subprocess.
- Collects candidate boundaries, enforces candidate cap.

5. `evaluator_workers`
- Deterministic evaluators only.
- No planning/stop logic inside workers.

6. `recorder`
- Persists cycle decision, script source, failure reasons, and realized deltas.

7. `problem_adapter`
- Encodes P1/P2/P3 objective/constraint/frontier semantics.

## Closed Loop (ASCII)
```text
+-----------------+
| Read DB State   |
| pending/running |
| frontier deltas |
+--------+--------+
         |
         v
+------------------------+
| Diagnose Last Cycle    |
| worked vs failed       |
| explore/exploit mode   |
+-----------+------------+
            |
            v
+------------------------+
| LLM Code Generation    |
| Python proposal script |
+-----------+------------+
            |
            v
+------------------------+
| Sandbox + Validate     |
| execute, collect, dedup|
+-----------+------------+
            |
            v
+------------------------+
| Enqueue Bounded Batch  |
| validated candidates   |
+-----------+------------+
            |
            v
+------------------------+
| Workers Evaluate       |
| DB status + metrics    |
+-----------+------------+
            |
            v
+------------------------+
| Record Outcomes (SSOT) |
| script + scorecards    |
+-----------+------------+
            |
            +-------> repeat until stop policy
```

## Action Semantics
- The LLM generates arbitrary Python proposal scripts each cycle (no fixed action enum).
- Governor enforces bounds: candidate cap, sandbox timeout, forbidden imports.
- No error swallowing: every LLM or sandbox failure is logged at ERROR, recorded in DB with full context, and fed into the next cycle's execution traces.
- Circuit breaker stops the run after N consecutive failures (no silent degradation).

## Problem Adapters (P1/P2/P3)
- P1:
- Objective: minimize `max_elongation`.
- Feasibility: aspect ratio, average triangularity, iota edge constraints.
- Frontier delta: best feasible objective improvement.

- P2:
- Objective: maximize `minimum_normalized_magnetic_gradient_scale_length`.
- Feasibility: aspect ratio, iota edge, log10_qi, mirror ratio, max_elongation constraints.
- Frontier delta: best feasible objective improvement.

- P3:
- Objective: multi-objective frontier (compactness vs coil-simplicity proxy).
- Feasibility: iota, log10_qi, mirror ratio, flux compression, vacuum well constraints.
- Frontier delta: hypervolume plus feasible objective movement.

## Memory and Records
Use both, but with strict authority:
- Authoritative memory: SQLite + artifacts.
- Non-authoritative memory: LLM session/scratchpad.

Persist per cycle:
- State snapshot (queue + frontier metrics).
- Proposal script source + hash.
- Failure reason + traceback (if LLM/sandbox failed).
- Outcome deltas.

## Keep vs Prune (Code-Focused)
Keep:
- Core governor loop.
- Deterministic worker and enqueue primitives.
- Memory schema and DB access layer.
- Decision client with code-generation interface.
- One adapter interface with P1/P2/P3 implementations.

Prune aggressively:
- Duplicate/legacy governor branches.
- Silent override paths with weak observability.
- Redundant novelty/router layers not required by core loop.
- Dead scripts/tests tied to removed paths.

Do not prune:
- `docs/`, `artifacts/`, and historical run evidence.

## Stop Policy
Governor stops when any is true:
- Manual stop signal set.
- Problem target reached.
- No-improvement window exceeded.
- Runtime/cycle budget exhausted.
- Queue health circuit-break condition triggered.

## Success Criteria
- Same harness runs P1/P2/P3 by configuration only.
- Cycle decisions are auditable end-to-end.
- Failure reasons are always persisted with full context.
- Harness reproduces manual-style gains without manual intervention.
- Code surface remains small enough for one-pass reasoning.
