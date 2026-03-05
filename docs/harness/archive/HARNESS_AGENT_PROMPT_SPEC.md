# Harness Agent Prompt Spec (SSOT)

Date: 2026-03-04
Document Role: Prompt/runtime contract
Status: **Superseded** by `docs/harness/HARNESS_CODEGEN_PLAN.md` (2026-03-04)
Owner: Harness maintainers
Last Updated: 2026-03-04
Applies To: `chore/docs-cleanup`

Related docs:
- `docs/harness/AUTONOMOUS_HARNESS_PLAN.md` (strategy)
- `docs/harness/HARNESS_PORTING_BLUEPRINT.md` (architecture)
- `docs/harness/HARNESS_CLEANUP_EXECUTION_PLAN.md` (execution)
- `docs/harness/HARNESS_DOC_INDEX.md` (index + status)

## Purpose
Define one canonical prompt/runtime contract for the autonomous harness agent so behavior is auditable, bounded, and portable across P1/P2/P3.

## Non-Negotiables
- LLM is planner only.
- Governor is single loop owner and single stop authority.
- Workers are deterministic evaluators only.
- SQLite plus artifacts are SSOT.
- Every cycle is bounded and explicitly recorded.

## Challenge Context Block (Required in Observation)
The cycle observation passed to the decision client must include:
- `challenge.name = "constellaration"`
- `challenge.deterministic_validity_required = true`
- `challenge.purpose = "Recover feasibility first, then improve objective frontier under hard physics constraints."`

## Problem Contract Block (P1/P2/P3)
The observation must include problem-scoped objective, constraints, action allowlist, mutation budget, and policy thresholds from `ai_scientist/problem_profiles.py`.

P1 contract:
- Objective: minimize `max_elongation`
- Feasibility constraints: `aspect_ratio`, `average_triangularity`, `iota_edge`
- Allowed actions: `repair | jump | global_restart`

P2 contract:
- Objective: maximize `minimum_normalized_magnetic_gradient_scale_length`
- Feasibility constraints: `aspect_ratio`, `iota_edge`, `log10_qi`, `mirror`, `max_elongation`
- Allowed actions: `repair | bridge | global_restart`

P3 contract:
- Objective: maximize feasible frontier quality (HV + feasible movement)
- Feasibility constraints: `iota`, `log10_qi`, `mirror`, `flux`, `vacuum`
- Allowed actions: `repair | bridge | global_restart`

## Tool and Action Surface (What Agent Can Actually Do)
The planning agent does not run physics directly. It can only emit bounded intent consumed by governor components:
- Choose one action from allowed set.
- Choose one target constraint from allowed constraint names.
- Emit bounded mutation edits under profile caps.
- Optionally request restart plan from allowed restart enums.

Runtime components available to execute that intent:
- Governor loop
- DB state reader
- Action compiler
- Enqueue executor
- Deterministic workers
- Recorder
- Stop controller

## Closed-Loop Protocol (Required Order)
1. Read snapshot from DB and health counters.
2. Diagnose what worked and failed in recent windows.
3. Build fresh cycle observation.
4. Request decision from decision client.
5. Validate decision against profile contract.
6. Compile bounded deterministic enqueue commands.
7. Enqueue and let workers evaluate.
8. Record effective action and outcome deltas.
9. Apply stop controller.
10. Repeat if stop conditions are not met.

## Decision Output Schema (LLM -> Governor)
Required fields:
- `action` (string)
- `target_constraint` (string)
- `mutations` (array of mutation edits)
- `expected_effect` (string)

Optional fields:
- `restart_plan` (enum)

Mutation edit schema:
- `parameter_group` (string, non-empty)
- `normalized_delta` (finite number)

## Allowed Enums
Action enum:
- `repair`
- `bridge`
- `jump`
- `global_restart`

Restart enum:
- `soft_retry`
- `degraded_restart`
- `global_restart`
- `circuit_break`

## Validation and Fallback Rules
- Reject unknown action, unknown constraint, malformed mutation, duplicate mutation groups, or cap violations.
- Reject `global_restart` with non-empty mutations.
- If LLM output is invalid and fallback is enabled, governor applies deterministic policy path and records fallback reason.
- If hard restart policy is active (`global_restart` or `circuit_break`), policy cannot be downgraded by LLM output.

## Authority and Hidden State Rules
- DB/artifacts are authoritative run memory.
- LLM session memory is advisory only.
- Fresh DB-derived context is required every cycle.
- No hidden side loops or hidden stop owners.
- Harness path selection is explicit (`legacy` or `new`), never implicit.

## Required Per-Cycle Record
Persist at least:
- Cycle id and snapshot hash.
- Harness path (`legacy|new`).
- Input source (`llm` vs `fallback_policy`).
- Raw intent payload.
- Validated/effective action.
- Override or fallback reason.
- Queue deltas and frontier deltas.
- Stop-controller decision for that cycle.

## Stop Controller Contract
Campaign stops when any condition is true:
- Manual stop signal.
- Problem target reached.
- No-improvement window exceeded.
- Runtime or cycle budget exhausted.
- Queue-health circuit break triggered.

## Minimal Prompt Template (Reference)
Use this shape when constructing decision requests:

```json
{
  "challenge": {
    "name": "constellaration",
    "deterministic_validity_required": true,
    "purpose": "Recover feasibility first, then improve objective frontier under hard physics constraints."
  },
  "problem": {
    "problem": "p1|p2|p3",
    "objective": {},
    "constraints": [],
    "allowed_actions": [],
    "mutation_budget": {},
    "autonomy_policy": {}
  },
  "phase": {},
  "frontier": {},
  "lessons": {},
  "state": {},
  "output_contract": {
    "json_only": true,
    "required_fields": [
      "action",
      "target_constraint",
      "mutations",
      "expected_effect"
    ]
  }
}
```

## Acceptance Criteria
- One engineer can trace any cycle from observation to outcome without guessing.
- Any invalid LLM output resolves deterministically with explicit fallback reason.
- P1/P2/P3 run under one shared loop with only profile differences.
- Prompt/runtime behavior matches this file and not tribal memory.
