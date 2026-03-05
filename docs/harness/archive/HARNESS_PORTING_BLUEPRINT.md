# Harness Porting Blueprint (P1/P2/P3)

Date: 2026-03-04
Document Role: Architecture blueprint
Status: **Superseded** by `docs/harness/HARNESS_CODEGEN_PLAN.md` (2026-03-04)
Owner: Harness maintainers
Last Updated: 2026-03-04
Applies To: `chore/docs-cleanup`

Related docs:
- `docs/harness/AUTONOMOUS_HARNESS_PLAN.md` (strategy)
- `docs/harness/HARNESS_CLEANUP_EXECUTION_PLAN.md` (execution)
- `docs/harness/HARNESS_DOC_INDEX.md` (index + status)

## Purpose
Define the concrete target architecture for one autonomous harness across P1/P2/P3.

## Scope
- In scope: governor control loop, decision transport reliability, adapterized problem logic, deterministic execution, stop controller.
- Out of scope: changing official physics evaluator semantics.

## Runtime Session Model
- One persistent decision client/gateway process for reliability.
- Fresh DB-derived observation each cycle for reasoning context.
- Transport state persists (auth profile cooldown/failover).
- Optimization state authority remains in DB/artifacts only.

## Harness Path Selection Contract
- Runtime selector: `--harness-path legacy|new`.
- Optional env override: `HARNESS_PATH=legacy|new`.
- Default during migration: `legacy`.
- Per-cycle recorder field: `harness_path` (required).

## Target Control Flow
```text
Governor loop (single owner)
  -> Read DB snapshot + health
  -> Stop-controller checks
  -> Build fresh observation
  -> LLM decision client (simple retry early; profile/cooldown/failover after Phase 5)
  -> Validate + compile bounded commands
  -> Enqueue
  -> Workers evaluate
  -> Record outcomes + override/fallback reasons
  -> next cycle
```

## Component Contracts
1. Governor
- Owns cycle boundaries, effective action selection, and stop decisions.

2. Problem Adapter
- Interface:
- `objective_utility(row)`
- `frontier_delta(prev_snapshot, now_snapshot)`
- `target_reached(snapshot)`
- `select_focus_partner(candidates)`

3. Decision Client
- Input: `{cycle_id, snapshot_hash, problem_profile, observation}`.
- Output: schema-valid intent or explicit fallback reason.
- Early phases: synchronous call + bounded retry + deterministic fallback.
- Phase 5+: auth profile ordering, cooldown, retries, failover.

4. Action Compiler
- Converts intent to deterministic bounded enqueue payloads.
- Enforces mutation/action caps.

5. Worker
- Claims `pending`, sets `running:*`, evaluates, persists metrics/artifacts.
- Never decides stop/restart policy.

6. Recorder
- Persists cycle request context, effective action, and realized deltas.
- Persists `harness_path` for per-cycle attribution during strangler runs.
- Enables replay/resume without hidden state.

## Action/Restart Contract
- Action set: `repair | bridge | jump | global_restart`.
- Restart plan set: `soft_retry | degraded_restart | global_restart | circuit_break`.
- If policy overrides LLM intent, reason must be recorded.

## Stop Controller Contract
Governor stops on:
- Manual stop flag.
- Target reached for active problem.
- Stagnation/no-improvement window.
- Runtime/cycle budget.
- Queue-health circuit break.

## Problem-Specific Frontier Rules
- P1: feasible objective improvement (minimize elongation).
- P2: feasible objective improvement (maximize gradient scale length proxy).
- P3: hypervolume + feasible objective movement.

## Reliability Requirements
- LLM transport failures must not crash campaign when fallback is enabled.
- Every cycle has explicit source attribution: LLM vs fallback policy.
- Queue state interpretation must treat `running:*` correctly.

## Non-Negotiables
- One loop owner.
- Bounded batches.
- Deterministic workers.
- SSOT in DB/artifacts.
