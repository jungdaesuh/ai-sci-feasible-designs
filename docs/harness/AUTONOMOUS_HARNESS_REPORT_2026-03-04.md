# Autonomous Harness Report (2026-03-04)

## Scope
This report summarizes:
1. What happened during the successful manual frontier push.
2. Why the current harness behavior has felt unreliable.
3. What architecture is recommended for autonomous closed-loop operation.
4. What was documented in the harness planning docs.

## 1) Manual Push Outcome (What Worked)

### Baseline and Result
- Baseline `best_obj_feas`: `5.249796918788695`
- New `best_obj_feas`: `5.276789734453542`
- Improvement: `+0.026992815664847`
- Winning candidate: `candidate_id=1615`
- Winning route: `manual_wave3/near_micro_smooth`
- Winning feasibility: `0.009966988704582569` (feasible)

### Process That Moved the Frontier
- Wave 1 and Wave 2 used broader blend/scale-group style moves.
- Those waves improved exploration volume but did not cross feasibility boundary.
- Postmortem of evaluated rows identified near-feasible, high-objective cluster.
- Wave 3 switched to boundary-hugging micro-perturbation around those parents.
- Small local geometry smoothing crossed the feasibility threshold while preserving objective.

### Operational Lesson
Broad exploration was not enough near the boundary; targeted micro-repair around near-feasible high-objective seeds produced the breakthrough.

## 2) Why "It Was Failing" for 3 Days

### Primary Failure Pattern
The system was not consistently operating as a strict closed loop with reliable intervention logic.

### Contributing Factors
- Observability confusion (`running:*` status interpretation) masked true queue state.
- Policy overrides and fallback paths could dilute LLM-guided interventions.
- Overgrown runtime/control surfaces created hidden state and harder diagnosis.
- Stale/legacy runtime artifacts and extra code paths increased noise.

### Effect
The harness often spent budget in low-yield cycles without fast enough adaptive correction toward feasibility-boundary crossing.

## 3) Recommended Autonomous Harness Shape

## Control Model
- One governor process as sole loop owner.
- Deterministic workers for evaluation.
- LLM used as strategist each cycle (not removed).
- SSOT memory in DB/artifacts; LLM context is non-authoritative helper.

## Closed Loop
1. Observe snapshot and health.
2. Diagnose what worked/failed.
3. Decide bounded next action.
4. Enqueue bounded batch.
5. Evaluate via workers.
6. Record deltas/outcomes.
7. Stop or continue by stop policy.

## LLM Integration Recommendation
Use clawdbot-style reliability characteristics for transport/auth:
- profile order
- cooldown
- failover
- retry discipline

Keep cycle prompts fresh and compact from current DB state (not an ever-growing hidden prompt history).

## Stop Authority
Governor owns all stop conditions:
- manual stop
- target reached
- stall window exhaustion
- time/budget cap
- queue/circuit-break safety

## 4) Documentation Status

> **Note (2026-03-04):** The blueprint and cleanup plan referenced below were superseded by the code-generation harness plan (`docs/harness/HARNESS_CODEGEN_PLAN.md`). They are archived at `docs/harness/archive/`. See `docs/harness/HARNESS_DOC_INDEX.md` for current doc status.

The harness docs were created/updated and aligned:
- `docs/harness/AUTONOMOUS_HARNESS_PLAN.md` (still active — updated for codegen interface)
- `docs/harness/archive/HARNESS_PORTING_BLUEPRINT.md` (superseded)
- `docs/harness/archive/HARNESS_CLEANUP_EXECUTION_PLAN.md` (superseded)
- `docs/harness/HARNESS_DOC_INDEX.md`

These originally defined:
- TT-style explicit closed-loop control
- persistent decision client + fresh cycle context
- clawdbot-style LLM transport reliability goals
- governor as single stop authority
- generalization target across P1/P2/P3 (not P3-only design)

## 5) Practical Next Step

> **Note (2026-03-04):** The next step is now to implement the code-generation harness per `docs/harness/HARNESS_CODEGEN_PLAN.md`. The blueprint/cleanup plan referenced in the original report are superseded.

Follow `docs/harness/HARNESS_CODEGEN_PLAN.md` — Phase 1 MVP (~1535 lines, 12 files in `harness/` package).
