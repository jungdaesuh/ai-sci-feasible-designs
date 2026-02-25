# Implementation Backlog (Master)

## Scope

This backlog converts the active plans into execution milestones:

- `docs/INTEGRATED_EVOLUTION_PLAN_P1_P2_P3.md`
- `docs/AI_SCIENTIST_UNIFIED_ROADMAP.md`
- `docs/CODEX_NATIVE_SUBSCRIPTION_INTEGRATION.md`

## Epics and Milestones

## Epic 1 — Data plane foundation

- **M1.1 Schema migration**: add lineage/novelty/operator/model-route fields required by adaptive evolution.
- **M1.2 Shared enqueue library**: replace duplicated enqueue logic in `scripts/p3_propose.py` and `scripts/p3_enqueue_submission.py`.
- **M1.3 Telemetry baseline**: record operator usage, novelty reject rate, fallback-path usage.

**Definition of done**
- New schema is backward-compatible and migration-tested.
- Enqueue callsites use a single shared module.
- Telemetry visible in per-cycle reporting.

## Epic 2 — P3 adaptive governor

- **M2.1 Flagged adaptive governor path** in `scripts/p3_governor.py`.
- **M2.2 Parent-group selector + operator-family bandit + novelty gate**.
- **M2.3 A/B validation** against static recipe path with fixed budget.

**Definition of done**
- Deterministic evaluator remains unchanged (`scripts/p3_worker.py`).
- Adaptive mode beats or matches baseline HV/feasible-yield targets.
- Rollback path exists and is validated.

## Epic 3 — P1/P2 lightweight adaptation

- **M3.1 Adaptive restart seed selector** for P1/P2.
- **M3.2 Constrained novelty gating** around restarts.
- **M3.3 Fixed-budget comparison** against current ALM/NGOpt loops.

**Definition of done**
- No regression on best feasible objective at fixed budget.
- Feasible hit rate is maintained or improved.

## Epic 4 — Provider swappability

- **M4.1 Add `codex_native` provider wiring** in config/runtime.
- **M4.2 Add codex-native aliases** for instruct/thinking tiers.
- **M4.3 Smoke and planner validation** for Grok/Kwaipilot/Codex-native switching.

**Definition of done**
- `MODEL_PROVIDER` and alias switching works without code edits.
- Docs and smoke commands match runtime behavior exactly.

## Epic 5 — Controlled retirement

- **M5.1 Remove static P3 path** after sustained adaptive superiority.
- **M5.2 Remove deprecated evaluation wrappers** after caller migration.
- **M5.3 Remove duplicate/legacy paths** after zero-usage window.

**Definition of done**
- No runtime callers depend on retired paths.
- Full tests and smoke checks pass after removal.

## Post-Upgrade Cleanup Checklist

- [ ] Remove temporary feature flags that are no longer needed.
- [ ] Remove deprecated governor logic superseded by adaptive path.
- [ ] Remove duplicate enqueue implementations and keep one SSOT module.
- [ ] Remove deprecated evaluation wrappers after all callers migrate.
- [ ] Remove transitional provider/config notes once codex-native is fully live.
- [ ] Delete stale docs/examples that reference retired commands or aliases.
- [ ] Run final repo-wide docs/code consistency review.
- [ ] Run validation suite (tests + targeted smoke runs) and record results.
- [ ] Freeze final defaults in `configs/model.yaml` and document rollback policy.
- [ ] Publish migration notes: what changed, what was removed, how to reproduce.

## Cleanup Exit Criteria

- No dead-path usage observed during the agreed stabilization window.
- All docs point to active runtime paths only.
- One-command onboarding and one-command smoke checks work end-to-end.
- Upgrade can be audited from telemetry + reports without manual reconstruction.
