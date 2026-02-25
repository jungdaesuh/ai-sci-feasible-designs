# Implementation Backlog (Master)

## Scope

This backlog converts the active plans into execution milestones:

- `docs/INTEGRATED_EVOLUTION_PLAN_P1_P2_P3.md`
- `docs/AI_SCIENTIST_UNIFIED_ROADMAP.md`
- `docs/CODEX_NATIVE_SUBSCRIPTION_INTEGRATION.md`

## Epics and Milestones

## Progress log

- **2026-02-25:** Completed Epic 4 (provider swappability) baseline:
  - `codex_native` provider entry + `codex-native-*` aliases exist in `configs/model.yaml`.
  - Runtime supports role-level model alias overrides via env vars for `planning`, `literature`, `analysis`.
  - Unit tests cover provider request building and role overrides.
  - Remaining work for “native subscription” is still tracked in `docs/CODEX_NATIVE_SUBSCRIPTION_INTEGRATION.md` (auth/profile management + local adapter server).
- **2026-02-25:** Completed Epic 1 data-plane core (M1.1) and partial telemetry (M1.3):
  - `candidates` schema now persists `lineage_parent_hashes_json`, `novelty_score`, `operator_family`, `model_route`.
  - WorldModel write/read hooks and report context now include P3 data-plane metadata.
  - `scripts/p3_propose.py`, `scripts/p3_enqueue_submission.py`, and `scripts/p3_governor.py` now write/read route/operator/lineage metadata.
  - Memory tests cover migration + persistence + summary aggregation.
  - Remaining Epic 1 gaps were shared enqueue SSOT module (M1.2) and telemetry counters for novelty reject/fallback usage.
- **2026-02-25:** Completed Epic 1 remaining work:
  - Shared enqueue SSOT module landed in `ai_scientist/p3_enqueue.py`.
  - `scripts/p3_propose.py` and `scripts/p3_enqueue_submission.py` now use the shared enqueue path.
  - Telemetry baseline now includes novelty reject counters and static/adaptive/fallback path counters in governor summaries and cycle reporting.
- **2026-02-25:** Epic 2 remains open:
  - `--adaptive` feature flag scaffold is now landed in `scripts/p3_governor.py` with static rollback.
  - Remaining Epic 2 work is adaptive policy logic (M2.2) + fixed-budget A/B validation (M2.3).
- **2026-02-25:** Epic 3 remains open:
  - P1/P2 adaptive restart seed selection + novelty gating are not integrated yet.

## Epic 1 — Data plane foundation

- [x] **M1.1 Schema migration**: add lineage/novelty/operator/model-route fields required by adaptive evolution. *(Done 2026-02-25)*
- [x] **M1.2 Shared enqueue library**: replace duplicated enqueue logic in `scripts/p3_propose.py` and `scripts/p3_enqueue_submission.py`. *(Done 2026-02-25)*
- [x] **M1.3 Telemetry baseline**: record operator usage, novelty reject rate, fallback-path usage. *(Done 2026-02-25)*

**Definition of done**
- New schema is backward-compatible and migration-tested.
- Enqueue callsites use a single shared module.
- Telemetry visible in per-cycle reporting.

## Epic 2 — P3 adaptive governor

- [x] **M2.1 Flagged adaptive governor path** in `scripts/p3_governor.py`. *(Done 2026-02-25; scaffold delegates to static recipe while preserving rollback.)*
- [ ] **M2.2 Parent-group selector + operator-family bandit + novelty gate**.
- [ ] **M2.3 A/B validation** against static recipe path with fixed budget.

**Definition of done**
- Deterministic evaluator remains unchanged (`scripts/p3_worker.py`).
- Adaptive mode beats or matches baseline HV/feasible-yield targets.
- Rollback path exists and is validated.

## Epic 3 — P1/P2 lightweight adaptation

- [ ] **M3.1 Adaptive restart seed selector** for P1/P2.
- [ ] **M3.2 Constrained novelty gating** around restarts.
- [ ] **M3.3 Fixed-budget comparison** against current ALM/NGOpt loops.

**Definition of done**
- No regression on best feasible objective at fixed budget.
- Feasible hit rate is maintained or improved.

## Epic 4 — Provider swappability

- [x] **M4.1 Add `codex_native` provider wiring** in config/runtime. *(Done 2026-02-25)*
- [x] **M4.2 Add codex-native aliases** for instruct/thinking tiers. *(Done 2026-02-25)*
- [x] **M4.3 Smoke and planner validation** for Grok/Kwaipilot/Codex-native switching. *(Done 2026-02-25; unit tests + smoke harness path)*

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
- [x] Remove duplicate enqueue implementations and keep one SSOT module. *(Done 2026-02-25)*
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
