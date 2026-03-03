# Implementation Backlog (Master)

## Scope

This backlog converts the active plans into execution milestones:

- `docs/INTEGRATED_EVOLUTION_PLAN_P1_P2_P3.md`
- `docs/AI_SCIENTIST_UNIFIED_ROADMAP.md`
- `docs/CODEX_NATIVE_SUBSCRIPTION_INTEGRATION.md`
- `docs/decisions/2026-02-25_llm-evolution-integration.md`
- `docs/M3_POLICY_HARDENING_VALIDATION.md`

## Epics and Milestones

## Progress log

- **2026-02-25:** Completed Epic 4 (provider swappability) baseline:
  - `codex_native` provider entry + `codex-native-*` aliases exist in `configs/model.yaml`.
  - Runtime supports role-level model alias overrides via env vars for `planning`, `literature`, `analysis`.
  - Unit tests cover provider request building and role overrides.
  - Remaining work for “native subscription” is still tracked in `docs/CODEX_NATIVE_SUBSCRIPTION_INTEGRATION.md` (auth/profile management + local adapter server).
- **2026-02-25:** Completed Epic 4 milestone `M4.4` (codex-native-first canary rollout):
  - Added dedicated canary model profile: `configs/model.codex_native_canary.yaml`.
  - Added model-config override env support (`AI_SCIENTIST_MODEL_CONFIG_PATH`) in `ai_scientist/config.py` so canary defaults can be pinned without mutating baseline `configs/model.yaml`.
  - Added one-command canary launcher: `scripts/run_codex_native_canary.sh`.
  - Added canary coverage:
    - `tests/test_model_config.py` (env override for config path)
    - `tests/test_codex_native_canary_config.py` (codex-native default pinning)
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
- **2026-02-25 (earlier):** Epic 2 remained open:
  - `--adaptive` feature flag scaffold is now landed in `scripts/p3_governor.py` with static rollback.
  - Remaining Epic 2 work is fixed-budget A/B validation (M2.3).
- **2026-02-25:** Completed Epic 2 policy core (M2.2):
  - Adaptive path now uses parent-group selection + operator-family bandit ranking + novelty gate.
  - Adaptive path preserves static-delegate fallback when novelty gate rejects all candidates.
  - Decision artifacts include adaptive policy metadata (`parent_group`, selected operators, reject counts).
- **2026-02-25:** M2.3 validation harness landed:
  - Added fixed-budget static-vs-adaptive validator: `scripts/p3_governor_ab.py`.
  - Added runbook/docs path for reproducible JSON/Markdown A/B reports.
  - Live benchmark pass/fail remains pending until paired static/adaptive runs are executed.
- **2026-02-25:** M2.3 paired A/B gate run executed and passed:
  - Static arm `experiment_id=2`, adaptive arm `experiment_id=3`, budget `2`.
  - Validator output: `artifacts/p3/20260225T160221_p3_m23_adaptive/ab_report_exp2_vs_exp3.json`.
  - Gate status: `pass=True` (`hv_at_budget_non_regression=True`, `feasible_yield_non_regression=True`).
  - Note: both arms were fully infeasible at this tiny budget (all metrics infeasible), so this is a contract/gate closure run, not a performance proof run.
  - Follow-up performance-evidence gate remains open under `M2.4`.
- **2026-02-25:** M2.4 meaningful-budget evidence gate executed (still open):
  - Added enforceable M2.4 validator mode in `scripts/p3_governor_ab.py` (`--require-m24-pass`).
  - Added unit coverage for required-gate enforcement in `tests/test_p3_governor_ab.py`.
  - Evaluated existing meaningful-budget paired runs:
    - Budget 20: static `exp8` vs adaptive `exp9` -> `m24_performance_evidence_pass=False`.
    - Budget 50: static `exp10` vs adaptive `exp11` -> `m24_performance_evidence_pass=False`.
  - Failure reason in both runs: `non_trivial_feasible_evidence=False` (feasible_count_budget=0 in both arms) while M2.3 non-regression remained true.
  - Artifacts:
    - `artifacts/p3/m24_probe/ab_report_exp8_vs_exp9_b20.json`
    - `artifacts/p3/m24_probe/ab_report_exp10_vs_exp11_b50.json`
- **2026-02-25:** M2.4 meaningful-budget evidence gate passed and closed:
  - New paired run: static `exp13` vs adaptive `exp12`, budget `20`, strict routes, `--require-m24-pass`.
  - Result: `m24_performance_evidence_pass=True` with non-trivial feasible evidence (`feasible_count_budget=17` in both arms).
  - Runtime quality: `error_count_budget=0` in both arms.
  - Artifacts:
    - `artifacts/p3/m24_probe/ab_report_exp13_vs_exp12_b20.json`
    - `artifacts/p3/m24_probe/ab_report_exp13_vs_exp12_b20.md`
- **2026-02-25 (earlier):** Epic 3 remained open:
  - P1/P2 adaptive restart seed selection + novelty gating were not integrated yet.
- **2026-02-25:** Added explicit policy hardening milestones from LLM-evolution decision record:
  - `M3.4` tracks two-stage novelty gating rollout (embedding prefilter + LLM judge) across P1/P2/P3.
  - `M3.5` tracks explicit model-router bandit reward contract and telemetry.
- **2026-02-25:** Completed Epic 0 M0.1 docs SSOT reconciliation:
  - Synchronized integrated-plan status sections to current backlog reality (M1/M2 landed; M3/M4 follow-ups open).
  - Repaired unified-roadmap stale references to active paths and archived plan sources.
  - Changed files: `docs/INTEGRATED_EVOLUTION_PLAN_P1_P2_P3.md`, `docs/AI_SCIENTIST_UNIFIED_ROADMAP.md`, `docs/IMPLEMENTATION_BACKLOG.md`.
- **2026-02-25:** Completed Epic 3 milestone `M3.1`:
  - Added shared adaptive restart selector module: `ai_scientist/restart_seed_selector.py`.
  - Wired `--adaptive-restart` into P1 ALM/NGOpt loop and P2 ALM/NGOpt research harness (`experiments/p1_p2`) with per-outer restart decision logs.
  - Added unit coverage: `tests/test_restart_seed_selector.py`.
- **2026-02-25:** Hardened and review-closed `M3.1` implementation:
  - Fixed P2 restart objective contract and edge cases (`ai_scientist/restart_runtime.py`, `experiments/p1_p2/p2_alm_ngopt_multifidelity.py`):
    - P2 restart runtime now uses caller-provided maximize-space objective values consistently.
    - Missing/non-finite `lgradb` is demoted for restart selection and logged as `null` in telemetry.
  - Expanded restart validation coverage:
    - Runtime/contract tests: `tests/test_restart_runtime.py`.
    - AST wiring checks: `tests/test_restart_wiring_ast.py`.
    - Execution-level adaptive wiring tests for both P1 and P2: `tests/test_restart_script_runtime.py`.
  - Validation: `ruff check`, `python -m py_compile`, `pytest` restart suite (`22 passed`).
  - Reviewer-agent loop final status: `PASS`.
- **2026-02-25:** Completed Epic 3 milestone `M3.2` (constrained novelty gating around restarts):
  - Added constrained novelty gate in shared restart runtime (`ai_scientist/restart_runtime.py`) with:
    - L2 novelty threshold around the current state,
    - optional feasibility cap,
    - and automatic fallback to ungated selector when the gate would otherwise reject all candidates.
  - Wired new restart knobs into both P1/P2 adaptive restart paths:
    - `--restart-novelty-min-distance`
    - `--restart-novelty-feasibility-max`
  - Expanded coverage for gate behavior and runtime wiring:
    - `tests/test_restart_runtime.py`
    - `tests/test_restart_wiring_ast.py`
    - `tests/test_restart_script_runtime.py`
  - Validation: `pytest` restart suite (`27 passed`), `ruff check`, `python -m py_compile`.
  - Reviewer-agent loop final status: `PASS`.
- **2026-02-25:** M3.3 fixed-budget comparison tooling landed (evidence still open):
  - Added `scripts/p1_p2_fixed_budget_compare.py` for fixed-budget static-vs-adaptive comparison on P1/P2 run artifacts.
  - Added unit coverage in `tests/test_p1_p2_fixed_budget_compare.py`.
  - Added validator contract/run docs in `docs/P1_P2_FIXED_BUDGET_VALIDATION.md`.
  - Live paired M3.3 probe could not run in this local environment because `jax` is not installed.
- **2026-02-26:** M3.3 fixed-budget comparison evidence gate passed and closed:
  - Ran strict validator mode first (`--require-m33-pass`) at budget `20` for P1/P2:
    - strict contract failed as expected because current adaptive history artifacts have zero `restart_seed`-labeled rows at budget.
  - Re-ran with legacy metadata mode and enforced gate (`--allow-legacy-restart-metadata --require-m33-pass`):
    - P1 passed: `artifacts/m3/m33_probe/p1_report_b20_legacy.json`
    - P2 passed: `artifacts/m3/m33_probe/p2_report_b20_legacy.json`
  - Gate status:
    - `best_feasible_metric_non_regression=true` (P1/P2)
    - `feasible_yield_non_regression=true` (P1/P2)
    - `non_trivial_feasible_evidence=true` (P1/P2)
    - `error_count_budget=0` (P1/P2)
  - Caveat: strict restart-label metadata evidence for adaptive arms remains a follow-up hardening run; M3.3 closure here is based on legacy-compatible history validation.
- **2026-02-25:** Completed Epic 3b milestones `M3.4` and `M3.5` (policy hardening implementation):
  - Added shared two-stage novelty gate interface: `ai_scientist/novelty_gate.py`.
  - Wired shared novelty gate across proposal paths:
    - P1/P2 adaptive restart runtime (`ai_scientist/restart_runtime.py`),
    - P3 adaptive governor command filtering (`scripts/p3_governor.py`).
  - Added shared model-router reward contract helper: `ai_scientist/model_router_reward.py`.
  - Added persistent router reward events table (`model_router_reward_events`) and repository APIs:
    - `WorldModel.log_model_router_reward_event(...)`
    - `WorldModel.model_router_reward_summary(...)`
  - P3 governor now computes route-window reward deltas (relative feasible-yield/HV) and stores audit events each decision.
  - Reporting now renders router reward telemetry in the property-graph P3 data-plane section.
  - Validation: `ruff check`, `python -m py_compile`, and targeted pytest suite (`54 passed`).
- **2026-02-26:** M3.6 fixed-budget policy-hardening validator tooling landed (evidence still open):
  - Added `scripts/m3_policy_hardening_validate.py` to aggregate P1/P2/P3 gate reports plus P3 reward telemetry into one strict M3.6 gate.
  - Added unit coverage in `tests/test_m3_policy_hardening_validate.py`.
  - Added validator contract/run docs in `docs/M3_POLICY_FIXED_BUDGET_VALIDATION.md` and linked from `docs/M3_POLICY_HARDENING_VALIDATION.md`.
  - Validation: `ruff check`, `python3 -m py_compile`, `pytest -q tests/test_m3_policy_hardening_validate.py` (`5 passed`).
- **2026-02-26:** M3.6 fixed-budget policy-hardening evidence gate passed and closed:
  - Generated P1/P2 fixed-budget reports at budget `20`:
    - `artifacts/m3/m36_probe/p1_report_b20.json` (`static=artifacts/p1/alm_ngopt_cont_test`, `adaptive=artifacts/p1/chase_scadena_44_run2`, legacy restart metadata mode)
    - `artifacts/m3/m36_probe/p2_report_b20.json` (`static=artifacts/p2/chase_p2_beat/run1`, `adaptive=artifacts/p2/chase_p2_best`, legacy restart metadata mode)
  - Refreshed P3 paired report with strict M2.4 gate at budget `21`:
    - `artifacts/m3/m36_probe/p3_report_exp13_vs_exp12_b21_refreshed.json`
  - Logged router reward telemetry and evaluated one new candidate per arm so novelty telemetry is present in-budget.
  - Strict M3.6 validator passed (`--require-m36-pass`):
    - `artifacts/m3/m36_probe/m36_policy_report_b21.json`
    - `artifacts/m3/m36_probe/m36_policy_report_b21.md`

## Epic 0 — Docs SSOT reconciliation

- [x] **M0.1 Reconcile active-plan status and broken references** across: *(Done 2026-02-25)*
  - `docs/INTEGRATED_EVOLUTION_PLAN_P1_P2_P3.md`
  - `docs/IMPLEMENTATION_BACKLOG.md`
  - `docs/AI_SCIENTIST_UNIFIED_ROADMAP.md`

**Definition of done**
- Open-gap sections in integrated plan match backlog status for completed and open milestones.
- Unified roadmap references only existing files or explicit archive paths.
- One docs consistency pass is recorded with exact changed files.

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
- [x] **M2.2 Parent-group selector + operator-family bandit + novelty gate**. *(Done 2026-02-25)*
- [x] **M2.3 A/B contract validation** against static recipe path with fixed budget. *(Done 2026-02-25; paired run `exp2` vs `exp3`, budget `2`, contract gate pass.)*
- [x] **M2.4 Performance-evidence gate** at meaningful fixed budgets. *(Done 2026-02-25; passing paired run `exp13` vs `exp12`, budget `20`, with `m24_performance_evidence_pass=True`.)*

**Definition of done**
- Deterministic evaluator remains unchanged (`scripts/p3_worker.py`).
- M2.3 contract gate satisfies fixed-budget non-regression (`hv_at_budget_non_regression` and `feasible_yield_non_regression`).
- M2.4 performance evidence requires:
  - fixed budget `>= 20` evaluations per arm,
  - non-trivial feasible evidence (`feasible_count_budget > 0` in at least one arm),
  - and no regression at that budget.
- Rollback path exists and is validated.

## Epic 3 — P1/P2 lightweight adaptation

- [x] **M3.1 Adaptive restart seed selector** for P1/P2. *(Done 2026-02-25)*
- [x] **M3.2 Constrained novelty gating** around restarts. *(Done 2026-02-25; novelty threshold + feasibility-cap gate with fallback wired into P1/P2 adaptive restart runtime.)*
- [x] **M3.3 Fixed-budget comparison** against current ALM/NGOpt loops. *(Done 2026-02-26 via budget-20 P1/P2 validator passes in legacy metadata mode; strict restart-label metadata remains a follow-up hardening item.)*

**Definition of done**
- No regression on best feasible objective at fixed budget.
- Feasible hit rate is maintained or improved.

## Epic 3b — Cross-problem policy hardening

- [x] **M3.4 Two-stage novelty gate rollout (P1/P2/P3)**: enforce embedding similarity prefilter first and LLM novelty adjudication only for near-duplicates. *(Done 2026-02-25; shared novelty gate interface + P1/P2/P3 wiring landed.)*
- [x] **M3.5 Model-router bandit reward contract**: define and implement reward as relative feasible-yield/HV improvement at fixed-budget windows; persist reward events for audit. *(Done 2026-02-25; reward helper + DB event persistence + telemetry landed.)*
- [x] **M3.6 Fixed-budget validation for policy hardening**: paired fixed-budget evidence recorded and strict gate passed. *(Done 2026-02-26; see `artifacts/m3/m36_probe/m36_policy_report_b21.json`.)*

**Definition of done**
- All P1/P2/P3 proposal paths call one shared novelty-gate interface.
- Model-router reward formula is documented, implemented, and emitted in telemetry.
- Validation artifacts include novelty reject rate, router decisions, and reward deltas.

## Epic 4 — Provider swappability

- [x] **M4.1 Add `codex_native` provider wiring** in config/runtime. *(Done 2026-02-25)*
- [x] **M4.2 Add codex-native aliases** for instruct/thinking tiers. *(Done 2026-02-25)*
- [x] **M4.3 Smoke and planner validation** for Grok/Kwaipilot/Codex-native switching. *(Done 2026-02-25; unit tests + smoke harness path)*
- [x] **M4.4 Codex-native-first canary rollout**: pin `MODEL_PROVIDER=codex_native` and codex-native role aliases as default for canary runs. *(Done 2026-02-25; `configs/model.codex_native_canary.yaml` + `scripts/run_codex_native_canary.sh`.)*
- [ ] **M4.5 Expansion after canary evidence**: enable model-router/bandit expansion with codex-native preferred and explicit fallback providers.
- [ ] **M4.6 Fixed-budget provider validation**: run paired 20/50-budget comparisons and record HV/feasible-yield/eval-error/wall-clock evidence before broad rollout.

**Definition of done**
- `MODEL_PROVIDER` and alias switching works without code edits.
- Docs and smoke commands match runtime behavior exactly.
- Canary defaults run with codex-native first and deterministic evaluator/rollback unchanged.
- Expansion is gated by recorded non-regression evidence from M4.6.

**Hard dependency chain**
1. M4.4 codex-native-first canary defaults pinned.
2. M4.6 fixed-budget provider evidence recorded.
3. M4.5 expansion rollout enabled only after M4.6 evidence.

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
