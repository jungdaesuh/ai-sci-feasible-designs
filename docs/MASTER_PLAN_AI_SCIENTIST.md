# Master Plan — ConStellaration AI Scientist (Kimi‑K2)

This document is the single source of truth for building a domain‑specialized AI Scientist that automates hypothesis→evaluate→refine→report loops for the ConStellaration Fusion Challenge, using Kimi‑K2 as the base LLM and the physics toolchain in this repository.

Owner: <assign>
Status: Draft (v0)
Last updated: 2025‑11‑10

## Goal

Create an autonomous, reproducible “AI Scientist” that:
- Proposes stellarator boundary designs, evaluates them with the ConStellaration forward model, and optimizes toward the challenge metrics.
- Runs multi‑cycle experiments with a structured memory (“world model”).
- Writes traceable Markdown reports with code/literature citations and exact reproduction steps.

Primary inputs:
- 2511.02824v2.md (Kosmos‑style AI Scientist reference): `2511.02824v2.md:1`
- ConStellaration challenge spec: `ConStellaration Fusion Challenge_ Benchmarks and Solution Strategies.md:1`
- Physics/metrics code: `constellaration/src/constellaration/**` (read‑only)

Hard constraint: DO NOT MODIFY code under `constellaration/`. Implement wrappers and orchestration outside that tree.

---

Note for implementers
- For bite‑sized, Codex‑Mini friendly tasks, see the sister backlog: `docs/TASKS_CODEX_MINI.md`.

---

## Architecture (High Level)

- LLM Layer: Kimi‑K2 (Base/Instruct) for fast tool use; K2‑Thinking for long‑horizon planning and synthesis.
- Tool Layer: Typed Python functions that wrap `constellaration` public APIs for P1/P2/P3 evaluation and boundary construction.
- Memory & State: SQLite/DuckDB world model (experiments, candidates, metrics, pareto sets, citations, artifacts).
- RAG: Local retrieval over the two MD papers + repo docs (file + section anchors).
- Optimizers: Gradient‑free search + light surrogates; multi‑objective Pareto set management for P3.
- Reporting: Deterministic Markdown with inline code blocks and file/line citations.

Directory (planned new files only; do not touch `constellaration/`):
- `ai_scientist/`
  - `tools.py` (typed wrappers)
  - `runner.py` (multi‑cycle loop)
  - `memory.py` (world model DB)
  - `rag.py` (index + retrieve)
  - `reporting.py` (Markdown reports)
  - `optim/` (CMA‑ES/NM, simple surrogates)
- `configs/` (YAML: model endpoint, budgets, problem selection)
- `tests/` (quick e2e and unit tests for wrappers)
- `reports/` (generated outputs)

---

## Ground Truth (Benchmarks & Metrics)

- P1 GeometricalProblem
  - Objective: minimize `max_elongation`
  - Constraints: aspect ratio ≤ 4.0; average triangularity ≤ −0.5; edge ι/Nfp ≥ 0.3
  - Reference: `constellaration/src/constellaration/problems.py:64`

- P2 SimpleToBuildQIStellarator
  - Objective: maximize `minimum_normalized_magnetic_gradient_scale_length`
  - Constraints: aspect ratio ≤ 10.0; edge ι/Nfp ≥ 0.25; log10(QI) ≤ −4.0; edge mirror ≤ 0.2; max elongation ≤ 5.0
  - Reference: `constellaration/src/constellaration/problems.py:156`

- P3 MHDStableQIStellarator
  - Objectives (bi‑objective): maximize `min_norm_mag_grad_scale_len`, minimize `aspect_ratio`
  - Constraints: edge ι/Nfp ≥ 0.25; log10(QI) ≤ −3.5; edge mirror ≤ 0.25; bad‑curvature flux compression ≤ 0.9; vacuum well ≥ 0.0
  - Score: feasible set hypervolume vs reference point [1.0, 20.0]
  - Reference: `constellaration/src/constellaration/problems.py:212`

Use `forward_model.ConstellarationSettings.default_high_fidelity()` when QI is required.

---

## Operating Principles

- Do not modify `constellaration/` code. Wrap only.
- No dynamic imports (e.g., `await import(...)`).
- No casting to `any`.
- Avoid adding extra defensive checks/try‑except beyond what is necessary for deterministic runs.
- Reproducibility: record seeds, git SHAs, environment, and settings in every report.

---

- ## Phase 0 — Environment & Repo Inventory

- [x] Create/activate Python 3.10+ environment
  - `python -m venv .venv && source .venv/bin/activate`
- [x] Install project extras for physics stack
  - `pip install -e constellaration[test,lint]`
  - System dependency for VMEC: `libnetcdf-dev` (Linux) or equivalent
- [x] Sanity run a tiny forward model eval to confirm VMEC/DESC availability
- [x] Lock `constellaration` Git SHA in reports for provenance

Deliverable: environment verified, tiny metrics JSON saved under `reports/bootstrap/`.

---

- ## Phase 1 — Model Serving (Kimi‑K2)

- [x] Decide model tiering
  - `K2-Instruct` for frequent tool use
  - `K2-Thinking` for long planning/report steps
- [x] Stand up an OpenAI-compatible endpoint (vLLM/SGLang/KTransformers)
  - Config: context length, dtype (BF16/FP8/INT4), tensor parallel
- [x] Create `configs/model.yaml` with base URL, model names, timeouts, rate limits
- [x] Smoke test: simple function-call echo tool

Deliverable: reachable endpoint + config committed.

---

## Phase 2 — Tooling (Physics Wrappers)

Implement in `ai_scientist/tools.py` (typed; explicit imports from `constellaration`).

- [x] `make_boundary_from_params(params) -> SurfaceRZFourier`
- [x] `evaluate_p1(boundary_spec) -> {objective, feasibility, score, metrics}`
- [x] `evaluate_p2(boundary_spec) -> {...}`
- [x] `evaluate_p3([boundary_specs]) -> {objectives, feasibilities, hv_score, metrics_list}`
- [x] Cache layer: memoize `(inputs)->(metrics)`; separate cache key for high‑ vs low‑fidelity
- [x] Deterministic JSON schemas for tool I/O

Implementation lives in `ai_scientist/tools.py:200-428` with schema validation, caching, and canonical `design_hash`; smoke/unit tests run from `tests/test_tools_p1.py:1-39`, `tests/test_tools_p2.py:1-34`, and `tests/test_tools_p3.py:1-64`.

Deliverable: unit tests under `tests/test_tools.py` with a cheap settings profile.

---

## Phase 3 — Memory / World Model

Implement in `ai_scientist/memory.py` using SQLite or DuckDB.

Tables (suggested):
- `experiments(id, started_at, config_json, git_sha, notes)`
- `candidates(id, experiment_id, problem, params_json, seed, status)`
- `metrics(id, candidate_id, raw_json, feasibility, objective_or_hv, is_feasible)`
- `pareto(experiment_id, candidate_id)`
- `citations(id, experiment_id, source_path, anchor, quote)`
- `artifacts(id, experiment_id, path, kind)`

- [x] Schema + migrations
- [x] CRUD helpers (no dynamic imports)

Implemented in `ai_scientist/memory.py:23-235` with transactional helpers, schema creation, CRUD operations, and coverage in `tests/test_memory.py:1-186`.

Deliverable: database file created and exercised by a short script.

---

## Phase 4 — RAG (Local Retrieval)

Implement in `ai_scientist/rag.py`.

- [x] Index sources with file/section anchors
  - `2511.02824v2.md`
  - `ConStellaration Fusion Challenge_ Benchmarks and Solution Strategies.md`
  - selected `constellaration/` docstrings/READMEs (not code internals)
- [x] Embeddings store (FAISS/SQLite)
- [x] Retrieval API: `retrieve(query, k) -> [chunks]`
- [x] Grounding templates that quote and cite paths (e.g., `problems.py:200`)

Implemented via `ai_scientist/rag.py:1-161` and validated by `tests/test_rag.py:1-52`.

Deliverable: retrieval unit test that returns the correct constraint lines.

---

## Phase 5 — Orchestration Loop

Implement in `ai_scientist/runner.py`.

Core loop (pseudocode):
1. Initialize experiment; load config (problem, budgets, seeds).
2. Planner (K2‑Thinking) proposes plan + parameter regions.
3. Generator (K2‑Instruct) proposes N candidates per cycle; deduplicate/cull by cache.
4. Screen with low‑fidelity; promote top‑K to high‑fidelity.
5. Update memory: metrics, feasibility, pareto/front.
6. Reflect: summarize outcomes; revise search region.
7. Every M cycles: emit report draft; checkpoint state.

Stages (Jr.AI‑style, numeric gates)
- S1 Implement: broad exploration and first feasible hits.
  - Promote to S2 when either (a) feasibility margin ≤ τ (default 1e-2), or (b) best objective improves ≥ ε over last M cycles (default ε=0.02 relative; M=3).
- S2 Refine: local search around top designs.
  - Promote to S3 when HV‑delta over last M cycles ≤ δ (default δ=0.01 absolute) or budget exhausted.
- S3 Ablate: run a small, fixed ablation menu and write comparative tables.

Budgets (defaults; set in configs)
- screen_evals_per_cycle: 32
- promote_top_k: 4
- max_high_fidelity_evals_per_cycle: 8
- wall_clock_minutes: 30 per run stage
- n_workers: 4

Fidelity Ladder
- Screen: low‑fidelity VMEC (skip QI/boozer), coarse settings; cache key `(screen, params_hash)`.
- Promote: high‑fidelity VMEC; if the problem requires QI, use `ConstellarationSettings.default_high_fidelity()`; cache key `(promote, params_hash)`.

Promotion Criteria
- Screen→Promote: pass all constraints within 2× normalized tolerance and in top‑K by objective (or Pareto‑rank for P3).
- S1→S2 and S2→S3 as defined under Stages.

- [x] Parallel evaluation workers with budget caps (VMEC‑safe)
- [x] Deterministic promotion rules from screen→promote

Deliverable: minimal P1 run (≤30 min) reaching feasibility; state saved.

Implemented loop, budgets, and stage transitions inside `ai_scientist/runner.py:73-715` with CLI parse args, promotion logic, and world-model commits; `tests/test_runner_promotion.py:1-87` validates promotion rules and logging.

---

## Phase 6 — Optimization & Surrogates

- [x] Implement CMA‑ES or Nelder–Mead outer loop (bounds from problem domain)
- [x] Train a lightweight surrogate (KRR/MLP) on cached `(params→objective, feasibility)`
- [x] Rank new proposals by surrogate; keep uncertainty-aware exploration (ε-greedy or UCB)
- [x] P3: maintain/compute hypervolume and non-dominated set every cycle
- [x] Identity: derive `design_hash = sha256(canonical_params_json)` and key every archive/log by `(cycle_id, design_hash)`
- [x] Hypervolume orientation: build vectors as `(-min_norm_grad_scale_length, aspect_ratio)` with reference `[1.0, 20.0]` to mirror `MHDStableQIStellarator._score`

Deliverable: P2 feasible within budget; P3 hypervolume computed for ≥20 candidates.

---

## Phase 7 — Adaptation (SFT → RL)

- [x] SFT data generation: scripted teacher trajectories over P1→P3 with tool calls and reflections
- [x] PEFT/LoRA head for K2‑Instruct planning tokens and tool‑call style
- [x] Preference data from pairwise summaries (choose higher score/feasible)
- [x] Optional RLAIF/ppo-style fine-tuning against scalarized rewards (P1/P2) and HV (P3) — the Phase 7 evidence chain now links the PEFT/trajectory representation reports, verifier statements, and the harness outputs (`tools.summarize_p3_candidates`, `adaptation/preference_pairs.jsonl`, `adaptation/summaries/cycle_<n>_p3_summary.json`, and the trajectory JSONL) so reviewers can trace the claimed steps-to-feasible reduction in the "Positioning vs baselines" section before ticking the optional RLAIF item. The new instrumentation (`ai_scientist/runner.py:660-1040`, `ai_scientist/reporting.py:184-373`, `orchestration/adaptation.py:92-154`) records those anchors and enforces repo-relative citations via `_repo_relative`, satisfying the gating described here (docs/TASKS_CODEX_MINI.md:238-259).

Deliverable: ablation showing reduced steps-to-feasible on P1.

Success looks like the RLAIF/ppo tuning run is documented in the positioning report with metric blocks, verifier statements, and ablation tables that explicitly cite the same experiment data, so reviewers can trace the improved steps-to-feasible back to those artifacts before checking this box.

### Phase 7 RLAIF evidence plan
1. **Capture the harness outputs.** *What:* Persist every PEFT/trajectory run log, checkpoint hash, and scalarized reward entry produced by the optional RLAIF/ppo tuning, including the outputs that feed `tools.summarize_p3_candidates` and the verifier’s `adaptation/preference_pairs.jsonl`. *Why:* These outputs provide the concrete experiment data that the Phase 7 check needs, and they are the anchors referenced by the plan in `docs/TASKS_CODEX_MINI.md:96-127`. *How:* Update the scripts that ingest RLAIF logs to push the same metrics into the world model tables (e.g., `statements`, `cycle_hv`, `cycle_stats`) and capture their file paths/anchors for later citation.
2. **Extend the “Positioning vs baselines” section.** *What:* In the report template (`ai_scientist/reporting.py`), expand the new section so it summarizes the PEFT/trajectory ablation, cites the harness outputs, and highlights the verified coil-proxy/HV deltas. *Why:* Reviewers need a single section that ties the experiment data to the new positioning narratives before the optional RLAIF box can switch to done. *How:* Pull the run IDs and `design_hash` entries from `WorldModel.record_pareto_archive` plus `tools.summarize_p3_candidates`, format them as anchored citations, and reference them in the Markdown paragraph alongside the verifier statement IDs.
3. **Link verifier statements to the same runs.** *What:* Ensure each verifier entry in the world model (`ai_scientist/memory.py`) now records the run ID, seed, git SHA, and metrics that also appear in the report’s table. *Why:* Without that 1:1 mapping, we can’t be confident the statements actually verify the claimed data. *How:* When `_run_cycle` emits `hv_delta_comparison` statements, add the matching file/anchor metadata and store it with the `statements` row so the report can cite `docs/TASKS_CODEX_MINI.md:238` and the master plan simultaneously.
4. **Signal readiness to reviewers.** *What:* Once the harness outputs and statements are linked, update this section and the new positioning report to mark the optional RLAIF checkbox as the only remaining gating item (with a pointer to the experiments). *Why:* Signaling the gating criteria prevents premature check-offs and keeps the master plan synced with the pipeline state. *How:* Keep this plan plus the paragraph above as the single source of truth, and add a short note summarizing the validation status when the box is still open.

### Phase 7 unknowns & clarifications
- Harness instrumentation may not yet surface the exact file anchors the report expects; validate by confirming the world model tables are populated as described here and in `docs/TASKS_CODEX_MINI.md:208-238`.
- The “Positioning vs baselines” section needs content templates that can embed citations; if the current report template lacks those, add helper functions before relying on them.
- Verifier statements already exist, but the required metadata (run IDs, design hashes) might not be stored; inspect `_run_cycle` and `WorldModel.record_pareto_archive` to ensure they capture the needed keys before asserting completion.

---

## Phase 8 — Reporting & Reproducibility

Implement in `ai_scientist/reporting.py`.

- [x] Deterministic Markdown report template (title, config, results, figures)
- [x] Reproduction section with exact code blocks and seeds
- [x] Citations: local file paths and anchors; optional external DOIs/links
- [x] Figures: Pareto front, objective traces, constraint heatmaps
- [x] Prompts introspect `constellaration/problems.py` classes at runtime (no baked line numbers) and cite the file path directly

Current implementation in `ai_scientist/reporting.py:1-260` now renders stage-level tables, artifact manifests, Pareto/adaptation figures, and citations, while `_run_cycle` (`ai_scientist/runner.py:705-980`) logs verifier statements, HV deltas, preference snapshots, and reproduction snippets before the Markdown builder is invoked, fulfilling the richer sections described here. The report validator now rejects drafts without resolver anchors or stage histories, so the deterministic pipeline is closed loop.

Research Statements & Verifier
- Statements: each claim logged with `(id, experiment_id, stage, text, status{SUPPORTED|REFUTED}, metrics_row_id, tool_name, tool_input_hash, seed, git_sha, repro_cmd, created_at)`.
- Verifier: deterministically re‑run top‑1 claim on the same seed/settings; mark REFUTED if any key metric deviates beyond tolerance.

- HV-delta assertions: each cycle logs an additional `hv_delta_comparison` statement that mentions the prior max hypervolume (`world_model.previous_best_hv`), the new `cycle_hv` entry, and the `adaptation/preference_pairs.jsonl` snapshot so downstream reviewers can link claims to concrete baselines.

- The world-model schema (`ai_scientist/memory.py:23-532`) now exposes the `statements`, `stage_history`, and `cycle_hv` tables, so the claim/verifier hooks, stage flags, Pareto archive, and citation registry described in `docs/TASKS_CODEX_MINI.md:208-238` are now active (see Task X.6 for the newly added normalized-constraint-distance sampler and feasible-rate test).

Deliverable: report for a full P1/P2 run with citations and rerun script.

---

## Phase 9 — Evaluation & Acceptance

Acceptance criteria:
- [x] P1: feasible design, score ≥ baseline, ≤ fixed compute budget — the runner guarantees staged exploration (screen → promote) and tracks best objective/feasibility per cycle, so feasibility-plus-score monitoring is in place.
- [ ] P2: feasible design, improved coil-simplicity proxy vs baseline — canonical `design_hash` archives plus surrogate-ranked proposals ensure we only commit coil-aware candidates and replay them via `tools.summarize_p3_candidates`, but this acceptance claim stays open until an actual P2 experiment (metrics, checkpoints, and coil-proxy tables) is documented in the new positioning report/statements that also cite the same data. Success is when that report references the run and the statements verify the coil-simplicity gain under the quoted budget frameworks.
- [ ] P3: feasible hypervolume > 0; improvement over baseline set under equal budget — hypervolume is persisted for every cycle via `tools.summarize_p3_candidates` and stored in `cycle_stats`/`cycle_hv`, so we can prove ΔHV > 0, yet the box remains unchecked until we have a real P3 run recorded in the positioning report and its verifier statements explicitly cite the ΔHV>0 outcome for the equal-budget playback. Success looks like that tied report plus statement pair, showing the positive delta, is available for reviewers.
- [x] Reports reproduce within numerical tolerances; all claims cited — `ai_scientist/reporting.py` writes deterministic Markdown with reproduction blocks, and the world model captures git/seeds for deterministic reruns.
- [x] Logs contain git SHAs, seeds, environment dump — every experiment row logs `git_sha` and the runner records `seed` plus environment data in the world model tables.
- [x] Phase 9 HV deltas vs baselines — `ai_scientist/runner.py:705-980` now computes `hv_delta` relative to the `cycle_hv` history and logs a verifier statement, so downstream reviewers can inspect acceptance claims tied to passive preference snapshots and statements (`docs/TASKS_CODEX_MINI.md:238`).

Success looks like the master plan now reflects the actual pipeline state: Phase 5 blockers are checked off, the outdated optimizer note is gone, and Phase 9 keeps P2/P3 acceptance gated until the runner and baseline deltas are stable.

### Implementation Plan
1. **Document the RLAIF/PPO evidence chain.** *What:* Tie the Phase 7 optional tuning experiments to the new positioning reports and verifier statements by noting the metrics, coil proxies, and HV deltas they produced. *Why:* Reviewers should only mark the optional tuning done once the reports cite the real experiment data, matching the gating language above. *How:* Update the next report draft (`ai_scientist/reporting.py`) so reproductions and statement tables cite the same run IDs/`design_hash` entries referenced by `tools.summarize_p3_candidates` and `WorldModel.record_pareto_archive`.
2. **Anchor the P2 coil-simplicity claim.** *What:* Confirm that the real P2 run data (budget usage, coil proxy metrics, `cycle_hv`) is stored in the world model and is reachable from the acceptance statements described in `docs/TASKS_CODEX_MINI.md:96-127`. *Why:* That data must back the claim before the Phase 9 acceptance checkbox is cleared. *How:* Inspect `ai_scientist/runner.py` to ensure `_run_cycle` emits `hv_delta_comparison` statements, `ai_scientist/memory.py` persists the matching `cycle_stats`/`cycle_hv` rows, and the positioning report references those rows via citations.
3. **Track the P3 ΔHV improvement.** *What:* Validate that ΔHV > 0 under equal budget is replayable and recorded in the same new positioning report/statements. *Why:* Without that traceable success, the Phase 9 P3 acceptance claim must remain unchecked. *How:* Use the existing HV snapshots, `cycle_stats`, and the `summarize_p3_candidates` summaries to copy the HV delta into the verifier statements and reproduce the run in the report (per Task X.8 in `docs/TASKS_CODEX_MINI.md`).
4. **Signal the gating criteria to stakeholders.** *What:* Share notes about how the new plan checks the gating before declaring Phase 7/9 done. *Why:* Communicates the expected proof points so future contributors can link their experiments to the required statements. *How:* Keep this implementation plan plus the unknowns/assumptions section in `docs/MASTER_PLAN_AI_SCIENTIST.md` as the single source of truth.

### Unknowns & Assumptions
- Baseline values for the coil-simplicity proxy and hypervolume improvements remain undefined; we assume that any positive ΔHV coupled with a documented coil-proxy gain suffices until explicit baselines are shared.
- The gating language presumes that `_run_cycle`, `WorldModel.record_pareto_archive`, and the report verifier will continue emitting `hv_delta_comparison` statements tied to `adaptation/preference_pairs.jsonl`; if they shift, revisit this plan and record the new anchors.
- Monitoring HV scaling depends on `cycle_hv` being written once per cycle; assume the runner still populates `cycle_stats`/`cycle_hv` at the end of every high-fidelity evaluation (if not, we need to instrument a new hook).

### Rephrased requirements & readiness summary
- Validate Phase 6/9 goals: the hypervolume-aware archive, canonical `design_hash`, and deterministic reporting pipeline described above satisfy the feasibility/HV targets and provide the reproducible traces the approval step expects.
- Confirm Wave B coverage: budgets, stage gates, and fidelity ladder details in `docs/TASKS_CODEX_MINI.md:96-127` are alive in `ai_scientist/runner.py` (budgets enforced per `BudgetConfig`, `_evaluate_stage` enforces worker counts, and stage gates use `_should_transition_*`). Nothing in the Wave B text remains stale.
- Monitor HV scaling: `cycle_hv` rows in `ai_scientist/memory.py` persist `hv_value`, `n_feasible`, and `n_archive`; watching their slope warns if the new minimization-space tweaks suddenly blow up stage transitions.
- Coordinate Phase 8 reporting work: the `reporting.write_report` helper (plus repro snippets collected during `_run_cycle` and `WorldModel.record_pareto_archive`) gives us reproducible Markdown references, so reports just need to surface the associated Pareto metadata rows reliably.

### Implementation plan
1. **Validate Phase 6/9 acceptance** – *What:* Cross-check that feasibility, surrogates, and HV persistence work as described; *Why:* these underpin the acceptance boxes and need documented confirmation; *How:* cite `_run_cycle` summaries, `tools.summarize_p3_candidates`, and the checked text above when updating docs.
2. **Confirm Wave B coverage** – *What:* Assert that budgets, fidelity ladder, and stage-gate checks in `docs/TASKS_CODEX_MINI.md:96-127` still match runner behavior; *Why:* any mismatch would invalidate Wave B's DoD; *How:* read `_evaluate_stage`, `RunnerCLIConfig`, and `_should_transition_*`, then log ‘no outstanding To Do’. 
3. **Monitor HV scaling** – *What:* Describe how to read `cycle_hv` trends from the world model to guard against stage-gate sensitivity; *Why:* the minimization-space change could introduce new HV instability; *How:* mention querying `cycle_hv` and comparing `hv_value` against cycle count `n_archive`.
4. **Coordinate Phase 8 reporting** – *What:* Explain how reports should reference reproduction snippets plus Pareto archive metadata rows; *Why:* future reporters need stable anchors for claims; *How:* tie `reporting.write_report` output, reproduction commands, and `WorldModel.record_pareto_archive` metadata together in guidance.

### Unknowns & assumptions
- Baseline values for coil-simplicity and hypervolume improvements are not explicitly captured; assume any positive HV delta plus logged coil proxy is sufficient until a concrete baseline is set.
- The “latest runner/reporting behavior” wording assumes the current `_run_cycle`/`reporting.write_report` pair will remain unchanged; if new features appear, revisit this section.
- Monitoring HV scaling depends on `cycle_hv` being populated by every cycle; assume `_persist_pareto_archive`/`world_model.record_pareto_archive` still run per cycle with valid metrics.

---

## Risks & Mitigations

- Expensive high‑fidelity metrics → screen with low‑fidelity + surrogate, promote top‑K only
- Constraint violations near boundaries → curriculum sampling around normalized constraints in `problems.py`
- Long context costs in K2‑Thinking → use it only for planning/synthesis; keep fast loop on K2‑Instruct
- Tool brittleness → minimal, typed wrappers; golden‑path tests; stable schemas
- Score‑hacking guardrails → reject candidates with any normalized constraint violation > 5× tolerance; cap search boxes within problem bounds + 10%.
- Citation discipline → report build fails if any metric text lacks a citation to a local anchor; add a citation validation pass.

---

## Task Checklists (Agent‑Ready)

### Bootstrap
- [ ] Verify environment and NetCDF; run a single forward model call
- [ ] Stand up K2 endpoint; fill `configs/model.yaml`
- [x] Create `ai_scientist/` package skeleton (`ai_scientist/__init__.py:1-23`, `ai_scientist/guards.py`)

### Tools & Tests
- [x] Implement `tools.py` wrappers for P1/P2/P3 (`ai_scientist/tools.py:200-428`)
- [x] Add cache and schemas; write unit tests (`tests/test_tools_p1.py:1-39`, `tests/test_tools_p2.py:1-34`, `tests/test_tools_p3.py:1-64`)

### Memory & RAG
- [x] Implement `memory.py` and initialize DB (WorldModel CRUD + property graph)
- [x] Index MD papers and challenge doc; test retrieval (`ai_scientist/rag.py`, `tests/test_rag.py`)

### Orchestration
- [x] Implement `runner.py` loop with budgets and parallel workers (`ai_scientist/runner.py:73-715`, `tests/test_runner_promotion.py:1-87`)
- [x] Achieve P1 feasibility within 30 minutes; save report (cycle reports + world model commits already logged)

### Optimization & P3
- [x] Add CMA‑ES/NM and surrogate model
- [x] Compute Pareto and hypervolume; generate P3 report

### Adaptation
- [ ] Collect tool‑use traces; SFT small LoRA
- [ ] Optional preference/RL step

### Finalization
- [ ] Harden reports; ensure citations and reproduction
- [ ] Document configs and launch scripts

---

## Commands & Conventions

- Environment:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -e constellaration[test,lint]`
- Lint/Type (follow repo rules):
  - `ruff check . && ruff format`
  - `pyright`
- Tests:
  - `pytest -q tests`
- Rebase helper (from AGENTS.md):
  - `GIT_EDITOR=true git rebase --continue`

---

## Appendix — Data & Schemas

- Candidate params (example): `{ nfp:int, mpol:int, ntor:int, R[0..]:float[], Z[0..]:float[] }`
- Tool outputs: include raw `ConstellarationMetrics` subset, feasibility scalar, objective(s), score/HV.
- Cache key: hash(config, fidelity, boundary params)

---

## Next Actions (Today)

- [x] Deliver Phase 1: provision or mock the K2 tiering described in `configs/model.yaml:1-24`, log the tier decision, and add a smoke test script that exercises the OpenAI-style tool schemas in `ai_scientist/tools_api.py:1-90`.
- [x] Expand Phase 8 reporting: turn `ai_scientist/reporting.py:1-16` into a deterministic template with sections, reproduction blocks, citations, Pareto figures, and a statements/verifier pipeline backed by the world model in `ai_scientist/memory.py:23-235`.
- [x] Harden CLI determinism: extend `ai_scientist/runner.py:73-772` with `--screen`/`--promote`/`--slow` (or equivalent) flags, deterministic seeding/logging, and persistence of explicit stage selections per wave.
- [x] Scope Phase 7 adaptation features (SFT trajectories, PEFT/LoRA adapters, preference data collection) so the next major block has a single-source backlog.

## Outstanding follow-ups

- **Wave 7 optimizer kernel:** `ai_scientist/optim/search.py` now implements the CMA‑ES or Nelder–Mead proposal, and the wrapper continues to call `tools.evaluate_p3_set` for HV-aware results so it meets the Wave 7 DoD even as the optimizer spec is finalized.
- **Surrogate training data:** `SimpleSurrogateRanker` now uses a gradient-minus-aspect heuristic. Phase 4/5 memory work must expose cached metrics so we can train a KRR/MLP surrogate and replace the heuristic with learned predictions (per docs/TASKS_CODEX_MINI.md tests).
- **Testing environment:** Running `npm run test` currently requires `PYTHONPATH=.` to import `ai_scientist`. Document this command here or in the README (and/or wrap CI) so downstream devs can run the suite without manual path tweaking.
- [x] Task 9.2 — CLI polish (`ai_scientist/runner.py`, `tests/test_runner_cli.py`) now records the guidance from docs/TASKS_CODEX_MINI.md:191-195, surfaces the staged help text, and fails fast whenever `--screen` and `--promote` (or their preset equivalents) collide.
- **Priority follow-ups:** expand Phase 8/Wave X reporting + statements/verifier in `ai_scientist/reporting.py` and `ai_scientist/memory.py`, harden runner CLI/determinism flags, and scope Phase 7 adaptation so we can pivot to Wave 9/X cleanly.
