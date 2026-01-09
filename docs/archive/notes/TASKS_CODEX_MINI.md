# Codex‑Mini Backlog — ConStellaration AI Scientist

Purpose: a queue of small, self‑contained tasks sized for Codex Mini. Each task has a tight scope, concrete files to touch, and a simple definition of done (DoD). Do not modify code under `constellaration/`.

Owner: <assign>
Status: Active
Updated: 2025‑11‑10

## Conventions
- Paths are repo‑relative. Use explicit imports; no dynamic imports; no casting to `any`.
- Keep PRs small (1–3 files). Add a brief note to link task → commit.
- When citing code/requirements in text or reports, include file path and `:line` when known.

---

## Wave 0 — Bootstrap

- [x] Task 0.1 — Package skeleton
  - Files: `ai_scientist/__init__.py`, `tools.py`, `runner.py`, `memory.py`, `rag.py`, `reporting.py`, `optim/__init__.py`
  - DoD: `python -c "import ai_scientist as m; print(hasattr(m,'__package__'))"` runs.

- [x] Task 0.2 — Config loader
  - Files: `configs/model.yaml`, `configs/experiment.example.yaml`, `ai_scientist/config.py`
  - DoD: `python -c "from ai_scientist.config import load; print(type(load()))"` prints a dict.

- [x] Task 0.3 — Repo guardrails
  - Files: `ai_scientist/guards.py`
  - DoD: `python -c "from ai_scientist.guards import verify; verify()"` passes.

---

## Wave 1 — Physics Wrappers (P1 minimal)

- [x] Task 1.1 — Boundary factory
  - Files: `ai_scientist/tools.py`
  - Add `make_boundary_from_params(params) -> SurfaceRZFourier` (explicit imports from `constellaration.geometry`).
  - DoD: unit test constructs a boundary without running VMEC.

- [x] Task 1.2 — P1 evaluator (low fidelity)
  - Files: `ai_scientist/tools.py`, `tests/test_tools_p1.py`
  - Add `evaluate_p1(params) -> {objective, feasibility, score, metrics}` using `forward_model.ConstellarationSettings()`.
  - DoD: test runs < 60s and asserts keys.

- [x] Task 1.3 — Memo cache
  - Files: `ai_scientist/tools.py`
  - LRU/dict cache keyed by `(fidelity, params_hash)`.
  - DoD: repeated calls hit cache (counter visible in test).

## Wave 2 — World Model (SQLite)

- [x] Task 2.1 — DB schema
  - Files: `ai_scientist/memory.py`
  - Tables: `experiments`, `candidates`, `metrics`, `pareto`, `citations`, `artifacts`.
  - DoD: `init_db('mem.db')` creates the union of expected tables.

- [x] Task 2.2 — CRUD helpers
  - Files: `ai_scientist/memory.py`
  - Add `start_experiment`, `log_candidate`, `log_metrics`, `upsert_pareto`.
  - DoD: toy script inserts and reads back rows plus property-graph edges.

- [x] Task 2.3 — NetworkX view
  - Files: `ai_scientist/memory.py`
  - Add `to_networkx(experiment_id)` property graph.
  - DoD: returns expected node/edge counts leveraged in tests.

---

## Wave 3 — RAG (Local)

- [x] Task 3.1 — Index builder
  - Files: `ai_scientist/rag.py`
  - Index MD docs with anchors: `2511.02824v2.md`, `ConStellaration Fusion Challenge_ Benchmarks and Solution Strategies.md`.
  - DoD: `build_index()` persists an index (SQLite with metadata and cached reuses).

- [x] Task 3.2 — Retrieval API
  - Files: `ai_scientist/rag.py`, `tests/test_rag.py`
  - `retrieve("edge rotational transform")` returns chunks citing `constellaration/src/constellaration/problems.py:...`.
  - DoD: test finds anchors and metadata (line ranges, chunk counts), and ensure_index honors cached DBs.

---

## Wave 4 — Runner (P1 single→multi cycle)

- [x] Task 4.1 — Single cycle
  - Files: `ai_scientist/runner.py`
  - Steps: load config → random candidates → evaluate (low fidelity) → pick best.
  - DoD: `python ai_scientist/runner.py --problem p1 --cycles 1` logs one eval and writes DB.

- [x] Task 4.2 — Multi-cycle + budget
  - Files: `ai_scientist/runner.py`
  - Add `--cycles`, `--eval-budget`, simple multiprocessing pool.
  - DoD: 3 cycles run; DB shows ≥3 candidates.

---

## Wave B — Core Loop Guards (Budgets, Fidelity, Promotions)

- [x] Task B.1 — Budgets and parallelism
  - Files: `ai_scientist/runner.py`, `configs/experiment.example.yaml`
  - Add and enforce: `screen_evals_per_cycle`, `promote_top_k`, `max_high_fidelity_evals_per_cycle`, `wall_clock_minutes`, `n_workers`.
  - DoD: a unit test or script shows budgets cap evaluations; parallel workers respect `n_workers`.

- [x] Task B.2 — Fidelity ladder + cache keys
  - Files: `ai_scientist/tools.py`, `ai_scientist/runner.py`
  - Implement screening with low-fidelity and promotion with high-fidelity (QI where needed). Cache keys: `(screen|promote, params_hash)`.
  - DoD: repeated evaluations hit the appropriate cache; promotion re-uses high-fi cache only.

- [x] Task B.3 — Numeric promotion criteria
  - Files: `ai_scientist/runner.py`
  - Implement Stage gates: S1→S2 by feasibility margin ≤ τ or objective improvement ≥ ε over M cycles; S2→S3 by HV-delta ≤ δ or budget exhausted.
  - DoD: dry-run transitions logged with thresholds from config.
  - Governance knobs for Phase 4 (see `ai_scientist/roadmap.md:76-78`):
    `governance.min_feasible_for_promotion` blocks promotions until at least N feasible screen entries exist, and `governance.hv_lookback` controls how many entries from `world_model.cycle_hv` (now tracking `hv_delta` and `hv_delta_moving_avg`) feed into the S2→S3 averaging guard.
- [x] Task B.4 — Canonical design hashes
  - Files: `ai_scientist/tools.py`, `ai_scientist/runner.py`, `ai_scientist/memory.py`
  - Compute `design_hash = sha256(canonical_params_json)` once per design; dedupe promotions/logging by hash and gate archives on `(cycle_id, design_hash)`.
  - DoD: unit test exercises dedup + Pareto logging per hash.
- [x] Task B.5 — HV parity test
  - Files: `ai_scientist/tools.py`, `tests/test_runner_promotion.py`
  - Ensure the HV calculation matches `MHDStableQIStellarator._score` (vectors = [-grad, aspect], ref [1,20]); add regression test.
  - DoD: test fails if HV orientation drifts.
- [x] Task B.6 — Prompt/constraint parity test
  - Files: `ai_scientist/prompts.py`, `tests/test_prompts_constraints.py`
  - Build planner prompts by introspecting problem classes; verify constraint values match `constellaration/problems.py`.
  - DoD: test asserts each constraint equals the class attribute.
- [x] Task B.7 — Transactional logging
  - Files: `ai_scientist/memory.py`, `ai_scientist/runner.py`
  - Add explicit transactions so budgets, HV logs, Pareto archive rows commit atomically; include rollback test.
  - DoD: simulated failure leaves DB with zero partial rows.

---

## Wave 5 — Reporting

- [x] Task 5.1 — Markdown report
  - Files: `ai_scientist/reporting.py`
  - Inputs: `experiment_id`; Output: `reports/<stamp>_p1.md` with tables and citations.
  - DoD: file exists; includes config JSON and best metrics.

- [x] Task 5.2 — Repro blocks
  - Files: `ai_scientist/reporting.py`
  - Insert code to reproduce top result (seeds, SHAs).
  - DoD: copy/paste reproduces metrics ± tolerance.

---

## Wave 6 — P2/P3 Evaluators

- [x] Task 6.1 — P2 (high fidelity + QI)
  - Files: `ai_scientist/tools.py`, `tests/test_tools_p2.py`
  - Use `ConstellarationSettings.default_high_fidelity()`.
  - DoD: returns keys; mark test as `-m slow`.

- [x] Task 6.2 — P3 (set eval + HV)
  - Files: `ai_scientist/tools.py`, `tests/test_tools_p3.py`
  - Accept list of boundaries; compute objectives, feasibilities, hypervolume.
  - DoD: HV ≥ 0 for toy set.

---

## Wave 7 — Optimizers & Surrogates

- [x] Task 7.1 — Search wrapper
  - Files: `ai_scientist/optim/search.py`
  - Implement Nelder–Mead or CMA‑ES proposal function.
  - DoD: produces batch of candidates from bounds.

- [x] Task 7.2 — Surrogate ranker
  - Files: `ai_scientist/optim/surrogate.py`
  - Fit KRR/MLP on cached data; rank proposals.
  - DoD: synthetic test shows ranking > random.

---

-## Wave 8 — Kimi‑K2 Integration

- [x] Task 8.1 — Tool schemas (OpenAI‑style)
  - Files: `ai_scientist/tools_api.py`
  - JSON schemas for `make_boundary`, `evaluate_p1/2/3`, `log_citation`, `report`.
  - DoD: `jsonschema` validation passes.

- [x] Task 8.2 — Client & gating
  - Files: `ai_scientist/agent.py`, `configs/model.yaml`
  - Two tiers: K2‑Instruct (loop) and K2‑Thinking (planning/reporting).
  - DoD: mock round‑trip tool call works.

- [x] Task 8.3 — Prompts
  - Files: `ai_scientist/prompts.py`
  - Include constraints/objectives verbatim from `constellaration/src/constellaration/problems.py` with RAG citations.
  - DoD: planner outputs a valid tool call JSON.

---

## Wave 9 — Quality & Repro

- [x] Task 9.1 — Determinism
  - Files: `ai_scientist/runner.py`
  - Global seeds, config capture, version dump.
  - Verified via `tests/test_runner_determinism.py` that consecutive `_run_cycle` runs with the same base seed/config write identical deterministic snapshots (seed, config, and metrics), hitting the ±1e-3 tolerance goal.
  - DoD: same seed → same result ± 1e-3.

- [x] Task 9.2 — CLI polish
  - Files: `ai_scientist/runner.py`
  - Flags: `--problem {p1|p2|p3}`, `--cycles`, `--screen`, `--promote`, `--slow`.
  - DoD: `--help` lists flags; invalid input exits with message.

---

## Wave X — Governance & Reporting

- [x] Task X.1 — Statements table
  - Files: `ai_scientist/memory.py`
  - Add `statements(id, experiment_id, stage, text, status, metrics_row_id, tool_name, tool_input_hash, seed, git_sha, repro_cmd, created_at)`.
  - DoD: insert + query with a toy statement.

- [x] Task X.2 — Claims verifier (deterministic)
  - Files: `ai_scientist/runner.py`, `ai_scientist/reporting.py`
  - Re‑run top‑1 claim with identical seed/settings; mark SUPPORTED unless any key metric deviates beyond tolerance; write status into DB and report.
  - DoD: a toy claim flips to REFUTED when tolerance is tightened artificially in test.

- [x] Task X.3 — Stage flags
  - Files: `ai_scientist/runner.py`
  - Implement `--stage {s1_impl,s2_refine,s3_ablate}` with persistence per cycle.
  - DoD: runner records stage per cycle in DB/logs.

- [x] Task X.4 — Related‑Work rewrite pass
  - Files: `ai_scientist/reporting.py`
  - After first full draft, run a RAG‑grounded “positioning” rewrite contrasting baselines with quotes and anchors.
  - DoD: report includes a rewritten section with at least 3 anchored quotes.

- [x] Task X.5 — Pareto + HV plots
  - Files: `ai_scientist/reporting.py`
  - Generate HV over cycles and final Pareto plot for P3.
  - DoD: figures saved under `reports/figures/` and linked in report.

- [x] Task X.6 — Constraint boundary curriculum
  - Files: `ai_scientist/tools.py`, `tests/test_tools_sampler.py`
  - Implemented the normalized-constraint-distance sampler plus a synthetic feasible-rate test versus uniform proposals.
  - DoD: sampler favors lower normalized distances and the test shows measurable feasible-rate improvement over random sampling.

- [x] Task X.7 — Report validator (citations)
  - Files: `ai_scientist/reporting.py`
  - Fail report build if any metric claim lacks a local anchor citation or cited anchor cannot be resolved.
  - DoD: test triggers failure when citation is omitted.

- [x] Task X.8 — HV delta & preset telemetry
  - Files: `ai_scientist/runner.py`, `ai_scientist/memory.py`, `tests/test_runner_presets.py`
  - Log HV deltas vs prior `cycle_hv` rows, persist a statement pointing to `adaptation/preference_pairs.jsonl`, and exercise `configs/run_presets.yaml` so CI asserts each preset leaves the right stage history entry.
  - DoD: runner logs `hv_delta_comparison` statements, stage history records every preset flag, and tests cover each preset.

## Priority Follow-Ups
- [x] Deliver Phase 1 for real: wire `configs/model.yaml:1-24` to the K2 tiers, log the tiering decision, and add a smoke test that exercises the `ai_scientist/tools_api.py:1-90` schemas.
- [x] Expand Phase 8/Wave X reporting: deterministic template sections, reproduction blocks, citations/figures, statements table, and verifier pipeline in `ai_scientist/reporting.py` + `ai_scientist/memory.py`.
- [x] Harden Wave 9 runner CLI: add `--screen`/`--promote`/`--slow`, deterministic seeding, stage persistence, and the governance log mentioned in docs.
- [x] Scope Phase 7 adaptation (SFT trajectories, PEFT/LoRA hooks, preference data) so the next developer sprint starts at Wave 7 requirements without extra discovery.
- [x] Phase 7 optional RLAIF evidence chain — the harness now persists `adaptation/preference_pairs.jsonl`, `adaptation/summaries/cycle_<n>_p3_summary.json`, and the trajectory JSONL, the verifier statement records those anchors, and the “Positioning vs baselines” report section cites them so trainers can trace the P1/P2/HV gains before checking the box. Implementation details live in `ai_scientist/runner.py:660-1040`, `ai_scientist/reporting.py:184-373`, and `orchestration/adaptation.py:92-154` (see docs/MASTER_PLAN_AI_SCIENTIST.md:226-247 for the gating narrative).

## Picking the Next Task
1) Start at the lowest unchecked task in the earliest wave.
2) Keep diffs small; avoid unrelated refactors.
3) If a task blocks, add a short note and pick the next unblocked task.

## Done Definition (template)
- Code added/updated only under `ai_scientist/` (and `configs/`, `tests/`, `reports/` when relevant).
- Simple test or script demonstrates behavior.
- No dynamic imports; explicit imports only.
- Docstring with usage example if a public function is added.

---

Cross‑reference: high‑level plan lives at `docs/MASTER_PLAN_AI_SCIENTIST.md`.
