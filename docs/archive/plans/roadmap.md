# AI Scientist Implementation Roadmap

This roadmap turns `ai_scientist/improvement-plan.md` into an actionable backlog. Each group is self-contained, includes file pointers, and lists concrete acceptance criteria so any coding agent can execute tasks without hunting for extra context.

## Repository & Environment Context

- **Repo root:** `/Users/suhjungdae/code/software/proxima_fusion/RL-feasible-designs`
- **Working area:** `ai_scientist/` package (do **not** edit `constellaration/` per project guidelines)
- **Entry CLI:** `python -m ai_scientist.runner --config <path> --problem <p1|p2|p3>`
- **Key modules:**
  - Candidate flow: `ai_scientist/runner.py`
  - Tooling + samplers: `ai_scientist/tools.py`
  - Memory/db: `ai_scientist/memory.py`
  - Surrogate helpers: `ai_scientist/optim/surrogate.py`
  - Adapter PEFT hooks: `ai_scientist/adapter.py`
  - Agents + tool gating: `ai_scientist/agent.py`
  - RAG utilities: `ai_scientist/rag.py`
  - Reporting + summaries: `ai_scientist/reporting.py`, `ai_scientist/adaptation_helpers.py`

Use Python 3.10+, existing venv instructions (see repository README). Never introduce dynamic imports or `any` casts. Keep changes inside `ai_scientist/` unless explicitly stated.

---

## Phase 1 – Smarter Candidate Generation & Ranking

- [x] **Introduce constraint-aware candidate helper**
  - Create `_propose_p3_candidates_for_cycle` in `ai_scientist/runner.py` using the recipe from `improvement-plan.md`.
  - Pull near-feasible Stage‑3 entries via `_latest_evaluations_by_design` or direct DB queries and feed them to `tools.normalized_constraint_distance_sampler`.
  - Mix ~70% sampler output with 30% `_generate_candidate_params` random exploration; expose ratios via config for tuning.
  - Acceptance: unit test stub referencing `tests/test_tools_sampler.py`, plus logging that prints sampler vs random counts per cycle when `runtime.verbose` (add flag if missing).

- [x] **Train + apply surrogate ranker each cycle**
  - Use `ai_scientist/optim/surrogate.SimpleSurrogateRanker` (or inline minimal linear solver) to fit on `world_model.surrogate_training_data(target="hv", problem=cfg.problem)`.
  - Build feature vectors from candidate params (flatten boundary arrays) or reuse cached metrics if available.
  - Rank combined candidate pool and keep top `cfg.budgets.screen_evals_per_cycle` before hitting `_evaluate_stage`.
  - Acceptance: add regression test that injects fake `world_model` history and asserts ordering improves against random baseline.

- [x] **Expose curriculum knobs via config**
  - Extend `ai_scientist/config.py` experiment schema with `proposal_mix` `{"constraint_ratio": float, "jitter_scale": float, ...}`.
  - Update CLI parsing and default YAML so new helper can be tuned without code edits.

## Phase 2 – Close the Adaptation Loop (PEFT + Preference Logs)

- [x] **Implement real LoRA loading in `adapter.py`**
  - Store adapters under `reports/adapters/<tool>/<stage>/adapter.safetensors` (choose deterministic path).
  - Update `AdapterState.load_lora_weights` to check for files, call backend-specific loader (e.g., HF PEFT, ggml) before tool invocation.
  - Update `AdapterState.push_updates` to persist new weights produced during run (if runtime supports) or enqueue metadata into `reports/adapters/queue.jsonl`.
  - Document env toggle `AI_SCIENTIST_PEFT=1` in module docstring and CLI help.

- [x] **Wire preference logs into an offline fine-tune script**
  - Build `scripts/update_adapters.py` (or similar) that reads `world_model` preference tables (see `memory.WorldModel.append_preference_pair`) and exports supervised datasets.
  - Script should output LoRA bundles matching the naming convention above so `adapter.with_peft` picks them up next run.
  - Acceptance: dry-run script with sample DB, confirm new adapter timestamps show up in logs.

## Phase 3 – Agentized Planning & Tool Governance

- [x] **Add planning agent driver**
  - Create `ai_scientist/planner.py` hosting a `PlanningAgent` class.
  - Use `agent.provision_model_tier(role="planning")` to fetch the correct gate + system prompt.
  - Provide structured context (P3 summary, last N cycles, current budgets, relevant RAG snippets) as JSON sections in the agent prompt.
  - Allow the agent to call `make_boundary`, `evaluate_p3`, and new `retrieve_rag` tool (below) via `tools_api` schemas.
  - Runner should be able to switch between deterministic loop and agent-driven loop via CLI flag (`--planner agent` vs `--planner deterministic`).

- [x] **Expose RAG retrieval as a tool**
  - Register `retrieve_rag` in `ai_scientist/tools_api.py` + `tools_api_smoke.py` with params `{query: str, k: int}`.
  - Implementation should use `rag.retrieve` with the challenge doc index `ai_scientist/rag_index.db`.
  - Permit tool for planning/report roles in `configs/model.yaml` agent gates.

- [x] **Refine AgentGate enforcement**
  - Ensure `agent.validate_tool_call` is invoked before every tool execution routed via agents.
  - Add telemetry logging (role, tool, context hash) to ease debugging.

## Phase 4 – Governance & Budget Adaptation

- [x] **Strengthen stage-gating logic**
  - In `_run_cycle` after computing `p3_summary`, require `feasible_count >= cfg.governance.min_feasible_for_promotion` before allowing promote stage.
  - Track a moving average HV delta across `cfg.governance.hv_lookback` cycles (persist in `world_model.record_cycle_hv`). Gate S2→S3 only when average < epsilon.
  - Document the new knobs in config + README snippet.

- [x] **Adaptive budget allocator**
  - Maintain running stats (HV slope, feasibility rate, cache hit rate) and adjust `screen_evals_per_cycle`, `promote_top_k`, `max_high_fidelity_evals_per_cycle` within configured min/max bounds.
  - Implementation idea: `BudgetController` struct in `runner.py` that produces per-cycle overrides consumed by `_evaluate_stage`.

-## Phase 5 – Observability & Performance

- [x] **Parallelism defaults + safeguards**
  - Set `_evaluate_stage` default executor to `ProcessPoolExecutor` unless config explicitly says `thread`.
  - Ensure VMEC/forward model doesn’t oversubscribe threads (set `OMP_NUM_THREADS=1` inside worker initializer if needed).

- [x] **Cache telemetry**
  - Expose `tools.get_cache_stats(stage)` via new logging hook each cycle (write to `reports/cache_stats.jsonl`).
  - Add CLI flag `--log-cache-stats` to toggle verbosity.

## Phase 6 – Documentation & DX

- [x] **Refresh `improvement-plan.md`**
  - Cross-link each section here to the original rationale paragraphs so future contributors understand *why* we made a change.
  - Add summary at top referencing this roadmap file for actionable tasks.

- [x] **Author onboarding snippet**
  - Create `docs/ai_scientist_onboarding.md` with quickstart commands, env vars, and explanation of the new agent planner vs deterministic runner.
  - Reference this roadmap and specify the order to tackle tasks (Phases 1→6).

---

### Execution Tips

- Implement and land tasks phase-by-phase; each bullet is independent enough for a short PR.
- Every task should come with at least one unit or smoke test (runner integration can rely on fixture DBs in `tests/`).
- Keep config defaults conservative to preserve current behavior; introduce new features behind flags.
- Update `reports/` artifacts or logging only when feature-enabled to avoid noise in existing pipelines.

Once Phase 1 completes you can start measuring HV lift relative to baseline random search; later phases build on that signal.
