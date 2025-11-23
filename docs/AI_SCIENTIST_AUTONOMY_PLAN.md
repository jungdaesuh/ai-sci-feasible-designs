AI Scientist – Full Autonomy & Productionization Plan
=====================================================
Goal
----
Turn `ai_scientist` into a self-running system that (a) proposes and refines stellarator boundaries with zero human-in-the-loop, (b) learns from its own runs (adapters + surrogate + curriculum), and (c) stays operable/recoverable for 12–24h unattended jobs.

Scope Boundaries
----------------
- Only touch `ai_scientist/` plus new scripts/docs/configs; **never modify `constellaration/`**.
- Keep VMEC/forward-model interface as black box; wrap, don’t patch.
- Prefer small, testable increments; default to flags so legacy behavior remains available.

Fast-Path Checklist (execute in order)
--------------------------------------
1) Enable smarter candidate pool each cycle (constraint sampler + surrogate ranker on by default).
2) Route cycles through the planning agent (LLM/tool gated) instead of deterministic loop.
3) Close the adapter loop: consume preference logs, emit LoRA bundles, auto-load next run.
4) Add a supervisor/daemon for long unattended runs with crash-resume and metrics logging.
5) Harden evaluation safety (timeouts, NaN clamps, cache of failures) and observability.

Detailed Workstream Plan
------------------------
### 1. Candidate Generation & Ranking (Week 1)
- **Tasks**
  - Wire `_propose_p3_candidates_for_cycle` to always mix `normalized_constraint_distance_sampler` (~70%) with random jitter (~30%); ratios from config (`proposal_mix`).
  - Train `SimpleSurrogateRanker` each cycle when history ≥ MIN_SURROGATE_HISTORY; rank pooled candidates; keep top `screen_evals_per_cycle`.
  - Add exploration decay schedule: reduce random fraction as feasibility rate rises.
- **Artifacts**: Updated `runner.py`, config schema/documentation; logs show sampler vs random counts per cycle.
- **Tests**: surrogate ordering improvement vs random (synthetic data); sampler obeys jitter caps.

### 2. Agent-Driven Planning Loop (Week 1–2)
- **Tasks**
  - Add runner flag `--planner agent` (default stays deterministic until validated).
  - `PlanningAgent.plan_cycle` must invoke an LLM via `agent.provision_model_tier(role="planning")`, pull RAG snippets (`retrieve_rag` tool), and emit structured tool calls (`make_boundary`, `evaluate_p3`).
  - Enforce `AgentGate.validate_tool_call` before every tool routed from the agent; log `role/tool/context_hash`.
  - Expand `tools_api` schemas; add `propose_boundary` tool for perturbing seeds with bounded norms.
- **Artifacts**: `runner.py`, `planner.py`, `agent.py`, `tools_api.py`, config prompt snippets.
- **Tests**: smoke test with fake provider returning scripted tool calls; ensure gates reject unauthorized tools.

### 3. Adaptation / PEFT Loop (Week 2)
- **Tasks**
  - Implement real load/save in `adapter.AdapterState` reading/writing `reports/adapters/<tool>/<stage>/adapter.safetensors`; gate via `AI_SCIENTIST_PEFT=1`.
  - Build `scripts/update_adapters.py` to ingest SQLite preference pairs (`memory.WorldModel.append_preference_pair`), train small DPO/RLAIF or regression, and emit LoRA bundles.
  - Record adapter version in run metadata; load newest at startup.
- **Artifacts**: `adapter.py`, new script, docs section in onboarding.
- **Tests**: adapter loader picks up fresh weights; updater script dry-run on fixture DB.

### 4. Supervisor & Recoverability (Week 2–3)
- **Tasks**
  - Add `scripts/daemon.py`: loop over cycles, checkpoint budgets/HV summaries to `reports/run_state.json`, restart failed evals, honor wall-clock budget.
  - CLI flag to resume from checkpoint; ensure cache and DB reuse.
  - Worker init sets `OMP_NUM_THREADS=1`; per-task timeout/kill for forward model.
- **Artifacts**: daemon script, updates to `runner.py` to accept resume state.
- **Tests**: simulated crash resume; timeout returns penalized metrics without hanging pool.

### 5. RAG Expansion & Context Hygiene (Week 3)
- **Tasks**
  - Index domain docs: `2506.19583v1.md`, `ConStellaration Fusion Challenge_*.md`, `2511.02824v2.md` (Kosmos), roadmap files. Refresh `rag_index.db`.
  - Add `retrieve_rag` tool output to planner prompt as structured JSON; cap token budget.
- **Artifacts**: refreshed index, prompt snippets in `planner.py`/configs.
- **Tests**: smoke retrieval returns expected doc titles; planner prompt size below configured context length.

### 6. Safety, Governance, Observability (Week 3)
- **Tasks**
  - Wrap evaluate_p* with timeouts, NaN/Inf clamps, and cache penalized failures.
  - Enforce `governance.min_feasible_for_promotion` and HV slope gate before promotions.
  - Emit `reports/cache_stats.jsonl`, `reports/budget_overrides.jsonl`, per-cycle `p3_summary.json`; optional Prometheus textfile exporter.
- **Artifacts**: `tools.py`, `runner.py`, logging plumbing.
- **Tests**: NaN metrics produce penalized outputs; budget controller reacts to synthetic HV deltas.

### 7. CI, Testing, and DX (Week 3–4)
- **Tasks**
  - Add pytest slices for planner path, surrogate ranking, adapter loader, budget controller.
  - CI job installs sklearn + runs fast tests; heavy VMEC guarded by env flag.
  - Update `docs/ai_scientist_onboarding.md` and `run_protocol.md` with agent mode, adaptor update flow, supervisor usage.
- **Artifacts**: tests, CI config snippet, doc updates.
- **Tests**: new suites passing locally + CI; lint unchanged.

Milestones & Acceptance
-----------------------
- **M1 (end Week 1)**: Smarter candidate pools enabled by flag; surrogate ranking measurable lift on synthetic benchmark; deterministic loop still supported.
- **M2 (end Week 2)**: Agent-driven loop runs for ≥3 cycles without manual input; planner gated tool calls logged; adapter updater script produces weights.
- **M3 (end Week 3)**: Daemon can resume after simulated crash; eval safety clamp prevents hangs; RAG index refreshed and used in planner prompts.
- **M4 (end Week 4)**: CI green with new tests; docs/onboarding updated; default run supports 12h autonomous job with adapter loading.

Operational Playbook (post-merge)
---------------------------------
- Nightly: run `scripts/update_adapters.py --db reports/ai_scientist.sqlite --out reports/adapters` after long jobs.
- Long run: `python -m ai_scientist.runner --config configs/experiment.autonomy.yaml --planner agent --resume-from reports/run_state.json` (daemon can wrap).
- Monitoring: tail `reports/cache_stats.jsonl` and `reports/budget_overrides.jsonl`; check p3_summary hypervolume trend.

Dependencies / Constraints
--------------------------
- No dynamic imports or `any` casts; keep changes inside `ai_scientist/` + new scripts/docs.
- Preserve legacy behavior behind flags; default configs stay conservative.
- World model source of truth is `ai_scientist/memory.py` (PropertyGraph + statements/citations/candidates/cycles). The older boundary-only `world_model.py` is removed.
