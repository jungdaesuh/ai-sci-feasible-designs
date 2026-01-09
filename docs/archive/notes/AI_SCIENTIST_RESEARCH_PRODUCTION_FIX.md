# AI Scientist: Research‑Production Remediation Plan

Purpose: turn the current mostly‑stubbed orchestration into a research‑grade, end‑to‑end system that can (1) ground plans in the provided papers, (2) search the ConStellaration space with seeds consistent with the benchmarks, (3) run coherent multi‑agent loops over long horizons, and (4) close the preference→LoRA adaptation loop without touching the upstream `constellaration/` code.

Scope and ownership
- Owners: `ai_scientist` maintainers (no changes in `constellaration/`).
- Target branch: mainline of this repo; keep features behind config flags where they change defaults.
- Success criteria: checklists per section; “done” requires commands run + artifacts written under `reports/`.

Prerequisites
- venv active; repo root = `/Users/suhjungdae/code/software/proxima_fusion/ai-sci-feasible-designs`.
- Obey AGENTS.md: no dynamic imports, no `any` casts, no extra try/except padding.

## 1) Ground everything in the two papers (RAG rebuild)
Objective: make planner/reporting cite external sources (2506.19583v1, 2511.02824v2) instead of internal notes.

Steps
1. Convert PDFs to markdown (keeps anchors stable):
   - `markitdown 2506.19583v1.pdf -o docs/papers/2506.19583v1.md`
   - `markitdown 2511.02824v2.pdf -o docs/papers/2511.02824v2.md`
2. Rebuild the RAG index with those and the existing positioning docs:
   ```bash
   python - <<'PY'
from ai_scientist import rag
sources = [
    'docs/papers/2506.19583v1.md',
    'docs/papers/2511.02824v2.md',
    'docs/MASTER_PLAN_AI_SCIENTIST.md',
    'docs/TASKS_CODEX_MINI.md',
    'docs/AI_SCIENTIST_PRODUCTION_PLAN.md',
]
rag.ensure_index(sources=sources, index_path='ai_scientist/rag_index.db')
print('indexed', sources)
PY
   ```
3. Smoke check retrieval:
   - `python - <<'PY'
from ai_scientist import rag
for q in ('hypervolume baseline', 'Kosmos world model', 'QI residual constraints'):
    print(q, rag.retrieve(q, k=2, index_path='ai_scientist/rag_index.db'))
PY`

Exit criteria
- `ai_scientist/rag_index.db` contains `chunks` table (no longer empty).
- Planner/tool `retrieve_rag` returns snippets from both papers.

## 2) Align seeds/config with ConStellaration benchmarks
Objective: stop exploring non‑standard 1‑field‑period shapes; start from rotating‑ellipse / dataset seeds so feasibility >0%.

Steps
1. Store a canonical seed JSON (rotating ellipse) under `configs/seeds/rotating_ellipse_p3.json` (generated via `constellaration.initial_guess.generate_rotating_ellipse` once, then frozen).
2. Update experiment configs (new preset `configs/experiment.p3.prod.yaml`):
   - `boundary_template.n_field_periods: 3`
   - `boundary_template.seed_path: configs/seeds/rotating_ellipse_p3.json`
   - `boundary_template.perturbation_scale: 0.01` (tighter around seed)
   - `proposal_mix.constraint_ratio: 0.7`, `exploration_ratio: 0.3`, `jitter_scale: 0.01`, `surrogate_pool_multiplier: 3.0`
   - `budgets.screen_evals_per_cycle: 24`, `promote_top_k: 6`, `max_high_fidelity_evals_per_cycle: 4` (tunable but not tiny)
3. Add a CLI preset in README/onboarding: `python -m ai_scientist.runner --config configs/experiment.p3.prod.yaml --problem p3 --cycles 10`.

Exit criteria
- Running the preset logs >0 feasible S2 designs by cycle 2 and records a Pareto archive in `reports/`.
- HV improves over cycles (tracked in `cycle_hv` table).

## 3) Multi‑agent coherence (Kosmos‑like loop)
Objective: three cooperating roles with shared state: planning, literature, analysis. Use the existing world‑model DB and PropertyGraph.

Steps
1. Define roles and allowed tools in `configs/model.yaml`:
   - planning → `retrieve_rag`, `make_boundary`, `evaluate_p3`
   - literature → `retrieve_rag` only (writes notes into PropertyGraph)
   - analysis → `evaluate_p3`, `make_boundary`
2. Extend `ai_scientist/planner.py` to:
   - materialize a `PropertyGraph` snapshot each cycle from `memory.WorldModel` + new `literature_notes` table;
   - pass that graph JSON into the planning prompt (keeps long‑horizon context like Kosmos world model).
3. Implement `ai_scientist/tools_api.py` tool `write_note{stage, text, source}` that inserts into `world_model` and `PropertyGraph`.
4. In `runner.py`, gate each tool call through `agent.validate_tool_call` (already referenced) and route planning/literature calls before candidate generation.
5. Parallelize literature and analysis agents per cycle (ProcessPool or ThreadPool) but ensure `PropertyGraph` merges into the same SQLite transaction per cycle.

Exit criteria
- Cycle logs show planning + literature agent tool calls every cycle.
- `PropertyGraph` serialized snapshots stored in `reports/graphs/` and referenced in reports.

## 4) Long‑horizon & budgeting for 12h runs
Objective: support multi‑hour autonomy with safe checkpoints and adaptive budgets.

Steps
1. Add checkpointing: every cycle write `reports/checkpoints/cycle_<n>.json` with `world_model.dump_state()` + RNG seeds + budget snapshot. Provide `--resume-checkpoint` flag in runner to reload.
2. Turn on `adaptive_budgets.enabled` by default in the prod preset; tune bounds to ±50% of base.
3. Cap thread oversubscription in workers: set `OMP_NUM_THREADS=1` in executor worker init in `_evaluate_stage` to keep VMEC throughput predictable.
4. Add wall‑clock guard: if `wall_clock_minutes` exceeded, runner writes a “graceful stop” report and exits cleanly (no half‑written DB rows).

Exit criteria
- Can stop after 6h and resume from last checkpoint with identical Pareto archive/hypervolume.
- Cache hit rate logged per stage (`reports/cache_stats.jsonl`).

## 5) Close the preference→LoRA loop
Objective: make the existing adapter hooks real: offline trainer consumes preference pairs and produces loadable LoRA bundles.

Steps
1. Write `scripts/train_adapters.py`:
   - reads `reports/ai_scientist.sqlite` tables `statements` / `preferences` (already logged via runner);
   - builds SFT/DPO dataset (prompt, chosen, rejected) per tool/stage;
   - trains a lightweight PEFT adapter (use HF peft; no dynamic import);
   - saves to `reports/adapters/<tool>/<stage>/adapter.safetensors` and records metadata JSON.
2. Register a real loader in `ai_scientist/adapter.py` (e.g., huggingface peft) alongside the existing JSON loader; keep staging via `AI_SCIENTIST_ADAPTER_PERSIST_DIR`.
3. Cron or manual cadence: run trainer nightly over the DB; keep last N adapters and log SHA to `reports/adapters/queue.jsonl`.

Exit criteria
- `adapter.with_peft` reports `loaded=hf_peft` (or equivalent) for `evaluate_p3` before tool execution when `AI_SCIENTIST_PEFT=1`.
- Preference dataset artifact saved under `reports/adapters/datasets/` per run.

## 6) Validation & reporting
Objective: guarantee reproducibility and evidence in outputs.

Steps
1. Add a deterministic regression test: `tests/test_rag_indexing.py` that asserts top‑k retrieval hits both papers after index rebuild.
2. Add `tests/test_runner_checkpoint_resume.py` to verify resume keeps HV and archive identical for a toy forward model stub.
3. Update `docs/ai_scientist_onboarding.md` with:
   - new prod preset command;
   - RAG rebuild instructions;
   - PEFT toggle (`AI_SCIENTIST_PEFT=1`).
4. Reporting: extend `ai_scientist/reporting.py` to include
   - PropertyGraph summary section;
   - citations from RAG sources (paper/line spans) per statement.

Exit criteria
- CI passes the two new tests.
- Reports show sourced citations from the two papers, and include graph snapshots.

---
Use this document as the authoritative backlog for the “research‑production‑grade” hardening pass. Each numbered section is independent and can be delivered in small PRs, but full credit requires all exit criteria to be met.
