# AI Scientist – Unified Research‑to‑Production Roadmap

Purpose: merge autonomy, production-hardening, and research-grounding plans into a single, conflict-resolved blueprint that delivers an autonomous, physics-grounded AI scientist capable of producing feasible P2/P3 stellarator designs using the existing `ai_scientist` scaffold.

## Guiding Principles
- Single source of truth: constraints, margins, hashes, and schema are shared across evaluators, surrogates, caching, and logging.
- Reliability first: direction-correct objectives, NaN/Inf safe eval, deterministic feature ordering, schema-versioned hashes.
- Feasibility → objective: prioritize constraint satisfaction before HV/gradient gains; penalize failures deterministically.
- Lightweight, swappable surrogates: default RF bundle; Torch ensemble optional behind a flag.
- Autonomy behind flags: deterministic loop is default; agent/multi-agent, adapters, and daemon are opt-in presets.
- Grounding and observability: RAG cites external papers; per-cycle summaries, cache stats, budgets, seeds, and hashes logged.

## Resolved Defaults (conflict ledger)
- Surrogate: Production plan RF heads (classifier + regressor) implemented as `SurrogateBundle` default; uses `structured_flatten` with persisted `(mpol, ntor, schema_version)` and rounding before hash.
- Hash/caching: single `design_hash(params, schema_version, rounding=1e-6)` used everywhere (eval cache, world-model, surrogates).
- Mode defaults: deterministic planner loop by default; agent/multi-agent enabled via config flag/preset (`planner: agent`).
- RAG scope: unified index over both papers (2506.19583v1, 2511.02824v2) + MASTER plan + Production and Updated plans; capped tokens per retrieval.
- Adapters/PEFT: off by default; enabled via `AI_SCIENTIST_PEFT=1` and config path; loaders/trainers gated to avoid runtime surprises.
- Presets: namespaced configs (`configs/experiment.p3.prod.yaml`, `configs/experiment.p3.agent.yaml`); no silent overrides of defaults.

## Workstreams (deliver in order of dependency)

### 1) Reliability Backbone (Production + Updated)
- [x] Add `_safe_evaluate(compute, stage, maximize=False)`: direction-aware penalties (+1e9 for minimize, -1e9 for maximize), recursive NaN/Inf clamp, cache penalized failures.
- [x] Implement `compute_constraint_margins(metrics, problem)` returning signed violations; store margins + `max_violation` in results.
- [x] Persist schema and hash: `structured_flatten(params, schema)` with `(mpol, ntor, schema_version)` and rounding; reuse in caches/DB.
- [x] Tests: NaN produces penalty; minimize/maximize flags correct; flatten keeps R_cos(1,0) index stable across mpol/ntor.

### 2) World-Model (SQLite) + Logging
- [ ] Tables: `boundaries(hash, p, nfp, r_cos_blob, z_sin_blob, source, parent_id, created_at)`, `evaluations(boundary_id, stage, vmec_status, runtime_sec, metrics..., margins..., is_feasible, created_at)`, `cycles(cycle_idx, phase, p, new_evals, new_feasible, cumulative_feasible, hv, notes, created_at)`.
- [ ] Hash canonicalization (round 1e-6) before insert; store schema_version per record.
- [ ] Helpers: add/query feasible, near-feasible by `l2` margin, latest archive, cache stats.
- [ ] Tests: read/write roundtrip; resume keeps archive/HV identical on stub model.

### 3) Surrogates (Unified)
- [ ] `SurrogateBundle`: vectorizer (structured_flatten), scaler, RF classifier (feasibility prob), RF regressor (objective/HV); PyTorch ensemble behind flag.
- [ ] Training policy: fit if ≥8 samples; if <8 log “cold start” and preserve input order; regressors use feasible-only if ≥4 feasible else all.
- [ ] Ranking score: `E[value] = P(feasible) * corrected_objective`, sign inverted for minimize to keep higher=better.
- [ ] Timeout guard on fit/predict; retrain cadence K new points or every N cycles.
- [ ] Tests: feasible-high > infeasible-high for P2 (maximize) and inverse for P1 (minimize); perf: RF fit+predict 200 samples, schema 6x6 <0.2s CPU.

### 4) Feasibility-First Loop (Updated)
- [ ] Sampling: seed JSONs (rotating ellipse P3) + Gaussian low-mode perturbations; oversample 2× budget.
- [ ] Ranking: feasibility prob − constraint-distance + small uncertainty bonus; exploration ratio from config.
- [ ] Evaluate top-K at `stage="screen"`; failures → infeasible with max violation; log per-cycle feasible count + HV if feasible exists.
- [ ] Retrain per policy; record cycle summaries in `cycles` table and `reports/cycle_<n>.json`.

### 5) HV / Objective Phase
- [ ] P2: maximize `L∇B` with feasibility prob ≥ threshold.
- [ ] P3: maintain Pareto archive on (Aspect Ratio, L∇B); compute HV each cycle; optimizer NSGA-II or NGOpt; promote top-K from surrogate batch to `refine/final` by feasibility then objective.
- [ ] BudgetController adapts stage budgets using HV delta, feasibility rate, cache hit rate.

### 6) Planner & Agents (Autonomy + RPF)
- [ ] Planner flag `--planner agent`; deterministic remains default.
- [ ] Roles: planning (retrieve_rag, make_boundary, evaluate_p3), literature (retrieve_rag, write_note), analysis (evaluate_p3, make_boundary); gate tools via AgentGate and log role/tool/context_hash.
- [ ] PropertyGraph per cycle from world-model + literature notes; snapshot to `reports/graphs/`; feed to planner prompt.
- [ ] Daemon (`scripts/daemon.py`): checkpoint per cycle, wall-clock guard, resume flag, `OMP_NUM_THREADS=1` in workers.

### 7) RAG Rebuild (RPF)
- [ ] Convert PDFs with `markitdown 2506.19583v1.pdf -o docs/papers/2506.19583v1.md` and similarly for 2511.02824v2.
- [ ] Build unified index over papers + MASTER plan + Production + Updated/Unified plans at `ai_scientist/rag_index.db`.
- [ ] Tests: `tests/test_rag_indexing.py` asserts top-k hits from both papers.

### 8) Adapter Loop (Autonomy + RPF)
- [ ] `scripts/train_adapters.py`: read preferences DB, build SFT/DPO dataset, train PEFT adapter, save to `reports/adapters/<tool>/<stage>/adapter.safetensors` + metadata.
- [ ] Loader in `adapter.py`: load newest adapter when `AI_SCIENTIST_PEFT=1`; log adapter version in run metadata.
- [ ] Cadence: nightly/manual; keep last N adapters, track in `reports/adapters/queue.jsonl`.

### 9) Observability, Safety, CI
- [ ] Metrics/logs: per-cycle HV, feasible count, VMEC failure rate, retrain time, cache hit rate, budget overrides, cache stats JSONL.
- [ ] Reporting: include PropertyGraph summary and RAG citations with source spans.
- [ ] CI: add sklearn install step; run surrogate/eval/unit tests and RAG test; optional perf check for RF timing.

## Presets and Commands
- P3 production (deterministic): `python -m ai_scientist.runner --config configs/experiment.p3.prod.yaml --problem p3 --cycles 10`
- P3 agent/daemon: `python -m ai_scientist.runner --config configs/experiment.p3.agent.yaml --planner agent --problem p3 --cycles 10 --resume-from reports/checkpoints/cycle_*.json` (agent preset enables literature/analysis roles and daemon loop).
- RAG rebuild: `python - <<'PY'` block from RPF to rebuild `ai_scientist/rag_index.db` after converting PDFs.
- Adapter train: `AI_SCIENTIST_PEFT=1 python scripts/train_adapters.py --db reports/ai_scientist.sqlite --out reports/adapters`.

## Acceptance Criteria (end-to-end)
- P1/P2/P3 smoke runs: no crashes; failed evaluations penalized and cached; feasibility-first loop logs feasible points by cycle 2 (P3 preset).
- HV increases or plateaus gracefully; BudgetController adapts without exceeding wall-clock or eval budgets.
- Surrogate tests, RAG test, and checkpoint-resume test pass; CI green with sklearn dependency.
- Agent preset runs ≥3 cycles with gated tool calls and PropertyGraph snapshots; deterministic preset unchanged.
- Adapter path: when enabled, loader reports loaded adapter; trainer emits artifacts and metadata.

## Notes
- Never touch `constellaration/` code; wrap VMEC/forward model as black box.
- No dynamic imports or `any` casts; avoid extra try/except padding.
- Keep configs/presets conservative by default; experimental features stay behind explicit flags.

## Reference Map (where the details came from)
- Reliability backbone, surrogate RF spec, safe eval, schema/hash, tests: `docs/AI_SCIENTIST_PRODUCTION_PLAN.md` (Architecture Changes, Tests sections).
- Feasibility-first loop, SurrogateBundle abstraction, SQLite world-model schema, budgets/HV phase: `docs/AI_SCIENTIST_UPDATED_PLAN.md` (Fast Wins, Architecture sections).
- Planner/agent roles, PropertyGraph, daemon/checkpointing, adapter loop, multi-agent presets: `docs/AI_SCIENTIST_AUTONOMY_PLAN.md` (Planning Loop, Supervisor, Adaptation, Safety, CI sections).
- RAG rebuild commands, rotating-ellipse seed preset, benchmark-aligned configs, literature/analysis role definitions, PEFT trainer steps: `docs/AI_SCIENTIST_RESEARCH_PRODUCTION_FIX.md` (Sections 1–5).
- Additional context and constraints: `docs/MASTER_PLAN_AI_SCIENTIST.md`, `docs/ai_scientist_onboarding.md`, `docs/run_protocol.md` for run protocol and governance reminders.
