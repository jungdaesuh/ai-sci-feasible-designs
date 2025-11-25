# AI Scientist Onboarding

This snippet assumes you are working inside `/Users/suhjungdae/code/software/proxima_fusion/RL-feasible-designs` and have followed the repository README. It summarizes how to bootstrap the agent stack, what knobs to watch, and how the planner-driven workflow differs from the deterministic runner described in `ai_scientist/roadmap.md` (see especially Phase 3).

## Quickstart commands
1. Create/activate the virtualenv if it is not already active:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e .
   ```
2. Run `hatch` shells (optional, but keeps dependencies isolated):
   ```bash
   hatch shell
   hatch run lint
   ```
3. Launch a baseline deterministic run (Phase 1 entry point):
   ```bash
   python -m ai_scientist.runner --config configs/default.yaml --problem p1 --planner deterministic
   ```
4. Try the agent planner once the basics are stable:
   ```bash
   python -m ai_scientist.runner --config configs/default.yaml --problem p2 --planner agent
   ```
5. Repeat or script runs via `npm run test` (invokes `python -m pytest`) once the setup works.

## Environment variables to note
- `AI_SCIENTIST_PEFT=1` toggles LoRA/PEFT loading so the adapters in `reports/adapters/...` are materialized during inference (see Phase 2 in the roadmap).
- `AI_SCIENTIST_PLANNER_MODE=agent|deterministic` (optional) mirrors the `--planner` flag and lets shell wrappers persist the choice.
- `AI_SCIENTIST_LOG_CACHE_STATS=1` enables the cache telemetry mentioned in Phase 5’s observability work.
- `AI_SCIENTIST_BUDGET_MULTIPLIER=<float>` can be used to temporarily scale `cfg.budgets.screen_evals_per_cycle`/`promote_top_k` while tuning the adaptive budget controller from Phase 4.
- The repo-level `.env` file now ships ready-to-source overrides for `MODEL_PROVIDER`, `AI_SCIENTIST_INSTRUCT_MODEL`, and `AI_SCIENTIST_THINKING_MODEL`. Run `set -a; source .env` (or copy those lines into your shell profile) so the runner automatically aims at the Kat Coder tiers without editing `configs/model.yaml` again.
- `AI_SCIENTIST_REMOTE_PROVIDER=1` toggles live OpenRouter calls for the gate smoke test (see below). Supply `OPENROUTER_API_KEY` from the Kat Coder Pro listing and optionally `AI_SCIENTIST_ENDPOINT_URL=https://openrouter.ai/api/v1` to force a specific base URL.

### Live OpenRouter smoke test
1. Export `OPENROUTER_API_KEY` after creating a key for [`kwaipilot/kat-coder-pro:free`](https://openrouter.ai/kwaipilot/kat-coder-pro:free).
2. Run the smoke harness against the hosted `/chat/completions` endpoint:
   ```bash
   AI_SCIENTIST_REMOTE_PROVIDER=1 python tools/ci_tools_smoke.py
   ```
3. Use `AI_SCIENTIST_ENDPOINT_URL=<custom URL>` if you need to explicitly override the base URL (e.g., a relay or proxy). Otherwise the script falls back to the local stub endpoint described in Phase 1.

## Supervisor / Daemon Usage
For long-running, unattended jobs (Phase 4), use `scripts/daemon.py`. This wrapper ensures the runner respects wall-clock limits, restarts automatically after crashes, and resumes from the latest checkpoint.

```bash
# Run for 50 cycles with a 12-hour wall-clock limit, auto-resuming if interrupted
python scripts/daemon.py \
  --config configs/experiment.p3.prod.yaml \
  --problem p3 \
  --cycles 50 \
  --wall-clock-minutes 720 \
  --planner agent
```

The daemon will:
- Check `reports/` for the latest `cycle_*.json` checkpoint.
- Restart the runner process if it crashes (non-zero exit code).
- Force `OMP_NUM_THREADS=1` to prevent thread contention in workers.
- Terminate gracefully if the wall-clock limit is exceeded.

## Planner vs deterministic runner (Phase 3 reference)
- **Deterministic runner:** Hard-coded loops generate candidates, evaluate them, and promote stages based on fixed logic. This pathway is great for reproducing legacy comparison baselines (Exercises in Phases 1 and 5).
- **Agent planner:** Reads `world_model` summaries, Pareto snapshots, and RAG-context chunks, then calls gated tools through `ai_scientist/tools_api` (including `make_boundary`, `evaluate_p3`, and `retrieve_rag`). It is driven by the `--planner agent` flag and reflects the multi-agent architecture described in Section 3 of `ai_scientist/roadmap.md`.
- Running both modes side by side lets you compare how the planning agent adapts the curriculum and surrogate-enhanced proposals introduced in Phases 1–3.

## Phase order (Phase 1 → Phase 6)
1. **Phase 1 – Smarter Candidate Generation & Ranking:** Hook the surrogate helpers and constraint-aware sampler so you have quality proposals before spending compute.
2. **Phase 2 – PEFT & Preference Logs:** Make the adapter loader and offline LoRA updater work so LLMs learn from preference data.
3. **Phase 3 – Agentized Planning & Tool Governance:** Give the planner agent structured context plus RAG tools; gate tool calls for safety.
4. **Phase 4 – Governance & Budget Adaptation:** Add stage-gates, moving-average checks, and adaptive budget controllers to avoid fake progress.
5. **Phase 5 – Observability & Performance:** Default to process-level parallelism, enforce cache telemetry, and expose logging knobs.
6. **Phase 6 – Documentation & DX:** Refresh this onboarding text and `improvement-plan.md`, linking every section to its rationale so the next contributor can learn why each phase exists.

## What success looks like
- You can run the deterministic runner and the planner agent locally with the commands above.
- You understand the knobs from the roadmap (Phases 1–6) and can cite the motivation for each and where the documentation lives.
- The onboarding doc links back to `ai_scientist/roadmap.md`, lists the recommended phase order, and explains how to reproduce both workflows.

## References
- `ai_scientist/roadmap.md` (official phase-by-phase checklist, including the Phase 6 documentation tasks you are reading about now)
