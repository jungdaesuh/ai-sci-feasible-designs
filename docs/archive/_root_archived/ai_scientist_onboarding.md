# AI Scientist Onboarding

This snippet assumes you are working inside `/Users/suhjungdae/code/software/proxima_fusion/ai-sci-feasible-designs` and have followed the repository setup docs. It summarizes how to bootstrap the agent stack, what knobs to watch, and how the planner-driven workflow differs from the deterministic runner described in `docs/AI_SCIENTIST_UNIFIED_ROADMAP.md`.

## Quickstart commands
1. Create/activate the virtualenv if it is not already active:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[test,experiments]"
   # or with uv:
   uv sync --extra test --extra experiments
   ```
2. Launch a baseline deterministic run (Phase 1 entry point):
   ```bash
   python -m ai_scientist.runner --config configs/experiment.yaml --problem p1 --planner deterministic
   ```
3. Try the agent planner once the basics are stable:
   ```bash
   python -m ai_scientist.runner --config configs/experiment.yaml --problem p2 --planner agent
   ```
4. Validate local behavior with tests once setup works:
   ```bash
   python -m pytest tests/
   ```

## Environment variables to note
- `AI_SCIENTIST_PEFT=1` toggles LoRA/PEFT loading so the adapters in `reports/adapters/...` are materialized during inference (see Phase 2 in the roadmap).
- Planner/cache/budget tuning currently uses CLI flags, not env vars: `--planner`, `--log-cache-stats`, `--eval-budget`, and `--workers`.
- If you keep provider overrides in `.env`, create it from `.env.example` and then run `set -a; source .env` before launching the runner.
- `AI_SCIENTIST_REMOTE_PROVIDER=1` toggles live provider calls for the gate smoke test (see below). For current live usage, set `OPENROUTER_API_KEY`.

### Provider switch recipes (today)
- **OpenRouter + Grok (current defaults):**
  ```bash
  export MODEL_PROVIDER=openrouter
  export AI_SCIENTIST_INSTRUCT_MODEL=grok-planning-short-loop
  export AI_SCIENTIST_THINKING_MODEL=grok-planning-full
  ```
- **OpenRouter + Kwaipilot:**
  ```bash
  export MODEL_PROVIDER=openrouter
  export AI_SCIENTIST_INSTRUCT_MODEL=kwaipilot-planning-short-loop
  export AI_SCIENTIST_THINKING_MODEL=kwaipilot-planning-full
  ```
- **Codex-native (requires a local OpenAI-compatible adapter endpoint):**
  ```bash
  export CODEX_NATIVE_BEARER_TOKEN="..."
  scripts/run_codex_native_canary.sh
  ```
  If you need remote provider calls instead of the local mock endpoint:
  ```bash
  AI_SCIENTIST_REMOTE_PROVIDER=1 CODEX_NATIVE_BEARER_TOKEN="..." scripts/run_codex_native_canary.sh
  ```

### ChatGPT subscription integration status
- Target architecture is tracked in `CODEX_NATIVE_SUBSCRIPTION_INTEGRATION.md`.
- Goal: native Codex/ChatGPT OAuth + profile management + local OpenAI-compatible adapter.
- `codex_native` is partially wired (provider + aliases), but native OAuth/profile management and a bundled adapter server are not implemented yet.

### Live provider smoke test
1. Choose one provider:
   - **OpenRouter**: export `OPENROUTER_API_KEY` and use the provider/model defaults from `configs/model.yaml` (currently `x-ai/grok-4.1-fast:free`).
2. Run the smoke harness against the configured provider endpoint:
   ```bash
   AI_SCIENTIST_REMOTE_PROVIDER=1 python tools/ci_tools_smoke.py
   ```
3. Use `AI_SCIENTIST_ENDPOINT_URL=<custom URL>` if you need to explicitly override the base URL. Otherwise the script uses the selected provider's `base_url` from `configs/model.yaml`.

## Supervisor / Daemon Usage
For long-running runs (Phase 4), use `scripts/daemon.py`. This wrapper sets `OMP_NUM_THREADS=1`, auto-selects the latest checkpoint at startup when available, and performs one conditional retry on failure (only when a newer checkpoint appears).

```bash
# Run for 50 cycles with a 12-hour wall-clock notice threshold; restart the command to resume from latest checkpoint
python scripts/daemon.py \
  --config configs/experiment.p3.prod.yaml \
  --problem p3 \
  --cycles 50 \
  --wall-clock-minutes 720 \
  --planner agent
```

The daemon currently:
- Check `reports/` for the latest `cycle_*.json` checkpoint.
- Retry once if the runner exits non-zero and a newer checkpoint is available.
- Force `OMP_NUM_THREADS=1` to prevent thread contention in workers.
- Print an overrun message if elapsed wall-clock exceeds `--wall-clock-minutes` after the run finishes.

## Planner vs deterministic runner (Phase 3 reference)
- **Deterministic runner:** Hard-coded loops generate candidates, evaluate them, and promote stages based on fixed logic. This pathway is great for reproducing legacy comparison baselines (Exercises in Phases 1 and 5).
- **Agent planner:** Reads `world_model` summaries, Pareto snapshots, and RAG-context chunks, then calls gated tools through `ai_scientist/tools_api` (including `make_boundary`, `evaluate_p3`, and `retrieve_rag`). It is driven by the `--planner agent` flag and reflects the multi-agent architecture described in `docs/AI_SCIENTIST_UNIFIED_ROADMAP.md`.
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
- The onboarding doc links back to `docs/AI_SCIENTIST_UNIFIED_ROADMAP.md`, lists the recommended phase order, and explains how to reproduce both workflows.

## References
- `docs/AI_SCIENTIST_UNIFIED_ROADMAP.md` (active phase-by-phase checklist)
- `docs/archive/plans/roadmap.md` (historical roadmap snapshot)
