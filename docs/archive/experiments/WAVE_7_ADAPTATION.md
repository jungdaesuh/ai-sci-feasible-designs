# Wave 7 Adaptation Preparation

This note scopes the supervised fine-tuning (SFT) trajectory generation, PEFT/LoRA hook points, and preference-data plumbing so Wave 7 can begin without re-discovering the fundamentals.

## SFT Trajectory Generation

1. **Trajectory source**: reuse the deterministic reporting loop (ai_scientist/runner.py) so every cycle emits a JSONL snippet containing the prompt/tool sequence, normalized params, stage label, and best metrics from `reports/.../cycle_*.md`.
2. **Storage path**: ship a helper in `orchestration/` that appends to `reports/adaptation/trajectories/trajectory_{timestamp}.jsonl` with canonical (`seed`, `tool_input_hash`, `tool_name`, `reproduction_snippet`) keys.
3. **Normalization**: annotate each entry with the stage (`s1`, `s2`, `s3`) and cycle ID to let Wave 7 curricula fan out between exploration/refinement.

## PEFT/LoRA Hook Points

1. **Adapter entry**: wrap the inference client (ai_scientist.agent + ai_scientist.tools) so the `agent_gates` dispatch can insert a PEFT adapter stack before every tool call (matching docs/TASKS_CODEX_MINI.md:157-190 expectations).
2. **Checkpointing**: expose `ai_scientist.adapter_state` that proxies `load_lora_weights()` and `push_updates()` so the next sprint can pin stage-specific LoRA modules without touching constellaration/.
3. **Runtime toggles**: guard PEFT activation with new env flags (e.g., `AI_SCIENTIST_PEFT=1`) so Wave 7 can flip adapters inside CI without editing the scheduler.

## Preference Data Plumbing

1. **Preference capture**: record each governance decision (stage transitions, best metrics) as a tuple `(governance_stage, candidate_hash, reward_diff)` inside `reports/adaptation/preferences.csv`.
2. **Feedback loop**: when the verifier marks a statement `SUPPORTED` versus `REFUTED`, log the pair plus the associated reproduction command to a `preference_pairs.jsonl` file for future reward modeling.
3. **Metadata**: extend the SQLite world model to store `statement.status` (already done in ai_scientist/memory.py) so preference replayers can fetch ground-truth labels by cycle.

## Success Criteria

1. The runner writes trajectory JSONL entries and preference pairs to `reports/adaptation/...` each cycle (Wave 9/X reporting depends on the same data).
2. A stub `ai_scientist.adapter` module exposes `prepare_peft_hook()` and `apply_lora_updates()` without changing constellaration/ so Wave 7 engineers can plug in training variants.
3. Stage history, statement status, and preference metadata persist in SQLite for every cycle (see docs/TASKS_CODEX_MINI.md:206-238 and docs/MASTER_PLAN_AI_SCIENTIST.md:247-368 for governance/reporting tie-ins).

Maintaining these APIs and artifacts makes Wave 7 a matter of hooking the trainer onto the already-logged signals rather than re-inventing the data collection pipeline.
