# Multi-agent-evolve Agent Structure

## 1) High-level architecture
- The agent system is a Ray-orchestrated PPO runtime centered in `absolute_zero_reasoner/main_azr_ppo.py`.
- Execution mode is selected by `azr.task_type`:
  - `GeneralIORayPPOTrainer` (Proposer/Solver/Judge loop)
  - `CodeIORayPPOTrainer` (code generation/prediction with execution-based rewards)
- Roles are task-conditioned paths on a shared actor pipeline (`gen_*`, `pred_*`, `judge_general`) then merged for policy updates.

## 2) Core structure (where agents live)
- Entrypoint and runtime assembly:
  - `absolute_zero_reasoner/main_azr_ppo.py`
- Trainer/orchestration loops:
  - `absolute_zero_reasoner/trainer/ppo/azr_ray_trainer.py`
  - `absolute_zero_reasoner/trainer/ppo/reason_rl_ray_trainer.py`
- Role and prompt construction:
  - `absolute_zero_reasoner/data_construction/constructor.py`
  - `absolute_zero_reasoner/utils/prompt_manager.py`
  - `absolute_zero_reasoner/data_construction/initial_prompt_templates/default.json`
- Reward and evaluation wiring:
  - `absolute_zero_reasoner/rewards/reward_managers.py`
  - `scripts/evaluation/eval_ID.sh`
- Tool/executor integration:
  - `absolute_zero_reasoner/utils/code_utils/python_executor.py`

## 3) Execution flow
1. Launch via selfplay script (`scripts/selfplay/mae.sh`) into `python -m absolute_zero_reasoner.main_azr_ppo`.
2. `TaskRunner` initializes directories, checkpoint/resume state, batch sizing.
3. Worker-role map is created (`ActorRollout`, `Critic`, optional reward/ref policy workers).
4. Trainer loop generates per-role responses, computes rewards, and appends valid outputs into dataset manager state.
5. Role batches are concatenated for critic/actor updates; periodic eval/checkpoint save follows.

## 4) Extension/configuration points
- Role enablement and mixing are config-driven (`train_propose`, `train_solve`, `train_judge`, mix strategies) in `absolute_zero_reasoner/configs/*.yaml`.
- Prompt behavior is configurable through template JSON + `PromptManager`.
- Reward shaping knobs (format/diversity/intrinsic/judge behavior) are exposed in trainer configs.
- Resume/save/test cadence is controlled by config (`resume_mode`, frequencies, output dirs).
- Multi-turn tool interface exists in config (`multi_turn.tool_config_path`), defaulted off in typical runs.

## 5) Strengths and risks
- Strength:
  - Explicit multi-role co-evolution loop with persistent step-indexed state and recoverable checkpoints.
- Risk:
  - Config surface suggests multiple executors, but trainer path currently hard-restricts accepted executor values in parts of runtime code, creating extension/operator mismatch risk.
