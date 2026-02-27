# M3.4/M3.5 Policy Hardening Validation

This document captures the implementation contracts for:

- `M3.4` two-stage novelty gate rollout (`P1`/`P2`/`P3`)
- `M3.5` model-router bandit reward contract and telemetry

For fixed-budget cross-problem validation after these contracts, see:

- `docs/M3_POLICY_FIXED_BUDGET_VALIDATION.md` (`M3.6`)

## M3.4 Two-Stage Novelty Gate

Shared interface:

- `ai_scientist/novelty_gate.py::apply_two_stage_novelty_gate`

Contract:

1. Stage 1 embedding prefilter (`embedding_prefilter_min_distance`)
2. Stage 2 near-duplicate adjudication (`near_duplicate_distance`, judge called only for near-duplicates)
3. Optional fallback-to-ungated behavior per caller

Wiring:

- P1/P2 adaptive restart runtime: `ai_scientist/restart_runtime.py`
  - CLI knobs:
    - `--restart-novelty-min-distance`
    - `--restart-novelty-near-duplicate-distance`
    - `--restart-novelty-feasibility-max`
    - `--restart-novelty-judge-mode {disabled,heuristic}`
- P3 adaptive governor command gate: `scripts/p3_governor.py`
  - shared gate diagnostics embedded in `decision["adaptive_policy"]["novelty_gate"]`

## M3.5 Model-Router Reward Contract

Shared reward helper:

- `ai_scientist/model_router_reward.py::compute_model_router_reward`

Reward definition:

- weighted relative improvement of:
  - feasible yield per 100 evals
  - HV

P3 persistence/audit path:

- Event table: `model_router_reward_events`
- Inserted each governor decision in `scripts/p3_governor.py`
- Repository read/write APIs:
  - `WorldModel.log_model_router_reward_event(...)`
  - `WorldModel.model_router_reward_summary(...)`

Telemetry rendering:

- `ai_scientist/reporting.py` property-graph section now includes:
  - router reward event count
  - average/last reward
  - route-level reward event counts

## Validation Commands

```bash
ruff check ai_scientist/novelty_gate.py ai_scientist/model_router_reward.py ai_scientist/restart_runtime.py scripts/p3_governor.py ai_scientist/memory/schema.py ai_scientist/memory/repository.py ai_scientist/reporting.py scripts/p1_alm_ngopt_multifidelity.py experiments/p1_p2/p2_alm_ngopt_multifidelity.py tests/test_novelty_gate.py tests/test_model_router_reward.py tests/test_restart_runtime.py tests/test_restart_wiring_ast.py tests/test_restart_script_runtime.py tests/test_p3_governor_wiring.py tests/memory/test_memory.py tests/test_reporting_p3_data_plane.py
python3 -m py_compile ai_scientist/novelty_gate.py ai_scientist/model_router_reward.py ai_scientist/restart_runtime.py scripts/p3_governor.py ai_scientist/memory/schema.py ai_scientist/memory/repository.py ai_scientist/reporting.py scripts/p1_alm_ngopt_multifidelity.py experiments/p1_p2/p2_alm_ngopt_multifidelity.py tests/test_novelty_gate.py tests/test_model_router_reward.py tests/test_restart_runtime.py tests/test_restart_wiring_ast.py tests/test_restart_script_runtime.py tests/test_p3_governor_wiring.py tests/memory/test_memory.py tests/test_reporting_p3_data_plane.py
pytest -q tests/test_novelty_gate.py tests/test_model_router_reward.py tests/test_restart_runtime.py tests/test_restart_wiring_ast.py tests/test_restart_script_runtime.py tests/test_p3_governor_wiring.py tests/memory/test_memory.py tests/test_reporting_p3_data_plane.py
```
