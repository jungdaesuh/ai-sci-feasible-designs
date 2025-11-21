# P1 Orchestration TODOs

- [ ] P1 optimizer upgrade

  - [ ] Implement ALM + Nevergrad runner in orchestration (parallel, batched), reusing constellaration’s AL primitives
  - [ ] Add optional CMA-ES backend (pluggable optimizer switch)
  - [ ] Log every evaluation (objective, feasibility, metrics, boundary) to JSONL
  - [ ] Coarse → fine ladder: configure fidelity schedule (very_low → low)
  - [ ] Auto‑promote top‑K candidates to high_fidelity for certification
  - [ ] Persist certification outputs (high_fidelity metrics/artifacts)

- [ ] Resume + summaries

  - [ ] Add `--resume` using state.json (RNG state, best‑so‑far, budget remaining)
  - [ ] Periodic checkpointing of state.json during runs
  - [ ] Add `orchestration/summarize_run.py`

    - [ ] Plot/print best E vs evaluations
    - [ ] Feasibility norm histogram and pass rate by fidelity
    - [ ] Deltas after promotion (low → high fidelity)
    - [ ] Export CSV summaries for downstream analysis

  

  - P2 orchestration
    - New CLI runner wrapping SimpleToBuildQIStellarator;
      objective=max L_grad with QI/geom constraints; same logging/
      promotion flow.
  - P3 orchestration
    - Multi-objective (maximize L_grad, minimize A) with constraints;
      use NSGA-II (or pymoo) + feasibility filter; compute HV on
      feasible set; same logging.
  - Feasibility prefilter
    - Train a quick classifier (from our logs + provided dataset) to
      reject obvious infeasible candidates before VMEC; drop‑in gate
      for P2/P3.
  - Parallelism + budgets
    - Add multiprocessing pool + per‑fidelity budgets; cap walltime
      per eval; early abort on repeated VMEC failures.

## Acceptance Criteria (P1)

- Logging fields
  - Each JSONL record includes: `evaluation_id`, `boundary` (coefficients), `boundary_hash`,
    `objective`, `minimize`, `feasibility_norm`, `is_feasible`, `metrics`, `equilibrium`,
    `fidelity`, `duration_sec`, `success|error`, `alm_loss`, `lambda`, `rho`, `outer_iter`.
- Atomic checkpointing
  - Write `state.json.tmp` periodically (at least every `--workers` tells) and atomically rename
    to `state.json` to avoid corruption; update best artifacts immediately on improvement.
- Resume behavior
  - `--resume PATH` restores RNG state, optimizer state (`lambda`, `rho`, bounds, mask),
    ladder stage and promotions, best‑so‑far, and accurate budget counters (total/used/remaining).
