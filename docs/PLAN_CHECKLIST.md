# Orchestration Plan – Checklist

- [ ] P1 Optimizer Upgrade
  - [ ] Implement ALM + Nevergrad runner in `orchestration/` (parallel, batched)
  - [ ] Optional CMA‑ES backend (pluggable optimizer switch)
  - [ ] Log every evaluation (objective, feasibility, metrics, boundary) to JSONL
  - [ ] Fidelity ladder: `very_low_fidelity` → `low_fidelity` (configurable)
  - [ ] Auto‑promote top‑K candidates to `high_fidelity` for certification
  - [ ] Persist certification outputs (high_fidelity metrics/artifacts)

- [ ] Resume + Summaries
  - [ ] `--resume` using `state.json` (RNG state, best‑so‑far, budget remaining)
  - [ ] Periodic checkpointing of `state.json` during runs (atomic write)
  - [ ] `orchestration/summarize_run.py`
    - [ ] Best E vs evaluation index
    - [ ] Feasibility norm histogram and pass rate by fidelity
    - [ ] Low→high fidelity deltas (E, constraints like ι/Nfp)
    - [ ] CSV export of key columns

- [ ] P2 Orchestration (Simple‑to‑Build QI)
  - [ ] Runner wrapping `SimpleToBuildQIStellarator` (objective=max L_grad; QI/geom constraints)
  - [ ] Same JSONL logging/promotion pipeline
  - [ ] Feasibility gating and failure‑tolerant batching

- [ ] P3 Orchestration (MHD‑Stable QI, Multi‑Objective)
  - [ ] Multi‑objective search (maximize L_grad, minimize A) with constraints
  - [ ] Compute hypervolume on feasible set; log per‑point objectives and feasibility
  - [ ] Pareto set certification at high_fidelity

- [ ] Feasibility Prefilter (Data‑Driven)
  - [ ] Train quick classifier on logged/dataset points to reject obvious infeasible candidates
  - [ ] Integrate prefilter before VMEC call; log prefilter decisions

- [ ] Parallelism, Budgets, Robustness
  - [ ] Process pool for evaluations; cap workers; batch ask/tell for reproducibility
  - [ ] Per‑fidelity budgets and per‑eval walltime caps
  - [ ] Backoff on repeated VMEC failures; clear error logging

- [ ] Validation & Reproducibility
  - [ ] Seed handling (NumPy/optimizer) and record git commit in `config.json`
  - [ ] Deterministic runs with fixed seeds + batch mode
  - [ ] Smoke tests on small budgets; sanity checks for certification consistency

Goal
Enable a new developer to implement:

- P1 optimizer upgrade (ALM + Nevergrad/CMA-ES, coarse→fine ladder,
  full eval logging)
- Resume + summaries (state.json resume; summary script)

Repo Context To Know

- Code you must not modify: constellaration/ (physics + scoring).
- Public contracts to rely on:
  - constellaration/forward_model.py:1 → forward_model(boundary,
    settings) -> (metrics, wout)
  - constellaration/problems.py:113 (GeometricalProblem) →
    get_objective(metrics), compute_feasibility(metrics),
    is_feasible(metrics), \_score(metrics)
  - constellaration/forward_model.ConstellarationSettings presets:
    - P1 low/very_low fidelity: disable QI (use
      boozer_preset_settings=None, qi_settings=None)
    - P1 high fidelity: default_high_fidelity_skip_qi()
- Boundary type: constellaration/geometry/
  surface_rz_fourier.SurfaceRZFourier
- Initial guesses: constellaration/initial_guess.py:8 (e.g.,
  generate_rotating_ellipse)

Environment

- Python 3.10+, virtualenv recommended.
- Install project in editable mode: pip install -e
  constellaration[test,lint]
- System dep for VMEC++ and its Python binding already vendored under
  vmecpp/ (builds as part of install).
- Run-time: multi-core CPU helpful; set parallelism responsibly.

Existing Orchestration Layer (you will extend)

- orchestration/run_paths.py: run directory scaffold, JSON/JSONL
  helpers, git-hash capture.
- orchestration/evaluation.py: one-call evaluation wrapper for any
  fidelity; returns objective, feasibility, full metrics; handles
  VMEC failures.
- orchestration/perturb.py: safe Fourier perturbations.
- orchestration/random_search.py: annealed random-search baseline with
  JSONL logging.
- orchestration/run_p1_random_search.py: CLI to run P1 baseline and
  high-fidelity certification.

P1 Optimizer Upgrade – Context/Requirements

- Objective/constraints source of truth: GeometricalProblem (P1)
  minimizes max elongation E with tight constraints on A, δ, ι/Nfp.
- Fidelity ladder:
  - Search at very_low_fidelity and/or low_fidelity (no QI); promote
    best feasible to high_fidelity for certification.
- Optimizers:
  - ALM outer loop (penalize positive constraint violations).
  - Inner oracle: Nevergrad NGOpt (default) with parallel ask/tell
    and bounded box around current iterate; optional CMA-ES backend.
- Parameterization:
  - Optimize directly over SurfaceRZFourier coefficients (r_cos,
    z_sin); keep DC terms constrained if needed (e.g., don’t move
    major radius by fixing [0,0]).
  - Build mask of trainable coefficients; ravel to vector; unflatten
    back to boundary for evaluations.
- Parallelism model:
  - Use process pool (each eval in a separate process) to avoid
    sharing VMEC state; cap workers to available cores; batch ask/
    tell for reproducibility.
- Reproducibility:
  - Seed all RNGs; log seed and git hash; use batch mode when
    running parallel Nevergrad.

Resume + Summaries – Context/Design

- Logging schema (evaluations.jsonl):
  - Keys: evaluation_id, boundary (all coefficients), boundary_hash, objective, minimize,
    feasibility_norm, is_feasible, score, metrics (full), equilibrium summary,
    fidelity, duration_sec, success|error, outer_iter, alm_loss, lambda, rho.
- Resume state (state.json):
  - version: schema version
  - optimizer: {type: "ALM+NGOpt"|"CMA-ES", hyperparams, lambda: list[float], rho: float,
    bounds: {low: list[float], high: list[float]}, mask: list[bool]}
  - rng_state: NumPy bit generator state (via rng.bit_generator.state)
  - budget_total, budget_used, current_outer_iter, evals_done, budget_remaining
  - best_low, best_high: last known best records (full JSON snippets)
  - ladder: {stages: ["very_low_fidelity","low_fidelity"], current_stage,
    promotions_done, topk}
  - KPIs tracked for summaries: best E vs eval_id, pass rate by fidelity, promotion deltas
    (E_low − E_high, |ι/Nfp_target − ι/Nfp_high|).
  - Atomic checkpoints: write `state.json.tmp` every N tells (default N=workers), then atomically
    replace `state.json`.
  - CSV export columns: evaluation_id, fidelity, objective, feasibility_norm, is_feasible, duration_sec,
    outer_iter, alm_loss, lambda, rho, boundary_hash.

Acceptance Criteria (Done = ✅)

- P1 ALM+Nevergrad runner CLI under `orchestration/`:
  - Runs parallel ask/tell evaluations; writes JSONL with every eval; deterministic with given seed.
  - Supports fidelity ladder; writes `best_low_fidelity.json` and `best_high_fidelity.json`.
  - Logs include `alm_loss`, `lambda` (λ), `rho` (ρ), `outer_iter`, and a stable `boundary_hash`.
- Resume:
  - `--resume PATH` restores RNG, budgets, best‑so‑far, and continues seamlessly.
  - Periodic atomic checkpointing of `state.json` using tmp+rename.
- Summaries:
  - CLI prints KPIs and optionally emits simple plots/CSVs.
  Implementation Sketch (high level)

- New file: orchestration/alm_ngopt_p1.py
  - Build mask/ravel for SurfaceRZFourier
  - ALM loop:
    - Propose batch candidates via NGOpt within bounds around
      current x
    - Evaluate in process pool using
      orchestration.evaluation.evaluate_boundary
    - Compute ALM scalarized loss: f + sum(0.5ρ_i[max(0, λ_i/ρ_i +
      g_i)]^2 − 0.5\*(λ_i/ρ_i)^2)
    - tell() each candidate; update multipliers/penalties/bounds
      per loop
    - Log every evaluation to JSONL
  - Fidelity ladder:
    - Stage S∈{very_low,low}: run budget; track feasible
      elite set; promote top-K to high_fidelity using
      evaluation.evaluate_boundary
- New file: orchestration/summarize_run.py
  - Read runs/<ts>\_<tag>/evaluations.jsonl
  - Compute KPIs; print table; optional --plots for quick matplotlib
    figures
- Extend existing runner: add --resume PATH, --topk, --workers,
  --ladder "very_low,low".

Physics/Math Caveats To Observe

- ι/Nfp is an equilibrium output: small coefficient perturbations can
  flip feasibility; ALM + robust inner search handles non-smoothness.
- Keep P1 QI off (don’t run Boozer/QI metrics) for speed and
  consistency with problem definition.
- Certification must re-evaluate exactly the same boundary at high
  fidelity (no re-optimization).

Quick Start For New Dev

- Create venv; pip install -e constellaration[test,lint]
- Dry-run baseline: python -m orchestration.run_p1_random_search --tag
  smoke --budget 20 --seed 0 --fidelity low_fidelity
- Inspect runs/<ts>\_smoke/evaluations.jsonl to understand record shape
- Implement ALM+NGOpt in orchestration/alm_ngopt_p1.py using the
  contracts above
- Add --resume and write orchestration/summarize_run.py
- Verify: small budgets (≤50) on laptop; confirm logs, resume, and
  summary outputs; no changes under constellaration/
  
Defaults & Specs

- ALM (single‑objective, inequality constraints g(x) ≤ 0):
  - Initialization: λ₀ = 0 for each constraint; ρ₀ = 1.0; ρ_max = 1e6; growth γ = 2.0.
  - Update per outer loop: λ ← max(0, λ + ρ·ḡ). If any ḡ > τ_viol then ρ ← min(γ·ρ, ρ_max).
  - Stopping: max_i ḡ_i ≤ τ_feas and |f̄_t − f̄_{t−1}| ≤ τ_f for 2 loops or budget exhausted.
  - Thresholds: τ_viol = 1e‑2; τ_feas = 1e‑3; τ_f = 1e‑3.
- Vectorization & Bounds:
  - Freeze r_cos[0,0] and z_sin[0,0]; mask/ravel other coefficients.
  - Per‑dim bounds: [c₀ − S·σ₀, c₀ + S·σ₀] with S=5.0, σ₀ from |c₀|+ε or mode‑wise scale; clamp before VMEC.
- Nevergrad:
  - NGOpt; `num_workers=--workers`; `batch_mode=True`; budget per outer loop and increment up to `--budget-max`.
  - Seed NG RNG from NumPy seed for determinism.
- Promotion policy:
  - Maintain feasible elite set at search fidelity; every outer loop promote top‑K distinct boundaries.
  - K = max(3, min(20, ceil(0.1 · loop_budget))). Deduplicate by SHA‑1 of rounded coeff vector; tie‑break by lower objective, then shorter walltime.
- Checkpointing:
  - Write `state.json.tmp` every `workers` tells, then `os.replace` to `state.json`; persist best artifacts immediately on improvement.
- Summaries (`orchestration/summarize_run.py`):
  - KPIs: best E vs eval_id; feasibility pass rate by fidelity = (#feasible/#success) per fidelity; promotion deltas: E_low−E_high, |(ι/Nfp)_high − target|.
  - CSV columns: evaluation_id, fidelity, success, objective, minimize, feasibility_norm, is_feasible, duration_sec, alm_loss, outer_iter, lambda[], rho[], boundary_hash.
- CLI:
  - `python -m orchestration.run_p1_alm_ngopt --tag TAG --budget-initial 800 --budget-increment 400 --budget-max 4000 --seed 0 --workers 8 --ladder very_low_fidelity,low_fidelity --topk 10 --bound-scale 5.0 [--resume PATH]`
- Guardrails:
  - `workers = min(8, physical_cores)` by default. Per‑eval walltime cap (e.g., 300 s) and retry/backoff; mark failures clearly in logs.
