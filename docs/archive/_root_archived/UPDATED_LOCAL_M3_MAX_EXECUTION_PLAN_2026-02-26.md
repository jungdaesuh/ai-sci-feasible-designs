# Updated Local Execution Plan (M3 Max 36 GB)

Last updated: February 26, 2026
Repository: `ai-sci-feasible-designs` (`constellaration` + `vmecpp`)

## Purpose
This is an improved, repo-aligned execution plan for solving the ConStellaration benchmarks locally on Apple Silicon with strict VMEC budget control.

## Scope And Non-Goals
- [ ] Use this repo's actual problem definitions and metric names as SSOT.
- [ ] Keep all heavy physics validation in controlled VMEC batches.
- [ ] Do not modify upstream/original code in `constellaration` or `vmecpp`.
- [ ] Do not optimize for cloud workflows; optimize for local reliability first.

## Ground Truth Problem Definitions (Repo SSOT)
Reference: `constellaration/src/constellaration/problems.py`

- [ ] P1 `GeometricalProblem`
- [ ] Objective: minimize `max_elongation`.
- [ ] Constraints: `aspect_ratio <= 4.0`.
- [ ] Constraints: `average_triangularity <= -0.5`.
- [ ] Constraints: `edge_rotational_transform_over_n_field_periods >= 0.3`.

- [ ] P2 `SimpleToBuildQIStellarator`
- [ ] Objective: maximize `minimum_normalized_magnetic_gradient_scale_length`.
- [ ] Constraints: `aspect_ratio <= 10.0`.
- [ ] Constraints: `edge_rotational_transform_over_n_field_periods >= 0.25`.
- [ ] Constraints: `log10(qi) <= -4.0`.
- [ ] Constraints: `edge_magnetic_mirror_ratio <= 0.2`.
- [ ] Constraints: `max_elongation <= 5.0`.

- [ ] P3 `MHDStableQIStellarator` (multi-objective)
- [ ] Objectives: maximize `minimum_normalized_magnetic_gradient_scale_length`.
- [ ] Objectives: minimize `aspect_ratio`.
- [ ] Constraints: `edge_rotational_transform_over_n_field_periods >= 0.25`.
- [ ] Constraints: `log10(qi) <= -3.5`.
- [ ] Constraints: `edge_magnetic_mirror_ratio <= 0.25`.
- [ ] Constraints: `flux_compression_in_regions_of_bad_curvature <= 0.9`.
- [ ] Constraints: `vacuum_well >= 0.0`.

## Hardware-Constrained Strategy
- [ ] Keep VMEC budget finite and explicit.
- [ ] Use surrogate-first ranking for nearly all search steps.
- [ ] Validate only top-ranked and diverse candidates with VMEC.
- [ ] Prefer Docker for VMEC reliability on macOS.

## Phase 0: Environment Setup (Day 0)

### 0.1 Base Toolchain
- [ ] Install dependencies: `netcdf`, `cmake`, `gcc`, `libomp`.
- [ ] Install Docker Desktop and verify `docker run hello-world`.
- [ ] Create Python env (`uv` or conda) pinned to Python 3.11.

### 0.2 Repo Install
- [ ] Install `constellaration` package in editable mode.
- [ ] Confirm import path for forward model and problem classes.
- [ ] Record exact package versions in `artifacts/setup/requirements-lock.txt`.

### 0.3 VMEC Path Decision
- [ ] Primary: Dockerized VMEC execution path.
- [ ] Fallback: native VMEC only if Docker path is blocked.
- [ ] Save command templates in `artifacts/setup/vmec_commands.md`.

## Phase 1: Data And Surrogate Baseline (Days 1-2)

### 1.1 Dataset Ingestion
- [ ] Load Hugging Face dataset `proxima-fusion/constellaration`.
- [ ] Normalize boundary representation (`r_cos`, `z_sin`, periodicity fields).
- [ ] Split train/validation/test with seed lock.
- [ ] Persist split indices to `artifacts/data/splits.json`.

### 1.2 Feasibility + Objective Surrogates
- [ ] Train feasibility classifier (valid VMEC proxy).
- [ ] Train regressors for P1/P2/P3 metrics used in repo problems.
- [ ] Calibrate classifier probability threshold on validation set.
- [ ] Save metrics and calibration plots to `artifacts/surrogate/`.

### 1.3 Baseline Reproduction
- [ ] Run existing ALM/optimization examples once for smoke baseline.
- [ ] Compare surrogate ranking with known feasible points.
- [ ] Freeze baseline report in `reports/baseline_reproduction.md`.

## Phase 2: Candidate Generation Engine (Days 3-5)

### 2.1 Start With Existing Repo Generative Path
- [ ] Reuse bootstrap/generative utilities first (PCA + RF + GMM + MCMC flow).
- [ ] Reuse notebook logic from `constellaration/notebooks/generative_model_simple_QI.ipynb`.
- [ ] Ensure generated candidates satisfy structural boundary schema before scoring.

### 2.2 Optional Extensions (Only If Needed)
- [ ] Add LightGBM regressors if they improve validation error materially.
- [ ] Add tiny latent model only if MCMC/GMM proposal quality plateaus.
- [ ] Keep extensions modular and removable.

### 2.3 Candidate Filtering Policy
- [ ] Hard drop candidates violating deterministic geometric sanity checks.
- [ ] Rank by surrogate objective plus constraint margin.
- [ ] Enforce diversity in Fourier space before VMEC queueing.

## Phase 3: Hybrid Search Loop (Weeks 2-3)

### 3.1 Loop Contract (One Outer Iteration)
- [ ] Generate candidate pool from current proposal engine.
- [ ] Score all candidates with surrogate stack.
- [ ] Keep top-N by score and top-M by diversity.
- [ ] Submit fixed VMEC batch budget.
- [ ] Ingest VMEC outcomes into memory dataset.
- [ ] Retrain/calibrate surrogates.
- [ ] Save iteration report under `artifacts/loop/<timestamp>/`.

### 3.2 VMEC Budget Guardrails
- [ ] Global budget target: 60-80 VMEC evaluations total.
- [ ] Per-iteration budget target: 20-25 VMEC evaluations.
- [ ] Abort criteria: high repeated failure mode without new feasible points.
- [ ] Resume criteria: updated proposal distribution or threshold tuning completed.

### 3.3 Reliability And Recovery
- [ ] Log each VMEC job inputs, runtime, exit status, parsed metrics.
- [ ] Auto-retry only transient infrastructure failures.
- [ ] Never retry deterministic physics-invalid candidates unchanged.

## Phase 4: Multi-Objective Pareto Production (Week 4)

### 4.1 P3 Conditioning Strategy
- [ ] Sweep target compactness regimes (low AR to higher AR).
- [ ] For each regime, generate and filter candidates with same policy.
- [ ] Validate only final shortlist with VMEC.

### 4.2 Pareto Artifacts
- [ ] Produce nondominated set JSON with full boundary payloads.
- [ ] Produce objective scatter and constraint pass-rate plots.
- [ ] Produce final submission-ready file set under `artifacts/final_pareto/`.

## Daily Execution Checklist
- [ ] Start-of-day: review previous night VMEC logs and failures.
- [ ] Start-of-day: update feasible archive and retrain surrogates.
- [ ] Mid-day: run proposal generation + surrogate filtering.
- [ ] End-of-day: launch bounded VMEC overnight batch.
- [ ] End-of-day: snapshot progress report and next-day actions.

## Quality Gates (Must Pass)
- [ ] All candidate files parse as valid boundary schema.
- [ ] Surrogate validation metrics are tracked and non-regressing.
- [ ] VMEC batch logs are complete and reproducible.
- [ ] P1/P2/P3 evaluations use repo metric names exactly.
- [ ] Final outputs include constraint audit tables per problem.

## Risk Register And Mitigations
- [ ] Risk: macOS VMEC native instability.
- [ ] Mitigation: prefer Docker execution path and isolate VMEC runtime.

- [ ] Risk: surrogate overfitting and false positives.
- [ ] Mitigation: maintain held-out calibration set and confidence thresholding.

- [ ] Risk: wasted VMEC budget on near-duplicate candidates.
- [ ] Mitigation: enforce novelty filtering in coefficient space.

- [ ] Risk: drift from SSOT constraints.
- [ ] Mitigation: map all scoring to `problems.py` names only.

## Deliverables Checklist
- [ ] `reports/baseline_reproduction.md`
- [ ] `artifacts/data/splits.json`
- [ ] `artifacts/surrogate/*`
- [ ] `artifacts/loop/<timestamp>/*`
- [ ] `artifacts/final_pareto/pareto_set.json`
- [ ] `artifacts/final_pareto/constraint_audit.csv`
- [ ] `artifacts/final_pareto/summary.md`

## Suggested 4-Week Calendar
- [ ] Week 1: setup, baseline reproduction, surrogate calibration.
- [ ] Week 2: loop iterations 1-2 with fixed VMEC budget.
- [ ] Week 3: loop iterations 3-4 and constraint tightening.
- [ ] Week 4: P3 Pareto sweep and final artifact freeze.

## Definition Of Done
- [ ] Reproducible runs from clean environment notes.
- [ ] Bounded VMEC spend with full traceability.
- [ ] Feasible candidates for each benchmark problem.
- [ ] Final Pareto artifact bundle and run summary committed.
