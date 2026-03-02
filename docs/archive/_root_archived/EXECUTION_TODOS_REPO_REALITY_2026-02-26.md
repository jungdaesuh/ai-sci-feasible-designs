# Execution TODOs (Repo Reality, Ordered)

Last updated: 2026-02-26
Workspace: `ai-sci-feasible-designs`
Scope: turn the requested end-to-end workflow into executable, ordered tasks with explicit gap closure.

## Rules

- [ ] Do not modify upstream/original code in `constellaration/`.
- [ ] Do not modify upstream/original code in `vmecpp/`.
- [ ] Keep VMEC usage budgeted and fully logged.
- [ ] Treat problem constraints/metrics in `constellaration/src/constellaration/problems.py` as SSOT.

## Ordered TODOs

### 1. Runtime stack setup + verification capture

Status: `PARTIAL` (infra docs and scripts exist, but no standardized `artifacts/setup/` provenance bundle).

- [x] Reuse existing infra guidance in `docs/INFRASTRUCTURE_SETUP.md`.
- [ ] Create `artifacts/setup/` and write:
- [ ] `artifacts/setup/versions.txt` (python, uv, pip, docker, constellaration SHA, repo SHA).
- [ ] `artifacts/setup/commands.md` (exact setup and run commands used).
- [ ] `artifacts/setup/vmec_path.md` (native vs docker VMEC path selected, with rationale).
- [ ] Add a small script `scripts/capture_setup_provenance.py` to generate the files above deterministically.

### 2. Deterministic dataset splits + manifest

Status: `MISSING` (seed handling exists in multiple places, but no canonical train/val/test split artifact).

- [ ] Add `scripts/make_dataset_splits.py`.
- [ ] Inputs: HF dataset, split seed, optional stratification key.
- [ ] Outputs:
- [ ] `artifacts/data/splits.json` (indices/ids for train/val/test).
- [ ] `artifacts/data/splits_meta.json` (seed, dataset revision, timestamp, command).
- [ ] Determinism check: same seed must produce same manifest.

### 3. Baseline surrogate stack + calibration reports

Status: `PARTIAL` (`ai_scientist/optim/surrogate.py` and `scripts/train_offline.py` exist; report contract is not standardized).

- [x] Reuse `ai_scientist/optim/surrogate.py` for feasibility-first RF bundle baseline.
- [ ] Add `scripts/train_baseline_surrogates.py` as the canonical baseline entrypoint.
- [ ] Required outputs under `artifacts/surrogate/`:
- [ ] `metrics.json` (AUC/PR for feasibility, MAE/RMSE for objectives).
- [ ] `thresholds.json` (calibrated feasibility cutoff and selection policy).
- [ ] `calibration.csv` (probability bin calibration table).
- [ ] `model_registry.json` (paths + hashes of trained artifacts).

### 4. Baseline optimization smoke reproduction

Status: `PARTIAL` (`scripts/p1_alm_ngopt_multifidelity.py` exists; no fixed report target file).

- [x] Reuse `scripts/p1_alm_ngopt_multifidelity.py`.
- [ ] Add wrapper `scripts/run_baseline_smoke.py` that executes one reproducible smoke run and writes:
- [ ] `reports/baseline_reproduction.md`.
- [ ] `artifacts/baseline/run_meta.json`.
- [ ] Include expected/observed feasibility + objective summary.

### 5. Candidate generation path (PCA/RF/GMM/MCMC first)

Status: `PARTIAL` (notebook path exists in `constellaration/notebooks/generative_model_simple_QI.ipynb`; no canonical script in this repo).

- [x] Prioritize existing PCA/GMM/MCMC flow before adding new model complexity.
- [ ] Add script `scripts/generate_candidates_pca_gmm_mcmc.py`.
- [ ] Inputs:
- [ ] split manifest, trained surrogate, budget (`n_generate`, `n_filter`).
- [ ] Outputs:
- [ ] `artifacts/candidates/iter_<k>/generated.jsonl`.
- [ ] `artifacts/candidates/iter_<k>/filtered_top.jsonl`.
- [ ] `artifacts/candidates/iter_<k>/generation_meta.json`.

### 6. VMEC queue policy + logging contract

Status: `PARTIAL` (queue and workers exist: `scripts/p3_init_run.py`, `scripts/p3_propose.py`, `scripts/p3_worker.py`; policy contract not centralized).

- [x] Reuse existing SQLite queue model and artifacts.
- [ ] Add `docs/VMEC_QUEUE_POLICY.md` with fixed contract:
- [ ] per-iteration VMEC budget,
- [ ] total VMEC budget cap,
- [ ] retry policy (`transient` vs `deterministic physics failure`),
- [ ] mandatory metadata fields for every eval.
- [ ] Add `scripts/enforce_vmec_budget.py` guard that blocks enqueue/eval once budget is exhausted.

### 7. Hybrid loop iterations 1-4 (generate -> filter -> VMEC -> retrain)

Status: `MISSING` (pieces exist, but no single orchestrated 4-iteration driver with strict budgeting).

- [ ] Add `scripts/run_hybrid_loop.py`.
- [ ] Loop contract per iteration:
- [ ] generate candidates,
- [ ] surrogate filter,
- [ ] enqueue fixed VMEC batch,
- [ ] collect evals,
- [ ] retrain surrogate,
- [ ] emit iteration report.
- [ ] Outputs:
- [ ] `artifacts/loop/iter_01/*` through `artifacts/loop/iter_04/*`.
- [ ] `artifacts/loop/summary.json`.

### 8. P3 Pareto sweep + final shortlist validation

Status: `PARTIAL` (`scripts/p3_governor.py`, `scripts/p3_worker.py`, `scripts/p3_make_submission.py` exist; final bundle contract not frozen).

- [x] Reuse existing governor/worker pipeline for P3.
- [ ] Add shortlist/sweep wrapper `scripts/run_p3_pareto_sweep.py`.
- [ ] Produce:
- [ ] nondominated shortlist candidates,
- [ ] final VMEC validation batch,
- [ ] reproducible Pareto summary.

### 9. Final deliverable freeze

Status: `MISSING` (final artifact names requested but not enforced as a single freeze step).

- [ ] Add `scripts/freeze_final_deliverables.py`.
- [ ] Required outputs in `artifacts/final_pareto/`:
- [ ] `pareto_set.json`,
- [ ] `constraint_audit.csv`,
- [ ] `summary.md`,
- [ ] `run_provenance.json` (commands, SHAs, seeds, budgets).

## Execution Priority (Do First)

1. [ ] `scripts/capture_setup_provenance.py` + `artifacts/setup/*`
2. [ ] `scripts/make_dataset_splits.py` + `artifacts/data/splits.json`
3. [ ] `scripts/train_baseline_surrogates.py` + `artifacts/surrogate/*`
4. [ ] `scripts/run_baseline_smoke.py` + `reports/baseline_reproduction.md`
5. [ ] `scripts/generate_candidates_pca_gmm_mcmc.py`
6. [ ] `docs/VMEC_QUEUE_POLICY.md` + `scripts/enforce_vmec_budget.py`
7. [ ] `scripts/run_hybrid_loop.py` (iter 1-4)
8. [ ] `scripts/run_p3_pareto_sweep.py`
9. [ ] `scripts/freeze_final_deliverables.py`

## Exit Criteria

- [ ] Every required artifact path exists and is populated.
- [ ] VMEC total budget not exceeded.
- [ ] All problem metrics use SSOT names from `problems.py`.
- [ ] Final bundle is reproducible from captured setup and commands.
