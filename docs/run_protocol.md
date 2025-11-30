# P1 Run Protocol

This document captures the minimum steps and acceptance checks for every P1 optimization or verification run. Follow it to keep artifacts reproducible and comparable.

## 0. Prerequisites

- Activate the project environment (`.venv` or `hatch shell`).
- Export deterministic threading: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`.
- Ensure VMEC++ binaries are built and on the expected path.
- Vacuum is now the default physics setting (zero pressure/current in `ideal_mhd_parameters.py`). Before changing presets or after pulling this update, disable or clear the forward-model cache (`CONSTELLARATION_DISABLE_CACHE=1` or delete `~/.cache/constellaration/forward_model`) so no run reuses finite-pressure evaluations.

## 1. Select/prepare seeds

1. Source boundary: pick `artifacts/p1/<RUN_ID>/..._boundary.json`, or generate seeds with `scripts/make_tri_seeds_p1.py`.
2. Record provenance (input boundary, random seed, CLI flags) in `meta.json`.

## 2. Repair / optimization loop

Run the random-search baseline from the repo root via `python -m orchestration.run_p1_random_search --tag <TAG> [flags]`. The script adjusts `sys.path` so direct execution (`python orchestration/run_p1_random_search.py ...`) is also supported, but only when the ConStellaration package dependencies are installed (e.g., `pip install -e constellaration[test]`).

1. Launch `scripts/repair_p1_seeds.py` with an isolated `--output-dir artifacts/p1/<RUN_ID>` and `--log-dir constellaration/runs/<RUN_ID>`.
2. Keep per-run budgets in the log for reproducibility (optimizer, total evaluations, wall time).
3. After each iteration, confirm the projected boundary still satisfies |A−4.0|, |δ+0.6|, and |ι/Nfp−0.3| < 5×10⁻³; tighten projection constants if not.
4. Every evaluation stores:
   - Objective, feasibility, per-constraint deltas.
   - Cache hit/miss counts.
   - Wall-clock time.
   - VMEC preset used.

## 3. Verification steps

1. **Low fidelity**: `python -m constellaration.cli <boundary> --problem p1 --fidelity low_fidelity --skip-qi --output reports/p1/<RUN_ID>_low.json`.
   - Target acceptance: `is_feasible == true`. Archive as `low_feasible.json` if it is the current best.
2. **Bridge**: repeat with `--fidelity from_boundary_resolution` and `--skip-qi`. Archive as `<RUN_ID>_bridge.json`.
3. **Strict / high fidelity** (when bridge passes): run `scripts/strict_homotopy_verify.py` or direct CLI with `--fidelity high_fidelity --skip-qi`. Set `CONSTELLARATION_DISABLE_CACHE=1` for these calls. If VMEC++ fails, rerun with `return_outputs_even_if_not_converged=True` to capture the residual trace and store it under `artifacts/p1/<RUN_ID>/`.

## 4. Artifact bookkeeping

- Update `meta.json` with:
  - `git_sha`
  - Time stamps
  - Targets and tolerance values
  - Optimizer settings
  - Verification reports written
  - Physics profile summary (e.g., `pressure_profile: vacuum`, `toroidal_current: vacuum`).
- Maintain stable symlinks or copies:
  - `artifacts/p1/latest_boundary.json`
  - `reports/p1/latest_low.json`, `latest_bridge.json`, `latest_high.json`

## 5. Logging & notes

- Snapshot CLI commands and key metrics to `SESSION_LOG_YYYY-MM-DD.md`.
- Summarize learnings / issues in `legacy-docs/PROGRESS_P1.md` or an equivalent daily log.

## 6. Promotion checklist (before announcing a “best” boundary)

- [ ] Low-fidelity feasible report with `meta.json` cross-linked.
- [ ] Bridge report recorded and either feasible or rationale documented.
- [ ] Strict/high-fidelity attempt logged (converged or failure reason recorded, including residual trace if it failed).
- [ ] Artifacts, logs, reports pushed under run-specific directories.
- [ ] Context updated with new best boundary and next steps.

## ASO Mode (Agent-Supervised Optimization)

Enable with `--aso` flag:
```bash
python -m ai_scientist.runner --aso
```

ASO mode uses real ALM state for supervision decisions. See `ASO_V4_IMPLEMENTATION_GUIDE.md` for details.