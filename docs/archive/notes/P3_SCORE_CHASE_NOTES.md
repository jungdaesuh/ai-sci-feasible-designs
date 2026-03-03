# P3 (mhd_stable) — Score Chase Notes (Jan 2026)

## TL;DR (result)

- **New verified local record (P3 hypervolume): `135.417600`**
- **Baseline reproduced (leaderboard best seed set): `133.500512`**
- **Upload file (P3 / mhd_stable):**
  - `artifacts/p3/20260108T070652_p3_mirror_repair_compact/submissions/submission_best.json`
- **Local verification (official scorer):**
  - `source /opt/anaconda3/etc/profile.d/conda.sh && conda activate vmecpp310`
  - `python scripts/score_candidates.py --problem p3 artifacts/p3/20260108T070652_p3_mirror_repair_compact/submissions/submission_best.json`

## Problem definition (what “feasible” means)

Constraints (see `constellaration/src/constellaration/problems.py`):
- `iota_edge >= 0.25`
- `log10(qi) <= -3.5`
- `mirror_edge <= 0.25`
- `flux_compression <= 0.9`
- `vacuum_well >= 0`

Important nuance:
- `is_feasible()` uses a **relative tolerance** `1e-2` on **normalized constraint violations**.
  - That’s why a candidate with `log10(qi) = -3.4655` can still be feasible:
    - violation `(-3.4655 - -3.5) / 3.5 ≈ 0.0099 <= 0.01`.

Score:
- Hypervolume in 2D objectives: maximize `lgradB`, minimize `aspect_ratio`.
- Hypervolume computed in the transformed space `(-lgradB, aspect_ratio)` with reference point `[1, 20]`.

## Environment (must-have)

Use `conda activate vmecpp310`.
- The base env on this machine is Python 3.12 and `import vmecpp` fails with `libtorch.dylib` `@rpath` missing.
- `vmecpp310` (Python 3.10) imports cleanly and runs VMEC++ high fidelity.

## Reproducible run artifacts (no overwriting)

This run is fully persisted:
- SQLite DB: `reports/p3_world_model.sqlite` (WAL mode).
- Run directory: `artifacts/p3/20260108T070652_p3_mirror_repair_compact/`
  - `meta.json` (run metadata)
  - `candidates/*.json` + `candidates/*_meta.json` (immutable inputs + provenance)
  - `eval/*.json` (immutable high‑fidelity evaluation results)
  - `governor/*.json` (decision logs)
  - `submissions/submission_best.json` (best 16-point set)

tmux:
- Session: `p3_1` (workers + dashboard panes)
- `tmux attach -t p3_1`

## What we actually did (high-level)

1. **Seeded with leaderboard designs**
   - Extracted scadena “compact” seed (A) + scadena feasible-ish seed (B) from `/tmp/scadena_mhd_stable_submission.json`.
   - Enqueued NianRan1’s 16-point feasible seed set from `/tmp/nianran_mhd_stable_submission.json` to reproduce baseline HV.

2. **High-fidelity loop (systematic, restartable)**
   - Used a SQLite-backed queue + 6 high‑fidelity workers.
   - Each candidate was evaluated using `ConstellarationSettings.default_high_fidelity()` (VMEC++ + Boozer + QI + turbulence).
   - All results were written to DB + immutable JSON artifacts.

3. **Targeted repair strategy (compact point)**
   - Diagnosed the compact scadena boundary: **mirror ratio** was the dominant violation (`~0.2759` vs `0.25`).
   - Used structured perturbations (small, interpretable knobs) rather than random noise:
     - `axisym_z` scaling (affects axisymmetric Z coefficients, tends to reduce mirror)
     - targeted `|n|` group scaling (especially `|n|=3`)
     - limited blend sweeps between “compact” and “feasible” parents

4. **Submission assembly**
   - Built a best‑HV 16-design submission with `scripts/p3_make_submission.py`.
   - Verified the HV via the official scorer.

## Key discoveries (the “why it worked”)

### 1) Mirror is the compact bottleneck
For the compact scadena parent A:
- `mirror ≈ 0.2759` → normalized violation ≈ `(0.2759 - 0.25)/0.25 ≈ 0.1036`
- other constraints were close to fine (flux/vacuum/iota OK, `log10(qi)` borderline)

So the fastest path was:
- **repair mirror** without breaking the others.

### 2) Blending created a near-feasible “bridge” candidate
Blend between scadena A and B:
- Found a good bridge at **`t = 0.86`**:
  - candidate `fd65cb7243...` (meta: `move_family=blend`, `t=0.86`)
  - mirror became essentially within spec (just under 0.25)
  - remaining blocker became **QI** (`log10(qi)` slightly too high)

### 3) Small `|n|=3` scaling fixed QI without breaking mirror
From that bridge candidate (`fd65cb...`), scaling **all `|n|=3` columns by `1.04`**:
- produced candidate `a9e01ecab2...` (meta: `move_family=scale_groups`, `abs_n_3=1.04`)
- became feasible with:
  - `aspect=7.624698`
  - `lgradB=5.191823`
  - `mirror=0.2484`
  - `log10(qi)=-3.4655` (still within tolerance)
- This new compact feasible point expands the Pareto front and **boosts HV to 135.4176**.

## Concrete “winning” chain (the provenance)

- Parent A: scadena compact extracted to `/tmp/p3_parent_a.json`
- Parent B: scadena feasible-ish extracted to `/tmp/p3_parent_b.json`
- Bridge: `fd65cb72439e...` = `blend(t=0.86)` between A and B
  - meta file: `artifacts/p3/20260108T070652_p3_mirror_repair_compact/candidates/fd65cb72439e46888abc3a2bf7e0de997288ec494b61cc30a8c4767aa79c640d_meta.json`
- Final compact feasible: `a9e01ecab23a...` = `scale(|n|=3, factor=1.04)` applied to the bridge
  - meta file: `artifacts/p3/20260108T070652_p3_mirror_repair_compact/candidates/a9e01ecab23a0a5d99504adc4546ce5e6db259b7fcf60ca24382e5df1e7abca0_meta.json`

## Tools/scripts added for P3 (what each does)

- `scripts/p3_init_run.py`: creates a run dir + an `experiments` row in SQLite.
- `scripts/p3_propose.py`: enqueues structured candidates (`blend`, `scale_groups`) into SQLite.
- `scripts/p3_enqueue_submission.py`: enqueues an existing P3 submission list (seed set) into SQLite.
- `scripts/p3_worker.py`: claims `pending` candidates and runs **high‑fidelity** physics evaluation; writes eval JSON + DB rows.
- `scripts/p3_dashboard.py`: read‑only monitor for queue + feasibility + HV.
- `scripts/p3_make_submission.py`: greedy-selects up to 16 feasible points to maximize HV; writes submission JSON list of JSON strings.
- `scripts/p3_governor.py`: “intervention” loop that chooses near‑feasible focus points and proposes the next structured nudges.

## Next ideas (if we push further)

- Iterate around the winning compact point:
  - sweep `abs_n_3` around `1.04` with finer steps (e.g., `1.038..1.045`)
  - add joint sweeps that preserve mirror while improving QI (tiny `axisym_z` + `abs_n_3`)
- Search for an even more compact feasible point (lower aspect) by:
  - exploring blends slightly beyond `t=0.86` and repairing constraints with `abs_n_3` + minimal damping of other mode groups.
