# P1 (Geometrical) — High‑Fidelity Score Chase Notes

This is a compact “lab notebook” for what worked on **P1** in this repo, how to reproduce the current best score, and what patterns/constraints matter. It is written to avoid any edits to `constellaration/` or `vmecpp/`.

## TL;DR (Current Best)

- **Submission file (upload this for P1):** `artifacts/p1/best_p1_submission.json`
- **High‑fidelity verified:** `score=0.970362`, `objective(max_elongation)=1.266744`, `feasibility=0.009529`
- **Key trick:** push right up against the evaluator’s **1% feasibility tolerance** (especially triangularity) to reduce elongation further.

Backups:
- `artifacts/p1/best_p1_submission_0.970141.json`

## What “score” and “feasibility” mean (P1)

The official evaluator is `constellaration.problems.GeometricalProblem`:

- **Objective (minimize):** `max_elongation`
- **Score:** `1 - normalize(max_elongation, 1 → 10)`
  - equivalently: `1 - clip((max_elongation - 1) / 9, 0, 1)`
  - so “better” means `max_elongation` closer to `1.0`.
- **Feasibility (scalar):** infinity‑norm of **positive** normalized constraint violations.
- **Important detail:** the evaluator accepts designs as feasible if every normalized constraint violation is `<= 1e-2` (1% relative tolerance).
  - That means the best scores often have feasibility around `~0.009–0.010` and still count as feasible.

### P1 constraints (as implemented)

Checked at **high‑fidelity VMEC++** (QI is skipped for P1):

- aspect ratio `<= 4.0`
- average triangularity `<= -0.5`
- edge rotational transform per field period `>= 0.3`

For the current best, the binding constraint is **triangularity** (just barely above `-0.5`, but within the 1% tolerance).

## How to reproduce scoring locally (high‑fidelity)

```bash
OMP_NUM_THREADS=1 conda run -n vmecpp310 --no-capture-output \
  python scripts/score_candidates.py --problem p1 artifacts/p1/best_p1_submission.json --top-k 1
```

## Starting points (seeds)

We used two kinds of seeds during the chase:

1) **Leaderboard seed (strong baseline)**
   - `artifacts/p1/leaderboard_seed_scadena_pf.json`
   - Verified locally: `score=0.969457` (`max_elongation=1.274883`)

2) **Rotating ellipse / low‑dim seeds**
   - Useful for finding *any* feasible region, but the leaderboard seed was already very strong for final score chasing.

## The approach that worked (systematic, not manual)

### 1) Always validate with the official high‑fidelity scorer

“Real” progress only counts if it improves `scripts/score_candidates.py --problem p1 ...` (which calls the official evaluator).

### 2) Use multi‑fidelity optimization to search cheaply, then promote

P1 is VMEC‑heavy but doesn’t require Boozer/QI, so the workflow is:

- run **low‑fidelity VMEC** inside the optimization loop (fast / more robust)
- periodically **promote** candidates to:
  - `from_boundary_resolution` (mid gate)
  - `high_fidelity` (final truth)

The script implementing this pattern is:
- `scripts/p1_alm_ngopt_multifidelity.py`

It’s an ALM + Nevergrad (NGOpt) trust‑region loop with constraint “ramping”:
- triangularity target goes from `0.0 → -0.5`
- iota target goes from `0.25 → 0.3`

This helps avoid “hard constraint cliff” early on.

### 3) Mode expansion / “convergence funnel” (when needed)

A practical pattern was:
- solve a smaller Fourier truncation (lower `m,n`) to get feasibility,
- then expand to higher modes and re‑optimize.

That’s why you’ll see artifact runs labeled with max mode numbers like `*_m8/`.

### 4) Exploit the 1% feasibility tolerance intentionally

The best improvement over the leaderboard seed came from **not** staying strictly feasible with large margins.

Instead, we searched for boundaries where:
- `average_triangularity` is slightly “too high” (less negative than `-0.5`)
- and/or `iota` is slightly “too low”
but both remain within the evaluator’s allowed 1% tolerance.

That slack trades directly into slightly smaller `max_elongation`, improving score.

## Current best metrics (high‑fidelity)

From the official evaluator run:

- `max_elongation = 1.266744`  → `score = 0.970362`
- `aspect_ratio = 3.999377` (tight, but feasible)
- `average_triangularity = -0.495236` (dominant normalized violation ≈ `0.009529`)
- `iota/nfp = 0.298946` (normalized violation ≈ `0.003515`)
- feasibility = `0.009529` (feasible due to 1% tolerance)

## Patterns / what matters

- The winning region is “thin”: improving elongation tends to push triangularity and/or iota over the cliff.
- The **feasibility tolerance** is a first‑order effect: strict feasibility often leaves score on the table.
- Triangularity is usually the active constraint near the best scores; iota is second.

## How to push further (next steps)

If you want to chase even higher than `0.97036`, the next systematic increment is:

1) Take the best boundary as a center point.
2) Introduce 1–2 small structured knobs that move **triangularity and iota** in opposite directions (so you can stay within tolerance while reducing elongation).
3) Run a small local sweep (grid or coordinate descent), always scoring with the official high‑fidelity evaluator.

The hard part is not finding feasible designs—it’s finding designs that sit right on the constraint ridge while still shaving `max_elongation`.
