# P2 (Simple-to-build QI) — High‑Fidelity Score Chase Notes

This is a running “lab notebook” of what worked, what didn’t, and how to reproduce the current best **P2** submission found in this repo **without modifying** `constellaration/` or `vmecpp/`.

## TL;DR (Current Best)

- **Submission file (upload this for P2):** `artifacts/p2/best_p2_submission.json`
- **High‑fidelity verified:** `score=0.511862`, `objective(L∇B)=10.237250`, `feasibility=0.009941`
- **How it was produced:** a small **systematic** sweep of two “knobs” around the best leaderboard seed:
  - `sz` = scale axisymmetric `z_sin` terms (`n=0`, all `m>=1`) by ~`0.979`
  - `s4` = scale all `|n|=4` Fourier columns (both `r_cos` and `z_sin`, all `m`) by ~`1.17`

Backups:
- `artifacts/p2/best_p2_submission_0.511443.json`
- `artifacts/p2/best_p2_submission_0.500967.json`

## What “score” and “feasibility” mean (P2)

The official evaluator is `constellaration.problems.SimpleToBuildQIStellarator`:

- **Objective:** maximize `minimum_normalized_magnetic_gradient_scale_length` (call it `L∇B`).
- **Score:** `clip(L∇B / 20, 0, 1)` *but only if the design is feasible* (otherwise score is forced to `0.0`).
- **Feasibility (scalar):** the infinity‑norm of **positive** normalized constraint violations.
- **Important detail:** feasibility is *not* “must be 0”; the evaluator treats a design as feasible if every normalized constraint violation is `<= 1e-2` (1% relative tolerance).
  - So you’ll often see feasible designs with feasibility around `0.005–0.010`.

### P2 constraints (as implemented)

Constraints are checked on high‑fidelity VMEC++ (+ Boozer + QI):

- aspect ratio `<= 10.0`
- edge rotational transform per field period `>= 0.25`
- `log10(qi) <= -4.0` (with 1% relative tolerance in normalized space)
- edge magnetic mirror ratio `<= 0.2`
- max elongation `<= 5.0`

In practice, near the top scores the limiting constraint is almost always **QI**.

## How to reproduce scoring locally (high‑fidelity)

Use the official scorer script in this repo (it calls the constellaration evaluator, which runs VMEC++/QI):

```bash
OMP_NUM_THREADS=1 conda run -n vmecpp310 --no-capture-output \
  python scripts/score_candidates.py --problem p2 artifacts/p2/best_p2_submission.json --top-k 1
```

Notes:
- I frequently wrapped runs with `timeout 600` to avoid occasional long/hung evaluations.
- You may see `RuntimeWarning` from `constellaration/mhd/geometry_utils.py` (`fsolve` progress). It usually doesn’t affect correctness, but it can slow a run.

## Where the seed came from

The best public leaderboard seed we used was extracted into:

- `artifacts/p2/dmcxe_seed.json`

That seed matches the leaderboard’s high‑scoring “DMCXE” design (as reproduced by the local evaluator) and served as the starting point for controlled perturbations.

## The two “knobs” (structured perturbations)

We avoided random coefficient noise and instead applied **small, structured, physically‑meaningful scalings**.

### Knob 1 — `sz`: axisymmetric Z scaling

Definition:
- Multiply `z_sin[m, n=0]` by `sz` for all `m>=1`.
- This changes the axisymmetric shaping (roughly “squash/stretch” vertically) without altering non-axisymmetric content.

Observed effect:
- `sz < 1` tends to **increase** `L∇B` (better objective),
- but quickly degrades **QI** and increases elongation once pushed too far.

### Knob 2 — `s4`: `|n|=4` mode scaling

Definition:
- Multiply all Fourier columns with `|n|=4` (both `n=+4` and `n=-4`) by `s4`,
- applied to both `r_cos` and `z_sin`, across all poloidal indices `m`.

Observed effect:
- `s4 > 1` can improve objective, but the feasible window is narrow because QI becomes the limiter.

## Timeline of discoveries (what we tried, what worked)

### 1) Confirm baseline and understand the real limiter

We used the official high‑fidelity scorer (`scripts/score_candidates.py`) exclusively for “real” validation.

Key observation at high scores:
- aspect ratio, mirror ratio, and iota were comfortably feasible,
- max elongation was close but usually not the limiting constraint,
- **QI** was the critical limiter (log10(qi) hovering just above/below `-4` with the 1% tolerance).

### 2) Axisymmetric `z_sin` scaling alone can beat 0.50 (but not 0.51)

From the seed, decreasing `sz` slightly gave feasible improvements like:
- `sz=0.985` → `score≈0.5045` (feasible, but QI close to the edge).

Further decreasing `sz` raised `L∇B` but caused infeasibility (QI + elongation blow up).

### 3) Combine `sz` + `s4` to reach ≥ 0.51

A systematic grid sweep revealed a narrow feasible ridge where both knobs cooperate:

- `sz=0.980`, `s4=1.18` → `score=0.511443` (feasible)

### 4) Refine locally to get the current best

We tightened the search around the best region and found:

- `sz=0.9790`, `s4=1.17` → **`score=0.511862`** (feasible)

This is the current `artifacts/p2/best_p2_submission.json`.

## Pattern summary (what seems to matter)

- **Feasible region is tiny** near the top: improving `L∇B` usually worsens QI; you need a second knob to steer QI back.
- The evaluator’s 1% feasibility tolerance is material: “best” designs often run with feasibility ≈ `0.008–0.010` and still count.
- “Bulk scaling by n‑bands” is surprisingly powerful: it changes geometry/field properties in a more coherent way than random coefficient noise.

## How I would push further (next steps)

Right now, we’re stuck against **QI**. To go beyond ~0.512 without random luck:

1) Add 1–2 extra *control knobs* specifically to tune QI without killing `L∇B`, for example:
   - split `|n|=4` scaling into separate `s4_r` (for `r_cos`) and `s4_z` (for `z_sin`)
   - a small `|n|=3` knob (or selected m‑bands) purely to recover QI
2) Run a constrained local search (still KISS):
   - coarse grid → identify feasible ridge
   - fine grid / coordinate descent near the ridge
   - always validate with high‑fidelity scorer
3) Use `scripts/p2_alm_ngopt_multifidelity.py` *starting from* the current best boundary so the optimizer explores more degrees of freedom than 2 scalars, while still promoting candidates to high‑fidelity regularly.

The key is to keep the loop systematic: propose small structured deltas → evaluate → learn which constraint is active → introduce the minimum new knob needed to steer back.
