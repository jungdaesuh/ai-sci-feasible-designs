# Feasible Design Factory (P1/P2/P3) — System Design

This document describes a **factory** that can **mass-produce feasible** stellarator boundary designs (and keep them **diverse**) in a large design space, while using **high‑fidelity VMEC++** efficiently.

It is written to be **implementation-ready** in this repo:
- Treat `constellaration/` + `vmecpp/` as the **oracle** (do not modify).
- Use **persistent storage** (SQLite) + **immutable artifacts** (JSON files) so progress is never lost.
- Run many **single‑threaded** VMEC++ evaluations in parallel (throughput).
- Add ML **only where it reduces oracle calls** (optional but recommended at scale).

---

## 0) Goals and non-goals

### Goals
- Produce **many feasible designs** (P1/P2 single boundary, P3 Pareto set).
- Maintain **diversity** on purpose (not accidental).
- Improve **scores** without “getting stuck” in a tiny region of design space.
- Make runs **restartable** and **auditable** (full provenance).

### Non-goals
- Prove global optimality.
- Replace VMEC++ / change physics.
- Rely on brittle one-off scripts without logging.

---

## 1) Core abstraction: the oracle + normalized violations

Everything should revolve around one contract:

**Given boundary parameters** → run `constellaration` evaluation → get:
- objectives (what you want to improve)
- constraint violations (normalized like the benchmark)
- feasibility (∞-norm of positive normalized violations)

Feasibility in `constellaration` is:
- `is_feasible := all(normalized_violations <= 1e-2)`
- `feasibility := max(max(normalized_violations, 0))`

This “normalized constraint space” is the **control space** for the factory.

**Important detail:** for each constraint, `constellaration` normalizes by the absolute
target value. Example: for P3 `vacuum_well >= 0`, the target used for normalization
is `max(1e-1, vacuum_well_lower_bound)` → `0.1`. That’s why tiny negative vacuum
well values (e.g. `-2e-4`) can still be feasible within the 1% tolerance.

---

## 2) Factory architecture (modules)

### 2.1 Single Source of Truth (SSOT): RunSpec
Define a spec for each problem:
- parameterization: Fourier coefficient shape (mpol/ntor), symmetry flags, `n_field_periods`
- objective(s) and constraint thresholds
- multi-fidelity settings (low/med/high)
- budgets: max VMEC calls, wall-time, or compute-hours
- diversity descriptors (bins) and archive policy

**Rule:** every run produces a `meta.json` that includes the RunSpec + git SHAs.

### 2.2 Persistent store (SQLite) + immutable artifacts
Store:
- all candidates (params + provenance)
- all evaluations (metrics + constraint vector + oracle settings)
- archive state (Pareto set / top-K)
- batch logs (what was proposed, why)

Immutable files on disk per candidate:
- `candidates/<design_hash>.json` (boundary)
- `candidates/<design_hash>_meta.json` (parents + knobs + batch id)
- `eval/<design_hash>.json` (metrics + violations + feasibility + wall time + error)

This makes the system:
- restartable
- debuggable
- reproducible

### 2.3 Proposal engine (candidate generators)
Generate candidates via **move families** (structured knobs), not arbitrary noise.

Move families should be cheap to generate and easy to reason about:
- `blend(A,B,t)` convex interpolation between known-good designs
- `scale_group(|n|=k, factor)` multiply a toroidal-mode group by a factor
- `scale_low_modes(m<=M, |n|<=N)` gentle low-mode changes
- `targeted_axis_mode` / `targeted_nfp` only if supported by representation
- `local_jitter` *only* in subspaces known to be safe (rare)

Each candidate must record:
- parent hashes
- knob values
- move family name
- deterministic seed

### 2.4 Screening pipeline (multi-fidelity gate)
Use cheap filters to avoid wasting high-fidelity VMEC calls.

Recommended stages:
- Stage 0 (free): JSON/schema/symmetry checks; coefficient bounds; self-intersection heuristics (optional)
- Stage 1 (cheap): low-fidelity equilibrium settings (or “skip expensive metrics” where allowed)
- Stage 2 (expensive): full high-fidelity evaluation (VMEC++ + QI/metrics)

**Factory invariant:** high-fidelity is only for candidates that are “promising” under cheap screens.

### 2.5 Worker farm (parallel oracle calls)
Workers:
- pull one `pending` candidate
- run the oracle for the correct problem + fidelity
- persist results + artifacts

Parallelism strategy:
- run **many** single-threaded VMEC processes
- avoid oversubscription by forcing:
  - `OMP_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `VECLIB_MAXIMUM_THREADS=1`

On a laptop, this often beats fewer multi-threaded runs.

### 2.6 Governor (controller / intervention layer)
The governor is how you scale beyond brute force:
- monitors yield: feasible rate, failure modes, best score, Pareto hypervolume
- identifies the **bottleneck constraint** in the current region
- allocates budget across move families adaptively (bandit-style)
- proposes the next batches automatically

The governor should answer:
- “What should we try next?”
- “Where are we failing?”
- “Are we improving or plateauing?”

### 2.7 Archive / diversity manager
Maintain curated outputs instead of “the last best thing”.

For P1/P2 (single objective):
- keep top‑K feasible by score
- plus a “diversity reserve” (avoid one-cluster collapse)

For P3 (multi objective):
- maintain Pareto archive and compute hypervolume
- prefer candidates with high **expected hypervolume contribution**

---

## 3) Production strategy (no ML required)

### 3.1 Never start from random
Build a seed bank:
- leaderboard submissions
- best-known local submissions
- vetted dataset points

Random sampling is not a strategy; it’s a time sink.

### 3.2 Repair-first local search (systematic feasibility)
For a candidate with normalized violations `v`:
1) Find worst constraint index `argmax(max(v,0))`
2) Choose a move family linked to that constraint
3) Do a tiny 1D sweep (line search) on the knob (e.g., 6–20 points)
4) Re-evaluate the best 1–3 at high fidelity

This turns “feasibility” from luck into a deterministic procedure.

Practical mapping (empirical, not guaranteed):
- mirror too high → adjust `|n|` groups (often `|n|=3`) or blend toward mirror-safe parent
- QI marginal → reduce certain mode energy (or blend toward QI-safe parent); treat as the knife-edge constraint
- flux compression high → soften shaping (reduce higher-|n| magnitude)
- vacuum well negative → small nudges, often correlated with other constraints; aim for margin
- iota low → targeted low-mode modifications or blend toward iota-safe parent

### 3.3 Diversify by design
To mass-produce diverse feasible designs without ML:
- Use MAP-Elites style bins in **metrics space**:
  - example descriptors for QI-like boundaries: `(aspect_ratio bin, iota bin, mirror bin, flux bin)`
  - keep one elite per bin (best objective/score inside that bin)
- Run multiple seeds in parallel (seed-level diversity).
- Mix move families (blend + scale + mild jitter).

Output becomes a catalog, not a single local optimum.

---

## 4) Scaling with ML (optional, high leverage)

ML is useful when it **reduces expensive oracle calls**. The simplest useful ML stack:

### 4.1 Feasibility classifier
Train `p(feasible | x)` on your accumulated DB.
- Inputs:
  - raw coefficients (optionally standardized per mode)
  - or low‑dim embeddings (PCA) + a few handcrafted features (Fourier energy by mode group)
- Outputs:
  - probability of feasibility (or probability of “near feasible”)

Use it as a **gate**:
- only high-fidelity evaluate candidates with `p(feasible)` above a threshold
- sample a small fraction below threshold for exploration + calibration

### 4.2 Constraint regressors (predict margins)
Predict each normalized violation component.
Use this to propose repairs:
- “which knob would most reduce the worst violation?”
- “which parent blend moves us toward the feasible manifold?”

### 4.3 Active learning loop (self-improving)
Repeat:
1) propose many candidates cheaply (thousands)
2) rank by (predicted feasibility × expected improvement × novelty)
3) evaluate top‑N with VMEC++
4) retrain models on new labels

This is how you scale to “vast spaces”.

### 4.4 Generative model (diffusion/flow/VAE) for diversity
When you have a large feasible set:
- train a conditional generator to sample diverse designs satisfying easy-to-specify targets
  - conditions might be `aspect_ratio`, `iota`, `nfp`, and/or coarse bins
- keep the oracle as final validator

Generative models help you sample **diverse** candidates, not necessarily the best.

---

## 5) Per-problem factory modes (P1 vs P2 vs P3)

### P1 (geometrical): single boundary, maximize score ∈ [0,1]
Production mode:
- seed bank → repair feasibility → objective refinement (`max_elongation` down)
- keep margin: target feasibility ≤ 0.005 to avoid noise flips

### P2 (simple_to_build): single boundary, maximize score ∈ [0,1]
Production mode:
- hardest part is being simultaneously QI-good + stable + buildable proxy-good
- strategy:
  - maintain multiple feasible clusters (diversity) because improvements are local
  - systematic mode-group sweeps around best seeds (small % steps)
  - use regressors/classifier early if possible (big reduction in wasted VMEC calls)

### P3 (mhd_stable): submit a set; score = hypervolume
Production mode:
- your “product” is a **Pareto set**
- factory must:
  - preserve feasible archive
  - propose candidates that expand the front (HV contribution)
  - fill gaps in objective space (diversity in Pareto sense)

Practical: once you find a new feasible compact point, it can unlock a large HV jump.

---

## 6) Concrete run layout and operations (this repo)

### 6.1 Directory conventions
Recommended convention:
- `artifacts/<problem>/<RUN_ID>/`
  - `meta.json`
  - `candidates/`
  - `eval/`
  - `batches/`
  - `submissions/`
- `reports/<problem>_world_model.sqlite` (or one DB with `problem` column)

### 6.2 Minimal operational commands (P3 implemented today)
We already have a working loop for P3 in `scripts/`:
- `scripts/p3_init_run.py` → create run dir + experiment row
- `scripts/p3_enqueue_submission.py` → seed from known Pareto set
- `scripts/p3_propose.py` → propose batches (structured move families)
- `scripts/p3_worker.py` → high-fidelity worker (VMEC/QI)
- `scripts/p3_dashboard.py` → monitor queue + HV
- `scripts/p3_make_submission.py` → build best submission JSON
- `scripts/p3_governor.py` → decide next proposals based on DB

Operational discipline:
- always log run_id + experiment_id
- never overwrite artifacts; always append

### 6.3 Suggested extension to P1/P2 (same factory)
Mirror the P3 pattern:
- `p1_*` and `p2_*` scripts that share:
  - the same DB schema
  - the same candidate hash + artifact layout
  - the same worker pattern
  - problem-specific evaluation call

The main difference:
- P1/P2 submit a single boundary; P3 submits a list.

### 6.4 Runbooks for P1/P2 (docs-only, mirror P3)
This section is the **concrete operating procedure** to run P1/P2 like P3.

It assumes you create `scripts/p1_*.py` and `scripts/p2_*.py` by copy‑adapting the existing P3
scripts (recommended), with these defaults:
- P1:
  - DB: `reports/p1_world_model.sqlite`
  - run root: `artifacts/p1/`
  - evaluator: `constellaration.problems.GeometricalProblem()`
- P2:
  - DB: `reports/p2_world_model.sqlite`
  - run root: `artifacts/p2/`
  - evaluator: `constellaration.problems.SimpleToBuildQIStellarator()`

#### P1 runbook (geometrical)
1) Activate the working environment:
```bash
conda activate vmecpp310
```

2) Initialize a run directory (template: `scripts/p3_init_run.py`):
```bash
python scripts/p1_init_run.py --tag p1_factory --workers 6
```
Record the printed values:
- `RUN_DIR=...` (under `artifacts/p1/`)
- `EXPERIMENT_ID=...`

3) Seed the queue with known-good boundaries:
```bash
python scripts/p1_enqueue_seed.py --experiment-id $EXPERIMENT_ID --run-dir $RUN_DIR --batch-id 0 artifacts/p1/best_p1_submission.json
```
Recommended: enqueue 10–100 diverse seeds (leaderboard + dataset + your best).

4) Propose a first batch (template: `scripts/p3_propose.py`):
```bash
python scripts/p1_propose.py --experiment-id $EXPERIMENT_ID --run-dir $RUN_DIR --batch-id 1 --seed-base 100000
```
Suggested move families for P1:
- small `scale_group(|n|=k)` sweeps (k ∈ {1,2,3})
- `blend(A,B,t)` between two feasible parents
- optional low-mode-only tweaks (avoid wild high-mode changes)

5) Start N high-fidelity workers in parallel (template: `scripts/p3_worker.py`):
```bash
conda run -n vmecpp310 python scripts/p1_worker.py --experiment-id $EXPERIMENT_ID --run-dir $RUN_DIR --worker-id 1
```
Run multiple workers in separate tmux panes/windows (`worker-id` 1..N). Keep thread caps at 1.

6) Monitor progress (template: `scripts/p3_dashboard.py`):
```bash
python scripts/p1_dashboard.py --experiment-id $EXPERIMENT_ID
```

7) Produce a submission candidate (single boundary JSON object):
```bash
python scripts/p1_make_submission.py --experiment-id $EXPERIMENT_ID --run-dir $RUN_DIR --output $RUN_DIR/submissions/submission_best.json
```

8) Verify with the official scorer before upload:
```bash
python scripts/score_candidates.py --problem p1 $RUN_DIR/submissions/submission_best.json
```

#### P2 runbook (simple_to_build)
Same structure as P1, but with two key differences:
- The feasibility bottleneck is usually **QI**, so keep multiple clusters/seeds alive.
- If QI computation is expensive, use a **two-stage worker**:
  1) `default_high_fidelity_skip_qi()` to reject obvious geometry/mirror/vacuum/flux failures fast
  2) `default_high_fidelity()` only for survivors (final truth for P2 scoring)

Suggested run sequence:
```bash
conda activate vmecpp310
python scripts/p2_init_run.py --tag p2_factory --workers 6

python scripts/p2_enqueue_seed.py --experiment-id $EXPERIMENT_ID --run-dir $RUN_DIR --batch-id 0 artifacts/p2/best_p2_submission.json
python scripts/p2_propose.py --experiment-id $EXPERIMENT_ID --run-dir $RUN_DIR --batch-id 1 --seed-base 200000

conda run -n vmecpp310 python scripts/p2_worker.py --experiment-id $EXPERIMENT_ID --run-dir $RUN_DIR --worker-id 1
python scripts/p2_dashboard.py --experiment-id $EXPERIMENT_ID
python scripts/p2_make_submission.py --experiment-id $EXPERIMENT_ID --run-dir $RUN_DIR --output $RUN_DIR/submissions/submission_best.json
python scripts/score_candidates.py --problem p2 $RUN_DIR/submissions/submission_best.json
```

Upload format reminder:
- P2 submission is a **single boundary JSON object**, not a list.

---

## 7) “Design factory” success metrics

Track these continuously (dashboard):
- throughput: high-fidelity evals/hour
- feasible yield: feasible / evaluated
- average feasibility margin (want slack, not knife-edge)
- best score (P1/P2) or best hypervolume (P3)
- novelty: distance-to-archive distribution
- failure taxonomy: dominant failing constraint over time

When the factory is healthy:
- feasible yield is stable (not collapsing to 0)
- archive keeps growing in diversity
- scores improve monotonically (or plateau with clear reason)

---

## 8) Scaling on a single workstation (practical)

On Apple Silicon laptop-class hardware:
- prefer many independent workers with strict thread caps
- start with 4 workers, measure wall time, then increase to 6–10
- watch memory pressure; VMEC runs can spike RAM depending on settings

If running overnight:
- cap max evaluations / wall time per batch
- keep the governor conservative (avoid exploding queue)

---

## 9) Concrete problem specs (P1/P2/P3) — thresholds, objectives, score math

This section is copied from the source of truth in
`constellaration/src/constellaration/problems.py`. Keeping the exact thresholds
here is useful for designing “repair knobs” that target the **worst** normalized
violation.

### P1: `GeometricalProblem`
- **Objective (minimize):** `max_elongation`.
- **Score:** `1 - normalize(max_elongation, [1, 10])` clipped to `[0, 1]`.
- **Constraints (feasible if each normalized violation ≤ 1e-2):**
  - `aspect_ratio <= 4.0`
  - `average_triangularity <= -0.5`
  - `edge_rotational_transform_over_n_field_periods >= 0.3`

### P2: `SimpleToBuildQIStellarator`
- **Objective (maximize):** `minimum_normalized_magnetic_gradient_scale_length`.
- **Score:** `normalize(L∇B, [0, 20])` clipped to `[0, 1]`.
- **Constraints:**
  - `aspect_ratio <= 10.0`
  - `edge_rotational_transform_over_n_field_periods >= 0.25`
  - `log10(qi) <= -4.0`
  - `edge_magnetic_mirror_ratio <= 0.2`
  - `max_elongation <= 5.0`

### P3: `MHDStableQIStellarator`
- **Objectives (Pareto):**
  - maximize `minimum_normalized_magnetic_gradient_scale_length` (call it `L∇B`)
  - minimize `aspect_ratio` (call it `A`)
- **Score:** hypervolume of feasible points in `(−L∇B, A)` with
  reference point `(1.0, 20.0)`.
- **Constraints:**
  - `edge_rotational_transform_over_n_field_periods >= 0.25`
  - `log10(qi) <= -3.5`
  - `edge_magnetic_mirror_ratio <= 0.25`
  - `flux_compression_in_regions_of_bad_curvature <= 0.9`
  - `vacuum_well >= 0.0` (normalized by `0.1` for feasibility)

---

## 10) Persistence (SQLite) — concrete schema + state machine

We already have a robust “WorldModel” SQLite schema in:
- `ai_scientist/memory/schema.py`

If you build a factory in this repo, reuse this schema instead of inventing a new one.
It already supports:
- restartability (`candidates.status` queue)
- provenance (`experiments.git_sha`, `experiments.constellaration_sha`)
- Pareto archive bookkeeping (`pareto_archive`, `cycle_hv`)
- artifact indexing (`artifacts` table)

### Candidate state machine
Use `candidates.status` as the queue:
1) `PENDING` (proposer enqueued it)
2) `RUNNING` (worker claimed it)
3) `DONE` (worker wrote `metrics` row + artifact paths)
4) `FAILED` (worker wrote an error payload + artifact paths)

Rule: status is monotone; never “rewrite history”. If you want to re-evaluate, enqueue
a new candidate row with a new `seed` (same params are okay if you include a “rerun” tag
in the metadata).

### Recommended SQLite settings (concurrency)
The schema already enables WAL:
```sql
PRAGMA journal_mode=WAL;
```
Operationally:
- each worker uses its own sqlite connection
- commits are small and frequent (one candidate at a time)
- artifacts are written to disk first, then DB rows are inserted pointing to them

---

## 11) The “knob library” — concrete move families that scale

The goal of move families is to make exploration **low dimensional** and **safe**.
This is exactly what some papers refer to as “hand-designed parameterizations”:
you deliberately restrict the search space to a structured subset that’s known to
contain feasible points.

### Representation reminder (Fourier grids)
Boundaries are stored as matrices:
- `r_cos[m][n]`, `z_sin[m][n]`
- `m` indexes poloidal mode number
- `n` indexes toroidal mode number, stored with an offset so that `n=0` is the middle
  column (e.g. for `ntor=4`, valid n are `-4..4` and `n=0` sits at column index `4`).

This makes it cheap to define “groups” like `|n|=1` or “axisymmetric” (`n=0`).

### Move family A: `blend(A, B, t)` (crossover)
Given two parents (same shape arrays):

`child = (1−t)·A + t·B`

Use:
- **repair**: blend toward a parent that satisfies the violated constraint
- **Pareto fill**: blend between a compact and a simple parent to fill gaps on the front

Typical sweep: `t ∈ {−0.05, 0.0, 0.05, …, 1.05}` with clamping disabled (allow mild
extrapolation) but keep steps tiny (±5%).

### Move family B: `scale_n_band(|n|=k, factor)` (structured perturbation)
Multiply all coefficients with `abs(n)==k` by `factor` (for both `r_cos` and `z_sin`).
Example:
- scale `|n|=1` by `1.01`
- scale `|n|=3` by `1.02`

Why it works:
- it changes “3D shaping strength” without destroying the boundary
- it’s only 1–3 parameters, so you can do systematic sweeps

Typical sweep grid (start small): `factor ∈ {0.98, 0.99, 1.00, 1.01, 1.02}`.

### Move family C: `scale_high_m(m>=M, factor)` (smoothing)
Multiply all coefficients with `m>=M` by `factor<1` to reduce high-frequency shaping.
Useful when:
- VMEC convergence gets flaky
- flux compression is too high
- elongation spikes (P1/P2)

### Move family D: “mode transplant” (copy a safe group)
Copy one group from B into A:
- axisymmetric spine (`n=0` column)
- a specific `|n|=k` band
- or “high m only”

This is a powerful *repair tool* when you know which parent is safe for a constraint.

### Move family E: “repair line search” (feasibility-first)
For a near-feasible candidate:
1) identify the worst constraint (largest normalized violation component)
2) choose one knob that correlates with that constraint
3) run a 1D sweep on that knob
4) keep top‑K and re-evaluate at full fidelity

This is the minimal, systematic version of “keep nudging until it works”.

---

## 12) Multi-fidelity gating (concrete, using official settings)

Use the official `ConstellarationSettings` presets:
- `default_high_fidelity()` (VMEC++ + Boozer + QI + turbulence metrics)
- `default_high_fidelity_skip_qi()` (VMEC++ only; skips Boozer/QI)

Practical gate for P2/P3:
1) run `default_high_fidelity_skip_qi()` first
2) if geometry/mirror/iota/vacuum/flux are already bad → reject early
3) only for survivors: run `default_high_fidelity()` to compute QI and final score

This is not “cheating”: QI isn’t approximated; it’s simply deferred to avoid spending
QI compute on obvious losers.

---

## 13) Governor (how the factory “learns what works” without an LLM agent)

The governor is a controller that turns evaluation logs into the next batch of proposals.
It can be purely algorithmic.

### 13.1 What the governor tracks (minimum dashboard)
Per batch (or per N evals), compute:
- feasible rate (`is_feasible` fraction)
- median feasibility (want it trending down)
- distribution of “worst violated constraint”
- best score (P1/P2) or best hypervolume (P3)
- per move-family win rates (how often it improves feasibility / score)

### 13.2 Constraint-bottleneck targeting (the key loop)
If 70% of near-feasible candidates fail on mirror, you don’t “optimize better”:
you allocate most of the next budget to **mirror-repair knobs**.

Concrete policy (KISS):
- Let `c*` be the most common worst constraint among the last W candidates.
- Pick 2 move families known to affect `c*` (e.g. `scale_n_band(|n|=3)` + `blend`).
- Generate a small parameter sweep grid around the current best parent(s).
- Evaluate; keep top‑K by feasibility, then top‑K by objective/score.

### 13.3 Budget allocation across move families (bandit-lite)
Keep an exponentially weighted moving average (EWMA) per move family:
- `Δfeasibility` improvement vs parent
- `Δscore` improvement vs parent (only for feasible candidates)
- `ΔHV` contribution when candidate enters the Pareto archive (P3)

Allocate next batch sizes proportional to the EWMA, with:
- minimum exploration floor (e.g. 10% split across “non-winners”)
- maximum cap per family (avoid mode collapse)

No ML is required for this; it’s just bookkeeping + adaptive scheduling.

### 13.4 LLM agents (optional)
LLMs can be useful *assistants*, but they are not the bottleneck for this challenge.
In practice, the bottleneck is **high‑fidelity physics evaluation** (VMEC++ + derived metrics),
and the highest leverage comes from:
- good seeds
- structured knobs
- persistent logging + an adaptive governor

If you use `ai_scientist/`, treat its LLM pieces as “orchestration glue”, not as a physics
replacement:
- Useful: summarize failures, propose new move families, write/adjust scripts, explain results.
- Not useful: reliably “guess” feasible designs without the oracle; replacing VMEC/QI evaluation.

---

## 14) ML augmentation (optional, high leverage at scale) — ideas from `2511.20445v1.md`

This repo can already win leaderboards with “repair-first” low-dimensional knobs.
ML becomes valuable when you want to:
- increase feasible yield (fewer wasted VMEC calls)
- increase diversity (sample far from current clusters without falling off the manifold)

Useful ideas from `2511.20445v1.md` (diffusion models for stellarator stage‑I design):
- **Learn a feasibility region** with a classifier and sample it (the paper notes
  classifier + MCMC as a viable constraint-satisfaction approach).
- **Conditional generation**: train a generator to sample designs given conditions
  (they use `(ι, A, nfp, N)` for quasisymmetry; for ConStellaration you’d condition on
  a small set like `(A, ι, mirror, L∇B)` or coarse bins).
- **Fourier-feature scaling / canonicalization**: high-order modes matter less;
  treat “Fourier energy by band” as a core feature for ML and for safe perturbations.

Concrete “starter ML” that pays off quickly:
1) Train `p(feasible | x)` on your DB (features: per-band energies + PCA of coefficients).
2) Train regressors for each normalized violation component.
3) Use the models only as a **ranker** for which candidates to evaluate next.

Keep the oracle as the final validator.

---

## 15) Output formats (avoid common submission/visualization confusion)

Different consumers expect different JSON shapes:

- **P1/P2 submission to HF benchmark**: a single boundary JSON object with keys
  `r_cos`, `z_sin`, `n_field_periods`, `is_stellarator_symmetric` (and optional null
  `r_sin`/`z_cos`).
- **P3 submission to HF benchmark**: a JSON list of JSON-encoded boundary strings
  (the benchmark expects a *set* of candidates).
- **HF boundary-explorer visualization**: expects a single boundary object; it cannot
  load a P3 multi-boundary list. Extract one boundary first (e.g. via
  `scripts/p3_extract_boundary.py`).

---

## 16) When you hit a score plateau

A plateau usually means one of:
- you’re stuck on the boundary of feasibility (no margin)
- your move families are too weak (can’t reach new basins)
- you need a better seed bank
- you need ML gating (wasting too many high-fidelity calls)

The “factory” response:
- increase diversity pressure (new bins, new seed families)
- add targeted repair operators for the dominant constraint
- add a feasibility classifier if eval budget is the bottleneck

---

## 17) Deliverables the factory should produce

At minimum, every run should output:
- `docs/FINAL_SUBMISSIONS.md`-style record:
  - uploaded files + SHA256
  - leaderboard result filenames + scores
  - exact commands to re-verify locally
- a reproducible artifact directory with immutable eval logs
- a curated archive (best feasible set / Pareto front)
