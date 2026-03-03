# P3 (MHD‑stable QI) — Systematic, Persistent Search Loop (tmux + WorldModel)

This is the **plan** for running a systematic, restartable, “learn what works” search for **P3** designs using:

- the repo’s existing **WorldModel SQLite** (`ai_scientist/memory/*`) for persistence, and
- a **tmux** workflow with **6 HF workers** evaluating VMEC++/Boozer/QI.

No commands are executed as part of this document; it’s a runbook + design spec.

---

## Scope / constraints

- **Do not modify** `constellaration/` or `vmecpp/`.
- All runs must be **append‑only**: no overwriting logs/results; every run gets a timestamped directory and DB rows.
- Use **high‑fidelity official physics** for validation (VMEC++ + Boozer + QI as required by P3).
- Default concurrency target: **6 workers**.

---

## What P3 is optimizing (source of truth)

P3 is `constellaration.problems.MHDStableQIStellarator` (multi‑objective).

- **Objectives (Pareto space):**
  - maximize `minimum_normalized_magnetic_gradient_scale_length` (call it `L∇B`)
  - minimize `aspect_ratio` (call it `A`)
- **Score:** hypervolume of the set of **feasible** designs in the 2D space `(−L∇B, A)` with reference point `(1.0, 20.0)`.
- **Feasibility:** infinity‑norm of positive normalized constraint violations, with a **relative tolerance** of `1e-2` (1%).

Constraints:
- `edge_rotational_transform_over_n_field_periods >= 0.25`
- `log10(qi) <= -3.5`
- `edge_magnetic_mirror_ratio <= 0.25`
- `flux_compression_in_regions_of_bad_curvature <= 0.9`
- `vacuum_well >= 0.0`

Practical implication:
- A “nearly feasible” design often has **one killer constraint** (frequently mirror or QI). The fastest HV gains come from **repairing** a high‑leverage point (especially a compact one) rather than random exploration.

---

## Strategy: treat P3 as “constraint repair + Pareto enrichment”

We run a loop that:
1) proposes batches of candidates via a small set of **structured move families** (“knobs”),
2) evaluates them with the **official high‑fidelity evaluator**,
3) archives feasible candidates and their Pareto contributions,
4) learns which move families improve which constraints, and allocates future budget accordingly.

Key design principles:
- **Not brute force.** Use structured perturbations and low‑dimensional searches.
- **Feasibility‑first.** If a candidate’s worst violation is mirror, don’t optimize HV; optimize mirror (while preserving QI/flux/vacuum/iota).
- **Restartable.** The loop can be stopped and resumed without losing state.

---

## Persistence: reuse the existing WorldModel SQLite (recommended)

Use a dedicated DB:
- `reports/p3_world_model.sqlite` (recommended; keeps P3 isolated)

### Tables we will use (already exist)

We will reuse the existing schema in `ai_scientist/memory/schema.py`:

- `experiments`: one row per P3 run (stores config JSON, git sha, notes).
- `candidates`: one row per proposed boundary.
  - `problem='p3'`
  - `params_json` stores the boundary JSON (`r_cos`, `z_sin`, `n_field_periods`, `is_stellarator_symmetric`, optional `r_sin/z_cos`).
  - `status` drives the worker queue: `pending → running → done|failed`
  - `design_hash` is the deterministic identity (use `ai_scientist.memory.hash_payload`).
- `metrics`: one row per evaluated candidate (raw high‑fid metrics + derived fields).
  - `feasibility` = official scalar feasibility (≤ `1e-2` means feasible)
  - `objective` = store `L∇B` for convenience (even though P3 is multiobjective)
- `pareto_archive`: store feasible points (design_hash, `aspect`, `gradient=L∇B`) per “batch id”.
- `cycle_hv` (optional but recommended): store the HV of the current feasible archive after each batch.
- `artifacts`: store file paths (candidate JSON, eval JSON, submission JSON).

### Artifact layout (never overwrite)

Each run uses:
- `RUN_ID = YYYYMMDDTHHMMSS_p3_<tag>`
- `artifacts/p3/<RUN_ID>/candidates/<design_hash>.json`
- `artifacts/p3/<RUN_ID>/eval/<design_hash>.json`
- `artifacts/p3/<RUN_ID>/batches/batch_<NNN>.jsonl` (append‑only)
- `artifacts/p3/<RUN_ID>/submissions/submission_<stamp>.json`

The DB is the index; artifacts are immutable blobs.

---

## Worker stability and parallelism (6 workers)

High‑fid P3 evaluation is VMEC++ + Boozer + QI. For stability and to avoid oversubscription, every worker process runs with math threading pinned to 1:

- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `VECLIB_MAXIMUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`

Parallelism choice:
- Prefer **process** parallelism for HF evaluation (each worker is its own process). Threads often underperform or destabilize due to mixed Python/native workloads.
- With an M3 Max (10 performance cores), 6 workers is a safe default that leaves headroom for OS + dashboard + proposer.

---

## tmux workflow (recommended session layout)

Create a dedicated tmux session, e.g. `p3`.

Windows:
1) `proposer`
   - Keeps `pending` queue depth near `2 × workers` (≈12) for steady utilization.
2) `workers`
   - 6 panes: `worker_id=1..6`
   - Each pane runs the same worker loop script.
3) `dashboard`
   - Periodic SQL summaries: best HV, feasibility rate, best feasible `A` vs `L∇B`, most common failing constraint, move family win rates.
4) `submission`
   - Builds a candidate set from `pareto_archive` and writes `submission_<stamp>.json`.

Restart semantics:
- If tmux dies, restart the same scripts; the queue is in SQLite (idempotent).

---

## Candidate generation: move families (structured “knobs”)

We explicitly restrict ourselves to a handful of **high signal** move families. Each proposed candidate must log:
- `move_family`
- `parents` (design_hashes)
- `knobs` (scalars / small vectors)
- `seed` and `batch_id`

Move families (initial set):

1) `blend(A, B, t)`
   - convex combination of two known good boundaries.
   - sweep small neighborhoods near known good `t` values.

2) `scale_groups(parent, knobs)`
   - scale coefficient **groups** (not per‑coefficient noise), e.g.:
     - axisymmetric only: `n=0` columns in `r_cos` / `z_sin`
     - non‑axisymmetric only: `n≠0`
     - by `|n|` bands (`|n|=1`, `|n|=2`, `|n|=3`, …)
     - by `m` bands (`m>=M` smoothing)

3) `replace_mode_group(A, B, group)`
   - copy a specific mode group from B into A (e.g., only `|n|=k`, or only axisymmetric terms).

4) `fd_local(parent, basis)`
   - finite‑difference sensitivity on a **tiny** parameterization (6–12 dims), then step along directions that reduce the **worst constraint**.
   - Feasibility‑first scalarization: minimize `max_violation`, then optimize Pareto objectives second.

Learning signal:
- After each batch, compute for each move family:
  - feasibility rate
  - average feasibility improvement vs parent
  - average improvement on the *worst constraint*
  - Pareto contribution count (new non‑dominated points)
  - HV delta when added to current archive

Then allocate the next batch budget using a simple bandit rule (e.g., softmax over recent HV deltas, with a minimum exploration floor).

---

## Subagents: where they help (short-lived, parallel, machine-readable)

Subagents should be used only for bounded tasks; they output **JSON** into `/tmp/codex-subtasks/<task>.json`.

Suggested subagents:

1) `leaderboard`
   - Fetch and summarize public `constellaration-bench-results` for `mhd_stable`.
   - Output: list of top submission seeds + Pareto points + constraint margins.

2) `physics`
   - Inspect which parts of the pipeline compute mirror/QI/flux and summarize likely levers.
   - Output: ranked list of knob ideas (mode groups) and risk notes.

3) `optimizer`
   - Propose a minimal constrained local method (trust-region over a 6–12D knob vector) for “repair this one violated constraint” problems.
   - Output: recommended knob parameterization + step schedule.

All subagent outputs should be treated as *suggestions*; the main session is responsible for implementing the actual loop.

---

## Implementation tasks (next steps; no execution yet)

1) Add P3 lab scripts under `scripts/`:
   - `scripts/p3_propose.py` (enqueue candidates)
   - `scripts/p3_worker.py` (claim → evaluate HF → write metrics/artifacts)
   - `scripts/p3_dashboard.py` (SQL summaries)
   - `scripts/p3_make_submission.py` (build submission JSON from archive)
2) Add a short tmux runbook:
   - `docs/P3_TMUX_RUNBOOK.md` with the exact commands for 6 workers.
3) Add a reproducibility header written once per run:
   - git SHA, python env, thread env vars, seed, and seed sources (files + HF refs).

Acceptance criteria for “ready to run”:
- Proposer/worker scripts are idempotent (safe to restart).
- Every evaluated candidate has:
  - a candidate JSON artifact,
  - an eval JSON artifact,
  - a `metrics` row in SQLite,
  - and `pareto_archive` entries for feasible points.
- Dashboard can compute HV from the feasible archive and show per‑constraint failure rates.
