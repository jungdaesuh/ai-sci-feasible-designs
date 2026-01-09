# P3 tmux Runbook (6 workers)

This runbook launches a **restartable** P3 high‑fidelity loop using:

- `reports/p3_world_model.sqlite` (WorldModel SQLite)
- `artifacts/p3/<RUN_ID>/...` (immutable artifacts per run)
- 1 proposer + 6 workers + 1 dashboard pane

It does **not** require modifying `constellaration/` or `vmecpp/`.

---

## 0) One‑time environment (recommended)

Use the VMEC++/constellaration environment you already validated (example name used in this repo: `vmecpp310`).

Export deterministic threading (important for stability and to avoid oversubscription):

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

---

## 1) Initialize a new run

```bash
python scripts/p3_init_run.py \
  --db reports/p3_world_model.sqlite \
  --run-root artifacts/p3 \
  --tag mirror_repair
```

This prints:
- `experiment_id=...`
- `run_dir=artifacts/p3/<RUN_ID>`

Keep those two values; they’re used everywhere.

---

## 2) Start tmux

```bash
tmux new -s p3
```

Create windows:
- `proposer`
- `workers`
- `dashboard`
- `submission`

---

## 3) Proposer (enqueue candidates)

If your “parent” file is a P3 submission list (JSON list of JSON strings), extract a single boundary first:

```bash
python scripts/p3_extract_boundary.py <SUBMISSION_LIST.json> --index 0 --output /tmp/p3_parent_a.json
python scripts/p3_extract_boundary.py <SUBMISSION_LIST.json> --index 2 --output /tmp/p3_parent_b.json
```

### Option A (manual): enqueue a blend sweep

Example: enqueue a **blend sweep** between two parent boundaries:

```bash
python scripts/p3_propose.py \
  --db reports/p3_world_model.sqlite \
  --experiment-id <EXPERIMENT_ID> \
  --run-dir <RUN_DIR> \
  --batch-id 1 \
  --family blend \
  --parent-a /tmp/p3_parent_a.json \
  --parent-b /tmp/p3_parent_b.json \
  --t-min 0.86 --t-max 0.92 --t-step 0.002
```

Notes:
- The proposer is safe to rerun; it skips duplicates by `design_hash`.
- `batch-id` is a logical number used for analysis and `pareto_archive` grouping.

### Option B (recommended): run the Governor (auto-propose + interventions)

The governor is a lightweight “autopilot” that:
- watches the SQLite DB,
- identifies the highest‑leverage near‑feasible candidate,
- proposes the next batch (repair moves + small blend sweep),
- and keeps the queue filled for 6 workers.

Dry‑run (prints what it would enqueue):

```bash
python scripts/p3_governor.py \
  --db reports/p3_world_model.sqlite \
  --experiment-id <EXPERIMENT_ID> \
  --run-dir <RUN_DIR>
```

Continuous mode (keeps queue filled; still dry‑run unless `--execute`):

```bash
python scripts/p3_governor.py \
  --db reports/p3_world_model.sqlite \
  --experiment-id <EXPERIMENT_ID> \
  --run-dir <RUN_DIR> \
  --workers 6 \
  --loop
```

Bootstrap (for the very first batch, before any metrics exist):

```bash
python scripts/p3_governor.py \
  --db reports/p3_world_model.sqlite \
  --experiment-id <EXPERIMENT_ID> \
  --run-dir <RUN_DIR> \
  --workers 6 \
  --loop \
  --bootstrap-parent-a /tmp/p3_parent_a.json \
  --bootstrap-parent-b /tmp/p3_parent_b.json \
  --bootstrap-t-min 0.86 --bootstrap-t-max 0.92 --bootstrap-t-step 0.002
```

When you are ready to actually enqueue proposals, add `--execute`.

---

## 4) Workers (6 panes)

In 6 separate panes:

```bash
python scripts/p3_worker.py \
  --db reports/p3_world_model.sqlite \
  --experiment-id <EXPERIMENT_ID> \
  --run-dir <RUN_DIR> \
  --worker-id 1
```

Repeat with `--worker-id 2..6`.

Each worker:
- claims one `pending` candidate at a time,
- runs high‑fidelity VMEC++/Boozer/QI,
- writes `artifacts/p3/<RUN_ID>/eval/<design_hash>.json`,
- inserts rows into SQLite (`metrics`, `pareto_archive` when feasible),
- marks the candidate `done` or `failed`.

---

## 5) Dashboard

```bash
python scripts/p3_dashboard.py \
  --db reports/p3_world_model.sqlite \
  --experiment-id <EXPERIMENT_ID> \
  --interval-sec 10
```

This prints:
- queue depth / worker progress
- current best estimated HV (from feasible archive)
- most common failing constraint
- best compact feasible point and best high‑L∇B point

---

## 6) Make a submission file (any time)

```bash
python scripts/p3_make_submission.py \
  --db reports/p3_world_model.sqlite \
  --experiment-id <EXPERIMENT_ID> \
  --output <RUN_DIR>/submissions/submission.json \
  --max-points 16
```

Optional local verification with the repo’s official scorer:

```bash
python scripts/score_candidates.py --problem p3 <RUN_DIR>/submissions/submission.json
```
