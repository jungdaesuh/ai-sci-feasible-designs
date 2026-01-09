# AI-Sci Feasible Designs (ConStellaration)

This repo contains tooling, artifacts, and reproducible evaluations for finding **feasible stellarator plasma boundary designs** for the ConStellaration Fusion Challenge (P1/P2/P3), using the official `constellaration` evaluators for scoring.

## Best known results (local, verified)

All results below are verified via `scripts/score_candidates.py` (high fidelity).

| Problem | Metric | Best (this repo) | Artifact |
|---|---:|---:|---|
| P1 | score (higher better) | 0.970362 (max_elongation=1.266744) | `artifacts/p1/best_p1_submission.json` |
| P2 | score (higher better) | 0.511862 (L∇B=10.237250) | `artifacts/p2/best_p2_submission.json` |
| P3 | hypervolume (higher better) | 135.417600 | `artifacts/p3/20260108T070652_p3_mirror_repair_compact/submissions/submission_best.json` |

Details and provenance: `docs/RESULTS.md`.

## Method (what actually produced the records)

The best results in this repo were found via an **LLM-guided high-fidelity search loop**:

- GPT‑5.2 xhigh proposed **structured, interpretable perturbations** (e.g., mode-group scaling, blending between parents).
- Candidates were evaluated and certified using the **official high-fidelity physics pipeline** (VMEC++ + Boozer + QI where applicable).
- Candidates and evaluations were persisted as artifacts and/or in SQLite for reproducibility.

This repo also contains longer-term plans for more “principled” optimization (surrogates, autonomy), but the headline results above come from the high-fidelity search workflow.

## Reproduce / verify

Prereq: a working VMEC++ runtime env (this repo commonly uses `conda activate vmecpp310`).

Verify P1:

```bash
OMP_NUM_THREADS=1 conda run -n vmecpp310 --no-capture-output \
  python scripts/score_candidates.py --problem p1 artifacts/p1/best_p1_submission.json --top-k 1
```

Verify P2:

```bash
OMP_NUM_THREADS=1 conda run -n vmecpp310 --no-capture-output \
  python scripts/score_candidates.py --problem p2 artifacts/p2/best_p2_submission.json --top-k 1
```

Verify P3:

```bash
OMP_NUM_THREADS=1 conda run -n vmecpp310 --no-capture-output \
  python scripts/score_candidates.py --problem p3 \
  artifacts/p3/20260108T070652_p3_mirror_repair_compact/submissions/submission_best.json --top-k 1
```

## Repo layout

- `scripts/`: scoring and search helpers
- `artifacts/`: best submissions and run outputs
- `reports/`: SQLite world model / logs
- `docs/`: public-facing documentation (older material in `docs/archive/`)
