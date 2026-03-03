# Final Submissions (ConStellaration Bench)

This file records the exact artifacts submitted to the Hugging Face **ConStellaration Bench** leaderboard, plus the minimal info needed to reproduce/verify them locally.

## Leaderboard Entries (CreativeEngineer)

| Problem | Score | Submission time (UTC) | Submission filename | Result filename |
|---|---:|---|---|---|
| P1 (`geometrical`) | 0.9701409584192382 | 2026-01-07T23:03:00.745924 | `geometrical/2026-01-07T23-03-00.745924_geometrical.json` | `geometrical/2026-01-07T23-03-00.745924_geometrical_results.json` |
| P2 (`simple_to_build`) | 0.5118625445566263 | 2026-01-07T22:54:40.936252 | `simple_to_build/2026-01-07T22-54-40.936252_simple_to_build.json` | `simple_to_build/2026-01-07T22-54-40.936252_simple_to_build_results.json` |
| P3 (`mhd_stable`) | 135.41759815376096 | 2026-01-08T10:07:00.611582 | `mhd_stable/2026-01-08T10-07-00.611582_mhd_stable.json` | `mhd_stable/2026-01-08T10-07-00.611582_mhd_stable_results.json` |

## Local Files That Were Uploaded

These are the exact local JSON files used as uploads in the Space UI.

| Problem | Local path | SHA256 |
|---|---|---|
| P1 | `artifacts/p1/best_p1_submission.json` | `9b19e167611d2937a92015eecfe159542a048ec5982540b1dd32ec2371181367` |
| P2 | `artifacts/p2/best_p2_submission.json` | `50747612fe4e75d2c815f79499218ec4d8cc0b0d9b5c028a2c720bfe5773ecaa` |
| P3 | `artifacts/p3/20260108T070652_p3_mirror_repair_compact/submissions/submission_best.json` | `5c1b60ec18115f9fa1963428095141bc9b552625fecd4294125325ccee88f63a` |

Notes:
- P1/P2 uploads are **single boundary objects** (`r_cos`, `z_sin`, `n_field_periods`, ...).
- P3 upload is a **JSON list of JSON-encoded boundary strings** (a Pareto set). This is required for `mhd_stable` scoring (hypervolume).

## Local Verification (High-Fidelity)

High-fidelity scoring uses VMEC++ (and QI where required). On this machine, run verification inside the working conda env:

```bash
conda activate vmecpp310
```

Then:

```bash
# P1: single boundary
python scripts/score_candidates.py --problem p1 artifacts/p1/best_p1_submission.json

# P2: single boundary
python scripts/score_candidates.py --problem p2 artifacts/p2/best_p2_submission.json

# P3: list of boundaries (Pareto set)
python scripts/score_candidates.py --problem p3 artifacts/p3/20260108T070652_p3_mirror_repair_compact/submissions/submission_best.json
```

## Boundary Explorer vs Bench (Common Confusion)

- The **bench** Space (`constellaration-bench`) accepts:
  - `geometrical`: one boundary JSON object
  - `simple_to_build`: one boundary JSON object
  - `mhd_stable`: a JSON list of boundary JSON strings
- The **boundary explorer** Space (`boundary-explorer`) upload mode accepts **only one boundary JSON object**.
  - It cannot visualize `mhd_stable` multi-boundary submissions.

If you want to visualize a single boundary from the P3 set, extract one:

```bash
python scripts/p3_extract_boundary.py \
  artifacts/p3/20260108T070652_p3_mirror_repair_compact/submissions/submission_best.json \
  --index 0 \
  --output /tmp/p3_one_boundary.json
```

Optional local checksum for the extracted file (example):
- `/tmp/p3_one_boundary.json` is derived data; only `submission_best.json` is the submission artifact of record.
