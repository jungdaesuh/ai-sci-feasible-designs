# Results

This document summarizes the best known locally-verified results in this repo and how to reproduce them.

## P1 (Geometrical)

- Best: score=0.970362, max_elongation=1.266744
- Submission: `artifacts/p1/best_p1_submission.json`
- Verification:
  - `python scripts/score_candidates.py --problem p1 artifacts/p1/best_p1_submission.json --top-k 1`
- Notes:
  - The evaluator uses ~1% relative tolerance on normalized violations; top solutions often sit near that ridge.

## P2 (Simple-to-build QI)

- Best: score=0.511862, L∇B=10.237250
- Submission: `artifacts/p2/best_p2_submission.json`
- Verification:
  - `python scripts/score_candidates.py --problem p2 artifacts/p2/best_p2_submission.json --top-k 1`
- Notes:
  - QI is typically the binding constraint near top scores.

## P3 (MHD-stable multi-objective)

- Best: hypervolume=135.417600 (verified locally)
- Baseline reproduced: hypervolume=133.500512
- Submission (16-point set): `artifacts/p3/20260108T070652_p3_mirror_repair_compact/submissions/submission_best.json`
- Verification:
  - `python scripts/score_candidates.py --problem p3 artifacts/p3/20260108T070652_p3_mirror_repair_compact/submissions/submission_best.json --top-k 1`
- Provenance (high-level):
  - Mirror constraint was the compact bottleneck; a blend bridge plus a small `|n|=3` scaling produced a feasible compact point that improved HV.
