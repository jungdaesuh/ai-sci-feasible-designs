# P3 Governor Fixed-Budget A/B Validation

Use this to compare **static** vs **adaptive** governor runs at the same high-fidelity
evaluation budget.

## Inputs

- SQLite DB with P3 experiments (`reports/p3_world_model.sqlite` by default).
- One static-arm `experiment_id`.
- One adaptive-arm `experiment_id`.

## Command

```bash
python scripts/p3_governor_ab.py \
  --db reports/p3_world_model.sqlite \
  --static-experiment-id <STATIC_EXPERIMENT_ID> \
  --adaptive-experiment-id <ADAPTIVE_EXPERIMENT_ID> \
  --budget 120 \
  --require-m24-pass \
  --max-error-rows-budget 0 \
  --output-json artifacts/p3/ab/ab_report.json \
  --output-md artifacts/p3/ab/ab_report.md
```

Notes:
- `--budget 0` means “use max shared budget” (`min(static_evals, adaptive_evals)`).
- `--budget` must be `>= 0` (`0` is the only auto-budget sentinel).
- If a positive `--budget` is larger than the shared eval window, it is clipped to
  `min(static_evals, adaptive_evals)`.
- `--max-error-rows-budget` defaults to `0`; by default both arms must have no runtime
  eval errors within the compared budget window.
- `--require-m23-pass` fails the command when M2.3 contract gate is false.
- `--require-m24-pass` fails the command when M2.4 performance-evidence gate is false.
- `--m24-min-budget` must be `>=20` (you can tighten it above 20, but not lower it).
- Hypervolume reference point defaults to `(x=1.0, y=20.0)`, matching P3 scripts.
- By default, static/adaptive route-family checks are strict; use
  `--allow-legacy-route-metadata` only for old experiments without `model_route`.
  Strict mode requires:
  - static arm rows at budget are all static-route,
  - adaptive arm rows at budget are all adaptive-route,
  - adaptive arm has at least one non-fallback adaptive row.

## What is measured

At fixed budget:
- Eval error rows.
- Feasible yield per 100 evals.
- Hypervolume at budget.
- Best feasible `L_gradB` at budget.

Also reported:
- Final hypervolume over all available evals.
- Route/operator usage summaries at budget (from data-plane telemetry).

## Gate used for M2.3

`pass = hv_at_budget_non_regression && feasible_yield_non_regression`

Where:
- `hv_at_budget_non_regression` means adaptive HV at budget ≥ static HV at budget.
- `feasible_yield_non_regression` means adaptive feasible yield/100 ≥ static yield/100.

## Gate used for M2.4

`m24_performance_evidence_pass = m23_contract_pass && budget_meets_m24_minimum && non_trivial_feasible_evidence`

Where:
- `budget_meets_m24_minimum` means `budget_used >= m24_min_budget` (default 20).
- `non_trivial_feasible_evidence` means at least one arm has `feasible_count_budget > 0`.

## Expected outputs

- JSON report with full metrics and deltas.
- Markdown summary for review artifacts.
