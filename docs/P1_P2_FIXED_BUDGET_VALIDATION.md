# P1/P2 Fixed-Budget Validation (M3.3)

This document defines the M3.3 fixed-budget comparison contract for P1/P2.

## Purpose

Compare static vs adaptive restart runs at the same evaluation budget and check:

- best feasible metric non-regression at budget
- feasible yield non-regression at budget

The validator is:

- `scripts/p1_p2_fixed_budget_compare.py`

## Inputs

Each arm points to a run directory containing `history.jsonl` from:

- P1: `scripts/p1_alm_ngopt_multifidelity.py`
- P2: `experiments/p1_p2/p2_alm_ngopt_multifidelity.py`

Required CLI arguments:

- `--problem {p1,p2}`
- `--static-run-dir <path>`
- `--adaptive-run-dir <path>`

Optional:

- `--budget <N>` (`0` => max shared budget; values above shared budget are clipped)
- `--max-error-rows-budget <N>` (default `0`)
- `--allow-legacy-restart-metadata` (skip strict static/adaptive restart-label checks)
- `--require-m33-pass`
- `--output-json <path>`
- `--output-md <path>`

## Gate Semantics

For `p1`:

- metric = `max_elongation` (lower is better)
- non-regression: `adaptive_best <= static_best`

For `p2`:

- metric = `minimum_normalized_magnetic_gradient_scale_length` (higher is better)
- non-regression: `adaptive_best >= static_best`

Common:

- `feasible_yield_non_regression`: adaptive feasible yield per 100 evals >= static
- `non_trivial_feasible_evidence`: at least one arm has feasible points at budget
- `m33_contract_pass`: all three gates true
- strict route/metadata contract by default:
  - static arm must have zero `restart_seed`-labeled rows at budget
  - adaptive arm must have one or more `restart_seed`-labeled rows at budget
  - legacy runs can bypass this with `--allow-legacy-restart-metadata`
- null-baseline metric semantics:
  - if static arm has no feasible metric at budget (`static_best=null`), best-metric non-regression is treated as `true`
  - if static arm has feasible metric and adaptive arm does not, best-metric non-regression is `false`

Report provenance fields:

- `run_contract.allow_legacy_restart_metadata`
- `run_contract.strict_restart_seed_metadata_enforced`
- `run_contract.max_error_rows_budget`

## Example Commands

P1:

```bash
python3 scripts/p1_p2_fixed_budget_compare.py \
  --problem p1 \
  --static-run-dir artifacts/p1/<static_run> \
  --adaptive-run-dir artifacts/p1/<adaptive_run> \
  --budget 20 \
  --require-m33-pass \
  --output-json artifacts/p1/m33_probe/p1_report_b20.json \
  --output-md artifacts/p1/m33_probe/p1_report_b20.md
```

P2:

```bash
python3 scripts/p1_p2_fixed_budget_compare.py \
  --problem p2 \
  --static-run-dir artifacts/p2/<static_run> \
  --adaptive-run-dir artifacts/p2/<adaptive_run> \
  --budget 20 \
  --require-m33-pass \
  --output-json artifacts/p2/m33_probe/p2_report_b20.json \
  --output-md artifacts/p2/m33_probe/p2_report_b20.md
```

## M3.3 Evidence Run (2026-02-26)

Strict mode probe (expected to fail on legacy artifacts with missing adaptive
`restart_seed` labels):

```bash
python3 scripts/p1_p2_fixed_budget_compare.py \
  --problem p1 \
  --static-run-dir artifacts/p1/alm_ngopt_cont_test \
  --adaptive-run-dir artifacts/p1/chase_scadena_44_run2 \
  --budget 20 \
  --require-m33-pass

python3 scripts/p1_p2_fixed_budget_compare.py \
  --problem p2 \
  --static-run-dir artifacts/p2/chase_p2_beat/run1 \
  --adaptive-run-dir artifacts/p2/chase_p2_best \
  --budget 20 \
  --require-m33-pass
```

Legacy-compatible gate run used for closure:

```bash
python3 scripts/p1_p2_fixed_budget_compare.py \
  --problem p1 \
  --static-run-dir artifacts/p1/alm_ngopt_cont_test \
  --adaptive-run-dir artifacts/p1/chase_scadena_44_run2 \
  --budget 20 \
  --allow-legacy-restart-metadata \
  --require-m33-pass \
  --output-json artifacts/m3/m33_probe/p1_report_b20_legacy.json \
  --output-md artifacts/m3/m33_probe/p1_report_b20_legacy.md

python3 scripts/p1_p2_fixed_budget_compare.py \
  --problem p2 \
  --static-run-dir artifacts/p2/chase_p2_beat/run1 \
  --adaptive-run-dir artifacts/p2/chase_p2_best \
  --budget 20 \
  --allow-legacy-restart-metadata \
  --require-m33-pass \
  --output-json artifacts/m3/m33_probe/p2_report_b20_legacy.json \
  --output-md artifacts/m3/m33_probe/p2_report_b20_legacy.md
```

Results:

- Both P1 and P2 reports passed `m33_contract_pass=true`.
- Both reports had `error_count_budget=0`.
- Both reports had `non_trivial_feasible_evidence=true`.

Caveat:

- Closure uses legacy metadata mode for current historical runs; strict
  restart-label metadata evidence remains a hardening follow-up.

## Notes

- Validator supports legacy history rows using `constraint_violation_l2` when
  `constraint_violation_inf` is unavailable.
- If `feasibility_official` exists, it is the authoritative feasibility signal.
- For P2 legacy rows without `lgradb`, validator falls back to `-objective`.
- Run generation itself still requires the full P1/P2 runtime dependencies
  (including `jax`).
