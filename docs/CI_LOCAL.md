# Local CI (Docker)

Use this to reproduce the GitHub Actions test job locally.

## Prereqs
- Docker Desktop (or Docker Engine)

## Run
```bash
./scripts/run_ci_local.sh
```

## Notes
- This uses `python:3.10-bullseye` to match CI’s Python version.
- The script installs the same system build deps and Python packages as `.github/workflows/ci.yml`.
- Pip cache is mounted from `~/.cache/pip` to speed up repeat runs.
