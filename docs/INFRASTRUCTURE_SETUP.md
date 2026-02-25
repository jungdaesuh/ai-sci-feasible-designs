# Infrastructure & Execution Guide

This guide explains how to prepare a reproducible runtime for the AI Scientist. The Python logic is self-contained; the only missing piece is the compiled *vmecpp* binaries (and their dependencies) that `constellaration.forward_model` relies on. Use this document whenever you need to respin the environment or share it with teammates.

## 1. System dependencies

1. **Compiler toolchain:** `gcc`/`g++` and `gfortran` (targeting the discipline of VMEC++).
2. **NetCDF:** Install the system packages (e.g., `libnetcdf-dev` on Debian/Ubuntu or `netcdf-fortran` on macOS via `brew`).
3. **MPI (optional):** If you plan to build VMEC++ with MPI, install `mpich`/`openmpi` and keep the MPI runtime available for spawned workers.

These dependencies are captured in the recommended Dockerfile (see Section 3); locally, install them once and keep them in your PATH.

## 2. Environment variables

Create a local `./.env` from `./.env.example` once, then source it before launching any AI Scientist runner or training script:

```sh
cp .env.example .env
set -a
source .env
set +a
```

The local `.env` file enforces:

- threading caps (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`)
- model-provider overrides (`MODEL_PROVIDER`, `AI_SCIENTIST_INSTRUCT_MODEL`, `AI_SCIENTIST_THINKING_MODEL`)

For native ChatGPT-subscription integration design (no OpenClaw dependency), see `CODEX_NATIVE_SUBSCRIPTION_INTEGRATION.md`.

Apply the same approach in CI/test scripts if you fork or wrap the runner.

## 3. Docker reference image

Use a container that already packages the physics stack.

### Current runtime (implemented): OpenRouter-backed aliases

Example snippet:

```Dockerfile
FROM proxima-fusion/constellaration:latest

WORKDIR /app
COPY . /app

ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MODEL_PROVIDER=openrouter \
    AI_SCIENTIST_INSTRUCT_MODEL="grok-planning-short-loop" \
    AI_SCIENTIST_THINKING_MODEL="grok-planning-full"

RUN pip install -e ".[test,experiments]" && \
    pip install -e constellaration[test,lint]

CMD ["python", "-m", "ai_scientist.runner"]
```

### Codex-native runtime (partially implemented): local adapter required

The `codex_native` provider and `codex-native-*` aliases exist in `configs/model.yaml`, but this repo does not ship the local OpenAI-compatible adapter server or the OAuth/profile management yet.

Target-state sketch (requires an external adapter endpoint):

```Dockerfile
ENV MODEL_PROVIDER=codex_native \
    AI_SCIENTIST_INSTRUCT_MODEL=codex-native-short-loop \
    AI_SCIENTIST_THINKING_MODEL=codex-native-full \
    CODEX_NATIVE_BEARER_TOKEN="..."
```

Only switch to this once you have an adapter running and have completed the remaining rollout tasks in `CODEX_NATIVE_SUBSCRIPTION_INTEGRATION.md`.

This image ensures:

- VMEC++ is built and linked (the base image ships the binaries).
- System threading limits match the `.env`.
- The AI Scientist always invokes the configured provider models.

Build/run with:

```sh
docker build -t ai-scientist .
docker run --rm -v "$PWD":/app -e OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" ai-scientist
```

For native ChatGPT-subscription mode, implement the local Codex-compatible adapter and auth/profile flow described in `CODEX_NATIVE_SUBSCRIPTION_INTEGRATION.md` before switching production runs.

## 4. Best practices

- Keep `constellaration/` read-only; only modify orchestration under `ai_scientist/`.
- Record seeds, git SHAs, and the environment (`.env` values) in every report (see `docs/archive/plans/MASTER_PLAN_AI_SCIENTIST.md` and `docs/AI_SCIENTIST_UNIFIED_ROADMAP.md`).
- When experimenting with new OpenRouter models, edit `configs/model.yaml` or override via env vars rather than code.
