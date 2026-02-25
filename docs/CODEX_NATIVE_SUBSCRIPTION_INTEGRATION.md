# Codex-Native ChatGPT Subscription Integration (No OpenClaw)

## Objective

Add an OpenClaw-like integration pattern in this repo, but without OpenClaw:

1. OAuth
2. Credential/profile management
3. OpenAI-compatible HTTP adapter layer

## Scope and constraints

- Use ChatGPT/Codex subscription auth natively.
- Do not require OpenClaw gateway/runtime.
- Keep planner and optimization workflows unchanged.
- Keep provider wiring inside `ai_scientist/model_provider.py` and config.

## Current status

- This architecture is **planned** and not yet fully implemented.
- Existing runs should continue using current provider setup (e.g., OpenRouter) until rollout completes.
- `openclaw` provider entries and related tests are still present for backward compatibility during transition.

### Runtime status matrix

| Mode | Status | Current selection path |
|---|---|---|
| OpenRouter + Grok aliases | Implemented | `MODEL_PROVIDER=openrouter`, `AI_SCIENTIST_INSTRUCT_MODEL=grok-planning-short-loop`, `AI_SCIENTIST_THINKING_MODEL=grok-planning-full` |
| OpenRouter + Kwaipilot aliases | Implemented | `MODEL_PROVIDER=openrouter`, `AI_SCIENTIST_INSTRUCT_MODEL=kwaipilot-planning-short-loop`, `AI_SCIENTIST_THINKING_MODEL=kwaipilot-planning-full` |
| Codex-native aliases/provider | Partially implemented | `MODEL_PROVIDER=codex_native`, `AI_SCIENTIST_INSTRUCT_MODEL=codex-native-short-loop`, `AI_SCIENTIST_THINKING_MODEL=codex-native-full` (requires a local OpenAI-compatible adapter endpoint + `CODEX_NATIVE_BEARER_TOKEN`) |

### Role mapping overrides

By default, `configs/model.yaml` pins some roles (planning/literature/analysis) via `role_map`. To force those roles to use the same alias family you are switching to, set:

- `AI_SCIENTIST_ROLE_PLANNING_MODEL`
- `AI_SCIENTIST_ROLE_LITERATURE_MODEL`
- `AI_SCIENTIST_ROLE_ANALYSIS_MODEL`

## Target architecture

### 1) OAuth layer

- Add a `codex_native` auth command set (login/status/logout).
- Use Codex-authenticated session state on the host (ChatGPT sign-in path).
- Resolve access credentials at runtime without storing raw secrets in repo files.

### 2) Credential/profile management

- Add profile metadata store (provider, mode, account label, last refresh, cooldown/health).
- Add profile resolution order and fallback rotation.
- Separate metadata from sensitive material:
  - metadata in app config/state
  - secrets in OS keychain or secure local store

### 3) HTTP compatibility adapter

- Add local endpoint:
  - `POST /v1/chat/completions`
  - optional streaming SSE
- Translate OpenAI chat payloads to internal planner request envelope.
- Normalize responses back to OpenAI-compatible shape.
- Support stable session routing via `user` field or explicit session key.

Note: `codex_native` calls send standard OpenAI chat payloads and use `X-AI-Scientist-Tool-Name` as an optional request header for internal tracing/tool tagging. Other providers may still include legacy `tool_call` metadata in the request body.

## Repo integration points

- Provider configuration: `configs/model.yaml`
- Provider invocation: `ai_scientist/model_provider.py`
- Planner callsites: `ai_scientist/planner.py`
- Smoke validation: `tools/ci_tools_smoke.py`
- Tests:
  - `tests/test_model_provider.py`
  - new tests for auth profiles and adapter endpoint

## Rollout plan

1. Add auth/profile primitives behind feature flag (`MODEL_PROVIDER=codex_native`).
2. Add non-streaming `/v1/chat/completions` adapter and smoke test.
3. Add streaming support and profile fallback/cooldown.
4. Cut over planner roles to `codex_native` provider.
5. Remove obsolete OpenClaw-specific docs/config from this repo branch.
