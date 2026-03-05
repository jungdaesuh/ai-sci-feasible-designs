# Harness Implementation Tracker

Date: 2026-03-04
Document Role: Build-order checklist for the code-generation harness
Status: Phase A complete (M0-M1)
Design Spec: `docs/harness/HARNESS_CODEGEN_PLAN.md`

## How to Use This File

1. Work **top-down** through milestones (M0 → M7). Each milestone lists its modules.
2. Check off each module when: code is written, tests pass, and acceptance criteria are met.
3. **Gates** block progression — do not start the next phase until the gate is green.
4. Update the Phase checkboxes in `HARNESS_CODEGEN_PLAN.md` when a full phase completes.

---

## Dependency Graph

```
Layer 0 (no internal deps)        Layer 1             Layer 2                Layer 3                    Layer 4
┌─────────────────────┐     ┌──────────────┐   ┌──────────────────┐   ┌──────────────────────┐   ┌──────────────┐
│ M0: types.py        │     │ M2: state_   │   │ M3: sandbox.py   │   │ M4: diagnosis.py     │   │ M5: governor │
│ M1: __init__.py     │────▶│     reader   │──▶│     decision_    │──▶│     observation.py    │──▶│     .py      │
│     problem_adapter │     │              │   │     client (File) │   │     prompt_templates/ │   │              │
│     recorder        │     └──────────────┘   └──────────────────┘   └──────────────────────┘   └──────────────┘
│     auth (stub)     │                                                                               │
└─────────────────────┘                                                                               ▼
                                                                                                ┌──────────────┐
                                                                                                │ M6: experi-  │
                                                                                                │     ence.py  │
                                                                                                └──────────────┘
                                                                                                      │
                                                                                                      ▼
                                                                                                ┌──────────────┐
                                                                                                │ M7: auth     │
                                                                                                │   (full) +   │
                                                                                                │   CodexClient│
                                                                                                └──────────────┘
```

---

## Milestone 0: Shared Types

- [x] **`harness/types.py`** (~60 lines)
  - **Purpose:** 6 frozen dataclasses shared across 8+ modules. Prevents circular imports between state_reader ↔ diagnosis ↔ observation.
  - **Internal deps:** None (leaf module).
  - **External deps:** `dataclasses`, `typing`.
  - **Key types:**
    - `CycleSnapshot` — frontier value, queue counts, near-feasible candidates, parent paths
    - `CycleDiagnosis` — exploration_mode, binding_constraints, feasible_yield, objective_delta
    - `ProposalScript` — source: str, model: str, latency_ms: int
    - `CandidateBundle` — boundary: dict, metadata: dict
    - `EnqueueResult` — inserted: int, skipped: int
    - `StopDecision` — should_stop: bool, reason: str | None
  - **Acceptance:** All 6 classes instantiate with positional args, are frozen, repr works.
  - **Test:** `tests/harness/test_types.py`
    - `test_cycle_snapshot_frozen`
    - `test_all_types_instantiate`

---

## Milestone 1: Foundation (Layer 0)

- [x] **`harness/__init__.py`** (5 lines)
  - **Purpose:** Package marker + version string.
  - **Acceptance:** `import harness` succeeds, `harness.__version__` is a string.

- [x] **`harness/problem_adapter.py`** (~100 lines)
  - **Purpose:** Wraps `ProblemProfile` with frontier/target logic for P1/P2/P3.
  - **Internal deps:** `harness.types.CycleSnapshot`.
  - **External deps:** `ai_scientist.problem_profiles`.
  - **Key functions:**
    - `objective_value(metrics_row: dict) -> float` — direction-normalized objective extraction
    - `frontier_delta(prev: CycleSnapshot, now: CycleSnapshot) -> float` — P1/P2: scalar delta, P3: HV delta
    - `target_reached(snapshot: CycleSnapshot) -> bool` — has problem target been met?
    - `binding_constraints(candidates: list[dict]) -> list[str]` — closest-to-threshold constraints
  - **Acceptance:** P2 adapter returns correct objective sign; frontier_delta is positive when improving.
  - **Test:** `tests/harness/test_problem_adapter.py`
    - `test_p2_objective_value_sign`
    - `test_frontier_delta_positive_on_improvement`
    - `test_target_reached_true_when_exceeded`
    - `test_binding_constraints_returns_closest`

- [x] **`harness/recorder.py`** (~90 lines)
  - **Purpose:** Creates `harness_cycles` table (additive, never modifies existing tables) and persists cycle audit records.
  - **Internal deps:** `harness.types.CycleSnapshot`, `harness.types.EnqueueResult`.
  - **External deps:** `sqlite3`.
  - **Key functions:**
    - `ensure_harness_table(conn) -> None` — `CREATE TABLE IF NOT EXISTS harness_cycles`
    - `record_cycle(conn, cycle_id, ...) -> None` — INSERT one audit row
  - **Acceptance:** Table is created on first call; second call is idempotent; row persists with all fields.
  - **Test:** `tests/harness/test_recorder.py`
    - `test_ensure_table_idempotent`
    - `test_record_cycle_persists`
    - `test_record_cycle_with_error_context`

- [x] **`harness/auth.py`** (~120 lines) — **stub only in M1**
  - **Purpose:** Define `CodexCredentials` dataclass and `load_codex_credentials` / `refresh_if_expired` signatures. Bodies raise `NotImplementedError` with message "Full auth in M7".
  - **Internal deps:** None.
  - **Acceptance:** Import succeeds; calling functions raises `NotImplementedError`.
  - **Test:** `tests/harness/test_auth.py`
    - `test_load_credentials_stub_raises`
    - `test_refresh_stub_raises`

---

## Milestone 2: State Reading (Layer 1)

- [x] **`harness/state_reader.py`** (~170 lines)
  - **Purpose:** Query DB for cycle snapshot + diverse parent selection (frontier best, near-feasible best, stepping stone via cosine distance).
  - **Internal deps:** `harness.types.CycleSnapshot`.
  - **External deps:** `sqlite3`, `numpy` (cosine similarity).
  - **Key functions:**
    - `read_snapshot(conn, problem_adapter) -> CycleSnapshot`
    - `select_diverse_parents(conn, frontier_id, n=3) -> list[Path]` — cosine distance diversity
  - **Acceptance:** Returns valid snapshot from existing DB; parent set has 3 distinct candidates; stepping stone is maximally distant from frontier.
  - **Test:** `tests/harness/test_state_reader.py`
    - `test_read_snapshot_from_seeded_db`
    - `test_diverse_parents_returns_three`
    - `test_stepping_stone_is_distant`

---

## Milestone 3: Sandbox + FileClient (Layer 2)

- [ ] **`harness/sandbox.py`** (~210 lines)
  - **Purpose:** 3-layer safety (static analysis, filesystem confinement, post-exec validation) + novelty dedup (cosine sim > 0.95 rejection, runs in governor process post-subprocess).
  - **Internal deps:** `harness.types.CandidateBundle`.
  - **External deps:** `subprocess`, `tempfile`, `re`, `numpy`, `sqlite3` (read-only for dedup).
  - **Key functions:**
    - `validate_static(source: str) -> list[str]` — returns list of violations (empty = pass)
    - `execute_sandboxed(source: str, parents_dir: Path, timeout: int) -> list[CandidateBundle]`
    - `deduplicate_novel(candidates, conn, threshold=0.95) -> list[CandidateBundle]`
  - **Acceptance:**
    - Forbidden import script → rejected with violation list.
    - Valid script → produces candidate JSONs.
    - Timeout script → raises within timeout + 5s.
    - Duplicate candidate → filtered by novelty dedup.
  - **Test:** `tests/harness/test_sandbox.py`
    - `test_static_rejects_os_import`
    - `test_static_rejects_eval`
    - `test_static_allows_numpy`
    - `test_execute_valid_script_produces_candidates`
    - `test_execute_timeout_raises`
    - `test_novelty_dedup_filters_duplicate`

- [ ] **`harness/decision_client.py`** (~150 lines, **FileClient only** in M3)
  - **Purpose:** `DecisionClient` Protocol + `FileClient` implementation. `OpenAICodexClient` and `ClaudeClient` raise `NotImplementedError` ("Implemented in M7").
  - **Internal deps:** `harness.types.ProposalScript`.
  - **External deps:** `pathlib`.
  - **Key functions:**
    - `DecisionClient.request_proposal(observation: str, cycle_id: int) -> ProposalScript` (Protocol)
    - `FileClient.__init__(script_path: Path)`
    - `FileClient.request_proposal(...)` — reads file, returns ProposalScript
  - **Acceptance:** FileClient reads a script file and returns ProposalScript with source.
  - **Test:** `tests/harness/test_decision_client.py`
    - `test_file_client_reads_script`
    - `test_file_client_missing_file_raises`
    - `test_codex_client_stub_raises`

---

### Gate 1: Sandbox Safety (required before any live run)

- [ ] `test_static_rejects_os_import` passes
- [ ] `test_static_rejects_eval` passes
- [ ] Adversarial fixture script (`tests/harness/fixtures/adversarial_proposal.py`) is blocked
- [ ] Timeout fixture script (`tests/harness/fixtures/timeout_proposal.py`) terminates within limit
- [ ] Post-exec validation catches path traversal attempt

---

## Milestone 4: Analysis + Prompt (Layer 3)

- [ ] **`harness/diagnosis.py`** (~130 lines)
  - **Purpose:** Analyze last cycle outcomes + adaptive explore/exploit mode switching.
  - **Internal deps:** `harness.types.CycleSnapshot`, `harness.types.CycleDiagnosis`.
  - **External deps:** None beyond stdlib.
  - **Key functions:**
    - `diagnose_cycle(prev: CycleSnapshot, curr: CycleSnapshot) -> CycleDiagnosis`
    - Explore/exploit rule: frontier improved in last 3 cycles → exploit; no improvement for 5+ → explore
  - **Acceptance:** Returns `exploration_mode="exploit"` when frontier improved; `"explore"` after 5 stalls.
  - **Test:** `tests/harness/test_diagnosis.py`
    - `test_exploit_mode_on_improvement`
    - `test_explore_mode_after_stall`
    - `test_binding_constraints_identified`

- [ ] **`harness/observation.py`** (~160 lines)
  - **Purpose:** Build full LLM prompt from: profile + snapshot + diagnosis + diverse parents + execution traces + experience summary + explore/exploit guidance.
  - **Internal deps:** `harness.types.CycleSnapshot`, `harness.types.CycleDiagnosis`, `harness.prompt_templates.proposal`.
  - **External deps:** None beyond stdlib.
  - **Key functions:**
    - `build_observation(snapshot, diagnosis, parents, traces, experience, problem_adapter) -> str`
  - **Acceptance:** Returned prompt contains all 7 sections; parent paths are real; explore/exploit text matches mode.
  - **Test:** `tests/harness/test_observation.py`
    - `test_observation_contains_all_sections`
    - `test_observation_exploit_guidance`
    - `test_observation_explore_guidance`

- [ ] **`harness/prompt_templates/__init__.py`** (1 line)
  - **Purpose:** Sub-package marker.

- [ ] **`harness/prompt_templates/proposal.py`** (~100 lines)
  - **Purpose:** `SYSTEM_PROMPT` constant + `render_cycle_prompt()` function.
  - **Internal deps:** None.
  - **Key functions:**
    - `SYSTEM_PROMPT: str` — the system prompt for the stellarator optimization agent
    - `render_cycle_prompt(sections: dict[str, str]) -> str` — assemble sections into final prompt
  - **Acceptance:** `SYSTEM_PROMPT` is a non-empty string; `render_cycle_prompt` produces valid markdown.
  - **Test:** `tests/harness/test_prompt_templates.py`
    - `test_system_prompt_non_empty`
    - `test_render_cycle_prompt_assembles_sections`

---

## Milestone 5: Governor MVP (Layer 4)

- [ ] **`harness/governor.py`** (~220 lines)
  - **Purpose:** Main cycle loop: read → diagnose → observe → decide → sandbox → enqueue → record → experience → stop. `GovernorConfig` dataclass for CLI args. Entry via `__main__`.
  - **Internal deps:** All harness modules.
  - **External deps:** `argparse`, `sqlite3`, `time`, `logging`.
  - **Key functions:**
    - `GovernorConfig` — frozen dataclass with all CLI params
    - `run_governor(config: GovernorConfig) -> None` — the main loop
    - `run_one_cycle(config, conn, cycle_id) -> StopDecision`
  - **Acceptance:** With FileClient + seeded DB, runs 1 cycle end-to-end; harness_cycles row appears; stop decision is returned.
  - **Test:** `tests/harness/test_governor.py`
    - `test_single_cycle_end_to_end`
    - `test_stop_on_target_reached`
    - `test_circuit_breaker_after_consecutive_failures`

---

### Gate 2: Single-Cycle Correctness (required before multi-cycle)

- [ ] FileClient end-to-end: 1 cycle produces valid candidates, recorder persists audit row
- [ ] Rejected script: sandbox rejection is recorded with error context
- [ ] Manual verify: inspect `harness_cycles` row, confirm all fields populated

---

## Milestone 6: Experience Distillation

- [ ] **`harness/experience.py`** (~80 lines)
  - **Purpose:** Periodic distillation — summarize top-K vs worst-K patterns every N cycles. Random coin flip injection (default p=0.5).
  - **Internal deps:** `harness.types`, `harness.decision_client.DecisionClient`.
  - **External deps:** `random`.
  - **Key functions:**
    - `should_distill(cycle_id: int, interval: int = 10) -> bool`
    - `distill_experience(conn, client, top_k=5, worst_k=5) -> str`
    - `maybe_inject(experience: str | None, p: float = 0.5) -> str | None` — random coin flip
  - **Acceptance:** Distillation fires every N cycles; coin flip is ~50% over 100 samples (±15%).
  - **Test:** `tests/harness/test_experience.py`
    - `test_should_distill_every_n`
    - `test_distill_produces_string`
    - `test_coin_flip_rate`

---

## Milestone 7: Auth + Live Client

- [ ] **`harness/auth.py`** — full OAuth PKCE (builds on M1 stub)
  - **Purpose:** Complete OAuth PKCE flow against `https://auth.openai.com/oauth/authorize`. Token storage in `~/.harness/auth.json`. Auto-refresh.
  - **Key functions:**
    - `login() -> CodexCredentials` — interactive browser OAuth flow
    - `load_codex_credentials() -> CodexCredentials` — from `~/.harness/auth.json`
    - `refresh_if_expired(creds) -> CodexCredentials` — auto-refresh via refresh_token
  - **Acceptance:** `python -m harness.auth login` opens browser; tokens are stored; refresh works.
  - **Test:** `tests/harness/test_auth.py` (extends M1 tests)
    - `test_login_stores_credentials` (mocked browser)
    - `test_refresh_token_flow` (mocked HTTP)

- [ ] **`harness/decision_client.py`** — `OpenAICodexClient` (builds on M3 stub)
  - **Purpose:** Real Codex Responses API transport. Uses `httpx` + auth module.
  - **Key functions:**
    - `OpenAICodexClient.request_proposal(observation, cycle_id) -> ProposalScript`
  - **Acceptance:** With valid auth, returns a ProposalScript with Python source.
  - **Test:** `tests/harness/test_decision_client.py` (extends M3 tests)
    - `test_codex_client_request` (mocked HTTP)
    - `test_codex_client_auth_refresh_on_401` (mocked)

---

### Gate 3: Live Validation (required before unattended use)

- [ ] Run on P2 for >= 10 cycles with `OpenAICodexClient`
- [ ] Candidates are valid (pass `sanitize_candidate_boundary()`)
- [ ] Frontier moves (at least 1 improvement in 10 cycles)
- [ ] No sandbox escapes in audit trail
- [ ] Recorder captures full audit trail (all 10 cycles in `harness_cycles`)
- [ ] Compare frontier movement rate against Jan 5-9 manual baseline

---

## Implementation Notes

### types.py Rationale
`types.py` is a **new file** not in the original 12-module architecture. It exists solely to prevent circular imports: `state_reader` produces `CycleSnapshot`, `diagnosis` consumes it and produces `CycleDiagnosis`, `observation` consumes both. Without a shared types module, these three would form a circular dependency chain.

### Test Infrastructure
- **`tests/harness/conftest.py`** — shared fixtures:
  - `_make_harness_db()` — in-memory SQLite with schema + harness_cycles table
  - `_insert_candidate_with_metric(conn, ...)` — seed test data
  - Reuses patterns from existing `tests/test_p3_governor_contract.py`
- **Fixture scripts** (in `tests/harness/fixtures/`):
  - `valid_proposal.py` — writes 3 valid candidate JSONs
  - `adversarial_proposal.py` — attempts `os.system`, `eval`, path traversal
  - `timeout_proposal.py` — infinite loop

### File Count
13 modules + 10 test files + 3 fixture scripts + 1 conftest = **27 new files**
