# Autonomous Harness with Code-Generation Decision Interface

Date: 2026-03-04
Document Role: Implementation plan (code-generation harness)
Status: Draft
Owner: Harness maintainers

Related docs:
- `docs/harness/AUTONOMOUS_HARNESS_PLAN.md` (strategy — governor loop still applies)
- `docs/harness/CODEGEN_IDEAS_FROM_LITERATURE.md` (research synthesis — source of Phase 1/2 enhancements)
- `docs/harness/HARNESS_DOC_INDEX.md` (index + status)
- `docs/P1_SCORE_CHASE_NOTES.md`, `docs/P2_SCORE_CHASE_NOTES.md`, `docs/archive/notes/P3_SCORE_CHASE_NOTES.md` (prior art)

## Context

The Codex session (Jan 5-9) that found P1/P2/P3 records worked by: reading the codebase, inventing perturbation strategies (sz/s4 knobs, blend, scale_groups), writing Python scripts, evaluating with VMEC++, diagnosing binding constraints, and adapting. The original harness docs designed a schema-bounded decision interface (`{action: "repair", parameter_group: "abs_n_3", normalized_delta: 0.04}`) that cannot express what actually worked.

**Goal:** Build an autonomous harness where the LLM generates bounded Python proposal scripts each cycle instead of picking from a fixed action menu. Reuse existing SSOT infrastructure (DB, workers, enqueue, profiles). New `harness/` package at repo root.

## Architecture

```
harness/
├── __init__.py
├── types.py             # ~60 lines. Shared frozen dataclasses (prevents circular imports)
├── governor.py          # ~220 lines. Cycle loop + stop controller
├── state_reader.py      # Read DB snapshot, compute diagnostics
├── diagnosis.py         # Analyze last cycle + adaptive explore/exploit mode
├── sandbox.py           # Execute LLM-generated Python safely + novelty dedup
├── recorder.py          # Persist cycle decisions + outcomes to DB
├── experience.py        # Periodic experience distillation (every N cycles)
├── problem_adapter.py   # P1/P2/P3 adapter (wraps problem_profiles.py)
├── decision_client.py   # Abstract LLM interface + OpenAICodex/Claude/File impls
├── auth.py              # OAuth PKCE flow + token storage/refresh for ChatGPT subscription
├── observation.py       # Build LLM prompt: diverse parents + traces + explore/exploit
└── prompt_templates/
    └── proposal.py      # System prompt + cycle observation template
```

Reuse as-is (no modifications to existing tables or interfaces):
- `ai_scientist/memory/schema.py` — SQLite schema + init_db (existing tables unchanged)
- `ai_scientist/p3_enqueue.py` — enqueue_candidate, sanitize, hash, dedup
- `ai_scientist/problem_profiles.py` — P1/P2/P3 definitions (SSOT)
- `scripts/p3_worker.py` — claim/eval/persist (workers)
- `scripts/p3_propose.py` — blend + scale_groups CLI (available to sandbox)
- `scripts/p3_init_run.py` — run initialization
- `scripts/score_candidates.py` — official evaluator

Additive schema (new table, does not modify existing):
- `harness_cycles` table — created by `harness/recorder.py` via `CREATE TABLE IF NOT EXISTS`. Contains cycle audit records (script source, hashes, frontier deltas). Does not alter any existing table in `schema.py`.

## Why Code Generation Instead of Schema Selection

| What actually moved the frontier | Schema can express? | Code gen can express? |
|---|---|---|
| `sz` knob: scale axisym Z by 0.979 | Yes (parameter_group) | Yes |
| `s4` knob: scale \|n\|=4 by 1.17 | Yes (parameter_group) | Yes |
| 2D grid sweep over sz x s4 (50 candidates) | No (one intent per cycle) | Yes |
| blend(A, B, t=0.86) with specific parents | No (no parent_id field) | Yes |
| Chain: blend → scale → evaluate | No (one action per cycle) | Yes |
| Invent a new knob (split s4 into s4_r, s4_z) | No (fixed vocabulary) | Yes |
| Coordinate descent near feasibility ridge | No | Yes |
| Mode expansion (solve small truncation, expand) | No | Yes |

## Codegen Flow

```text
                    ┌──────────────────────────┐
                    │     GOVERNOR START        │
                    │     cycle_id = 0          │
                    └────────────┬─────────────┘
                                 │
            ┌────────────────────▼────────────────────┐
            │            1. STATE_READER               │
            │  DB → queue counts, frontier, parents    │
            │  → CycleSnapshot                         │
            └────────────────────┬────────────────────┘
                                 │
            ┌────────────────────▼────────────────────┐
            │            2. DIAGNOSIS                  │
            │  last cycle outcomes, binding constraints │
            │  frontier improved? → exploit / explore  │
            │  → CycleDiagnosis                        │
            └────────────────────┬────────────────────┘
                                 │
            ┌────────────────────▼────────────────────┐
            │            3. OBSERVATION                │
            │  profile + snapshot + diagnosis           │
            │  + 3 diverse parents + execution traces   │
            │  + experience summary + explore/exploit   │
            │  → prompt string                         │
            └────────────────────┬────────────────────┘
                                 │
            ┌────────────────────▼────────────────────┐
            │         4. DECISION_CLIENT               │
            │  prompt → LLM (Codex / Claude)           │
            │  → ProposalScript (Python source)        │
            └──────────┬─────────────────┬────────────┘
                       │                 │
                    success         fail (ERROR)
                       │                 │
                       │    ┌────────────▼───────────┐
                       │    │  RECORD FAILURE         │
                       │    │  error type + traceback  │
                       │    │  → harness_cycles table  │
                       │    │  → next cycle traces     │
                       │    │  circuit breaker++       │
                       │    └────────────┬───────────┘
                       │                 │
                       │          ┌──────▼──────┐
                       │          │ breaker > N? │
                       │          └──┬───────┬──┘
                       │           yes       no
                       │            │         │
                       │      ┌─────▼────┐    │
                       │      │  STOP    │    │
                       │      │  (error) │    │
                       │      └──────────┘    │
                       │                      │
                       └────────┬─────────────┘
                                │
            ┌───────────────────▼─────────────────────┐
            │         5. SANDBOX (sandbox.py)             │
            │  [subprocess]                              │
            │    L1: static analysis (forbidden imports) │
            │    L2: filesystem confinement (tempdir)    │
            │  [governor process, post-subprocess]       │
            │    L3: post-exec validation (no escapes)   │
            │    novelty dedup (cosine sim > 0.95, DB)   │
            │  → list[CandidateBundle]                   │
            └──────────┬─────────────────┬────────────┘
                       │                 │
                   candidates       fail (ERROR)
                       │                 │
                       │    ┌────────────▼───────────┐
                       │    │  RECORD FAILURE         │
                       │    │  error type + traceback  │
                       │    │  + rejected script src   │
                       │    │  circuit breaker++       │
                       │    └────────────┬───────────┘
                       │                 │
                       │          ┌──────▼──────┐
                       │          │ breaker > N? │
                       │          └──┬───────┬──┘
                       │           yes       no
                       │            │         │
                       │      ┌─────▼────┐    │
                       │      │  STOP    │    │
                       │      │  (error) │    │
                       │      └──────────┘    │
                       │                      │
                       └────────┬─────────────┘
                                │
            ┌───────────────────▼─────────────────────┐
            │            6. ENQUEUE                    │
            │  enqueue_candidate() per valid candidate │
            │  → EnqueueResult(inserted, skipped)      │
            └───────────────────┬─────────────────────┘
                                │
            ┌───────────────────▼─────────────────────┐
            │       7. WORKERS EVALUATE (async)        │
            │  pending → running:* → done/failed       │
            │  (existing p3_worker.py, already running) │
            └───────────────────┬─────────────────────┘
                                │
            ┌───────────────────▼─────────────────────┐
            │            8. RECORDER                   │
            │  script source, hash, frontier delta     │
            │  → harness_cycles table                  │
            └───────────────────┬─────────────────────┘
                                │
                        cycle_id % N == 0?
                           ╱          ╲
                         yes           no
                          │             │
            ┌─────────────▼──────────┐  │
            │  9. EXPERIENCE         │  │
            │  DISTILLATION          │  │
            │  top-K vs worst-K      │  │
            │  → LLM summary stored  │  │
            └─────────────┬──────────┘  │
                          │             │
                          └──────┬──────┘
                                 │
            ┌────────────────────▼────────────────────┐
            │         10. STOP CONTROLLER              │
            │  target reached?  stall > N cycles?      │
            │  budget exhausted?  STOP file?            │
            │  circuit break?                          │
            └──────────┬─────────────────┬────────────┘
                       │                 │
                     stop            continue
                       │                 │
            ┌──────────▼──────┐    sleep(N sec)
            │  GOVERNOR EXIT  │     cycle_id++
            │  (log reason)   │          │
            └─────────────────┘          │
                                         └──── back to 1
```

## The Cycle (one iteration, detailed)

```
1. STATE_READER
   - Query DB: pending/running/done counts, recent metrics, frontier
   - Query: near-feasible candidates (feasibility < 0.02, sorted by objective)
   - Query: last cycle's candidates and their outcomes
   -> Returns: CycleSnapshot dataclass

2. DIAGNOSIS
   - Compare last cycle's candidates vs outcomes
   - Identify binding constraint per near-feasible cluster
   - Compute: feasible yield rate, objective delta, frontier movement
   - Surface: "N candidates within epsilon of feasibility on constraint X"
   - Adaptive mode (from AdaEvolve):
     - If frontier improved in last 3 cycles → exploration_mode="exploit"
     - If no improvement for 5+ cycles → exploration_mode="explore"
   -> Returns: CycleDiagnosis dataclass (includes exploration_mode)

3. OBSERVATION (build LLM prompt)
   - Problem profile (constraints, objectives, scoring formula)
   - Snapshot (frontier value, queue health, near-feasible clusters)
   - Diagnosis (what worked, binding constraints, exploration_mode)
   - Diverse parent set (from GEA/DGM archive selection):
     - Top-1 frontier candidate (best feasible objective)
     - Top-1 near-feasible candidate (closest to boundary, different region)
     - 1 "stepping stone" from archive: sample from feasible candidates outside
       the near-feasible cluster, weighted by Fourier coefficient cosine distance
       from the frontier candidate (maximize design-space diversity)
   - Execution traces from last 3 cycles (from GEA):
     - Abbreviated script snippets that worked/failed
     - Specific failure modes ("all 12 violated QI with avg 0.03")
   - Experience summary (if available, from periodic distillation)
   - Explore/exploit guidance:
     - exploit → "refine near current best, small deltas (0.01-0.03)"
     - explore → "try novel approaches, larger perturbations, new knob families"
   - Available tools (numpy, constellaration.surface, parent paths)
   - Prior art (chase notes patterns: knob families, blend, grid sweeps)
   - Output contract: write candidate JSONs to STAGING_DIR
   -> Returns: str (the full prompt)

4. DECISION_CLIENT (LLM generates proposal script)
   - Send observation to LLM (OpenAI Codex subscription or Claude API)
   - Receive: Python source code string
   - LLM can write anything: grid sweeps, new knob definitions,
     blend logic, coordinate descent, mode expansion...
   -> Returns: ProposalScript(source: str, model: str, latency_ms: int)

5. SANDBOX (validate + execute)
   - Static validation:
     - No forbidden imports (os.system, subprocess, shutil.rmtree, etc.)
     - No DB writes (sqlite3.connect with write, UPDATE, DELETE, INSERT)
     - No network calls (urllib, requests, socket)
   - Prepare sandbox environment:
     - STAGING_DIR: tempdir for candidate outputs
     - PARENTS_DIR: read-only access to run_dir/candidates/
     - Inject: numpy, json, math, copy, pathlib, constellaration.surface
   - Execute with timeout (default 120s)
   - Collect candidate JSONs from STAGING_DIR
   - Validate each: sanitize_candidate_boundary(), max N candidates
   [governor process, post-subprocess — sandbox.py has DB read access here]
   - Novelty dedup (from ShinkaEvolve): compute cosine similarity of
     flattened r_cos+z_sin against last 50 DB candidates; reject if sim > 0.95
   -> Returns: list[CandidateBundle(boundary, metadata)]

6. ENQUEUE
   - For each validated candidate:
     - enqueue_candidate(conn, boundary=..., move_family="harness_gen",
       model_route=f"harness/{model}/{cycle}", knobs=script_metadata)
   - Record: batch_id, script hash, candidate count
   -> Returns: EnqueueResult(inserted: int, skipped: int)

7. WORKERS EVALUATE (existing p3_worker.py, already running)
   - Workers claim pending -> running:* -> done/failed
   - Write metrics to DB + eval JSON to disk

8. RECORDER
   - Persist to harness_cycles table:
     - cycle_id, snapshot_hash
     - proposal_script source (full text)
     - script_hash, model, latency
     - candidate_count, enqueue result
     - pre/post frontier values, delta
     - experience_summary (if distillation ran this cycle, else NULL)
     - stop controller decision
   -> Enables audit replay: review what was decided and why.
      Note: This is audit replay (what script ran, what it produced),
      not execution replay (re-run and get bit-identical output).
      Exact execution replay requires seeded RNG — see prompt template.

9. EXPERIENCE DISTILLATION (every N cycles, e.g., 10)
   - Query DB: top-K feasible candidates + worst-K failures
   - Send both to LLM: "What Fourier mode patterns distinguish
     successful vs failed designs? What should the next script focus on?"
   - Store response in harness_cycles.experience_summary
   - Inject into observation prompt via random coin flip (default p=0.5,
     configurable) — random beats deterministic because it provides a natural
     ablation signal (cycles with vs without experience summary)
   - Cost: 1 extra LLM call every 10 cycles
   -> Returns: Optional[str] (experience summary or None)

10. STOP CONTROLLER
   - target_reached: problem_adapter.target_reached(snapshot)
   - stall: no frontier improvement for N cycles (configurable)
   - budget: max_cycles or max_runtime_sec exceeded
   - manual: STOP file exists in run_dir
   - circuit_break: consecutive LLM or sandbox failures > threshold
   -> Returns: StopDecision(should_stop: bool, reason: str | None)
```

## Decision Client Interface

```python
class DecisionClient(Protocol):
    def request_proposal(self, observation: str, cycle_id: int) -> ProposalScript: ...

class OpenAICodexClient(DecisionClient):
    """V1 primary. OpenAI Codex Responses API via ChatGPT Plus/Pro subscription OAuth.
    Endpoint: https://chatgpt.com/backend-api/codex/responses
    Model: gpt-5.3-codex (or gpt-5.3-codex-spark)
    Auth: OAuth PKCE flow → Bearer JWT token (same as clawdbot/openclaw).
    No per-token API billing — uses flat subscription."""

class ClaudeClient(DecisionClient):
    """V2. Direct Anthropic API. Needs ANTHROPIC_API_KEY."""

class FileClient(DecisionClient):
    """Dev/test: reads a pre-written script from a file."""
```

### V1 Transport: OpenAI Codex Responses API (clawdbot/openclaw pattern)

The first version calls the OpenAI Codex Responses API directly, authenticated via ChatGPT Plus/Pro subscription OAuth. This is the same transport clawdbot/openclaw uses — no per-token API billing, flat subscription cost.

**Key details (from clawdbot source):**
- **Provider:** `openai-codex`
- **API type:** `openai-codex-responses`
- **Model:** `gpt-5.3-codex` (also available: `gpt-5.3-codex-spark`)
- **Endpoint:** `https://chatgpt.com/backend-api/codex/responses`
- **Auth:** OAuth PKCE flow against `https://auth.openai.com/oauth/authorize`
- **Token:** Bearer JWT with `chatgpt_account_id` claim, refreshed every ~8 hours

**OAuth flow (ported from clawdbot `openai-codex-oauth.ts` → Python):**

```python
# One-time setup: run OAuth PKCE flow to get tokens
# 1. Generate code_verifier + code_challenge (S256)
# 2. Open browser to https://auth.openai.com/oauth/authorize
#    with client_id="app_EMoamEEZ73f0CkXaXp7hrann", redirect_uri, PKCE challenge
# 3. User signs in → callback receives authorization_code
# 4. Exchange code for tokens at https://auth.openai.com/oauth/token
# 5. Store: access_token (JWT), refresh_token, expires_at
# 6. Extract chatgpt_account_id from JWT payload

# Stored in ~/.harness/auth.json:
# {
#   "access_token": "eyJ0eXAi...",
#   "refresh_token": "...",
#   "expires_at": 1709595600,
#   "account_id": "..."
# }
```

**Invocation pattern:**

```python
import httpx
import time
from harness.auth import load_codex_credentials, refresh_if_expired

class OpenAICodexClient:
    def __init__(self, model: str = "gpt-5.3-codex"):
        self.model = model
        self.endpoint = "https://chatgpt.com/backend-api/codex/responses"

    def request_proposal(self, observation: str, cycle_id: int) -> ProposalScript:
        creds = load_codex_credentials()  # from ~/.harness/auth.json
        creds = refresh_if_expired(creds)  # auto-refresh via refresh_token

        t0 = time.monotonic()
        response = httpx.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {creds.access_token}",
                "chatgpt-account-id": creds.account_id,
                "OpenAI-Beta": "responses=experimental",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": observation},
                ],
            },
            timeout=120.0,
        )
        content = extract_response_text(response.json())
        script = extract_python_block(content)
        latency_ms = int((time.monotonic() - t0) * 1000)
        return ProposalScript(source=script, model=self.model, latency_ms=latency_ms)
```

**Auth management:**
- OAuth tokens stored in `~/.harness/auth.json` (access, refresh, expires, account_id)
- Auto-refresh when token expires (~8h lifetime) via `https://auth.openai.com/oauth/token` with `grant_type=refresh_token`
- One-time interactive setup: `python -m harness.auth login` (opens browser for OAuth)
- No `OPENAI_API_KEY` needed — flat subscription billing

**Error handling (v1 — no error swallowing):**

Every failure is logged at ERROR level, recorded in `harness_cycles` with full context (error type, traceback, script source if available), and fed into the next cycle's execution traces so the LLM can see what went wrong.

- HTTP timeout → ERROR log, record in DB, circuit breaker++
- 429 rate limit → ERROR log, sleep, retry once, then record failure + circuit breaker++
- 401/403 auth error → ERROR log, attempt token refresh, retry once, then **stop run** (auth is not transient)
- Empty or unparseable response → ERROR log, record full response body in DB, circuit breaker++
- Circuit breaker threshold (default: 3 consecutive failures) → **stop run**

No silent continues. Every failed cycle still runs the recorder (with error context) and stop controller.

Config via CLI: `--decision-client openai-codex|claude|file --decision-model gpt-5.3-codex`

## Sandbox Safety

The sandbox enforces three layers: static analysis, filesystem confinement, and post-execution validation.

### Layer 1: Static Analysis (pre-execution gate)

```python
FORBIDDEN_PATTERNS = [
    r'import\s+os', r'from\s+os\s+import', r'import\s+subprocess',
    r'import\s+shutil', r'import\s+socket', r'import\s+urllib',
    r'import\s+requests', r'import\s+importlib',
    r'__import__', r'eval\s*\(', r'exec\s*\(',
    r'getattr\s*\(', r'globals\s*\(', r'locals\s*\(',
    r'sqlite3', r'compile\s*\(',
]

ALLOWED_IMPORTS = {
    'numpy', 'json', 'math', 'copy', 'pathlib', 'itertools',
    'dataclasses', 'typing',
    'constellaration.geometry.surface_rz_fourier',
}
```

Static analysis is necessary but not sufficient — it catches obvious violations early but cannot prevent all evasion.

### Layer 2: Filesystem Confinement (execution environment)

- Create an isolated tempdir as `STAGING_DIR` with no symlinks to outside paths.
- Copy (not symlink) parent boundaries into a read-only `PARENTS_DIR` inside the tempdir.
- Set environment: strip `PATH`, `HOME`, `PYTHONPATH`; inject only `STAGING_DIR` and `PARENTS_DIR`.
- Execute: `subprocess.run([sys.executable, script_path], timeout=timeout_sec, cwd=staging_dir, env=restricted_env)`.
- The script writes JSON files to its cwd. Governor collects them after.

### Layer 3: Post-Execution Validation

After the subprocess exits, before collecting candidates:
- **Filesystem audit**: Walk `STAGING_DIR` and reject the cycle if any file was created outside it (check for path traversal via `..` or absolute paths in written files).
- **File count + size cap**: Reject if more than `max_candidates` files or any file exceeds 1 MB.
- **Content validation**: Each candidate JSON must pass `sanitize_candidate_boundary()` — rejects malformed or oversized boundaries.
- **No side effects**: Verify no files were modified in `PARENTS_DIR` (compare mtimes or hashes).

### Future Hardening (Phase 3+)

For production use, consider process-level isolation (`bubblewrap`, `nsjail`, or container) to enforce filesystem boundaries at the OS level rather than relying solely on post-exec checks.

## Problem Adapter

```python
class ProblemAdapter:
    """Wraps ProblemProfile with frontier/target logic."""

    def objective_value(self, metrics_row: dict) -> float:
        """Extract objective from a metrics row (direction-normalized)."""

    def frontier_delta(self, prev: CycleSnapshot, now: CycleSnapshot) -> float:
        """P1/P2: best feasible objective improvement. P3: HV delta."""

    def target_reached(self, snapshot: CycleSnapshot) -> bool:
        """Has the problem target been met?"""

    def binding_constraints(self, candidates: list[dict]) -> list[str]:
        """Which constraints are closest to threshold across near-feasible set?"""
```

## LLM Prompt Structure

```
You are an autonomous stellarator optimization agent. Your job is to write
a Python script that generates candidate boundary designs to push the
frontier for problem {problem}.

## Problem Definition
{problem_profile: objective, constraints, scoring formula}

## Current State
- Frontier: {best_feasible_objective}
- Queue: {pending}/{running}/{done} candidates
- Near-feasible cluster: {N} candidates within {epsilon} of feasibility
  - Binding constraint: {constraint_name} (avg violation: {value})
  - Best objective among near-feasible: {value}
- Mode: {exploration_mode} — {mode_guidance}

## Parent Candidates (diverse selection)
- Frontier best: {parent_1_path} (objective={value}, feasibility={value})
- Near-feasible best: {parent_2_path} (objective={value}, binding={constraint})
- Stepping stone: {parent_3_path} (unique mode structure, objective={value})

## Recent Execution Traces (last 3 cycles)
- Cycle {N-2}: {abbreviated_script_snippet} → {M} feasible / {K} total,
  frontier delta={delta}, failure: "{failure_mode}"
- Cycle {N-1}: {abbreviated_script_snippet} → ...
- Cycle {N}: {abbreviated_script_snippet} → ...

{experience_summary_if_available}

## Prior Art (what has worked before)
- P2 breakthrough: sz (axisym Z scale ~0.979) + s4 (|n|=4 scale ~1.17)
- P3 breakthrough: blend(A, B, t=0.86) -> scale(|n|=3, 1.04)
- Key pattern: broad exploration first, then micro-repair near boundary

## Available Tools
- Read parent boundaries from: {parents_dir}/*.json
- Write candidate JSONs to current directory (one file per candidate)
- Available: numpy, json, math, pathlib
- Available: constellaration.geometry.surface_rz_fourier.SurfaceRZFourier

## Constraints on Your Script
- Max {N} candidate files (excess will be truncated)
- Timeout: {T} seconds
- No network, no DB writes, no shell commands
- For reproducibility: seed numpy RNG with the provided cycle_id
  (e.g., `np.random.seed({cycle_id})`) so results can be replayed

## Output
Write Python that generates candidate boundary JSONs. Each file should contain:
{"r_cos": [[...]], "z_sin": [[...]], "n_field_periods": 3, "is_stellarator_symmetric": true}
```

## Implementation Phases

### Phase 1: Core loop (MVP)

| File | Lines (est) | What it does |
|---|---|---|
| `harness/__init__.py` | 5 | Package init |
| `harness/types.py` | 60 | Shared frozen dataclasses: CycleSnapshot, CycleDiagnosis, ProposalScript, CandidateBundle, EnqueueResult, StopDecision |
| `harness/governor.py` | 220 | Main loop: read → diagnose → decide → sandbox → dedup → enqueue → record → experience → stop |
| `harness/state_reader.py` | 170 | Query DB for snapshot + diverse parent selection (frontier, near-feasible, stepping stone) |
| `harness/diagnosis.py` | 130 | Analyze last cycle outcomes, binding constraints, adaptive explore/exploit mode |
| `harness/observation.py` | 160 | Build LLM prompt: diverse parents + execution traces + explore/exploit guidance + experience summary |
| `harness/sandbox.py` | 210 | Static validation + subprocess execution + candidate collection + novelty dedup (cosine sim) |
| `harness/experience.py` | 80 | Periodic experience distillation: summarize top-K vs worst-K patterns every N cycles |
| `harness/recorder.py` | 90 | Write cycle record to DB + experience_summary field |
| `harness/problem_adapter.py` | 100 | Wrap ProblemProfile with frontier/target logic |
| `harness/decision_client.py` | 150 | Protocol + OpenAICodexClient (v1, subscription OAuth) + ClaudeClient + FileClient |
| `harness/auth.py` | 120 | OAuth PKCE flow, token storage/refresh for ChatGPT subscription |
| `harness/prompt_templates/proposal.py` | 100 | Prompt template with chase notes + explore/exploit + traces + experience |
| **Total** | **~1595** | |

### Phase 2: Stop-controller hardening + worker pool + literature enhancements
- Stop-controller hardening (configurable thresholds, stall detection tuning, budget tracking)
- Worker pool supervision (reuse patterns from existing governor.py)
- `--autonomous` mode that spawns workers
- Surrogate pre-filter before VMEC++ (from DGM/SkyDiscover/ExLLM cascade evaluation):
  - Train RF/MLP on 158k dataset to predict feasibility
  - Gate: only enqueue candidates with predicted feasibility > threshold
  - Could cut VMEC++ evals by 50-80%
- Paradigm breakthrough detection (from SkyDiscover AdaEvolve):
  - If no frontier improvement for `stall_threshold` cycles, ask LLM for strategy shift
  - Store paradigm description, prepend to future observations until frontier moves
- Multi-model bandit selector (from ShinkaEvolve UCB1):
  - Track which model's scripts produce frontier improvements
  - UCB1 shifts probability toward more productive model over time

### Phase 3: Hardening
- Per-cycle manifest for audit replay and resume (script source + hash + snapshot already recorded; manifest adds runtime metadata for exact reconstruction)
- Self-evolving search strategy (from SkyDiscover EvoX / DGM): when stalled, LLM evolves the prompt template itself — see `CODEGEN_IDEAS_FROM_LITERATURE.md` §7 for design

### Phase 4: Transport reliability (clawdbot-style)
- Multiple OAuth sessions with profile rotation and cooldown (per clawdbot `auth-profiles.ts`)
- Failure classification per clawdbot `failover-error.ts`: rate_limit → exponential backoff, auth → rotate profile + refresh, timeout → retry, format → sanitize+retry, context_overflow → compact observation
- SSE streaming for faster first-token and partial recovery
- Reference: `clawdbot/src/agents/failover-error.ts`, `auth-profiles/oauth.ts`, `pi-auth-credentials.ts`

## CLI

```bash
# Initialize a run (reuse existing)
python scripts/p3_init_run.py --tag harness_v1 --workers 8

# Start workers (reuse existing)
for i in $(seq 1 8); do
  python scripts/p3_worker.py --problem p2 --db reports/p3_world_model.sqlite \
    --experiment-id 42 --run-dir artifacts/p3/... --worker-id $i &
done

# One-time: authenticate with ChatGPT subscription
python -m harness.auth login  # opens browser for OAuth PKCE flow

# Run the new harness governor (v1: OpenAI Codex via subscription OAuth)
python -m harness.governor \
  --problem p2 \
  --db reports/p3_world_model.sqlite \
  --experiment-id 42 \
  --run-dir artifacts/p3/... \
  --decision-client openai-codex \
  --decision-model gpt-5.3-codex \
  --max-cycles 100 \
  --max-candidates-per-cycle 16 \
  --sandbox-timeout 120 \
  --sleep-sec 30
```

## Verification

1. Unit test sandbox: known script -> verify candidate collection + safety rejection
2. Unit test state_reader: point at existing DB -> verify snapshot
3. Unit test problem_adapter: verify frontier_delta for P1/P2/P3
4. Integration test: one cycle with FileClient (pre-written script) -> end-to-end
5. Live test: OpenAICodexClient on P2 -> verify valid candidates + frontier movement

## Acceptance Gate

This harness is a new package (`harness/`), not a migration of the existing governor. It has its own readiness criteria before it replaces manual sessions:

### Gate 1: Sandbox Safety (required before any live run)
- Unit tests demonstrate that forbidden patterns are rejected.
- Post-execution validation catches path traversal and out-of-bounds writes.
- At least one adversarial test script attempts evasion and is blocked.

### Gate 2: Single-Cycle Correctness (required before multi-cycle)
- One cycle with `FileClient` (pre-written script) produces valid candidates end-to-end.
- Candidates pass `sanitize_candidate_boundary()` and are accepted by workers.

### Gate 3: Live Validation (required before unattended use)
- Run on P2 (most data, best understood) for >= 10 cycles with `OpenAICodexClient`.
- Verify: candidates are valid, frontier moves, no sandbox escapes, recorder captures full audit trail.
- Compare frontier movement rate against manual session baseline (Jan 5-9 Codex session).

### Rollback
- The codegen harness is an independent package. Rollback = stop running it, resume manual sessions or existing governor. No shared state is modified — it only adds candidates to the existing DB via `enqueue_candidate()`.

## Key Design Decisions

1. **LLM generates Python, not JSON intent** — fundamental change from harness docs
2. **Subprocess sandbox, not exec()** — safer, timeout-able, process-isolated
3. **Configurable LLM provider** — OpenAI Codex subscription and Claude API via Protocol
4. **Reuse all SSOT pieces** — no modifications to schema, enqueue, workers, profiles
5. **New harness/ package** — clean separation, no risk to existing code
6. **Chase notes as prior art in prompt** — encodes winning patterns from Jan sessions
7. **No error swallowing** — every failure is logged at ERROR, recorded in DB with full context, and fed into next cycle's traces; circuit breaker stops run after N consecutive failures

## Implementation Progress

Track detailed progress in `docs/harness/HARNESS_IMPL_TRACKER.md`. Summary:

- [ ] **Phase A: Foundation** (M0-M1) — types, problem_adapter, recorder, auth stub
- [ ] **Phase B: Core Pipeline** (M2-M4) — state_reader, sandbox, FileClient, diagnosis, observation, Gate 1
- [ ] **Phase C: Governor MVP** (M5-M6) — governor loop, experience distillation, Gate 2
- [ ] **Phase D: Live Client** (M7) — OpenAICodexClient, full auth, Gate 3

## Relationship to Strategy Doc

The strategy doc (`AUTONOMOUS_HARNESS_PLAN.md`) defines shared runtime rules (governor ownership, SSOT, bounded actions, stop policy). This plan implements those rules via code generation instead of the original schema-bounded interface. See the "Why Code Generation" table above for the full comparison.
