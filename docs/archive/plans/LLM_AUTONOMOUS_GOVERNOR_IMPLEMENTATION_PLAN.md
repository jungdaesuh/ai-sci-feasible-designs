# Lean Implementation Plan (KISS/YAGNI/SSOT/DRY/SOLID)

## Goal
Build a single autonomous governor loop for P1/P2/P3 that can recover from weak seeds by combining:
- Codex-driven decision/proposal,
- deterministic mutation translation + physics evaluation,
- harness-style control policy (retry/verify/frontier/stop).

## Locked Decisions (User-Selected)
- Scope: one unified loop for `P1/P2/P3` with `P3`-first integration and minimal `P1/P2` runtime restore (`1a` + shared-loop alignment).
- Governor shape: one unified governor (`2a`).
- LLM transport: codex subscription only (`3a`).
- P1 actions: keep `jump` and implement deterministic translation now (`4b`).
- Done criteria: compile + dry-run all three + live cycle per problem when deps exist (`5a`).

## Non-Negotiable Design Constraints
- KISS: one controller, one schema family, one governor loop.
- YAGNI: no legacy stack restore, no multi-agent graph, no new DB tables.
- SSOT:
  - `ai_scientist/llm_controller.py` is the only LLM decision API.
  - `ai_scientist/problem_profiles.py` is the only action/constraint/bounds config.
- DRY:
  - one decision schema + validator for P1/P2/P3.
  - one verifier gate implementation reused by all problems.
- SOLID:
  - SRP: controller decides, governor orchestrates, workers evaluate.
  - OCP: add per-problem profiles/translators without controller rewrite.
  - DIP: governor depends on controller interface, not provider-specific calls.

## Harness Port Policy (Compare-to-`run_harness_v2.sh`)
### Port (Required)
- Explicit stop policy: attempt/time/stagnation caps.
- Verifier gate: accept/reject cycle only from deterministic metrics.
- Frontier tracking: best-so-far objective/feasibility artifact per run.
- Retry escalation: `repair -> bridge/jump -> global_restart`.
- Startup recovery before first action: restore frontier, validate schema/version compatibility, replay only unevaluated pending tasks, and persist resume manifest.

### Do Not Port (Rejected as Bloat/Mismatch)
- Code-editing attempts by agents.
- Git worktree/branch per attempt lifecycle.
- Dual-model generator/verifier infrastructure.
- Commit promotion logic (`best commit checkout`) during search.
- Recursive self-rewrite controller loop.
- Unbounded novelty/semantic judge stacks not tied to numeric objective telemetry.

## Minimal Architecture
1. Shared controller (`ai_scientist/llm_controller.py`)
  - `build_observation(problem, rows, context) -> dict`
  - `decide(problem, observation, session_id) -> Decision`
  - `validate_decision(problem, decision) -> ValidatedDecision`
2. Shared problem profiles (`ai_scientist/problem_profiles.py`)
  - allowed actions/constraints/bounds for `p1`, `p2`, `p3`.
3. Unified governor orchestration
  - existing `scripts/p3_governor.py` becomes shared loop shape.
  - add `--problem p1|p2|p3` mode or thin wrappers.
4. Deterministic executors remain source of truth
  - enqueue path (existing propose/enqueue scripts),
  - physics worker path (existing worker scripts).

## Control Loop (Repo-Native)
1. Observe: fetch recent slice from DB (top/bad/diverse, violations, route reward, queue state).
2. Decide (LLM): controller injects profile-gated action set, then LLM chooses from allowed actions only.
3. Translate (deterministic): convert bounded mutations to proposal commands.
4. Execute (deterministic): enqueue + evaluate.
5. Verify (deterministic gate): compute realized delta, pass/fail/stagnation.
6. Reflect: log proposal vs realized outcome and operator stats.
7. Escalate or continue by policy.
8. Enforce budget/inference caps before next enqueue to avoid drift.

## Decision Contract (Shared)
- `action`:
  - p1: `repair | jump | global_restart`
  - p2: `repair | bridge | global_restart`
  - p3: `repair | bridge | global_restart`
- `target_constraint`: must be in profile allowlist.
- `mutations[]`: typed, bounded, finite, and capped by SSOT mutation budget policy.
- `expected_effect`: required target deltas (or explicit qualitative outcome key).
- Invalid output: hard reject + deterministic fallback policy.
- `restart_plan` (optional internal): `soft_retry`, `degraded_restart`, `global_restart`, `circuit_break`.
- Controller enforces profile-gated `action` allowlist both before prompt construction and at JSON validation time.

## Mutation Budget Contract (SSOT Defaults)
- `max_candidates_per_cycle = 8`
- `max_mutations_per_candidate = 6`
- `max_mutation_groups_per_candidate = 3`
- `repair` per-group normalized delta cap: `|delta| <= 0.15 * allowed_range`
- `bridge` per-group normalized delta cap: `|delta| <= 0.30 * allowed_range`
- `jump` per-group normalized delta cap: `|delta| <= 0.45 * allowed_range` (P1 only)
- Any overflow beyond caps => reject decision and fallback to deterministic policy.

## Determinism Contract (Seeds + Replay)
- Global run seed is created once at run init and persisted in run metadata.
- Candidate seed is deterministic from `(run_seed, cycle_id, candidate_index, operator)`.
- Same input state + same seeds must produce identical proposal translation and evaluator call args.
- Resume must reload persisted seeds; never regenerate seeds for already scheduled/evaluated candidates.

## Restart Trigger Policy (Typed, Explicit)
- `soft_retry`: transient evaluator/infra failure with `consecutive_transient_failures <= 2`.
- `degraded_restart`: `consecutive_transient_failures >= 3` or `queue_desync_events >= 1` in last 20 cycles.
- `global_restart`: `stagnation_cycles >= 8` (no accepted feasible-progress delta) or frontier integrity check failure.
- `circuit_break`: `budget_remaining <= 0`, schema/version incompatibility, or `invalid_llm_outputs >= 3` in last 20 cycles.
- Numeric defaults live in `problem_profiles` SSOT and can be tuned there only.

## Phase Switch Rule (Deterministic)
- Start in `feasibility_recovery`.
- Switch to `frontier_improvement` only when `accepted_feasible_last20 >= 3` and `dominant_violation_rate_last20 <= 0.20`.
- Revert to `feasibility_recovery` if accepted feasible candidates in last 10 verified cycles `= 0`.

## Lesson Memory Policy (Context-Bounded)
- Persist per-cycle outcome tuple: `(action, target_constraint, predicted_delta, realized_delta, feasibility_outcome)`.
- Maintain rolling operator success stats by dominant constraint family.
- Feed LLM only compact summaries (windowed aggregates + top failure signatures), not full raw history.
- Raw event logs stay append-only in deterministic storage; summaries are derived views.

## Prompt Context Contract (Per Cycle)
- Include overall challenge block (ConStellaration mission context and non-negotiable deterministic physics validity rules).
- Include problem nature block (`p1`/`p2`/`p3` objective shape, hard constraints, and phase intent).
- Include exact objective expression and optimization direction for active problem from `problem_profiles` SSOT.
- Include exact hard-constraint conditions and numeric thresholds (no paraphrased limits) from `problem_profiles` SSOT.
- Include explicit objective and purpose block (current phase target: feasibility-first vs frontier-score improvement).
- Include current state block (frontier snapshot, dominant violations, stagnation counters, remaining budget).
- Include lesson summary block (recent operator win/loss stats and prediction-vs-realization deltas).
- Include action contract block from problem profile (allowed actions, constraints, mutation bounds).
- Include strict output schema block (JSON-only response contract and rejection behavior).

### `jump` Deterministic Translator (P1, v1)
- Start from selected parent boundary.
- Apply bounded non-local group perturbations (larger than `repair`).
- Enforce hard caps: per-group scale, total edits, coefficient bounds.
- Validation failure: fallback to `global_restart`.

## Scope (P1/P2/P3)
1. Add shared controller + strict schema validation.
2. Integrate into P3 governor first (lowest integration risk).
3. Add verifier gate + stop policy + frontier artifact to governor loop.
4. Restore minimal P1/P2 runtime path for same loop contract.
5. Add per-problem translators (`bridge`/`jump`) and profile bounds.
6. Persist reflection in existing `scratchpad_events`.
7. Add lean recipe synthesis from successful interventions only.
8. Add typed restart semantics and compact resume manifest for interruption safety.

## Explicitly Out of Scope (No Bloat)
- Full legacy planner/provider/config restoration.
- New schema migrations/tables.
- Generic framework rewrite.
- Multi-provider routing in v1.
- Worktree-based attempt isolation.
- Hard-coded parent-edit/rewrite policies in the control core.

## Minimal File Touch List
- `ai_scientist/llm_controller.py` (new)
- `ai_scientist/problem_profiles.py` (new)
- `scripts/p3_governor.py` (controller + verifier + stop/frontier integration)
- `scripts/p1_governor.py` (new thin wrapper or unified mode)
- `scripts/p2_governor.py` (new thin wrapper or unified mode)
- `ai_scientist/memory/repository.py` (reuse existing scratchpad logging methods only)

## Validation Gates
- Deterministic evaluator path unchanged.
- `py_compile` on all changed files.
- Dry-run per problem with `--llm-enabled` on/off.
- One live cycle per problem only when deps are available.
- Replay check: resumed cycle must match pre-resume seed/candidate identity for unevaluated queue entries.
- Unit test: decision schema acceptance/rejection (valid JSON, invalid fields, overflow caps).
- Unit test: profile-gated action enforcement before prompt and at post-LLM validation.
- Unit test: restart transition thresholds (`soft_retry`/`degraded_restart`/`global_restart`/`circuit_break`).
- Unit test: deterministic phase-switch transitions (`feasibility_recovery` <-> `frontier_improvement`).
- Unit test: replay determinism over resume path (seed and candidate identity continuity).
- Codex transport unavailable:
  - codex-only mode => run-blocking config error,
  - otherwise deterministic fallback (if fallback mode explicitly enabled).

## Checkpoint Commits
1. `feat: add shared llm controller + problem profiles + strict schema`
2. `feat: integrate controller into p3 governor`
3. `feat: add verifier gate + stop policy + frontier artifact in governor loop`
4. `feat: add unified p1/p2 integration + p1 jump translator`
5. `feat: add cross-problem scratchpad reflection and lean recipe synthesis`
