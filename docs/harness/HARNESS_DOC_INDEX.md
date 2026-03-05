# Harness Doc Index

Date: 2026-03-04
Document Role: Index and status board
Status: Active
Owner: Harness maintainers
Last Updated: 2026-03-04
Applies To: `harness/` package implementation

## Purpose
Single index for the autonomous harness documentation set.

## Active Docs

| Doc | Role | Path |
|---|---|---|
| Strategy (north star) | Design principles, governor ownership, stop policy | `docs/harness/AUTONOMOUS_HARNESS_PLAN.md` |
| Code-generation harness plan | Implementation plan (active) | `docs/harness/HARNESS_CODEGEN_PLAN.md` |
| Literature ideas | Research synthesis feeding into codegen plan | `docs/harness/CODEGEN_IDEAS_FROM_LITERATURE.md` |
| Post-mortem report | Historical evidence from 2026-03-04 session | `docs/harness/AUTONOMOUS_HARNESS_REPORT_2026-03-04.md` |
| Implementation tracker | Checkboxes + build order | `docs/harness/HARNESS_IMPL_TRACKER.md` |

## Reading Order
1. Strategy (`AUTONOMOUS_HARNESS_PLAN.md`)
2. Codegen plan (`HARNESS_CODEGEN_PLAN.md`)
3. Implementation tracker (`HARNESS_IMPL_TRACKER.md`) — build order + checkboxes for implementers
4. Literature ideas (`CODEGEN_IDEAS_FROM_LITERATURE.md`) — reference for where enhancements came from

## Archived Docs (Superseded by Codegen Plan)

These docs described the original schema-bounded decision interface. They are preserved for historical reference but superseded by `HARNESS_CODEGEN_PLAN.md`.

| Doc | Original Role | Path |
|---|---|---|
| Porting blueprint | Architecture for schema-bounded harness | `docs/harness/archive/HARNESS_PORTING_BLUEPRINT.md` |
| Cleanup execution plan | 9-phase migration plan | `docs/harness/archive/HARNESS_CLEANUP_EXECUTION_PLAN.md` |
| Agent prompt spec | Prompt/runtime contract for schema decisions | `docs/harness/archive/HARNESS_AGENT_PROMPT_SPEC.md` |

## Shared Runtime Rules (Carry Forward)
- Governor is sole loop owner and stop authority.
- SQLite + artifacts are SSOT; LLM memory is advisory only.
- Fresh DB-derived context for every cycle decision.
- Bounded actions only (candidate cap per cycle).
- Persistent decision client for reliability.

## Change Policy
- Strategy changes rarely.
- Codegen plan updates during active implementation.
- If one changes, verify cross-links and index status in same commit.
