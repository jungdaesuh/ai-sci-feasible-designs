# Decision Record: LLM Evolution Integration for P1/P2/P3

Date: 2026-02-25
Status: Accepted (staged rollout)
Owners: AI Scientist runtime and docs maintainers

## Context

We want LLMs to be involved across all P1/P2/P3 loops while keeping deterministic physics evaluation as the scoring source of truth.

Cross-source synthesis from Group Evolving Agents, AlphaEvolve, DGM, ShinkaEvolve, and the `alma`/`ShinkaEvolve`/`dgm` repos converges on the same architecture pattern:

1. Archive-centric evolution with lineage.
2. Diversity-aware parent selection (novelty/child-penalty/islands).
3. LLM-driven proposal and mutation.
4. Deterministic staged evaluation gates for acceptance.
5. Explicit model routing policy (often bandit-based).
6. Safety and rollback contracts.

## Decision

Target state: adopt an LLM-in-the-loop runtime contract for all P1/P2/P3.
Current state remains staged; P1/P2 adoption is still open in backlog.

Target runtime contract:

1. LLMs are mandatory in proposal and strategy layers.
2. Deterministic evaluators remain mandatory decision authority.
3. Every cycle follows: propose -> novelty gate -> deterministic eval -> archive update -> routing reward update.
4. Static rollback path remains available until sustained non-regression gates are met.

Currently enforced scope: P3 adaptive governor path.
P1/P2 rollout remains staged and is tracked in backlog milestones `M3.1`-`M3.3`.

## Priority Recommendations

1. Fix docs SSOT drift first (integrated plan vs backlog status conflicts).
2. Land P1/P2 adaptive restart seed selection and novelty gating.
3. Apply two-stage novelty gate (embedding prefilter, then LLM judge) across problems.
4. Define model-router reward spec using relative feasible/HV improvement.
5. Add explicit evaluator-integrity and sandbox boundary policy to docs.
6. Require meaningful fixed-budget A/B evidence (for example 20/50/100 per arm) before performance claims.

## Compliance Snapshot (2026-02-25)

1. Strong: P3 A/B governance and deterministic evaluator contract.
2. Partial: P1/P2 adaptive LLM integration.
3. Needs cleanup: stale references and contradictory status sections in docs.

## Execution Links

1. Backlog tracking: `docs/IMPLEMENTATION_BACKLOG.md`
   - Policy milestones: `M3.4` (two-stage novelty gate) and `M3.5` (model-router bandit reward contract)
2. Unified roadmap: `docs/AI_SCIENTIST_UNIFIED_ROADMAP.md`
3. Integration plan: `docs/INTEGRATED_EVOLUTION_PLAN_P1_P2_P3.md`
