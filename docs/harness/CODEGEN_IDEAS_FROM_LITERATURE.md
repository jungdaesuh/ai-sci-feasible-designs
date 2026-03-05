# Extractable Ideas for the Codegen Harness (Literature Survey)

Date: 2026-03-04
Document Role: Research synthesis
Status: Active
Owner: Harness maintainers

Related docs:
- `docs/harness/HARNESS_CODEGEN_PLAN.md` (implementation plan this feeds into)
- `docs/harness/AUTONOMOUS_HARNESS_PLAN.md` (strategy)

## Sources Analyzed

| Source | Type | Core Idea |
|---|---|---|
| **DGM** (Darwin Gödel Machine) | Repo + paper (arXiv 2505.22954) | Self-referential self-improving agent; archive-based open-ended evolution; staged evaluation; Docker isolation |
| **ShinkaEvolve** (Sakana AI) | Repo + paper | Island model with elitism; novelty rejection sampling; meta-scratchpad; adaptive LLM ensemble via UCB1 bandits; EVOLVE-BLOCK markers |
| **Group Evolving Agents (GEA)** | Paper (arXiv 2602.04837) | Group-level evolution; Performance-Novelty selection criterion; experience sharing via evolutionary traces; Reflect → Evolve → Act pipeline |
| **SkyDiscover** | Repo + papers (AdaEvolve, EvoX) | Modular discovery framework; adaptive multi-island with UCB scheduling; self-evolving search strategies (meta-optimization); cascade evaluation; quality-diversity archives |
| **ExLLM** | Repo | Experience-enhanced LLM optimization; already has fusion/stellarator problem; NSGA-II multi-objective; code generation with sandbox exec |

---

## 1. Experience Distillation

**Sources:** ExLLM, ShinkaEvolve, GEA

All three systems converge on the same pattern: periodically ask the LLM to reflect on what's working and inject the summary into future prompts.

- **ExLLM**: Every ~100 evals, samples top-10 and worst-10, asks LLM to summarize patterns. Injected into 50% of future prompts.
- **ShinkaEvolve**: "Meta-scratchpad" — 3-step LLM analysis: individual summaries → global patterns → actionable recommendations.
- **GEA**: "Reflection module" — analyzes aggregated traces from all group members, produces evolution directives.

### Application to harness

→ Incorporated into codegen plan Phase 1. See `HARNESS_CODEGEN_PLAN.md`: `harness/experience.py` in Phase 1 table, "EXPERIENCE DISTILLATION" in cycle detail.

---

## 2. Adaptive Search Intensity

**Source:** SkyDiscover (AdaEvolve)

AdaEvolve tracks accumulated improvement signal `G` per island and adapts search intensity:

- High G (productive) → low intensity → exploit (small perturbations)
- Low G (stagnating) → high intensity → explore (large jumps)

Formula: `I = I_min + (I_max - I_min) / (1 + sqrt(G + ε))`

### Application to harness

→ Incorporated into codegen plan Phase 1. See `HARNESS_CODEGEN_PLAN.md`: `harness/diagnosis.py` in Phase 1 table, "DIAGNOSIS" in cycle detail.

---

## 3. Staged / Cascade Evaluation

**Sources:** DGM, SkyDiscover, ExLLM

All three use staged evaluation to reduce expensive oracle calls:

- **DGM**: 10 tasks → 50 tasks → 200 tasks (each gate requires threshold)
- **SkyDiscover**: `cascade_evaluation: true` with per-stage thresholds
- **ExLLM**: Surrogate filter → only send >80% feasibility-predicted candidates to VMEC++

### Application to harness

Enhance: `harness/sandbox.py` (post-execution validation).

After the sandbox produces candidates, add a pre-filter stage before enqueue:

1. **Geometric validation** (instant): non-intersecting surface, Fourier spectrum decay, R₀₀ ≈ 1.0
2. **Surrogate prediction** (if available): Random Forest/MLP feasibility score > threshold
3. **Enqueue** only candidates passing both gates

Could cut VMEC++ evaluations by 50-80%. The existing `sanitize_candidate_boundary()` is a start; adding a learned surrogate pre-filter is the real win.

---

## 4. Archive-Based Parent Selection

**Sources:** DGM, GEA, ShinkaEvolve

All systems maintain a growing archive of all viable solutions, not just the current frontier:

- **DGM**: Score-proportional selection with child-diversity penalty (parents with many children are down-weighted).
- **GEA**: Performance-Novelty criterion: `score(i) = α_i * sqrt(nov(i))` — balances quality with behavioral diversity.
- **ShinkaEvolve**: Power-law parent selection with novelty rejection sampling.

### Application to harness

→ Incorporated into codegen plan Phase 1. See `HARNESS_CODEGEN_PLAN.md`: `harness/state_reader.py` + `harness/observation.py` in Phase 1 table, "OBSERVATION" in cycle detail.

---

## 5. Novelty Rejection Sampling

**Source:** ShinkaEvolve

After generating candidates, compute cosine similarity of new Fourier coefficients against recent candidates in the DB. If similarity > 0.95, reject the candidate (it's a near-duplicate that wastes a VMEC++ eval).

### Application to harness

→ Incorporated into codegen plan Phase 1. See `HARNESS_CODEGEN_PLAN.md`: `harness/sandbox.py` in Phase 1 table, "SANDBOX" in cycle detail.

---

## 6. Multi-Model Ensemble with Bandit Selection

**Sources:** ShinkaEvolve (UCB1 bandit), SkyDiscover (weighted pool), DGM (multi-model)

- **ShinkaEvolve**: UCB1 bandit tracks which LLM produces best mutations; dynamically shifts selection probabilities.
- **SkyDiscover**: Weighted model pool with configurable ratios.
- **DGM**: Uses o1 for diagnosis (reasoning), Claude for code generation (tool use).

### Application to harness

Enhance: `harness/decision_client.py`.

The `DecisionClient` Protocol already supports multiple backends. Add a bandit selector:

- Track which model's scripts produced frontier improvements
- After N cycles, shift probability toward the more productive model
- Reward formula (from ShinkaEvolve): `r'_i = exp(max(r_i - r*_0, 0)) - 1`

This lets the harness auto-discover whether Claude or Codex is better for each problem type without manual A/B testing.

---

## 7. Self-Evolving Search Strategy (Meta-Optimization)

**Sources:** SkyDiscover (EvoX), DGM

Both systems evolve the search algorithm itself, not just solutions:

- **EvoX**: When solutions stagnate, LLM generates an entirely new `ProgramDatabase` class (selection/variation methods) as Python code, validates it, hot-swaps it.
- **DGM**: The agent modifies its own prompts, tools, and coding logic.

### Application to harness (future)

Instead of a fixed prompt template, let the LLM evolve the prompt template itself. When the harness stalls for N cycles:

1. Ask the LLM: "Here's the current prompt template and the last 10 cycle results. What should change about how I prompt you?"
2. LLM suggests modifications to the prompt structure.
3. Validate the new prompt produces parseable scripts.
4. Resume with the evolved prompt.

Most ambitious idea — save for Phase 3+ after the core loop is proven.

---

## 8. Execution Trace Sharing

**Source:** GEA (4-type traces)

GEA collects 4 types of traces per agent and shares them across the group:

1. Applied code patches
2. Predicted next-step patches
3. Execution logs
4. Outcome logs (failures + why)

### Application to harness

→ Incorporated into codegen plan Phase 1. See `HARNESS_CODEGEN_PLAN.md`: `harness/observation.py` in Phase 1 table, "OBSERVATION" in cycle detail, "Recent Execution Traces" in prompt template.

---

## 9. Paradigm Breakthrough Detection

**Source:** SkyDiscover (AdaEvolve paradigm module)

When global improvement rate drops below threshold for N iterations, trigger a "paradigm breakthrough" step:

- Ask LLM to generate a high-level strategy shift (not a candidate script, but a meta-strategy).
- E.g., "Switch from perturbing individual Fourier modes to blending entire designs" or "Focus on satisfying vacuum well constraint first, then optimize L_∇B."
- Inject the paradigm description into subsequent prompts.

### Application to harness

New check in `harness/governor.py`: `paradigm_check()`.

If no frontier improvement for `stall_threshold` cycles:

1. Send the full run history summary to the LLM.
2. Ask: "What fundamentally different approach should we try?"
3. Store the response and prepend it to future observations until the frontier moves again.

This maps directly to the chase notes pattern from the Jan Codex sessions — the breakthroughs came from paradigm shifts (discovering sz/s4 knobs, switching to blend strategies).

---

## Priority Ranking

| # | Idea | Effort | Impact | Phase | Harness Component |
|---|------|--------|--------|-------|-------------------|
| 1 | Experience distillation | Low (1 extra LLM call / 10 cycles) | High | Phase 1 | `harness/experience.py` (new) |
| 2 | Adaptive explore/exploit mode | Low (add field to diagnosis) | High | Phase 1 | `harness/diagnosis.py` |
| 3 | Novelty dedup before enqueue | Low (numpy cosine sim) | Medium | Phase 1 | `harness/sandbox.py` |
| 4 | Diverse parent set in prompt | Low (3 SQL queries) | High | Phase 1 | `harness/observation.py` |
| 5 | Cascade / surrogate pre-filter | Medium (train surrogate) | Very High | Phase 2 | `harness/sandbox.py` |
| 6 | Execution trace enrichment | Low (extend observation) | Medium | Phase 1 | `harness/observation.py` |
| 7 | Paradigm breakthrough detection | Medium (new module) | High | Phase 2 | `harness/governor.py` |
| 8 | Multi-model bandit selector | Medium (UCB1 tracking) | Medium | Phase 2 | `harness/decision_client.py` |
| 9 | Self-evolving search strategy | High (meta-optimization) | Speculative | Phase 3+ | `harness/prompt_templates/` |

Items 1–4 and 6 are low-hanging fruit for Phase 1 MVP. Items 5, 7, 8 are Phase 2 hardening. Item 9 is research-grade — wait until the core loop is battle-tested.

---

## Cross-Cutting Observations

### What all 5 systems agree on

1. **LLM generates code, not JSON intents.** Every system that works well lets the LLM write arbitrary code (ExLLM: numpy arrays, DGM: git diffs, ShinkaEvolve: SEARCH/REPLACE blocks, SkyDiscover: diff or full rewrite, GEA: framework patches). The codegen harness plan is already on the right track.

2. **Experience/reflection loops are essential.** All 5 systems invest in periodic LLM introspection on what patterns are working. This is the single most transferable idea.

3. **Diversity preservation prevents collapse.** Whether via islands (ShinkaEvolve, SkyDiscover), novelty scores (GEA), archive branching (DGM), or NSGA-II crowding distance (ExLLM) — every system actively fights convergence to a single lineage.

4. **Staged evaluation saves budget.** Every system with an expensive oracle gates access behind cheaper filters. The harness should do the same with VMEC++.

5. **Checkpoint everything.** All systems save full state per cycle/generation for resume and audit. The harness recorder already plans this.

### What the harness plan already has right

- Governor as sole loop owner (matches DGM's outer loop, SkyDiscover's Runner)
- SQLite + artifacts as SSOT (matches ShinkaEvolve's ProgramDatabase, DGM's archive)
- Sandbox isolation (matches DGM's Docker, ShinkaEvolve's EVOLVE-BLOCK enforcement)
- Chase notes as prior art in prompt (matches GEA's evolutionary traces)
- No error swallowing on LLM failure — circuit breaker stops run (matches SkyDiscover's graceful degradation)

### What was incorporated into the codegen plan (Phase 1)

These items from this survey have been integrated into `HARNESS_CODEGEN_PLAN.md`:
- Diversity preservation: novelty dedup in `sandbox.py`, diverse parent selection in `observation.py`
- Experience distillation loop: periodic LLM reflection in `experience.py`
- Adaptive explore/exploit mode: convergence signal in `diagnosis.py`
- Execution trace enrichment: abbreviated script snippets in `observation.py`

### What remains for Phase 2+

- Surrogate pre-filter before VMEC++ (cascade evaluation) — Phase 2
- Paradigm shift detection for deep stalls — Phase 2
- Multi-model bandit selection — Phase 2
- Self-evolving search strategy (meta-optimization) — Phase 3+
