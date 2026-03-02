AI Scientist – Production-Grade Improvement Plan
================================================

Goal
----
Deliver a robust, physics-aware optimization stack that (1) never rewards
failures (min/max direction safe), (2) learns from historical runs with stable
feature geometry, and (3) remains operable under compute and data sparsity
constraints while matching ConStellaration problem specs.

Design Principles
-----------------
- Determinism: fixed feature ordering tied to Fourier mode indices and a
  recorded truncation schema (mpol/ntor) per experiment.
- Direction-correct objectives: penalties respect maximize vs minimize stages.
- Early-signal learning: surrogates provide value with <20 samples.
- Safety-first evaluation: cache-safe, NaN/Inf-guarded, stage-aware wrappers.
- Actionable prompts: only mention knobs the optimizer can change.
- Measurable rollout: benchmarks and regression tests defined up front.

Architecture Changes (mandatory)
---------------------------------
1) Surrogate and Feasibility Gate
   - Replace ridge ranker with a two-head RandomForest (classifier + regressor).
   - Feature encoding: structured_flatten(params, schema) maps (m,n) -> index
     using experiment-specific mpol/ntor limits loaded from config/template.
     Persist the schema in experiment metadata to keep design_hash deterministic.
   - Data sources: use WorldModel metrics table; targets per problem:
       * P1: objective (max_elongation, minimize)
       * P2: objective (gradient, maximize)
       * P3: hv
   - Feasibility label: binary (feasibility <= DEFAULT_RELATIVE_TOLERANCE) plus
     continuous feasibility stored for analysis; threshold comes from
     tools._DEFAULT_RELATIVE_TOLERANCE.
   - Low-data behavior: fit if >=8 samples; with <8, return input order but log
     “cold start”. With >=4 feasible points, train regressor on feasible only;
     otherwise train on all.
   - Ranking score: expected value = P(feasible) * predicted_objective_value,
     sign corrected per stage (invert predicted_value for minimization before
     expectation to keep higher=better convention internally).

2) Evaluation Safety Layer
   - Wrap evaluate_p1/p2/p3 in _safe_evaluate(compute, stage, maximize=False)
     with:
       * Direction-aware worst_case_obj: +1e9 for minimize, -1e9 for maximize.
       * minize_objective flag set consistently with the stage.
       * Recursive NaN/Inf checks over metrics (scalars and arrays); on failure
         emit penalized result plus error string, not an exception.
       * Cache-safe: failed evaluations are cached with their penalized outputs
         to avoid re-running doomed shapes.

3) Feature Schema and Hashing
   - Serialize the (mpol, ntor, schema_version) alongside each experiment and
     use it in structured_flatten. If template changes, create a new schema and
     avoid mixing differently-shaped features in one model.
   - Ensure design_hash uses canonicalized params plus schema_version so caches
     and logs remain consistent.

4) Prompts
   - PHYSICS_HEURISTICS limited to Fourier amplitudes that the sampler can
     change: reduce high (m>2, n>2) modes for feasibility; increase m=1 to boost
     gradient (warn it raises aspect ratio); never change n_field_periods.
   - Keep governance/budget reminders intact to honor run protocol.

5) Dependency & Performance
   - Add scikit-learn>=1.3.0,<1.5 to project requirements; document optional
     GPU/non-GPU behavior (RF is CPU-ok).
   - Tree counts/depths tuned for <200 ms fit/predict on ~200 samples, feature
     dim <= schema_size*4; benchmark in tests.

6) Data Plumbing
   - Extend WorldModel.surrogate_training_data to safely select objective/hv/
     feasibility per problem; fall back to parsing metrics JSON when the column
     is null.
   - Log feasibility and objective for every evaluation so surrogate targets are
     populated for P1/P2 (not just hv).

7) Runner Integration
   - Swap to RobustSurrogateRanker; expose feature_dim/schema in config.
   - Use explicit cycle_index for exploration schedules instead of history//k.
   - If surrogate not fitted, emit a warning once per run and preserve input
     order.

8) Tests (must pass)
   - unit: structured_flatten keeps R_cos(1,0) at fixed index across varying
     mpol/ntor; NaN in metrics triggers penalty.
   - unit: surrogate rank orders feasible-high-objective above infeasible high
     objective for P2 (maximize) and the inverse for P1 (minimize).
   - unit: _safe_evaluate returns minimize_objective flag correctly per stage.
   - integ (fast): dummy forward_model returning NaNs is handled without crash;
     cache stores penalized result.
   - perf: RF fit+predict on 200 samples, schema 6x6, completes <0.2s on CPU.

Rollout Steps
-------------
1) Implement structured_flatten with schema persistence; update design_hash.
2) Add _safe_evaluate and wrap evaluate_p1/2/3; add recursive NaN checks.
3) Update WorldModel.surrogate_training_data to honor problem-specific targets.
4) Integrate RobustSurrogateRanker in runner; add config knobs and warnings.
5) Update prompts with constrained Fourier-only heuristics and keep governance
   text.
6) Add sklearn dependency to setup/requirements; document in TESTING.md.
7) Add tests listed above; run pytest slice for ai_scientist.
8) Benchmark runtime on small sample to confirm budget impact; adjust RF params
   if needed.

Exit Criteria
-------------
- P1 smoke run: no crashes; penalized failures do not outrank valid candidates.
- P2 smoke run: maximizing objective never rewards failed evaluations.
- P3 surrogate uses hv target and ranks without shape errors on mixed history.
- All new tests pass; CI adds sklearn install step.
