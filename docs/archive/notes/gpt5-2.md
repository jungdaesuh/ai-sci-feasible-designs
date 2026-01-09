    Read constraints.py, fidelity_controller.py, repository.py,
         objective_types.py, surrogate_v2.py, differentiable.py,
         problems.py, config.py, evaluation.py, forward_model.py,
         cycle_executor.py, adapter.py

─ Worked for 25m 14s ───────────────────────────────────────────────

• ## Atom of Thoughts (AoT) — Codebase Review of ai_scientist/

  ### Executive Summary (what’s true after reading the code and
  running tests)

  This repo has a genuinely promising systems architecture for
  autonomous stellarator design: a multi-stage loop (generate
  → screen → promote → archive), a reproducibility-oriented
  SQLite “world model”, caching, and multiple search engines
  (ALM/NGOpt, differentiable surrogate optimization, near-axis
  sampling, generative diffusion). The core constraint formulas
  in the centralized forward model are benchmark-consistent with
  constellaration.

  However, there are several high-severity scientific/engineering
  correctness gaps that will prevent “maximum correctness” and will
  very likely block “autonomous discovery of truly feasible P2/P3
  designs” under the strict constraints:

  1. Hypervolume reference point is inconsistent with the official
     benchmark.
     constellaration uses reference point [1.0, 20.0] in
     minimization space, while this code uses (-HV_REF_GRADIENT,
     HV_REF_ASPECT) with default gradient threshold 1.0 → (-1.0,
     20.0). That changes the score meaningfully and makes results
     not comparable to baselines.
     Evidence: constellaration/src/constellaration/problems.py:365
     vs ai_scientist/constraints.py:262 and ai_scientist/tools/
     hypervolume.py:33.
  2. The codebase uses the token “hv” for incompatible things
     (semantic drift):
      - cycle-level hypervolume (true HV) = p3_summary.hv_score
        stored in cycle_hv table,
      - per-candidate evaluation["hv"] = max(0, gradient - 1) (not
        hypervolume),
      - surrogate training target "hv" = the DB column metrics.hv
        which is the per-candidate value above.
        This makes the surrogate “HV training target” not actually
        hypervolume.
        Evidence: ai_scientist/fidelity_controller.py:266,
        ai_scientist/memory/repository.py:1076, ai_scientist/
        objective_types.py:62.
  3. The differentiable optimization loop ignores key P2/P3
     constraints and penalizes the wrong ones.
     In gradient_descent_on_inputs, P3 constraints mirror ratio
     and flux compression are explicitly unpacked and then ignored,
     while elongation is always penalized even though P3 has no
     elongation constraint. P2 constraints (aspect ratio, mirror
     ratio, iota) are also not enforced consistently.
     Evidence: ai_scientist/optim/differentiable.py:411 and
     ai_scientist/optim/differentiable.py:455.
  4. Neural surrogate ranking computes prob_feasible then does not
     use it in expected_value.
     For P3 (tiny feasible volume), this is a major algorithmic
     flaw: the ranker will happily push high objective + high
     uncertainty candidates even when predicted feasibility is low.
     Evidence: ai_scientist/optim/surrogate_v2.py:1161 and
     ai_scientist/optim/surrogate_v2.py:1197.
  5. Reproducibility snippet embedded in reports is syntactically
     broken.
     The code writes a “replay from SQLite” snippet that is missing
     the row = conn.execute(...) call and will not run. This matters
     because your process emphasizes deterministic replay.
     Evidence: ai_scientist/cycle_executor.py:905.
  6. The test suite currently fails (4 failures) in core areas.
     I ran: pytest -q -m "not slow and not integration" and got 4
     failed, 263 passed.
     Failures are about: surrogate output arity changes (4→6 heads)
     and P3Problem.get_objective semantic mismatch.
     Evidence: tests/optim/test_surrogate.py:10, tests/optim/
     test_surrogate_ensembles.py:95, tests/test_problems.py:180.

  Net: the physics constraint math in the forward model is mostly
  correct, but the optimization semantics (HV, constraints, training
  targets) are inconsistent enough that the system can optimize the
  wrong thing and/or report misleading progress.

  ———

  ## AoT Step 1 — Atomic decomposition + dependency graph

  ### Atoms (smallest independent review tasks)

  A0. Inventory & entry points (what runs; where the “main loop” is)
  A1. Architecture / control flow (runner → cycle executor →
  fidelity controller → world model)
  A2. Physics interface correctness (forward model + backend +
  constraint margins)
  A3. Scoring correctness (Pareto logic + hypervolume reference
  & dominance)
  A4. Data representation & canonicalization (flatten/unflatten,
  hashing, schema stability)
  A5. Surrogate models (RF bundle + neural operator ensemble:
  training, uncertainty, targets)
  A6. Optimization kernels (candidate proposal mixture, ALM bridge,
  gradient descent, prerelax, RL)
  A7. Output pipeline & reproducibility (DB schema, artifact
  logging, report correctness)
  A8. Engineering quality (deps, tests, determinism, perf
  bottlenecks, failure modes)

  ### Dependency graph (what must be understood before what)

  - A0 → A1
  - A1 → A7
  - A2 → A3
  - A4 → A5
  - A2 + A3 + A4 → A5 (surrogate targets depend on objective/
    feasibility semantics + vectorization)
  - A2 + A5 → A6 (optimization uses surrogate/constraints)
  - A1 + A2 + A3 + A7 → A8 (testing/perf/determinism depends on the
    whole stack)

  ———

  ## AoT Step 2 — Solve atoms (ruthless, evidence-based)

  ### Atom A0 — Inventory & entry points

  Primary runtime entrypoint

  - ai_scientist/experiment_runner.py orchestrates experiments.
  - ai_scientist/runner.py is a shim re-exporting experiment_runner
    for backward compatibility.

  Key subsystems

  - Evaluation: ai_scientist/forward_model.py + ai_scientist/
    backends/real.py / mock.py
  - Control loop: ai_scientist/cycle_executor.py, ai_scientist/
    fidelity_controller.py
  - Storage: ai_scientist/memory/repository.py, schema in
    ai_scientist/memory/schema.py
  - Candidate generation: ai_scientist/optim/samplers.py,
    ai_scientist/tools/design_manipulation.py
  - Surrogates: ai_scientist/optim/surrogate.py (RF) and
    ai_scientist/optim/surrogate_v2.py (NN ensemble)
  - Differentiable geometry: ai_scientist/optim/geometry.py
  - Differentiable optimization: ai_scientist/optim/
    differentiable.py
  - ALM/nevergrad bridge: ai_scientist/optim/alm_bridge.py

  Carmack-style note: The module list suggests the repo is half
  “research sandbox” and half “production runner”. That’s OK, but
  in this state the presence of many unused-but-imported heavy deps
  (gymnasium, pymoo, sklearn, pandas, nevergrad, datasets, etc.)
  increases breakage surface unless dependency management is strict.

  ———

  ### Atom A1 — Architecture & orchestration correctness

  Main loop

  - run_experiment() (ai_scientist/experiment_runner.py) creates:
      - BudgetController (adaptive budgets)
      - FidelityController (stage evaluation + promotions)
      - WorldModel (SQLite)
      - Surrogate + optional GenerativeModel
      - CycleExecutor to run each cycle

  Cycle orchestration

  - CycleExecutor.run_cycle():
      - generates candidate pool via Coordinator/ALM/samplers/
        generative, then
      - surrogate ranks to screen budget, then
      - evaluates “screen” stage, then selects promotions, then
        evaluates “promote” stage,
      - computes cycle Pareto/HV summary, logs artifacts, logs
        “supported/refuted” replay checks.

  Strong engineering points

  - Uses a single “world model” DB as SSOT for history and
    retraining.
  - Captures git SHA and constellaration SHA at experiment start,
    enabling reproducibility.
  - Has explicit stage history and gating logic (S1→S2→S3).

  Architectural hazards

  - Overloaded concept of “stage”: sometimes stage means governance
    stage (S1/S2/S3) and sometimes evaluation fidelity stage
    (“screen/promote/p3”). Confusion is visible in naming and causes
    subtle bugs.
  - Coordinator rebuilt every cycle (in CycleExecutor.run_cycle)
    even if not needed; this isn’t a correctness bug, but it’s
    unnecessary allocations / initialization overhead.

  ———

  ### Atom A2 — Physics evaluation & constraints (scientific
  correctness)

  #### What is correct

  Constraint normalization matches constellaration benchmark
  definitions
  In ai_scientist/forward_model.py:522, constraint margins are
  computed as normalized violations with denominators equal to the
  benchmark bounds, matching constellaration/src/constellaration/
  problems.py:113 onward. Examples:

  - P2 QI constraint uses (log10(qi) - (-4.0)) / 4.0
    matches constellaration/src/constellaration/problems.py:230.
  - P3 vacuum well uses normalization fallback 0.1 (because bound
    is 0.0)
    matches constellaration/src/constellaration/problems.py:396
    (uses max(1e-1, bound)).

  This is the single most important physics correctness pillar, and
  it’s largely right.

  Stage gating for expensive metrics is coherent
  ai_scientist/forward_model.py:557 decides which constraints to
  include based on stage string; tools._settings_for_stage similarly
  decides which constellaration settings to use. When you run stage
  "p2"/"p3", you’ll compute QI. When you run stage "promote", QI is
  skipped. That’s coherent if the config uses it intentionally.

  #### What is risky/wrong

  1. Preset mislabeling can silently disable QI constraints

  - ExperimentConfig.p3_high_fidelity() uses
    defaults.fidelity_ladder (likely screen/promote) (ai_scientist/
    config.py:410).
  - But tools._settings_for_stage("promote", ...) uses
    default_high_fidelity_skip_qi() (ai_scientist/tools/
    evaluation.py:290).
  - And compute_constraint_margins skips qi constraint for stage
    "promote" (ai_scientist/forward_model.py:558).

  Result: The “p3-high-fidelity” preset can run P3 without QI
  enforcement at all. That is catastrophic for scientific validity
  because QI is a defining constraint (Boozer transform + QI
  residual).

  2. EvaluationResult.dominates() is a stub
     ai_scientist/forward_model.py:396 always returns False. If
     any code ever uses it for Pareto set maintenance, it will be
     silently wrong. Today, it appears unused (Pareto is handled
     elsewhere), but stubs like this are landmines.

  ———

  ### Atom A3 — Hypervolume/Pareto correctness

  #### Hypervolume sign convention

  The code does the standard reduction:

  - natural objectives: maximize gradient, minimize aspect
  - minimization vector: (-gradient, aspect)
    This matches constellaration/src/constellaration/
    problems.py:355.

  So far so good.

  #### Critical mismatch: reference point

  - Official benchmark reference point: reference_point =
    np.array([1.0, 20.0]) in minimization space (constellaration/
    src/constellaration/problems.py:365).
  - This repo: reference point comes from get_hv_reference_point()
    → default (-1.0, 20.0) (ai_scientist/constraints.py:262, used in
    ai_scientist/tools/hypervolume.py:33).

  This changes the hypervolume number materially:

  - With ref (1, 20), any feasible point with gradient ≥ 0
    contributes.
  - With ref (-1, 20), only points with gradient > 1 contribute, and
    the dominated rectangle changes.

  If your goal is to compare against ALM-NGOpt baselines and the
  competition leaderboard, you must match the benchmark’s reference
  point exactly. If you want a “thresholded hypervolume”, fine—but
  then it’s not the benchmark score and cannot be used for claim-
  making.

  #### Pareto extraction implementation

  ai_scientist/tools/hypervolume.py uses an O(N²) dominance check
  (summarize_p3_candidates). For typical per-cycle candidate counts
  (~100–1000) that’s fine, but it’s quadratic; if you scale to 10k
  archive sizes it will hurt.

  Carmack-level note: This is fine for per-cycle; for long runs,
  store the Pareto set incrementally rather than recomputing from
  scratch.

  ———

  ### Atom A4 — Data representation & canonicalization

  #### Strong points

  - tools.structured_flatten enforces deterministic coefficient
    ordering (ai_scientist/tools/design_manipulation.py:197).
  - forward_model.compute_design_hash canonicalizes values with
    rounding (ai_scientist/forward_model.py:441). This is essential
    for caching and DB identity.
  - Symmetry enforcement when proposing perturbations is
    consistent with the masking conventions used by constellaration
    (ai_scientist/tools/design_manipulation.py:55).

  #### Risks

  - JSON-based hashing is expensive for large parameter dictionaries
    and repeated calls. You do it a lot (every candidate). For 100k
    evals, this becomes non-trivial overhead.
  - Using Python json.dumps(... allow_nan=True default) means NaN
    encodes as NaN (non-standard JSON). It’s stable for hashing, but
    it’s not portable across strict JSON parsers. This is acceptable
    internally but should be explicit.

  ———

  ### Atom A5 — Surrogate models (ML correctness + scientific
  validity)

  #### RF surrogate bundle (ai_scientist/optim/surrogate.py)

  Good

  - Feasibility-first structure: classifier for feasibility
    probability + regressors for objective and auxiliaries.
  - Proper handling of degenerate single-class training (fallback
    probability) (ai_scientist/optim/surrogate.py:346).
  - Uses P(feasible) weighting in expected value, which is correct
    in principle for rare-feasible search.

  Bad / risky

  - Dependencies (sklearn) aren’t declared in pyproject.toml. This
    is fine in your dev env but breaks portability.
  - Feature vectorization uses structured_flatten (good), but
    training labels come from feasibility thresholds; OK.
  - The scoring heuristic mixes pf, objective, constraint_distance,
    and an uncertainty proxy; it’s not calibrated, but can be
    acceptable.

  #### Neural operator surrogate (ai_scientist/optim/
  surrogate_v2.py)

  This is the most ambitious ML component: a hybrid coefficient-grid
  conv net + PointNet geometry branch + deep ensembles.

  Scientific/engineering strengths

  - Correct Fourier surface reconstruction math in
    _generate_point_cloud (uses trig separability; aligns with
    standard expansions).
  - Uses deep ensembles & bagging for epistemic uncertainty
    (Lakshminarayanan et al., 2017).
  - Uses log10 transform for QI during training (correct due to
    orders of magnitude spread).
  - Caches trig grids as buffers to avoid recomputation
    (ai_scientist/optim/surrogate_v2.py:110).

  High-severity flaws

  1. Ranking ignores feasibility probability.
     You compute prob_feasible (ai_scientist/optim/
     surrogate_v2.py:1162) and then ignore it in score
     (ai_scientist/optim/surrogate_v2.py:1197).
     For P2/P3, feasibility is the bottleneck; ignoring it wastes
     VMEC budget and kills convergence in rare-feasible regimes.
  2. Backward compatibility break: model output arity.
     The model now returns 6 heads (obj, mhd, qi, iota, mirror,
     flux), but tests and any older checkpoints expecting 4 heads
     break.
     Evidence: test failures and predict_torch slicing outputs[:6]
     (ai_scientist/optim/surrogate_v2.py:823).
  3. Target semantics ambiguity (“obj” head predicts what?)
     The NN is trained on target_values given by the runner.
     For P3, the runner chooses TargetKind.HV (ai_scientist/
     objective_types.py:78) but DB “hv” is not true hypervolume (see
     A3). So the “objective head” becomes a moving target.

  ML methodology gaps

  - No explicit train/val metrics logged, no calibration checks
    (reliability diagrams), no out-of-distribution detection. For an
    extrapolation-heavy domain (no feasible designs in dataset), you
    need epistemic uncertainty that actually correlates with error.
    Ensembles help, but only if trained and used correctly (and you
    use prob_feasible).

  ———

  ### Atom A6 — Optimization kernels (math + convergence +
  feasibility)

  #### Candidate generation

  - Uses mixtures of:
      - constraint-distance sampler from near-feasible history
        (ai_scientist/cycle_executor.py:1865)
      - near-axis sampler (ai_scientist/optim/samplers.py:84)
      - rotating ellipse template
      - generative diffusion (optional)
  - This is a reasonable exploration strategy for “feasible region
    is tiny”.

  Key risk: If your fidelity ladder stage is wrong (e.g., promote
  skipping QI), the “near-feasible history” is polluted with false
  positives and the constraint-distance sampler will push into the
  wrong manifold.

  #### ALM bridge (ai_scientist/optim/alm_bridge.py)

  - Correctly wraps
    constellaration.optimization.augmented_lagrangian_runner.objecti
    ve_constraints and provides step-wise outer iterations.
  - Uses ProcessPoolExecutor with spawn to avoid fork issues with
    JAX/torch (reasonable on macOS).

  Performance critique:

  - Spawning a new process pool per ALM step is extremely expensive
    on macOS. If budget per step is small, overhead dominates. You
    want a persistent pool or thread pool if VMEC allows it.

  #### Differentiable optimization (gradient_descent_on_inputs)

  This is currently scientifically inconsistent with benchmark
  constraints:

  - It unpacks surrogate predictions for iota/mirror/flux and then
    ignores them (ai_scientist/optim/differentiable.py:418).
  - It penalizes MHD vacuum well for all problems (including P2),
    even though P2 does not constrain vacuum well.
  - It penalizes elongation for all problems, even though P3 does
    not constrain elongation.
  - It does not enforce:
      - P2 aspect ratio constraint
      - P2 mirror ratio constraint (explicitly in benchmark)
      - P3 flux compression constraint
      - P3 mirror ratio constraint

  This matters because gradient descent is supposed to be your “fast
  inner loop” pushing designs toward feasibility before expensive
  VMEC. Right now it pushes toward a different feasibility region.

  Carmack-level performance critique
  Inside the inner loop (per step) it does:

  - NumPy array → Python list → structured_flatten → NumPy vector →
    new Torch tensor (ai_scientist/optim/differentiable.py:392–407)
    This is allocation-heavy and will thrash caches and PCIe
    transfers if device != cpu. You want to precompute the dense
    base vector once and do in-place updates on-device.

  #### Pre-relaxation (ai_scientist/optim/prerelax.py)

  Reasonable as a cheap geometry smoother. It uses mean curvature
  and elongation penalties. This is not a physics guarantee
  (VMEC feasibility is not implied), but it can reduce obviously
  pathological self-intersections.

  Physics note: mean curvature minimization is a geometric
  regularizer; it may bias away from physically relevant shaping if
  over-weighted.

  ———

  ### Atom A7 — Output pipeline & reproducibility

  World model is a strong design

  - Candidates, metrics, cycle summaries, HV archive are persisted
    to SQLite; good for long autonomous runs and postmortems.
  - Deterministic snapshots exist and are tested (tests/
    test_runner_determinism.py:88).

  But the report reproduction snippet is broken

  - ai_scientist/cycle_executor.py:905 builds a snippet missing the
    SQL execute call. This defeats the “press play and reproduce”
    workflow.

  Also, “best candidate” selection ignores feasibility

  - The cycle selects best_entry = min(latest_by_design.values(),
    key=_oriented_objective) (ai_scientist/cycle_executor.py near
    best_entry), which can pick infeasible designs if they have
    extreme objective values.
  - For scientific reporting, best should usually mean “best
    feasible” or “best trade-off” depending on problem.

  ———

  ### Atom A8 — Engineering quality, dependencies, tests,
  determinism, perf

  #### Tests (ground truth)

  Command executed:

  - pytest -q -m "not slow and not integration"

  Result:

  - 4 failed, 263 passed, 6 deselected

  Failures:

  - tests/optim/test_surrogate.py assumes 4-head model output but
    code now returns 6 heads (ai_scientist/optim/surrogate_v2.py).
  - tests/optim/test_surrogate_ensembles.py likewise.
  - tests/test_problems.py expects P3 objective = aspect ratio, but
    P3Problem.get_objective returns -L_gradB / AR (ai_scientist/
    problems.py:222).

  Interpretation:

  - This is classic “semantic drift”: code changed but tests and/
    or API contracts didn’t get updated. In a research repo this
    happens; in a production-grade autonomous scientist it’s
    unacceptable because it invalidates regressions.

  #### Dependency correctness

  pyproject.toml only declares numpy/jax/torch/pydantic, but the
  code imports:

  - sklearn, pandas, joblib (ai_scientist/optim/data_loader.py)
  - datasets (ai_scientist/datasets/sampler.py)
  - pymoo (ai_scientist/tools/hypervolume.py)
  - nevergrad (ai_scientist/optim/alm_bridge.py)
  - gymnasium (ai_scientist/rl_env.py)
  - yaml (ai_scientist/config.py)

  This is not a style issue; it’s correctness/portability. A fresh
  install of ai_scientist as a package will fail unless you happen
  to have those deps.

  #### Determinism

  - Seeds are set for numpy and Python random in run_experiment
    (ai_scientist/experiment_runner.py), but torch/JAX seeds are not
    globally controlled.
  - Process pools + non-deterministic scheduling will also change
    ordering and can change results if order-dependent logic exists.

  ———

  ## AoT Step 3/4 — Contracted findings (what matters, without the
  intermediate noise)

  ### What is scientifically correct / solid

  - Constraint margin formulas in ai_scientist/forward_model.py are
    aligned with constellaration benchmarks.
  - Fourier-to-real-space math and geometry utilities are well
    thought-out (trig separability; numerical stability fixes like
    l2 = det / l1 in elongation).
  - The overall loop architecture (history DB + staged evaluation +
    archiving) is the right shape for autonomous discovery.

  ### What is scientifically wrong or dangerously misleading
  (highest severity)

  1. Hypervolume reference mismatch vs benchmark
     Your HV score is not the same metric as the official benchmark
     when defaults are used. Any “we beat baseline HV” claim would
     be invalid until fixed/aligned.
  2. “hv” semantic drift breaks surrogate training and optimization
     goals
     The system claims “train surrogate on HV for P3”, but it’s
     not hypervolume. This undermines scientific intent and makes
     progress signals unreliable.
  3. Differentiable optimization does not enforce the benchmark
     constraints
     It ignores critical P3 constraints and includes non-benchmark
     penalties (elongation) that distort feasibility search.
  4. Neural surrogate ranking ignores feasibility probability
     This is algorithmically mismatched to rare-feasible search and
     will waste budget.
  5. Preset “p3-high-fidelity” can skip QI
     A preset labeled “high fidelity” must not omit QI for P3/P2.
     Right now it can.

  ———

  ## Critical Bugs / Errors (prioritized)

  ### High severity

  - Hypervolume ref-point mismatch: ai_scientist/constraints.py:262,
    ai_scientist/tools/hypervolume.py:33 vs benchmark
    constellaration/src/constellaration/problems.py:365.
  - “hv” semantic drift: ai_scientist/fidelity_controller.py:266,
    ai_scientist/memory/repository.py:1076, ai_scientist/
    objective_types.py:62.
  - Gradient descent constraints wrong/missing: ai_scientist/
    optim/differentiable.py:418 and ai_scientist/optim/
    differentiable.py:455.
  - Neural surrogate ranking ignores feasibility: ai_scientist/
    optim/surrogate_v2.py:1188–ai_scientist/optim/
    surrogate_v2.py:1203.
  - Preset “p3_high_fidelity” does not force fidelity_ladder:
    p3/p3: ai_scientist/config.py:410 and ai_scientist/tools/
    evaluation.py:290.
  - Test suite failures indicate broken contracts: tests/optim/
    test_surrogate.py:10, tests/test_problems.py:180.

  ### Medium severity

  - Report reproduction snippet broken: ai_scientist/
    cycle_executor.py:905.
  - EvaluationResult.dominates() always False: ai_scientist/
    forward_model.py:396.
  - Best candidate selection can ignore feasibility, impacting
    reporting and preference logs.

  ### Low severity

  - Missing optional dependency declarations in pyproject.toml
    (still important for portability, but doesn’t change physics
    truth).
  - Minor stage naming confusion and repeated Coordinator
    construction overhead.

  ———

  ## Physics & mathematics validation (with relevant equations)

  ### Boundary representation

  Your Fourier evaluation uses:

  - ( R(\theta,\zeta) = \sum_{m,n} R_{mn} \cos(m\theta - n
    N_{fp}\zeta) )
  - ( Z(\theta,\zeta) = \sum_{m,n} Z_{mn} \sin(m\theta - n
    N_{fp}\zeta) )

  The code implements trig separability:

  - ( \cos(a-b) = \cos a\cos b + \sin a\sin b )
  - ( \sin(a-b) = \sin a\cos b - \cos a\sin b )

  This is implemented correctly in both:

  - ai_scientist/optim/geometry.py:1 (real-space generation)
  - ai_scientist/optim/surrogate_v2.py:157 (cached-grid point cloud
    generation)

  ### Constraint normalization

  Matches constellaration normalized violations:

  - For bound (x \le x^): ( (x-x^)/|x^*| )
  - For bound (x \ge x^): ( (x^-x)/|x^*| )
  - For QI: ( (\log_{10}(QI)-\log_{10}QI^)/|\log_{10}QI^| )

  These appear correct in ai_scientist/forward_model.py:522 and
  match constellaration/src/constellaration/problems.py:113.

  ### Hypervolume correctness depends on the reference point

  Given minimization vector ( (-g, A) ) where (g) is gradient
  and (A) is aspect ratio, hypervolume uses a reference point (r)
  that must match the benchmark. Using a different reference point
  changes the score and therefore changes optimization incentives.

  Reference papers:

  - Deb et al., 2002 (NSGA-II, nondominated sorting)
  - Zitzler & Thiele, 1998/1999 (hypervolume as Pareto indicator)

  ———

  ## ML model assessment (surrogate validity & risks)

  - Surrogate-v2 architecture is plausible (spectral conv + point
    cloud encoder + ensembles), but the ranking and target semantics
    currently prevent it from functioning as a reliable rare-
    feasibility guide.
  - No explicit calibration/validation loop exists in the runner
    (no held-out dataset, no error vs uncertainty plots). Deep
    ensembles are only useful if your uncertainty correlates with
    error (Lakshminarayanan et al., 2017). Right now prob_feasible
    is computed but unused in ranking.

  ———

  ## Performance & scalability notes (Carmack-style)

  - gradient_descent_on_inputs performs repeated Python→NumPy→Torch
    conversions per step (ai_scientist/optim/differentiable.py:392).
    This is allocation-heavy and will dominate wall time for “fast”
    inner loops. Precompute and stay on-device.
  - Process pools are created repeatedly (batch evaluation, ALM
    step). On macOS spawn is expensive; for small batches this will
    be dominated by process startup.
  - Hypervolume and Pareto extraction are O(N²) per cycle. Fine for
    N~100, risky for N~10k.

  ———

  ## Concrete prioritized recommendations (to reach production-grade
  autonomous feasible discovery)

  ### P0 — Correctness integrity (must do before trusting results)

  1. Unify hypervolume reference point with benchmark
     Set HV reference point exactly as constellaration defines it,
     or explicitly rename your metric to avoid false claims.
     Target alignment: constellaration/src/constellaration/
     problems.py:365.
  2. Rename and separate “hv” fields everywhere
      - candidate_hv_proxy (if you keep grad-1)
      - cycle_hv_score (true hypervolume)
      - hv_reference_point stored with every cycle
        Then ensure TargetKind.HV really corresponds to a meaningful
        candidate-level target (e.g., marginal HV contribution).
  3. Fix differentiable optimization constraint set by problem
      - P2: enforce aspect ratio, iota, mirror ratio, elongation,
        log10(QI).
      - P3: enforce iota, mirror ratio, flux compression, vacuum
        well, log10(QI).
      - P1: enforce aspect ratio, triangularity, iota (and
        optionally elongation if used as objective).
        Remove penalties that are not in the benchmark unless they
        are explicitly labeled as “regularizers”.
  4. Use feasibility probability in neural surrogate ranking
     A bare minimum scoring should look like:
      - score = prob_feasible * (base_score + exploration_bonus)
        - penalty_terms
        Without this, rare-feasible search will not converge.
  5. Fix “p3-high-fidelity” preset fidelity ladder
     Any P2/P3 “high fidelity” preset must run stage "p2"/"p3" (or
     equivalent settings that compute QI), not "promote" if promote
     skips QI.

  ### P1 — Reliability and reproducibility
        a held-out set
  9. Add “active learning” acquisition functions:
      - e.g., expected feasibility improvement, constrained EI/UCB,
        or probability-of-feasibility–weighted EI.
  10. Use near-axis expansion + ALM as the feasibility “funnel”,
     then do surrogate-guided local refinement.

  ———

  ## Suggested next experiments / ablations (high information gain)

  1. HV reference ablation: compute HV with [1,20] vs [-1,20] on the
     same archive and quantify rank-order changes.
  2. Feasibility-weighted ranking ablation: in neural surrogate
     ranking, compare:
      - score = base_score only
      - score = prob_feasible * base_score
      - constrained UCB: prob_feasible * (mean + κ·std)
  3. Constraint-complete GD ablation: run gradient descent with (a)
     current penalties vs (b) benchmark-exact penalties and measure
     feasibility improvements.
  4. QI computation stage ablation: run with promote skipping QI vs
     promote including QI; quantify false-feasible rate.

  ———

  ## Key literature citations (for the claims above)

  - Hirshman & Whitson (1983): VMEC-style variational equilibrium
    methods.
  - Boozer (1981): Boozer coordinates (basis for QI evaluation in
    Boozer space).
  - Cary & Shasharina (1997): omnigenity / confinement properties
    (context for QI).
  - Garren & Boozer (1991): near-axis expansion for stellarator
    design (seed generation rationale).
  - Deb et al. (2002): NSGA-II, nondominated sorting and crowding
    distance.
  - Zitzler & Thiele (1998/1999): hypervolume indicator as Pareto
    set measure.
  - Lakshminarayanan et al. (2017): deep ensembles for predictive
    uncertainty.
  - Qi et al. (2017): PointNet (point cloud encoding used in
    surrogate-v2).
  - Shoemake (1992): uniform random rotations via quaternions (used
    in equivariance.random_rotation_matrix).

  ———

  If you want, I can follow up with a “fix plan” (still AoT style)
  that’s surgically limited to:

  - correcting HV reference + “hv” semantics,
  - making differentiable optimization constraint-complete per
    problem,
  - using prob_feasible in neural surrogate ranking,
  - repairing the reproduction snippet,
  - updating the failing tests to restore CI correctness.
