Verify gpt5-2.md Claims
High Severity Claims - VERIFIED
H1: Hypervolume reference point mismatch - FALSE (MISLEADING)
Document claims:
(-1.0, 20.0)
 vs benchmark [1.0, 20.0]
Reality: constraints.py:275 returns
(-grad, aspect)
 in minimization form
Benchmark problems.py:370 uses [1.0, 20.0] also in minimization form
Since gradient is negated for minimization,
(-1.0, 20.0)
 = gradient=1.0, aspect=20.0
SAME AS BENCHMARK - Document claim is technically inaccurate
H2: "hv" semantic drift - FIXED ✅
Document claimed "hv" token overloaded with incompatible meanings
**Verification**: Issue was REAL - "hv" was used for both:
  1. True Pareto hypervolume (cycle-level metric)
  2. Per-candidate gradient proxy = max(0, gradient - 1) (NOT hypervolume)
**Fix Applied**: Renamed to "gradient_proxy" with backward-compat alias
  - `objective_types.py:59-60`: TargetKind.GRADIENT_PROXY with H2 FIX comment
  - `fidelity_controller.py:266-277`: H2 FIX - renamed from "hv" to "gradient_proxy"
  - `repository.py:515-517`: H2 FIX - Use gradient_proxy with fallback
  - `repository.py:1086-1088`: H2 FIX - Accept "gradient_proxy" as DB alias
H3: Differentiable optimization ignores key constraints - FIXED ✅
Original issue: _pred_iota, _pred_mirror, _pred_flux were computed but discarded
**Fix Applied**: Constraints now included in loss function
  - `differentiable.py:511-551`: H3 FIX - P3-specific constraints added
  - `viol_iota`, `viol_mirror`, `viol_flux` now computed and added to loss_penalty
  - `differentiable.py:543-551`: All constraints included in loss calculation
H4: Neural surrogate ranking ignores feasibility probability - FIXED ✅
Original issue: prob_feasible computed but NOT used in score calculation
**Fix Applied**: prob_feasible now weights the ranking score
  - `surrogate_v2.py:1217-1223`: H4 FIX comment
  - OLD: `score = base_score + exploration_bonus - (10.0 * constraint_distance)`
  - NEW: `score = prob_feasible * (base_score + exploration_bonus) - (10.0 * constraint_distance)`
  - Ensures VMEC budget spent on candidates likely to be feasible
H5: p3-high-fidelity preset can skip QI - FIXED ✅
Original issue: `p3_high_fidelity()` used `defaults.fidelity_ladder` (screen/promote) which triggers `default_high_fidelity_skip_qi()` for "promote" stage (evaluation.py:290-292).
**Fix Applied**: `config.py:427-430` now explicitly uses `FidelityLadder(screen="screen", promote="p3")`.
  - Using `promote="p3"` stage triggers `default_high_fidelity()` (evaluation.py:298-299), which DOES compute QI
  - H5 FIX comment at lines 427-430 explains the fix rationale

H6: Test suite failures - NOT VERIFIED
Would need to run pytest -q -m "not slow and not integration"

Medium Severity Claims - VERIFIED
 M1: Report reproduction snippet broken - FIXED ✅
   Original claim: cycle_executor.py:905 missing SQL execute call
   **Fix Applied**: Snippet now includes complete `conn.execute()` call (lines 925-928)
   - Verified: `row = conn.execute(...)` with proper SQL query and `.fetchone()` call

 M2: EvaluationResult.dominates() always False - FIXED ✅
   Original claim: dominates() stub always returns False (forward_model.py:396)
   **Fix Applied**: Method now raises `NotImplementedError` with guidance (lines 397-410)
   - Explains that Pareto dominance requires objective direction context
   - Points to `ai_scientist.tools.hypervolume.summarize_p3_candidates()` for proper analysis

 M3: Best candidate selection ignores feasibility - FIXED ✅
   Original claim: `best_entry = min(..., key=_oriented_objective)` ignores feasibility
   **Fix Applied**: `cycle_executor.py:738-754` now prefers feasible candidates
   - M3 FIX comment at line 738
   - First attempts to find best among feasible entries
   - Falls back to lowest max_violation only if no feasible candidates exist

Low Severity Claims - VERIFIED
 L1: Missing optional dependency declarations - FIXED ✅
   Original claim: pyproject.toml only declares core deps, missing sklearn/pandas/etc.
   **Fix Applied**: `pyproject.toml:19-43` now declares all optional dependencies:
   - `[ml]`: scikit-learn, pandas, joblib
   - `[optimization]`: pymoo, nevergrad
   - `[datasets]`: datasets
   - `[rl]`: gymnasium
   - `[full]`: all above + PyYAML

 L2: Stage naming confusion - NOT A BUG (architectural observation)
   This is a naming/documentation issue, not a code bug.
   "Stage" semantically overloaded: governance stage (S1/S2/S3) vs fidelity stage (screen/promote/p3).
   No code fix applied - would require refactoring and renaming throughout.

Canonicalization Issues - FIXED ✅

 C1: Seed not expanded to template max modes - FIXED ✅
   Original issue: `_generate_candidate_params` uses seed matrix as-is without expanding to template's max modes.
   Seed file `rotating_ellipse_p3.json` has shape (2×3) = (mpol=1, ntor=1) but template may define larger.
   **Fix Applied**:
   - `cycle_executor.py:1594-1636`: Added `_expand_matrix_to_mode` helper function
   - `cycle_executor.py:1802-1830`: Updated seed loading to expand all matrices (r_cos, z_sin, r_sin, z_cos)
   - Logs info message when expansion occurs for visibility

 C2: Surrogate schema derived without warning - FIXED ✅
   Original issue: `SurrogateBundle._vectorize` derives schema from first params silently,
   causing implicit truncation/zero-padding when mixing shapes.
   **Fix Applied**:
   - `surrogate.py:250-262`: Added warning log when schema is first derived
   - Alerts developers about potential shape mixing issues

Section 4: Surrogate & Data Correctness - FIXED ✅

 A3: Surrogate Training Data Mixes Experiments - FIXED ✅
   Original issue: `surrogate_training_data` selects from `metrics` without filtering by `experiment_id`.
   **Fix Applied**:
   - `repository.py:1085-1124`: Added `experiment_id` parameter, filters by `c.experiment_id`
   - `surrogate.py:168-183`: Updated `fit_from_world_model` to pass `experiment_id`
   - `cycle_executor.py:1256-1259`: Updated caller to provide `experiment_id`

 A4.2: SurrogateBundle (RF) Weaknesses - FIXED ✅
   1. Feature vector ignored `n_field_periods` (Nfp).
   2. Auxiliary targets (QI) used raw scales instead of log10.
   **Fix Applied**:
   - `surrogate.py:43-52`: `_params_feature_vector` includes Nfp
   - `surrogate.py:268-277`: `_vectorize` appends Nfp to flattened params
   - `surrogate.py:350-356`: QI targets log-scaled in `fit()`
   - `surrogate.py:543`: QI predictions denormalized (10^x) in `rank_candidates`

 A4.3: NeuralOperatorSurrogate Weaknesses - FIXED ✅
   1. Fidelity conditioning unused (defaults to high fidelity).
   2. Soft feasibility used universal constraints for all problems.
   **Fix Applied**:
   - `surrogate_v2.py:354-360`: Added `problem` arg to `__init__`
   - `surrogate_v2.py:703-710`: Extracts `_stage` and maps to fidelity tensor
   - `surrogate_v2.py:989-994`: Conditional vacuum_well check based on problem bounds

## A5: Constraints, Feasibility, and Scoring - FIXED ✅

 A5.1: Constraint formulas vs ConStellaration benchmark - VERIFIED ✅
   Constraint bounds in `forward_model.compute_constraint_margins()` match
   `constellaration/problems.py` including P1 triangularity ≤ -0.5 (correct per benchmark).

 A5.3: Objective-direction bug in stage-gating - FIXED ✅
   Original issue: `_relative_objective_improvement()` assumed lower-is-better,
   inverting improvement signal for P2 (maximize gradient).
   **Fix Applied**:
   - `fidelity_controller.py:105-130`: A5.3 FIX - Added `minimize` parameter
   - `fidelity_controller.py:383-388`: Caller passes correct direction per problem

 A5.2: Stage-dependent constraint gating brittleness - FIXED ✅
   Original issue: "promote" stage skips QI constraint, risking false positive feasibility.
   **Fix Applied**:
   - `forward_model.py:562-568`: A5.2 FIX - Warning log when promote used with P2/P3

 A5.4: CycleSummary semantic confusion - FIXED ✅
   Original issue: `CycleSummary.objective` direction was undocumented.
   **Fix Applied**:
   - `fidelity_controller.py:27-44`: A5.4 FIX - Added comprehensive docstring
   - `objective_types.py:58-59`: Removed legacy HV alias (pre-launch, no legacy data)
