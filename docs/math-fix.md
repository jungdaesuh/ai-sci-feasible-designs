Implementation Plan: Correctness Bug Fixes (v3.2 — APPROVED)
Date: 2025-12-13
Status: ✅ APPROVED — Ready for Implementation
Revision Notes: Final amendments added per reviewer approval

Changes Since v3 (Final Critique) + v3.2 Amendments
Critique Point	Fix
softplus(0)≈0.693 destroys raw QI scale	Use abs() + eps for raw QI positivity
SurrogateBundle has no _objective_regressor	Gate on self._regressors.get("objective")
RL sanity table sign errors	Fixed arithmetic in table
Missing call site list for stage-aware changes	Explicit list provided
v3.2: RL obj_val becomes score, not AR	Compute AR from params via geometry
v3.2: cycle_executor scope issue	Use fm_settings.stage (in scope)
Confirmed Decisions
Screen Stage Constraints: Option B (reduced constraint set)
Coordinator Retraining Target: P2/P3 → score (maximize), P1 → objective (minimize)
RL Goal: Option C (feasibility priority + continuous improvement)
Proposed Changes (Final)
P0 Critical Fixes
[MODIFY] 
forward_model.py
Bug: 
max_violation()
 silently treats NaN as 0.0

Fix:

import math
def max_violation(margins: Mapping[str, float]) -> float:
    """Return maximum constraint violation.
    
    If any margin is non-finite (NaN/inf), returns inf (conservative infeasibility).
    """
    if not margins:
        return float("inf")
    
    # If ANY margin is non-finite, entire feasibility is undefined → infeasible
    for value in margins.values():
        if not math.isfinite(value):
            return float("inf")
    
    return float(max(0.0, max(margins.values())))
Note: No 
stage
 parameter added to avoid call-site migration. Logging is optional internal detail.

Stage-aware constraint checking:

def compute_constraint_margins(
    metrics: Any,
    problem: str,
    *,
    stage: str = "high",  # NEW: optional stage parameter
) -> Dict[str, float]:
    """Compute margins for constraints based on problem and stage.
    
    Args:
        metrics: Metrics object or dict
        problem: Problem type (p1, p2, p3)
        stage: Fidelity stage. Low-fidelity stages skip expensive constraints.
               Valid values: "screen", "low", "default" (low fidelity)
                            "promote", "high", "p2", "p3" (high fidelity)
    """
    metrics_map = (
        metrics.model_dump() if hasattr(metrics, "model_dump") else dict(metrics)
    )
    problem_key = problem.lower()
    is_low_fidelity = stage.lower() in ("screen", "low", "default")
    
    # ... existing margin computation ...
    
    # For P3:
    if problem_key.startswith("p3"):
        margins = {}
        
        # Geometric constraints (always required)
        margins["edge_rotational_transform"] = 0.25 - float(
            metrics_map.get("edge_rotational_transform_over_n_field_periods", float("nan"))
        )
        margins["edge_magnetic_mirror_ratio"] = float(
            metrics_map.get("edge_magnetic_mirror_ratio", float("nan"))
        ) - 0.25
        
        # Physics constraints (only at high fidelity)
        if not is_low_fidelity:
            # Vacuum well
            well = metrics_map.get("vacuum_well")
            margins["vacuum_well"] = -float(well) if well is not None else float("inf")
            
            # Flux compression
            flux_value = metrics_map.get("flux_compression_in_regions_of_bad_curvature")
            margins["flux_compression"] = float(flux_value) - 0.9 if flux_value is not None else float("inf")
            
            # QI (log scale)
            margins["qi_log10"] = _log10_margin(-3.5)
        
        return margins
Call Sites to Update:

File	Line	Current Call	New Call
forward_model.py
(internal)	N/A	Primary definition
backends/real.py
117	
compute_constraint_margins(metrics, settings.problem)
compute_constraint_margins(metrics, settings.problem, stage=settings.stage)
backends/mock.py
179	
compute_constraint_margins(metrics, problem)
compute_constraint_margins(metrics, problem, stage=stage)
cycle_executor.py
1395	tools.compute_constraint_margins(metrics, "p3")	tools.compute_constraint_margins(metrics, self.config.problem, stage=fm_settings.stage)
cycle_executor.py
1454	tools.compute_constraint_margins(metrics, "p3")	tools.compute_constraint_margins(metrics, self.config.problem, stage=fm_settings.stage)
cycle_executor.py
1705	tools.compute_constraint_margins(metrics, problem_type)	tools.compute_constraint_margins(metrics, problem_type, stage=stage) (if stage in scope)
tools/evaluation.py
251-252	Wrapper function	Add 
stage
 parameter and forward
tools/evaluation.py
312, 317	Internal calls	Add 
stage
 from context
[MODIFY] 
coordinator.py
Bug 1: Always uses minimize_objective=True
Bug 2: Wrong label structure

Fix (unchanged from v3, correct):

def _periodic_retrain(self, cycle: int, experiment_id: int, candidates: List[Dict[str, Any]]) -> None:
    # ...
    
    for cand in top_elites:
        params = cand.get("params", cand.get("candidate_params"))
        if params is None:
            continue
        
        eval_data = cand.get("evaluation", {})
        actual_metrics = eval_data.get("metrics", {})
        
        # Skip if metrics are missing (don't train on fabricated targets)
        if not actual_metrics:
            continue
        
        problem = self.cfg.problem.lower()
        if problem.startswith("p1"):
            target = eval_data.get("objective")
            minimize_obj = True
        else:
            # P2/P3: use score (higher = better)
            target = eval_data.get("score")
            if target is None:
                # Fallback: compute score from metrics (only if metrics exist!)
                grad = actual_metrics.get("minimum_normalized_magnetic_gradient_scale_length")
                aspect = actual_metrics.get("aspect_ratio")
                if grad is not None and aspect is not None:
                    target = float(grad) / max(1.0, float(aspect))
                else:
                    continue  # Skip: can't compute target
            minimize_obj = False
        
        if target is None:
            continue
        
        metrics_list.append({
            "candidate_params": params,
            "metrics": actual_metrics,
        })
        target_values.append(float(target))
    
    if metrics_list:
        self.surrogate.fit(metrics_list, target_values, minimize_objective=minimize_obj)
[MODIFY] 
differentiable.py
Bug 1: QI threshold too loose
Bug 2: softplus destroys raw QI scale

Corrected Fix (abs() + eps for raw QI):

The surrogate is trained on raw QI values. softplus(x) where x is small (e.g., 1e-4) returns ≈0.693, destroying the scale.

Instead, use abs() + eps which:

Preserves scale: abs(1e-4) + eps ≈ 1e-4
Has gradients everywhere except exactly 0: d/dx |x| = sign(x)
Enforces non-negativity
# Problem-dependent QI thresholds (raw values)
QI_THRESHOLD_P2 = 1e-4   # log10(qi) <= -4.0
QI_THRESHOLD_P3 = 3.16e-4  # log10(qi) <= -3.5
QI_EPS = 1e-12  # Small epsilon for numerical stability
def _get_qi_threshold(problem: str) -> float:
    if problem.lower().startswith("p2"):
        return QI_THRESHOLD_P2
    return QI_THRESHOLD_P3
# In optimization loop:
qi_threshold = _get_qi_threshold(problem)
# Use abs() + eps for scale-preserving positivity
# This keeps raw QI values in correct order of magnitude
# abs(1e-4) ≈ 1e-4, not 0.693 like softplus would give
qi_raw = pred_qi.squeeze()
qi_positive = qi_raw.abs() + QI_EPS  # Differentiable (except at 0), scale-preserving
c2 = torch.relu(qi_positive + beta * s_qi - qi_threshold)
Mathematical Verification:

>>> import torch
>>> qi = torch.tensor([1e-4, 1e-5, -1e-4])  # Raw QI values (last one negative from surrogate)
>>> qi.abs() + 1e-12
tensor([1.0000e-04, 1.0000e-05, 1.0000e-04])  # Correct scale preserved!
>>> import torch.nn.functional as F
>>> F.softplus(qi)
tensor([0.6932, 0.6932, 0.6931])  # Wrong! All ≈0.693
Bug 3: Always minimizes, wrong for P2

Fix (unchanged, correct):

def optimize_alm_inner_loop(
    ...,
    problem: str = "p3",  # NEW PARAMETER
) -> np.ndarray:
    
    qi_threshold = _get_qi_threshold(problem)
    
    for _ in range(steps):
        # ...
        
        # Objective term (pessimistic for both directions)
        if problem.lower().startswith("p2"):
            # P2: MAXIMIZE gradient → minimize -gradient
            # LCB: mean - β·std (pessimistic for maximization)
            obj_term = -(pred_obj.squeeze() - beta * std_obj.squeeze())
        else:
            # P1/P3: MINIMIZE objective
            # Pessimistic: mean + β·std
            obj_term = pred_obj.squeeze() + beta * std_obj.squeeze()
        
        # QI positivity with scale preservation
        qi_positive = pred_qi.squeeze().abs() + QI_EPS
        
        c2 = torch.relu(qi_positive + beta * s_qi - qi_threshold)
Call site update (cycle_executor.py:1369):

x_new_np = differentiable.optimize_alm_inner_loop(
    x_initial=np.array(state.x),
    scale=np.array(scale),
    surrogate=surrogate_model,
    alm_state=alm_state_dict,
    n_field_periods_val=initial_params_map.get("n_field_periods", 1),
    problem=self.config.problem,  # NEW
    steps=budget_per_step,
)
P1 High Priority Fixes
[MODIFY] 
rl_env.py
Bug 1: Uses raw QI, dead zone once feasible
Bug 2 (v3.2): After retraining change, obj_val becomes score (grad/aspect), not AR

Fix (Option C shaping + compute AR from params):

IMPORTANT

Since surrogate now predicts "score" for P2/P3, obj_val is no longer aspect ratio. We must compute AR directly from params via the geometry module.

from ai_scientist.optim import geometry
import torch
def _compute_score(self, vec: np.ndarray) -> float:
    # ... surrogate prediction gives obj_val, mhd_val, qi_val ...
    
    # QI in log scale (consistent with benchmark)
    QI_CLAMP_FLOOR = 1e-10
    qi_clamped = max(QI_CLAMP_FLOOR, qi_val)
    log_qi = np.log10(qi_clamped)
    qi_target = self.target_metrics.get("log10_qi_threshold", -4.0)
    
    qi_feasibility_penalty = max(0.0, log_qi - qi_target)
    qi_continuous = (log_qi - qi_target)
    
    mhd_violation = max(0.0, -mhd_val)
    mhd_continuous = -mhd_val
    
    # v3.2 FIX: Compute AR directly from params, NOT from obj_val
    # (obj_val is now score=grad/aspect after retraining change)
    ar_target = self.target_metrics.get("aspect_ratio", 8.0)
    try:
        r_cos = torch.tensor(self.current_params["r_cos"], dtype=torch.float32).unsqueeze(0)
        z_sin = torch.tensor(self.current_params["z_sin"], dtype=torch.float32).unsqueeze(0)
        nfp = self.current_params.get("n_field_periods", 1)
        computed_ar = float(geometry.aspect_ratio(r_cos, z_sin, nfp).item())
    except Exception:
        computed_ar = ar_target  # Fallback: no AR penalty if computation fails
    
    ar_deviation = abs(computed_ar - ar_target)
    
    cost = 0.0
    cost += 10.0 * qi_feasibility_penalty
    cost += 0.5 * qi_continuous
    cost += 5.0 * mhd_violation
    cost += 0.3 * mhd_continuous
    cost += 1.0 * ar_deviation
    
    return -cost  # Higher is better
Corrected Reward Scale Sanity Table:

Scenario	QI Feas (10x)	QI Cont (0.5x)	MHD (5x+0.3x)	AR (1x)	Cost	Reward (-cost)
Baseline: QI=-4, MHD=0, AR=8	0	0	0	0	0	0
QI=-3 (infeasible)	+10	+0.5	0	0	+10.5	-10.5
QI=-5 (better feasible)	0	-0.5	0	0	-0.5	+0.5
MHD=-0.05 (infeasible)	0	0	+0.25+0.015	0	+0.265	-0.265
MHD=+0.01 (feasible, good)	0	0	0-0.003	0	-0.003	+0.003
AR=12 (off target)	0	0	0	+4	+4	-4
Interpretation:

Negative reward = bad (cost positive)
Positive reward = good (cost negative)
Feasibility penalties dominate (10x, 5x) ✓
Continuous improvement gives small reward (~0.5) ✓
[MODIFY] 
surrogate.py
Bug: Ignores predicted objective

Corrected Fix (gate on actual _regressors dict):

From code inspection:

self._regressors is a dict[str, RandomForestRegressor]
Objective regressor is stored as self._regressors["objective"] (line 371)
self._trained means fit ran, but regressor might still be missing
def rank_candidates(
    self,
    candidates: Sequence[Mapping[str, Any]],
    *,
    minimize_objective: bool,
    exploration_ratio: float = 0.0,
) -> list[SurrogatePrediction]:
    # ...
    
    exploration_weight = max(0.0, float(exploration_ratio)) * 0.1
    
    if not self._trained:
        return self._heuristic_rank(candidates, minimize_objective)
    
    # ... prediction code ...
    
    # Gate objective weight on actual regressor existence
    # self._regressors is dict[str, RandomForestRegressor]
    objective_regressor_trained = (
        self._trained 
        and self._regressors.get("objective") is not None
    )
    
    if objective_regressor_trained:
        training_size = self._last_fit_count
        MIN_SAMPLES_FOR_OBJ = 32
        obj_weight = min(1.0, training_size / (MIN_SAMPLES_FOR_OBJ * 2))
    else:
        obj_weight = 0.0
    
    ranked: list[SurrogatePrediction] = []
    for i, candidate in enumerate(candidates):
        pf = prob[i]
        obj = objs[i]
        
        constraint_distance = float(candidate.get("constraint_distance", 0.0))
        constraint_distance = max(0.0, constraint_distance)
        
        # Classifier entropy (binary classification uncertainty proxy)
        uncertainty = float(pf * (1.0 - pf))
        
        # Expected value: feasibility-weighted objective
        base_score = self._expected_value(pf, obj, minimize_objective)
        
        # Composite score with ramped objective contribution
        score = (
            obj_weight * base_score 
            + (1.0 - obj_weight) * float(pf)
            - constraint_distance 
            + exploration_weight * uncertainty
        )
        
        ranked.append(SurrogatePrediction(...))
    
    return sorted(ranked, key=lambda item: item.expected_value, reverse=True)
P2 Low Priority Fixes (Unchanged)
problems.py
: Update constraint names
forward_model.py
: Remove duplicate cache_key
Verification Plan
Unit Tests
def test_max_violation_nonfinite():
    from ai_scientist.forward_model import max_violation
    assert max_violation({"a": 0.5, "b": float("nan")}) == float("inf")
    assert max_violation({"a": 0.5, "b": -0.1}) == 0.5
def test_screen_stage_skips_physics_constraints():
    from ai_scientist.forward_model import compute_constraint_margins
    
    # Screen stage: no flux/vacuum_well/QI required
    screen_metrics = {
        "edge_rotational_transform_over_n_field_periods": 0.3,
        "edge_magnetic_mirror_ratio": 0.2,
        # No flux, vacuum_well, qi
    }
    margins = compute_constraint_margins(screen_metrics, "p3", stage="screen")
    
    # Only geometric constraints returned
    assert "flux_compression" not in margins
    assert "vacuum_well" not in margins
    assert "qi_log10" not in margins
    assert "edge_rotational_transform" in margins
def test_qi_abs_preserves_scale():
    import torch
    QI_EPS = 1e-12
    
    qi = torch.tensor([1e-4, 1e-5, -1e-4], requires_grad=True)
    qi_positive = qi.abs() + QI_EPS
    
    # Scale preserved
    assert qi_positive[0].item() == pytest.approx(1e-4, rel=1e-6)
    assert qi_positive[1].item() == pytest.approx(1e-5, rel=1e-6)
    
    # Gradient flows
    loss = qi_positive.sum()
    loss.backward()
    assert qi.grad is not None
    assert all(g != 0 for g in qi.grad)
def test_surrogate_objective_weight_gating():
    from ai_scientist.optim.surrogate import SurrogateBundle
    
    bundle = SurrogateBundle()
    
    # Before training: no gating
    assert bundle._regressors.get("objective") is None
    
    # After training
    bundle.fit(mock_metrics_list, mock_targets, minimize_objective=True)
    assert bundle._regressors.get("objective") is not None
Integration Tests
def test_p2_optimization_increases_gradient():
    """P2 differentiable optimizer should increase gradient scale length."""
    # Setup with problem="p2"
    # Run optimize_alm_inner_loop
    # Assert final gradient > initial gradient
def test_screen_stage_produces_finite_feasibility():
    """Screen stage should produce some candidates with finite feasibility."""
    # Run FidelityController.evaluate_stage(stage="screen")
    # Assert at least some results have feasibility < inf
Summary Checklist
 QI uses abs() + eps (not softplus) for raw scale preservation
 SurrogateBundle gates on self._regressors.get("objective")
 RL sanity table arithmetic matches actual cost definition
 Explicit call site list for stage-aware 
compute_constraint_margins()
 Screen stage constraint policy: geometric only at low fidelity
 P2/P3 retraining uses score (maximize), P1 uses objective (minimize)
 Coordinator skips rows with missing metrics
 Differentiable optimization threads 
problem
 through call sites
 v3.2: RL env computes AR from params via geometry.aspect_ratio()
 v3.2: Call sites use fm_settings.stage (in scope)
Approval Status
✅ APPROVED — All critique points resolved + v3.2 amendments added:

✅ QI positivity: abs() + eps preserves raw scale
✅ SurrogateBundle: gates on self._regressors.get("objective")
✅ RL table: arithmetic corrected (positive cost = negative reward)
✅ Call sites: 7 locations listed, using fm_settings.stage (in scope)
✅ v3.2: RL env computes AR from params via geometry.aspect_ratio()
✅ v3.2: Call sites use fm_settings.stage instead of out-of-scope variable
Ready for implementation.