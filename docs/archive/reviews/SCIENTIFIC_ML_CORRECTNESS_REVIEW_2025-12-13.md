# Scientific & ML Correctness Review
Date: 2025-12-13
Scope: Math, physics, AI/ML, and logic correctness for `ai_scientist/`
Repo SHA: `ae4ed74ffa6ae30d9a46447c484948bb55e1eaa2`
Bundled `constellaration/` SHA: `3c97274d189335a7c827d482ad5a939dba540116`

## Executive Summary

**Verdict:** The core architecture is coherent and many key formulas and algorithmic building blocks are implemented in standard ways. However, there are a few **high-impact correctness risks** where the current implementation can optimize the *wrong* objective/constraints (especially for P2/P3), or can incorrectly mark incomplete/invalid evaluations as feasible.

### What is solid
- **Fourier surface evaluation** and indexing are consistent with a VMEC-style convention and the code is internally self-consistent.
- **Constraint margin sign convention** is consistent in the centralized forward model: positive means violation.
- **P3 hypervolume framing** uses the standard trick: maximize gradient + minimize aspect ratio is converted to minimizing `(-gradient, aspect)`.
- **DDPM-style diffusion sampling** and **PPO mechanics** are idiomatic, readable, and match common reference implementations.

### What needs attention (highest impact)
1. **Feasibility can be overstated when metrics are missing/NaN.**
   - NaN margins can be treated as non-violations because of how `max()` behaves in Python.
   - Some missing metrics (e.g., flux compression) default to “no violation”.
2. **Objective direction and labels are inconsistent across modules (especially P2).**
   - Some places treat P2 as maximize (natural objective), others negate for minimization.
   - Coordinator retraining currently hardcodes minimization and likely passes the wrong training payload shape.
3. **RL environment and differentiable optimizer are not benchmark-aligned for QI.**
   - Benchmarks constrain `log10(qi)` (e.g., `<= -4` / `<= -3.5`), but several ML components use raw `qi` with very different thresholds.

The rest of this document provides the concrete technical findings and recommended next steps.

---

## 1) Physics / Math Correctness

### 1.1 Fourier Surface Representation (Mostly Correct)
**Location:** `ai_scientist/optim/geometry.py` (`fourier_to_real_space`, `batch_fourier_to_real_space`)

The implemented parameterization matches a common VMEC-style representation:

- \( R(\theta, \zeta) = \sum_{m,n} R_{mn} \cos(m\theta - nN_{fp}\zeta) \)
- \( Z(\theta, \zeta) = \sum_{m,n} Z_{mn} \sin(m\theta - nN_{fp}\zeta) \)

**What is correct here:**
- Index mapping `n = n_idx - ntor` correctly maps array columns to signed toroidal mode number.
- The phase `m*theta - n*Nfp*zeta` is a standard convention and is consistent throughout this repo.
- Using only `r_cos` + `z_sin` corresponds to a stellarator-symmetric subspace.

**Caveat (wording precision):**
- It is safest to say “VMEC-style convention” rather than “VMEC++ exactly”, because sign conventions for the toroidal phase can differ across communities and the code itself notes this is convention-dependent.

### 1.2 Constraint Margins (Mostly Correct, But Missing/NaN Handling is Risky)
**Location:** `ai_scientist/forward_model.py` (`compute_constraint_margins`, `max_violation`)

The constraint margin convention is coherent:
- Upper bound constraints: `metric - limit`
- Lower bound constraints: `limit - metric`
- **Positive margin means violation**, and feasibility is based on `max(0, margins...)`.

**Key correctness risk (high impact):**
- **NaN margins can silently look feasible.** In Python, `max(0.0, float("nan"))` returns `0.0`. If any constraint margin becomes NaN (e.g., due to missing metrics or a propagation bug), it may be treated as “no violation”.
- **Missing flux compression is treated as satisfied** for P3: if `flux_compression_in_regions_of_bad_curvature` is absent, the code assigns `flux_margin = 0.0`, which can incorrectly pass feasibility.

**Recommendation:**
- Treat **any missing or non-finite metric used in constraints as a violation** (i.e., margin `+inf`) to avoid false feasibility.

### 1.3 Aspect Ratio Proxy (Correct as a Proxy)
**Location:** `ai_scientist/optim/geometry.py` (`aspect_ratio`)

The cross-section area uses a standard Green’s theorem form:
- \( A = \frac{1}{2}\int (R Z_\theta - Z R_\theta)\, d\theta \)

Then:
- \( r_{minor} \approx \sqrt{A/\pi} \)
- \( AR \approx R_{00} / \langle r_{minor}\rangle_{\zeta} \)

This is a reasonable geometric proxy for fast screening. It is not VMEC’s “true” aspect ratio, but it is consistent with the intended use (cheap geometry heuristic).

### 1.4 Mean Curvature (Reasonable Direction, Needs Validation)
**Location:** `ai_scientist/optim/geometry.py` (`mean_curvature`)

The implementation attempts a differential-geometry mean curvature based on first and second fundamental forms for the cylindrical embedding \( (R\cos\phi, R\sin\phi, Z) \).

**Important clarification:**
- The `R^2` term in `G = R_z^2 + R^2 + Z_z^2` is *expected* for this embedding because \( \partial_\zeta (R\cos\zeta, R\sin\zeta, Z) \) contributes an \(R^2\) term to the metric.

**What remains uncertain:**
- The correctness of the *full* second fundamental form terms in this coordinate system. These expressions are easy to get subtly wrong.

**Recommendation:**
- Add a unit test that compares `mean_curvature()` against a known analytic surface (e.g., a torus with fixed major/minor radii) or against a numerically estimated curvature on a simple parameterized surface.

---

## 2) AI/ML Correctness

### 2.1 Diffusion (DDPM-Style) Implementation (Correct Form)
**Location:** `ai_scientist/optim/generative.py` (`_build_noise_schedule`, `sample`)

The code uses a standard linear beta schedule:
- `beta = linspace(1e-4, 0.02, T)`
- `alpha = 1 - beta`
- `alpha_hat = cumprod(alpha)`

Sampling uses the common update:
- \( x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right) + \sqrt{\beta_t}z \)

**Caveat (precision of claim):**
- This is “DDPM-style” and consistent with common implementations; different derivations sometimes use a posterior variance \( \tilde{\beta}_t \) instead of \( \beta_t \). The current approach is standard and acceptable, just not identical to every paper’s exact variance choice.

### 2.2 PPO Implementation (Correct Mechanics)
**Location:** `ai_scientist/optim/rl_ppo.py`

The implementation contains the core PPO components in standard form:
- Orthogonal initialization
- Gaussian policy with learned log-std
- GAE-style advantage computation
- PPO clipped objective (in loss form, with negation)
- Value loss + entropy bonus + gradient clipping

This is mechanically correct and aligns with common PPO baselines.

### 2.3 Neural Operator Surrogate (Architecture OK; Benchmark Alignment is the Main Risk)
**Location:** `ai_scientist/optim/surrogate_v2.py`

The hybrid architecture (2D CNN over coefficient grid + PointNet over generated point cloud) is conceptually reasonable. Rotation augmentation gives approximate invariance and is not a correctness bug.

**Main correctness risk (higher than invariance/weights):**
- The benchmark feasibility constraints are defined on **`log10(qi)` thresholds** (P2: `<= -4`, P3: `<= -3.5`), but:
  - The surrogate fits QI head on raw `qi` (`surrogate_v2.py` collects `y_qi = metrics["qi"]`).
  - Ranking uses a feasibility proxy based primarily on `mhd_mean >= 0` and does not enforce QI feasibility thresholds.

This mismatch can produce a surrogate that is “good” on its own loss but misleading for benchmark feasibility.

**Recommendation:**
- Either train the surrogate QI head on `log10(qi)` directly, or consistently transform targets and thresholds to match how feasibility is defined in the benchmark logic.

### 2.4 RL Environment (Mechanically Fine; Objective/Constraint Semantics Are Not Aligned)
**Location:** `ai_scientist/rl_env.py`

The environment wiring (state/action spaces, reward as score delta) is fine. The issue is semantics:
- Reward penalizes raw `qi` magnitude, while benchmarks use `log10(qi)` thresholds.
- The interpretation of “objective” in RL is partly heuristic and can diverge from the actual benchmark objective and ALM state.

**Recommendation:**
- Update RL reward and termination criteria to use the same transformed metrics and feasibility gates as the benchmark evaluator (or make it explicit that RL is only a local geometry heuristic).

### 2.5 Differentiable Optimization (Mechanically Fine; QI Threshold Is Not Benchmark-Consistent)
**Location:** `ai_scientist/optim/differentiable.py`

The module implements gradient-based optimization over inputs through a differentiable surrogate.

**Main correctness risk:**
- QI feasibility is treated with a cutoff around `1e-2` on raw `qi` in places, while the actual benchmark constraints are orders of magnitude tighter and expressed in log space.

**Recommendation:**
- Replace raw-`qi` thresholds with `log10(qi)` feasibility constraints consistent with P2/P3 definitions.

---

## 3) Logic / Pipeline Correctness

### 3.1 Stage Evaluation, Pareto, and Hypervolume (Correct)
**Locations:**
- `ai_scientist/fidelity_controller.py`
- `ai_scientist/tools/hypervolume.py`
- `ai_scientist/cycle_executor.py`

The P3 Pareto and hypervolume computations are consistent with:
1) feasibility filtering (`feasibility <= tolerance`)
2) objective vectorization as `(-gradient, aspect)`
3) hypervolume computed in minimization space with a fixed reference point

### 3.2 Coordinator Retraining (High-Risk for Correctness)
**Location:** `ai_scientist/coordinator.py`

Two linked issues matter for scientific correctness:
1) The retraining call hardcodes `minimize_objective=True`, which is wrong for P2 (maximize).
2) The metrics payload used for retraining likely does not match what the surrogate expects (it may be passing a full candidate record instead of the metrics dict), which can cause the surrogate to train on defaults rather than real `vacuum_well`/`qi`.

**Recommendation:**
- Thread `problem` and `minimize_objective` into retraining consistently, and pass training rows with `metrics` equal to the actual evaluation metrics dict.

---

## 4) Tests Run (Local Verification)

The following targeted tests were executed and passed (they validate internal consistency, not full physics truth):

1) Forward model, problems, surrogates, differentiable optim:
- `python -m pytest -q tests/test_forward_model.py tests/test_problems.py tests/optim/test_surrogate_v2.py tests/optim/test_surrogate_bundle.py tests/optim/test_differentiable_optim.py`

2) Coordinator and geometry batch utilities:
- `python -m pytest -q tests/test_coordinator_retraining.py tests/test_coordinator_surrogate.py tests/optim/test_surrogate_uncertainty.py tests/optim/test_geometry_batch.py`

---

## 5) Prioritized Action Items

### P0 (Correctness / feasibility integrity)
1) **Make missing/NaN constraint metrics infeasible by default.**
   - Goal: avoid false “feasible” signals.
2) **Unify objective sign conventions (especially P2).**
   - Decide on one: either “objective is natural metric + explicit minimize flag” or “everything is minimization with negation”, and enforce it everywhere.

### P1 (Benchmark alignment for ML components)
3) Align RL environment and differentiable optimization with benchmark constraints:
   - Use `log10(qi)` where the benchmark does.
   - Match feasibility tolerances and objective direction.

### P2 (Physics validation)
4) Add an analytic/numerical regression test for `mean_curvature()`.
5) Consider documenting which geometry metrics are *proxies* vs VMEC truth metrics.

---

## Appendix: Useful File Pointers
- Central margins/objectives: `ai_scientist/forward_model.py`
- Problem definitions: `ai_scientist/problems.py`
- Geometry proxy metrics: `ai_scientist/optim/geometry.py`
- Hypervolume/Pareto utilities: `ai_scientist/tools/hypervolume.py`
- Neural surrogate: `ai_scientist/optim/surrogate_v2.py`
- Diffusion model: `ai_scientist/optim/generative.py`
- PPO: `ai_scientist/optim/rl_ppo.py`
- RL environment: `ai_scientist/rl_env.py`
- Differentiable optimization: `ai_scientist/optim/differentiable.py`
