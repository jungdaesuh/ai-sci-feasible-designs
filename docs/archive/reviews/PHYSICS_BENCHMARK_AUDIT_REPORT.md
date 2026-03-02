# Physics Benchmark Alignment Audit Report

**Date:** 2025-12-15
**Auditor:** Antigravity (AI Assistant)
**Scope:** `ai_scientist` codebase alignment with `constellaration` benchmark definitions.

## Executive Summary

The audit confirms that the `ai_scientist` codebase is aligned with the canonical benchmark definitions in `constellaration`. No critical misalignments were found regarding constraint names, sign conventions, ordering, or objective directions. The "Legacy" `hv` column in the database is correctly documented and handled via the `objective_types.py` SSOT.

## Audit Checklist Findings

### A. Canonical Definitions
*   **Status:** ✅ Aligned
*   **Verification:** `ai_scientist/problems.py` wrappers (P1, P2, P3) correctly map to `constellaration` definitions.
*   **Note:** P3 wrapper returns a scalarized objective (`aspect_ratio`) for `compute_objective`, but `cycle_executor` correctly uses `summarize_p3_candidates` for multi-objective analysis (Hypervolume).

### B. Constraint Names
*   **Status:** ✅ Aligned
*   **Verification:**
    *   `ai_scientist/constraints.py` serves as the Single Source of Truth (SSOT).
    *   Names match `constellaration/problems.py`.
    *   `forward_model.py` and `differentiable.py` use these canonical names.

### C. Constraint Ordering & Normalization
*   **Status:** ✅ Aligned
*   **Verification:**
    *   `differentiable.py` (L868+) constructs ALM vectors using `get_constraint_names(problem, for_alm=True)`, ensuring consistent ordering regardless of dict iteration order.
    *   `forward_model.compute_constraint_margins` normalizes violations using `(val - limit) / abs(limit)` (or reversed for lower bounds), ensuring **Positive = Violation**.

### D. Objective Directions & Types
*   **Status:** ✅ Aligned
*   **Verification:** `ai_scientist/objective_types.py` explicitly defines semantics:
    | Metric | P1 | P2 | P3 |
    | :--- | :--- | :--- | :--- |
    | **ALM Objective** | Min `max_elongation` | Min `20 - gradient` | Min `20 - gradient` |
    | **Physics Objective** | Min `max_elongation` | Max `gradient` | Min `aspect_ratio` (primary) |
    | **Ranking Score** | Max `-elongation` | Max `gradient/aspect` | Max `gradient/aspect` |
    *   Optimizers use `TargetKind` Enum to select the correct target (Standard Objective vs. Gradient Proxy).

### E. Surrogate Targets
*   **Status:** ✅ Aligned
*   **Verification:** `objective_types.get_training_target` correctly assigns `GRADIENT_PROXY` for P3, ensuring the surrogate models the correct physics goal (Simpler Coils & Compactness).

### F. Reporting & Hypervolume
*   **Status:** ✅ Aligned
*   **Verification:**
    *   `ai_scientist/tools/hypervolume.py` correctly converts natural units (Max Gradient, Min Aspect) to minimization form `(-gradient, aspect)` for pymoo.
    *   `reporting.py` reports the true set-level Hypervolume.
    *   The `metrics` table `hv` column (per-candidate) stores the `gradient_proxy` (scalar), which is permitted as a documented legacy alias.

### G. Feasibility (Rule R3)
*   **Status:** ✅ Aligned
*   **Verification:** Stage gating in `forward_model` allows "Screen" stage to return feasibility based on partial constraints, but `reporting.py` and `cycle_executor` clearly distinguish "Promoted" (High Fidelity) candidates from "Screened" ones throughout the pipeline.

## Conclusion

The `ai_scientist` codebase requires no changes to align with the physics benchmarks. The Single Source of Truth (`constraints.py` & `objective_types.py`) pattern is effectively preventing drift.
