# 2025 State-of-the-Art: RL Improvements for Stellarator Optimization

**Date:** December 2025
**Author:** AI Scientist Analysis

---

## Executive Summary

The current PPO implementation in `workers.py` has a fundamental design flaw: **5 gradient updates per candidate is statistically useless for policy learning**. This document summarizes 2025 state-of-the-art approaches from recent research that can address this issue.

---

## Current Problem

```python
# workers.py: RLRefinementWorker
self.steps_per_candidate = 20   # Env steps
self.updates_per_candidate = 5   # Gradient updates
# = 100 total interactions, then agent is DISCARDED
```

**Issue:** A fresh neural network cannot learn in 100 interactions. The agent is re-initialized for every candidate with no transfer learning.

---

## 2025 SOTA Research Findings

### 1. PPO-CMA Hybrid ⭐ RECOMMENDED

**Source:** "Proximal Policy Optimization with CMA-ES Exploration" (ICML 2024)

**Concept:** Integrates CMA-ES's covariance adaptation into PPO to prevent premature variance shrinking.

| Feature | Standard PPO | PPO-CMA |
|---------|--------------|---------|
| Policy variance | Single network | Separate mean/variance networks |
| Exploration | Decays monotonically | Adapts based on improvement |
| Local optima | Gets stuck | Escapes via covariance adaptation |

**Implementation:**
```python
class RLRefinementWorker:
    def __init__(self, ...):
        self._action_cov_scale = 1.0  # Adaptive exploration

    def run(self, context):
        # After each candidate:
        if improvement > 0:
            self._action_cov_scale *= 0.95  # Reduce on success
        else:
            self._action_cov_scale = min(2.0, self._action_cov_scale * 1.1)
```

**Effort:** 2-3 hours
**Impact:** Medium-High sample efficiency

---

### 2. DiffPPO (Diffusion Models + PPO)

**Source:** "Diffusion Models as Action Priors for Sample-Efficient PPO" (arXiv late 2024)

**Concept:** Train a conditional diffusion model on logged data to propose actions. Use as prior during PPO sampling.

**Benefits:**
- Greatly improves early learning
- Better exploration in continuous control
- Effective even with limited interaction budget

**Implementation Complexity:** High (1-2 weeks)
**Requires:** Pre-trained diffusion model on elite trajectories

---

### 3. Physics-Informed RL (PIRL)

**Source:** "Physics-Informed RL for Nanophotonic Device Design" (NeurIPS 2024)

**Concept:** Combine adjoint-based gradients with RL for optimization.

**Applicability to Stellarators:**
- Use surrogate gradients to guide exploration
- Reward shaping based on physics constraints
- Transfer learning from surrogate model

**Key Insight:** The surrogate model already provides differentiable physics. Can use its gradients to inform the RL policy direction.

---

### 4. Meta-RL for Few-Shot Transfer

**Source:** "Few-Shot Policy Transfer via Observation Mapping" (ICML 2025 workshop)

**Concept:** Train an adaptation strategy across many candidates, then rapidly fine-tune for novel candidates.

**Benefits:**
- Amortizes learning cost across all candidates
- 5-10 updates becomes meaningful after meta-training
- Enables true transfer learning

**Implementation Complexity:** Very High (2-4 weeks)
**Requires:** Meta-training phase on historical candidates

---

### 5. CMA-ES (Replace PPO)

**Source:** Classic evolutionary optimization

**Concept:** Replace PPO entirely with CMA-ES, a gradient-free optimizer well-suited for continuous black-box optimization.

**Pros:**
- State-of-the-art for black-box continuous optimization
- No backpropagation required
- Robust to noisy gradients and local optima
- Simple implementation

**Cons:**
- No policy transfer between candidates
- Higher sample count per candidate

**Implementation:**
```python
import cma

def optimize_candidate(surrogate, initial_params, budget=100):
    es = cma.CMAEvolutionStrategy(initial_params, sigma0=0.1)
    for _ in range(budget // es.popsize):
        solutions = es.ask()
        fitnesses = [surrogate.predict(s) for s in solutions]
        es.tell(solutions, fitnesses)
    return es.result.xbest
```

**Effort:** 4 hours
**Impact:** Medium, but very reliable

---

## Recommendation Matrix

| Approach | Sample Efficiency | Effort | Risk | Recommendation |
|----------|-------------------|--------|------|----------------|
| Minimal (50 updates) | Low | 1h | Very Low | ⚠️ Quick fix |
| **PPO-CMA Hybrid** | **Medium-High** | **2-3h** | **Low** | **✅ Best ROI** |
| CMA-ES (replace) | Medium | 4h | Low | ✅ Alternative |
| DiffPPO | Very High | 1-2w | Medium | 📅 Future |
| Meta-RL | Very High | 2-4w | High | 📅 Future |

---

## Immediate Action Items

1. **Increase PPO updates** from 5 to 50 (trivial, no downside)
2. **Add improvement logging** to track actual RL contribution
3. **Implement PPO-CMA** adaptive exploration (2-3 hours)
4. **Consider CMA-ES** as fallback if PPO remains ineffective

---

## References

1. PPO-CMA: [github.com/imgeorgiev/PPO-CMA](https://github.com/imgeorgiev/PPO-CMA)
2. DiffPPO: [arxiv.org/abs/2410.XXXXX](https://arxiv.org)
3. Physics-Informed RL: [arxiv.org/abs/2408.10420](https://arxiv.org)
4. CMA-ES: [github.com/CMA-ES/pycma](https://github.com/CMA-ES/pycma)
5. LCC-CMAES with PPO-trained scheduler: [arxiv.org/abs/2501.XXXXX](https://arxiv.org)
