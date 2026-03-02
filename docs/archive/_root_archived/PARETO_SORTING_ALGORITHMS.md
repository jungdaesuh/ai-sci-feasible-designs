# Pareto Sorting Algorithms Reference

This document explains the non-dominated sorting algorithms used in `ai_scientist/optim/search.py`.

## Problem Statement

Given N solutions with M objectives, partition them into **Pareto fronts**:
- Front 1: Non-dominated solutions
- Front 2: Non-dominated after removing Front 1
- ... and so on

## Current Implementation: Fast Non-Dominated Sort (NSGA-II)

**Complexity**: O(MN²)

**Reference**: Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
"A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II".
*IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.

### Algorithm
1. For each pair (i, j), determine domination relationship
2. Track `domination_count[i]` = number of solutions dominating i
3. Track `dominated_set[i]` = solutions that i dominates
4. Extract fronts by iteratively removing solutions with count = 0

## Alternative: Kung's Algorithm (First Front Only)

**Complexity**: O(N log N) for M=2 objectives, O(N log^(M-1) N) general

**Reference**: Kung, H.T., Luccio, F., & Preparata, F.P. (1975).
"On Finding the Maxima of a Set of Vectors".
*Journal of the ACM*, 22(4), 469-476.

### Algorithm (2 objectives)
1. Sort solutions by first objective: O(N log N)
2. Sweep through sorted list, tracking running minimum of second objective
3. Solution is non-dominated if its second objective ≤ running minimum

### Why We Don't Use It
- Only extracts **first front** efficiently
- Extracting all F fronts requires O(F × N log N) ≈ O(N² log N) worst case
- More complex implementation for marginal gain when F is large

## Complexity Comparison

| Algorithm | All Fronts | First Front Only |
|-----------|------------|------------------|
| Naive     | O(MN³)     | O(MN²)           |
| NSGA-II   | O(MN²)     | O(MN²)           |
| Kung's    | O(FN log N)| O(N log N)       |

For typical evolutionary optimization where N ≈ 100-1000 and we need all fronts,
NSGA-II's O(MN²) is the standard choice.

## Future Optimization Opportunities

1. **Kung's hybrid**: Use Kung's for first front, NSGA-II for remainder
2. **Parallel comparison**: Vectorize pairwise dominance checks with NumPy
3. **Incremental updates**: Maintain fronts across generations rather than recomputing
