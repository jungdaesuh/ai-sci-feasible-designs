# StellarForge: Architecture & Implementation Strategy

**Status:** Draft / Planning
**Date:** December 2, 2025
**Project:** ConStellaration Fusion Challenge (AI-Sci Feasible Designs)

---

## 1. Executive Summary

This document outlines the architectural pivot from online, cold-start learning to an offline, pre-trained generative pipeline ("StellarForge"). The goal is to solve the "Death Valley" problem in stellarator optimization—where random initialization yields 0% feasible designs—by pre-training a Generative AI on the ~160k sample ConStellaration dataset.

**Core Thesis:** The Generative Model is not a scratchpad; it is a **Library**. We must fill the library with books (Offline Training) so the scientist (Agent) can read them to get ideas (Inference), rather than asking the scientist to write the books from scratch while trying to read them.

## 2. The Tri-Hybrid Architecture

We are implementing a three-stage pipeline modeled after state-of-the-art approaches in molecular design (*BoltzGen*, *Diffusion for Fusion*):

1.  **The Dreamer (Generative Model):** A Conditional Diffusion Model trained on PCA-compressed latent representations of stellarator surfaces. It generates "warm start" candidates that are topologically plausible and conditioned on desired physics metrics (e.g., "High Stability").
2.  **The Critic (Surrogate Model):** A Neural Operator ensemble that instantly predicts physics metrics (Aspect Ratio, QI, Vacuum Well) for thousands of candidates, filtering out 99% of failures before they reach the expensive simulator.
3.  **The Engineer (RL/Optimizer):** A local refinement agent (PPO or ALM) that performs "micro-surgery" on the best candidates to push them over the feasibility threshold using gradients from the differentiable Critic.

## 3. Critical Technical Upgrades

Based on literature review (*Diffusion for Fusion 2025*, *LEGO-xtal 2025*), we are adopting three specific engineering upgrades:

### A. Latent Diffusion with PCA
*   **Problem:** Training directly on 661 raw Fourier coefficients is noisy; high-frequency modes dominate variance but matter little for topology.
*   **Solution:**
    1.  **Compress:** Use Principal Component Analysis (PCA) to reduce the 661-dimensional boundary vector to ~50 latent dimensions.
    2.  **Train:** Train the Diffusion Model on this smooth 50-dim latent space.
    3.  **Project:** Inverse-transform generated latents back to full geometry for evaluation.

### B. Geometric Pre-Relaxation
*   **Problem:** Generative models often produce shapes with minor local defects (e.g., self-intersections, kinks) that crash the physics solver (`VMEC++`).
*   **Solution:** Before any physics simulation, run 50 steps of gradient descent minimizing a purely geometric energy function (Surface Curvature + Self-Intersection Penalty) to "snap" the mesh into a valid configuration.

### C. Massive Batching & Filtering
*   **Problem:** Finding a valid stellarator is a "needle in a haystack" problem. Sampling 10 designs is statistically insufficient.
*   **Solution:** Scale the "Dreamer" to generate **10,000+ candidates** per cycle. Use the "Critic" (Surrogate) to aggressively rank them, passing only the top ~20 to the "Engineer" for refinement.

## 4. Implementation Roadmap

### Phase 1: Infrastructure (Current Focus)
*   [ ] **Update Config:** Add `checkpoint_path` and `device` fields to `GenerativeConfig` in `ai_scientist/config.py`.
*   [ ] **Training Script:** Create `scripts/train_generative_offline.py` to:
    *   Download/Load the HF dataset.
    *   Compute PCA basis.
    *   Train Conditional Diffusion on GPU.
    *   Save `generative_physics_v1.pt` (Model + PCA State).

### Phase 2: Integration
*   [ ] **Model Loading:** Update `experiment_setup.py` to load the pre-trained checkpoint and PCA basis.
*   [ ] **Worker Update:** Modify `ExplorationWorker` to use the PCA inverse transform and conditional sampling (e.g., `target_metrics={'vacuum_well': 0.01}`).

### Phase 3: The RL Loop (Future)
*   [ ] **RL Worker:** Implement `PPOAgent` in `workers.py` to refine candidates against the Surrogate reward signal.
*   [ ] **Active Learning:** Close the loop by feeding failed VMEC++ runs back into the Surrogate training set.

## 5. References & Inspirations
*   *Padidar et al. (2025)*: "Diffusion for Fusion" (PCA + Diffusion for Stellarators).
*   *Ridwan et al. (2025)*: "LEGO-xtal" (Geometric Pre-relaxation).
*   *Stark et al. (2025)*: "BoltzGen" (Unified Design/Fold Pipeline).
