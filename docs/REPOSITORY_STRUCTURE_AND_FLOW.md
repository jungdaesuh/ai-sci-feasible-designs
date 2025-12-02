# Repository Structure & Flow: AI Scientist for Fusion Design

This document provides a comprehensive overview of the `ai-sci-feasible-designs` repository, detailing its structure, the interaction between the AI agent and the physics engine, and the execution flow of the Agent-Supervised Optimization (ASO) loop.

## 1. Overview

This project is an autonomous AI system designed to solve the **ConStellaration Fusion Challenge**. Its goal is to design optimized plasma boundary shapes for stellarator fusion reactors.

The system employs a hierarchical agent architecture (`ai_scientist`) that orchestrates physics simulations (`constellaration`) to optimize for geometric complexity, coil simplicity, and MHD stability.

## 2. Repository Structure

The codebase is divided into two primary components: the **AI Agent** and the **Physics/Benchmark Library**.

### 📂 `ai_scientist/` (The Agent)

Contains the core logic for the autonomous research agent.

*   **`experiment_runner.py`**: The main entry point. Orchestrates the experiment loop, manages the lifecycle of cycles, and handles high-level configuration.
*   **`cycle_executor.py`**: executes a single "cycle" of the experiment (Plan → Run → Report). It integrates candidate generation, surrogate ranking, and physics evaluation.
*   **`coordinator.py`**: The "Brain" of the agent (Phase 5). It decides the high-level strategy (EXPLORE vs. EXPLOIT) and delegates tasks to workers.
*   **`planner.py`**: An LLM-powered module that "supervises" the optimization. It diagnoses failures in the numerical optimizer (ALM) and suggests hyperparameter adjustments (e.g., penalty weights).
*   **`workers.py`**: Specialized sub-agents used by the Coordinator:
    *   `ExplorationWorker`: Generates new candidates (via VAE, diffusion, or random sampling).
    *   `OptimizationWorker`: Refines candidates using gradient-based methods or ALM.
    *   `GeometerWorker`: Validates geometric constraints (checks for self-intersection, singularities).
*   **`surrogate_model/` & `optim/surrogate_v2.py`**: Neural networks (MLP, Neural Operators) that approximate physics results to save compute time.
*   **`fidelity_controller.py`**: Manages the "Fidelity Ladder" (Low → Bridge → High), ensuring expensive simulations are only run on promising candidates.
*   **`budget_manager.py`**: Tracks and limits resource usage (evaluations, wall-clock time).
*   **`memory/`**: Handles the "World Model" (SQLite database) to store experiment history, metrics, and artifacts.

### 📂 `constellaration/` (The Physics Engine)

A standalone library for stellarator physics and optimization benchmarks.

*   **`src/constellaration/geometry/`**: Defines the plasma boundary representation (`SurfaceRZFourier`).
*   **`src/constellaration/forward_model/`**: Wraps `VMEC++` (the MHD equilibrium solver). This is the "Oracle" that validates designs.
*   **`src/constellaration/problems.py`**: Defines the optimization benchmarks (P1, P2, P3) with their specific objectives and constraints.
*   **`vmecpp/`**: Contains the bindings or setup for the underlying C++/Fortran physics code.

### 📂 `docs/`
*   **`run_protocol.md`**: The standard operating procedure for running experiments.
*   **`AGENTS.md`**: High-level context and "personality" for the AI agent.
*   **`ASO_V4_IMPLEMENTATION_GUIDE.md`**: Details on the Agent-Supervised Optimization architecture.

---

## 3. System Architecture & The ASO Loop

The core of the system is the **Agent-Supervised Optimization (ASO)** loop. This loop iterates to refine designs, learning from previous results.

### The Flow of a Cycle

1.  **Initialization (`experiment_runner.py`)**
    *   The system loads the configuration (e.g., `P3` problem, `ASO` enabled).
    *   It initializes the **World Model** (database) and loads previous state if resuming.

2.  **Strategy Selection (`coordinator.py`)**
    *   The **Coordinator** analyzes the experiment history.
    *   It decides whether to **EXPLORE** (if stagnated) or **EXPLOIT** (if making progress).
    *   *Example*: If the Hypervolume (HV) hasn't improved recently, it switches to Exploration.

3.  **Candidate Generation (`workers.py`)**
    *   **Exploration**: The `ExplorationWorker` samples from a Generative Model (VAE/Diffusion) or uses geometric heuristics (Rotating Ellipse).
    *   **Exploitation**: The `OptimizationWorker` takes the best previous designs and refines them.
    *   **Filtering**: The `GeometerWorker` immediately discards geometrically invalid shapes (e.g., self-intersecting).

4.  **Surrogate Ranking (`surrogate_v2.py`)**
    *   Instead of simulating all candidates, the **Neural Operator Surrogate** predicts their performance.
    *   Candidates are ranked, and only the top $k$ are selected for the expensive physics simulation.

5.  **Optimization & Supervision (`planner.py` & `cycle_executor.py`)**
    *   Selected candidates undergo **Augmented Lagrangian Method (ALM)** optimization.
    *   **The Supervisor (LLM)**: The `planner.py` module monitors the ALM process.
        *   *Diagnostic*: "Constraint violation is increasing."
        *   *Action*: "Increase penalty parameters by 2x."
    *   This "Human-in-the-Loop" simulation allows the agent to recover from numerical instabilities without human intervention.

6.  **Physics Evaluation (The Fidelity Ladder)**
    *   **Low Fidelity**: Fast, coarse-grid VMEC++ run. (Filters out ~90% of failures).
    *   **Bridge**: Medium resolution.
    *   **High Fidelity**: Full resolution run for final verification.
    *   Results are stored in the **World Model**.

7.  **Learning & Reporting**
    *   The **Surrogate Model** is retrained on the new physics data.
    *   The **Generative Model** (VAE) is fine-tuned on successful designs.
    *   A report is generated (JSON metrics + Pareto Front plots).

## 4. Key Concepts

### 🪜 The Fidelity Ladder
To save compute, simulations are tiered:
1.  **Surrogate**: < 0.01s (Neural Net prediction).
2.  **Geometer**: < 0.1s (Geometric checks).
3.  **Low Fidelity**: ~1s (Coarse VMEC).
4.  **High Fidelity**: ~10-60s (Fine VMEC).

### 🧠 Agent-Supervised Optimization (ASO)
Standard numerical optimizers (like `scipy.minimize`) often fail in the complex stellarator landscape. ASO wraps these optimizers with an "AI Supervisor" that acts like a human expert: watching the convergence trace, adjusting hyperparameters dynamically, and restarting from new seeds when stuck.

### 🌍 The World Model
The `memory/` module acts as the agent's long-term memory. It stores:
*   **Candidates**: Every design ever evaluated.
*   **Metrics**: Physics results (QA, stability, etc.).
*   **Trajectory**: The path taken by the optimizer.
*   **Artifacts**: Configs, logs, and reports.

## 5. Getting Started

To run an experiment using the ASO loop:

```bash
# Run Problem 3 (Multi-objective) with ASO enabled
python -m ai_scientist.runner --problem p3 --aso --cycles 50
```

To resume an experiment:
```bash
python -m ai_scientist.runner --resume-from checkpoints/last_run.json
```
