This is a comprehensive analysis and refactoring plan to unify the semantic reasoning (LLM Agents) and numerical optimization (Solvers) in your stellarator optimization codebase. The goal is to achieve Architectural Harmony through a continuous Neuro-Symbolic Feedback Loop.

## Architectural Analysis: The Neuro-Symbolic Feedback Loop

The existing architecture suffers from a "split-brain" problem by toggling between EXPLORE (Agent) and EXPLOIT (Math). This prevents dynamic interaction and learning during the optimization process.

We propose a redesign based on the **Agent-Supervised Optimization (ASO)** pattern. In ASO, the optimization process is iterative, and the Agent actively supervises and steers the numerical solver based on real-time, semantically rich feedback.

### The Unified Loop Design

We bridge the semantic-numeric gap by transforming the roles of the components and the nature of their interaction:

1.  **The Planner as Supervisor:** The `PlanningAgent` is responsible for analyzing the optimizer's progress and issuing tactical directives.
2.  **The Coordinator as Optimization Manager:** The `Coordinator` is refactored. It no longer makes high-level strategy decisions (EXPLORE/EXPLOIT). Instead, it manages the optimization state (e.g., `AugmentedLagrangianState`), executes optimization chunks via Workers, and facilitates the feedback loop.

#### The Control Interface (Agent Orchestrates Math)

The `PlanningAgent` issues an `OptimizationDirective`. This structured object allows the Agent to dynamically `ADJUST` hyperparameters—such as increasing specific constraint weights (Soft ALM) if it observes the optimizer struggling with MHD stability, or adjusting the `penalty_parameters_increase_factor` (Standard ALM) if overall convergence is slow.

#### The Feedback Signal (Math Educates Agent)

We introduce a `generate_diagnostics` function in the `Coordinator`. This **Diagnostic Translator** converts the raw numerical state of the optimizer into a semantic report. It analyzes trends (e.g., "increasing\_violation" for specific constraints) and provides a clear status (e.g., "STAGNATION" or "FEASIBLE\_FOUND").

#### The Orchestration

The binary `decide_strategy` is removed. The `Coordinator.produce_candidates` now manages the ASO loop:

1.  **Math:** The `OptimizationWorker` executes a short burst of steps.
2.  **Translate:** The Coordinator generates diagnostics.
3.  **Reason (Neuro):** The Coordinator calls `Planner.analyze_optimizer_diagnostics` (an LLM call) to interpret the diagnostics.
4.  **Direct (Symbolic):** The Planner issues an `OptimizationDirective`.
5.  **Adapt:** The Coordinator applies the directive to the optimization settings.

This loop continues until convergence, budget exhaustion, or the Planner issues a `STOP` directive.

## Refactored Code

Below are the rewritten critical orchestration layers.

### 1\. `planner.py` (Refactored: The Supervisor)

We introduce the `OptimizationDirective` and the core supervision method `analyze_optimizer_diagnostics`.

````python
# ================================================================================
# File: planner.py (Refactored)
# ================================================================================

"""Agentized planning and optimization supervision."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Dict, Optional

from ai_scientist import agent as agent_module
from ai_scientist import config as ai_config
from ai_scientist import memory
from ai_scientist import rag
from ai_scientist import tools
from ai_scientist import tools_api

# (Keep existing PlanningOutcome and helper functions)
class PlanningOutcome:
    # (Structure remains the same as original, defined in the prompt)
    def __init__(
        self,
        context: Mapping[str, Any],
        evaluation_summary: Mapping[str, Any],
        boundary_summary: Mapping[str, Any],
        rag_snippets: Sequence[Mapping[str, str]],
        graph_summary: Mapping[str, Any] | None = None,
        suggested_params: Mapping[str, Any] | None = None,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.context = context
        self.evaluation_summary = evaluation_summary
        self.boundary_summary = boundary_summary
        self.rag_snippets = rag_snippets
        self.graph_summary = graph_summary
        self.suggested_params = suggested_params
        self.config_overrides = config_overrides

def _serialize_summary(summary: tools.P3Summary | None) -> Mapping[str, Any] | None:
    # (Implementation remains the same as original)
    if summary is None:
        return None
    return {
        "hv_score": summary.hv_score,
        "reference_point": list(summary.reference_point),
        "feasible_count": summary.feasible_count,
        "archive_size": summary.archive_size,
        "pareto_entries": [
            {
                **entry.as_mapping(),
                "design_hash": entry.design_hash,
                "stage": entry.stage,
            }
            for entry in summary.pareto_entries
        ],
    }

@dataclass
class OptimizationDirective:
    """Structured directive issued by the Planner to the Optimizer (Coordinator)."""
    action: str # CONTINUE, ADJUST, STOP
    config_overrides: Optional[Mapping[str, Any]] = None
    suggested_params: Optional[Mapping[str, Any]] = None
    reasoning: Optional[str] = None

class PlanningAgent:
    """Wraps the planning-tier gate and supervises the Neuro-Symbolic loop."""

    def __init__(
        self,
        *,
        config: ai_config.ModelConfig | None = None,
        rag_index: Path | str | None = None,
        world_model: memory.WorldModel | None = None,
    ) -> None:
        self.config = config or ai_config.load_model_config()
        # Introduce a specific gate for optimization supervision (The tactical brain)
        self.orchestration_gate = agent_module.provision_model_tier(
            role="orchestration", config=self.config
        )
        # Keep other gates (The strategic brain and analysis tools)
        self.planning_gate = agent_module.provision_model_tier(role="planning", config=self.config)
        self.literature_gate = agent_module.provision_model_tier(role="literature", config=self.config)
        self.analysis_gate = agent_module.provision_model_tier(role="analysis", config=self.config)
        
        self.world_model = world_model
        self.rag_index = Path(rag_index or rag.DEFAULT_INDEX_PATH)
        self.last_context: Mapping[str, Any] | None = None

    # ... (Keep existing helper methods: _hash_context, _validate_tool_call, retrieve_rag, write_note, evaluate_p3, etc.) ...
    # (We assume the utility methods from the original planner.py are kept, but omit them for brevity here)

    def analyze_optimizer_diagnostics(self, diagnostics: Dict[str, Any], cycle: int) -> OptimizationDirective:
        """
        Analyzes the optimizer's diagnostic narrative and decides the next tactical move.
        This is the core of the Neuro-Symbolic Feedback Loop (The Control Interface).
        """
        print(f"[Planner][Orchestration] Analyzing diagnostics at step {diagnostics.get('step', 0)}")

        # Fallback if LLM is disabled or unavailable
        if not self.config.agent_gates:
            return OptimizationDirective(action="CONTINUE", reasoning="LLM disabled.")

        # Use RAG to find strategies based on diagnostics
        rag_context = []
        if diagnostics.get("status") == "STAGNATION":
            rag_context = self.retrieve_rag("Strategies for escaping local minima or stagnation in ALM stellarator optimization", k=2)

        from ai_scientist import model_provider
        provider = self.config.get_provider()
        
        # Define the levers the agent can pull (The Control Schema)
        control_schema = {
            "config_overrides": {
                "description": "Dynamically adjust optimization parameters. Use this to steer the optimizer.",
                "examples": [
                    # Soft ALM: Adjusting specific constraint weights if the optimizer struggles with them
                    {"constraint_weights": {"vacuum_well_mhd": 100.0, "qi_log10": 5.0}}, 
                    # Standard ALM: Adjusting overall behavior based on convergence speed or constraint satisfaction
                    {"alm_settings": {"penalty_parameters_increase_factor": 10.0}}, # Be more aggressive on constraints
                    {"alm_settings": {"bounds_reduction_factor": 0.8}}, # Narrow the search space faster
                ]
            },
            "action": {
                "enum": ["CONTINUE", "ADJUST", "STOP"]
            },
            "reasoning": "A brief explanation of the decision."
        }

        system_prompt = (
            f"You are the Optimization Supervisor (Cycle {cycle}). Your role is to steer the numerical optimizer (Augmented Lagrangian Method) by analyzing its diagnostics and issuing directives.\n\n"
            "GOAL: Guide the optimizer towards feasibility (max_violation near zero) and optimality.\n\n"
            "DIAGNOSTICS INTERPRETATION:\n"
            "- 'status': Optimizer assessment (IN_PROGRESS, STAGNATION, FEASIBLE_FOUND).\n"
            "- 'narrative': Summary of the situation.\n"
            "- 'constraint_trends': Shows evolution of violations. 'increasing_violation' demands immediate action.\n\n"
            "PROTOCOL:\n"
            "1. Analyze the provided 'diagnostics' and 'rag_context'.\n"
            "2. Reason step-by-step. Is the optimizer struggling? Which constraints are violated?\n"
            "3. Formulate a directive using the 'control_schema'.\n"
            "   - If progress is good: 'action': 'CONTINUE'.\n"
            "   - If a constraint shows 'increasing_violation' or high 'violation': 'action': 'ADJUST'. Increase its 'constraint_weights' (Soft ALM) OR increase the global 'penalty_parameters_increase_factor' (Standard ALM).\n"
            "   - If 'STAGNATION' is detected: Consider adjusting 'bounds_reduction_factor' or other ALM settings. If severe, consider 'STOP' and suggesting a new seed.\n"
            "   - If 'FEASIBLE_FOUND' and objective is satisfactory: 'action': 'STOP'.\n"
            "4. Output a JSON object matching the OptimizationDirective structure.\n\n"
            f"CONTROL SCHEMA:\n{json.dumps(control_schema, indent=2)}\n\n"
            f"RAG CONTEXT (if any):\n{json.dumps(rag_context, indent=2)}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Diagnostics: {json.dumps(diagnostics, default=str)}"}
        ]

        try:
            # (LLM invocation and response parsing logic adapted from original planner.py)
            response = model_provider.invoke_chat_completion(
                provider,
                tool_call={"name": "issue_optimization_directive", "arguments": {}},
                messages=messages,
                model=self.orchestration_gate.provider_model
            )

            if response.status_code != 200:
                return OptimizationDirective(action="CONTINUE", reasoning="LLM error.")

            content = response.body.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # JSON extraction (robust handling)
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                 json_str = content.split("```")[1].split("```")[0].strip()

            directive_data = json.loads(json_str)
            directive = OptimizationDirective(
                action=directive_data.get("action", "CONTINUE"),
                config_overrides=directive_data.get("config_overrides"),
                suggested_params=directive_data.get("suggested_params"),
                reasoning=directive_data.get("reasoning", "N/A")
            )
            print(f"[Planner][Orchestration] Directive issued: {directive.action}. Reason: {directive.reasoning}")
            return directive

        except Exception as exc:
            print(f"[Planner][Orchestration] Failed: {exc}. Issuing CONTINUE directive.")
            return OptimizationDirective(action="CONTINUE", reasoning=f"Planner exception or JSON parsing error: {exc}")

    # Keep the original plan_cycle for high-level strategy and initial seed proposal
    def plan_cycle(
        self,
        # ... (arguments remain the same)
        *,
        cfg: ai_config.ExperimentConfig,
        cycle_index: int,
        stage_history: Sequence[Mapping[str, Any]],
        last_summary: tools.P3Summary | None,
        experiment_id: int | None = None,
    ) -> PlanningOutcome:
        # ... (The original plan_cycle logic remains here, as provided in the prompt's consolidated_code.txt)
        # This method defines the high-level goal for the cycle (e.g., "Focus on feasibility restoration")
        # and proposes the initial seed(s) and configuration that the Coordinator will use to start the ASO loop.
        # (Implementation omitted for brevity, assuming it matches the provided consolidated_code.txt)
        pass
````

### 2\. `coordinator.py` (Refactored: The Optimization Manager)

The `Coordinator` manages the optimization state, executes the ASO loop, translates diagnostics, and applies directives from the `PlanningAgent`.

```python
# ================================================================================
# File: coordinator.py (Refactored)
# ================================================================================

"""
The Coordinator manages the Neuro-Symbolic Feedback Loop, bridging the gap between 
Semantic Reasoning (Planner) and Numerical Optimization (Workers).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Mapping, Tuple
import numpy as np
import jax.numpy as jnp
from dataclasses import replace

from ai_scientist import config as ai_config
from ai_scientist import memory

# Import Planner type for hinting, avoiding circular imports during execution
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # These definitions must match planner.py
    from ai_scientist.planner import PlanningAgent, OptimizationDirective

# Assuming Workers are defined in ai_scientist.workers (as in the original structure)
# Note: OptimizationWorker implementation needs to support iterative execution (accepting state).
from ai_scientist.workers import OptimizationWorker, ExplorationWorker, GeometerWorker
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate
from ai_scientist.optim.generative import GenerativeDesignModel

# Import ALM State for type hinting (assuming access to constellaration)
try:
    from constellaration.optimization.augmented_lagrangian import AugmentedLagrangianState
except ImportError:
    # Define a placeholder if the library is not available for type hinting
    AugmentedLagrangianState = Any 

class Coordinator:
    """
    Manages the iterative optimization loop (ASO), supervised by the PlanningAgent.
    """

    def __init__(
        self, 
        cfg: ai_config.ExperimentConfig, 
        world_model: memory.WorldModel,
        planner: PlanningAgent, # The Planner is injected for supervision
        surrogate: Optional[NeuralOperatorSurrogate] = None,
        generative_model: Optional[GenerativeDesignModel] = None
    ):
        self.cfg = cfg
        self.world_model = world_model
        self.planner = planner
        
        # Initialize Workers (These execute the math/exploration)
        # Note: We assume OptimizationWorker is adapted to handle iterative calls.
        self.opt_worker = OptimizationWorker(cfg, surrogate)
        self.explore_worker = ExplorationWorker(cfg, generative_model)
        self.geo_worker = GeometerWorker(cfg)
        
        # State management for the ASO loop
        self.optimization_history: List[AugmentedLagrangianState] = []
        self.constraint_names = self._get_constraint_names(cfg.problem)

    # REMOVED: decide_strategy. Strategy is now iterative and supervised.

    def _get_constraint_names(self, problem_type: str) -> List[str]:
        """Helper to map constraint indices to semantic names for diagnostics."""
        p_key = (problem_type or "").lower()
        if p_key.startswith("p1"):
            return ["aspect_ratio", "average_triangularity", "edge_rotational_transform"]
        elif p_key.startswith("p2"):
            return ["aspect_ratio", "edge_rotational_transform", "edge_magnetic_mirror_ratio", "max_elongation", "qi_log10"]
        else: # P3 default
            # Renaming 'vacuum_well' to 'vacuum_well_mhd' for clarity in LLM prompts
            return ["edge_rotational_transform", "edge_magnetic_mirror_ratio", "vacuum_well_mhd", "flux_compression", "qi_log10"]

    def generate_diagnostics(self, alm_state: AugmentedLagrangianState, step: int) -> Dict[str, Any]:
        """
        (The Feedback Signal Implementation)
        Translates the raw numerical ALM state into a semantic diagnostic report.
        """
        diagnostics = {
            "step": step,
            "objective": float(alm_state.objective),
            "max_violation": float(jnp.max(jnp.maximum(0., alm_state.constraints))),
            "constraint_trends": {},
            "optimizer_health": {
                "bounds_norm": float(jnp.linalg.norm(alm_state.bounds)),
            },
            "narrative": []
        }

        # Analyze trends by comparing with the previous step
        prev_state = self.optimization_history[-1] if self.optimization_history else None

        # 1. Constraint Analysis
        struggling = False
        for i, name in enumerate(self.constraint_names):
            if i >= len(alm_state.constraints): continue

            violation = float(jnp.maximum(0., alm_state.constraints[i]))
            penalty = float(alm_state.penalty_parameters[i])
            trend = "stable"
            
            if prev_state and i < len(prev_state.constraints):
                prev_violation = float(jnp.maximum(0., prev_state.constraints[i]))
                # Define thresholds for trend detection
                if violation > prev_violation * 1.05 and violation > 1e-3:
                    trend = "increasing_violation"
                    struggling = True
                elif violation < prev_violation * 0.95:
                    trend = "decreasing_violation"

            diagnostics["constraint_trends"][name] = {
                "violation": violation,
                "penalty": penalty,
                "trend": trend
            }
        
        # 2. Progress Analysis
        objective_delta = 0.0
        if prev_state:
            objective_delta = float(alm_state.objective) - float(prev_state.objective)
        diagnostics["objective_delta"] = objective_delta

        # 3. Status Determination and Narrative
        FEASIBILITY_THRESHOLD = 1e-3
        STAGNATION_THRESHOLD = 1e-5
        
        if diagnostics["max_violation"] < FEASIBILITY_THRESHOLD:
            diagnostics["status"] = "FEASIBLE_FOUND"
            diagnostics["narrative"].append("Feasible region reached. Now optimizing objective.")
        # Detect stagnation: minimal objective change AND significant constraint violation
        elif prev_state and abs(objective_delta) < STAGNATION_THRESHOLD and diagnostics["max_violation"] > 0.05:
             diagnostics["status"] = "STAGNATION"
             diagnostics["narrative"].append("Stagnation detected: minimal objective change and high constraint violation.")
        else:
             diagnostics["status"] = "IN_PROGRESS"
             if struggling:
                 diagnostics["narrative"].append("Progressing, but struggling with specific constraints (see 'increasing_violation').")
             else:
                 diagnostics["narrative"].append("Progressing normally.")
        
        return diagnostics

    def apply_directive(self, directive: OptimizationDirective, current_cfg: ai_config.ExperimentConfig) -> Tuple[ai_config.ExperimentConfig, Dict[str, Any]]:
        """
        (The Control Interface Implementation)
        Applies the Planner's directive to the configuration and returns specific overrides for the worker.
        """
        if not directive.config_overrides:
            return current_cfg, {}

        print(f"[Coordinator] Applying Planner directive: {directive.action}. Reason: {directive.reasoning}")
        
        new_cfg = current_cfg
        worker_overrides = {}
        
        try:
            overrides = directive.config_overrides
            
            # Handle Soft ALM constraint weights (applied globally to config, affects objective formulation)
            if "constraint_weights" in overrides:
                new_weights = replace(new_cfg.constraint_weights, **overrides["constraint_weights"])
                new_cfg = replace(new_cfg, constraint_weights=new_weights)
                print(f"[Coordinator] Updated Constraint Weights.")
            
            # Handle Standard ALM settings (passed directly to worker for the next step)
            if "alm_settings" in overrides:
                worker_overrides["alm_settings"] = overrides["alm_settings"]
                print(f"[Coordinator] Applying ALM settings overrides for next step.")

            # Handle exploration mix (if applicable)
            if "proposal_mix" in overrides:
                 new_mix = replace(new_cfg.proposal_mix, **overrides["proposal_mix"])
                 new_cfg = replace(new_cfg, proposal_mix=new_mix)

        except Exception as exc:
            print(f"[Coordinator] Failed to apply directive: {exc}")
        
        return new_cfg, worker_overrides


    def produce_candidates(
        self, 
        cycle: int, 
        experiment_id: int, 
        n_candidates: int, # This now represents the total budget for the cycle
        template: ai_config.BoundaryTemplateConfig,
        initial_seeds: Optional[List[Dict[str, Any]]] = None,
        initial_config: Optional[ai_config.ExperimentConfig] = None
    ) -> List[Dict[str, Any]]:
        """
        (The Orchestration Implementation)
        Orchestrates the Agent-Supervised Optimization (ASO) loop.
        Replaces the old EXPLORE/EXPLOIT logic with the Neuro-Symbolic loop.
        """
        budget = n_candidates
        config = initial_config or self.cfg
        
        print(f"[Coordinator] Starting ASO Loop. Budget: {budget} evals.")

        # 1. Initialization
        # Determine starting seeds.
        if not initial_seeds:
            # If Planner didn't provide seeds, use ExplorationWorker to generate initial points.
            print("[Coordinator] No initial seeds provided by Planner. Generating exploratory seeds.")
            explore_ctx = {"n_samples": 5, "cycle": cycle}
            initial_seeds = self.explore_worker.run(explore_ctx).get("candidates", [])
            if not initial_seeds: 
                print("[Coordinator] Exploration failed to generate seeds. Stopping.")
                return []

        # Use GeometerWorker to validate seeds
        geo_ctx = {"candidates": initial_seeds}
        valid_seeds = self.geo_worker.run(geo_ctx).get("candidates", [])
        if not valid_seeds: return []

        # Select the primary seed for the supervised optimization run.
        active_seed = valid_seeds[0] 
        
        current_config = config
        evals_used = 0
        step = 0
        # Define the budget for execution between Planner consultations (Micro-cycle budget)
        INNER_LOOP_BUDGET = 10 

        # Initialize optimization context for the worker
        # Note: We assume OptimizationWorker is modified to handle iterative execution.
        opt_ctx = {
            "initial_guesses": [active_seed], 
            "budget": INNER_LOOP_BUDGET,
            "alm_settings_overrides": {},
            "continue_from_state": None
        }
        
        self.optimization_history = []

        # 2. The Neuro-Symbolic Loop
        while evals_used < budget:
            step += 1
            print(f"\n[Coordinator] ASO Step {step}. Evals used: {evals_used}/{budget}")

            # 2a. Execute Math (OptimizationWorker)
            # The worker executes the optimization chunk and returns the resulting state.
            res = self.opt_worker.run(opt_ctx)
            evals_used += res.get("evals_used", INNER_LOOP_BUDGET)
            alm_state_raw = res.get("final_alm_state")

            if not alm_state_raw:
                print("[Coordinator] OptimizationWorker failed or did not return state. Stopping.")
                break

            # Assuming worker returns compatible state object
            alm_state = alm_state_raw 

            # 2b. Translate (Diagnostic Translator)
            diagnostics = self.generate_diagnostics(alm_state, step)
            # Record the state *after* generating diagnostics (for trend analysis in the next step)
            self.optimization_history.append(alm_state)
            
            # 2c. Reason (Planner Supervision)
            directive = self.planner.analyze_optimizer_diagnostics(diagnostics, cycle)

            # 2d. Direct (Apply Directive)
            if directive.action == "STOP":
                print("[Coordinator] Planner requested stop.")
                break
            
            current_config, worker_overrides = self.apply_directive(directive, current_config)
            
            # Prepare next iteration context
            opt_ctx["continue_from_state"] = alm_state
            opt_ctx["budget"] = min(INNER_LOOP_BUDGET, budget - evals_used)
            
            # Apply the specific ALM settings overrides requested by the Planner
            if "alm_settings" in worker_overrides:
                opt_ctx["alm_settings_overrides"] = worker_overrides["alm_settings"]
            # Note: Changes to current_config (like constraint_weights) are automatically picked up by the worker in the next iteration.

            # Handle seed injection (if Planner determines the current trajectory is hopeless)
            if directive.suggested_params:
                print("[Coordinator] Injecting new seed from Planner. Resetting ALM state.")
                opt_ctx["initial_guesses"] = [{"params": directive.suggested_params}]
                opt_ctx["continue_from_state"] = None
                self.optimization_history = [] # Reset history for the new seed

        print(f"[Coordinator] ASO loop finished after {step} steps.")
        # Return the history of candidates generated during the optimization
        # The OptimizationWorker should track the candidates evaluated during its runs.
        return self.opt_worker.get_history()
```

### 3\. `runner.py` (Refactored: Simplified Executor)

The runner is streamlined. It initializes the Planner and Coordinator (injecting the Planner into the Coordinator) and initiates the Coordinator's ASO loop.

*Note: This is a snippet focusing on the changes within `_run_cycle` and the required initialization changes.*

```python
# ================================================================================
# File: runner.py (Refactored Snippet)
# ================================================================================

# ... (Imports remain similar) ...
from ai_scientist.coordinator import Coordinator
from ai_scientist import planner as ai_planner
from ai_scientist.optim.surrogate import BaseSurrogate
from ai_scientist.optim.surrogate_v2 import NeuralOperatorSurrogate
from ai_scientist.optim.generative import GenerativeDesignModel
# ...

# (Helper functions, BudgetController, etc. remain) ...

# IMPORTANT: Initialization sequence in the main execution block must be updated.

# Example initialization (replaces the logic in the main execution flow):
def initialize_architecture(cfg: ai_config.ExperimentConfig, world_model: memory.WorldModel):
    """Initializes the core components of the Neuro-Symbolic architecture."""
    
    # Initialize models
    surrogate_model = _create_surrogate(cfg)
    generative_model = _create_generative_model(cfg)

    # 1. Initialize Planner (The Supervisor)
    planner = ai_planner.PlanningAgent(
        config=cfg.model,
        rag_index=rag.DEFAULT_INDEX_PATH,
        world_model=world_model,
    )

    # 2. Initialize Coordinator (The Optimization Manager)
    # Inject the Planner into the Coordinator for supervision
    coordinator = Coordinator(
        cfg=cfg,
        world_model=world_model,
        planner=planner, # Injection point
        surrogate=surrogate_model if isinstance(surrogate_model, NeuralOperatorSurrogate) else None,
        generative_model=generative_model
    )
    
    return planner, coordinator

# The main loop would then use these components:
# planner, coordinator = initialize_architecture(cfg, world_model)
# for cycle_index in range(cfg.n_cycles):
#     _run_cycle(..., planner=planner, coordinator=coordinator, ...)


def _run_cycle(
    cfg: ai_config.ExperimentConfig,
    cycle_index: int,
    # ... other parameters
    # REFACTORED: Pass both Planner and Coordinator
    planner: ai_planner.PlanningAgent,
    coordinator: Coordinator,
    surrogate_model: BaseSurrogate,
    generative_model: GenerativeDesignModel | None = None,
    *,
    runtime: RunnerCLIConfig | None = None,
    budget_controller: BudgetController,
    # ... other parameters
) -> tuple[Path | None, dict[str, Any] | None, tools.P3Summary | None]:
    
    # (Initial setup for the cycle: evaluators, budgets)
    cycle_number = cycle_index + 1
    # ...
    
    # --- 1. High-Level Planning Phase ---
    # The Planner sets the initial strategy, seeds, and configuration for the cycle.
    
    # (Fetch history/summary for the planner)
    stage_history = [] # Placeholder
    last_p3_summary = None # Placeholder

    # Call the high-level planner to determine the starting point and strategy.
    planning_outcome = planner.plan_cycle(
        cfg=cfg,
        cycle_index=cycle_index,
        stage_history=stage_history,
        last_summary=last_p3_summary,
        experiment_id=experiment_id,
    )

    # Apply initial agent-driven config overrides suggested by the Planner
    active_cfg = cfg
    if planning_outcome.config_overrides:
        # (Logic to apply initial overrides as in original runner.py)
        # ...
        pass

    # (Budget snapshot logic remains)
    budget_snapshot = budget_controller.snapshot()
    # ... (Update active_budgets based on snapshot)

    # --- 2. Agent-Supervised Optimization (ASO) Loop ---
    
    # Prepare initial seeds from the planning outcome
    initial_seeds = []
    if planning_outcome.suggested_params:
        # (Logic to format suggested_params into initial_seeds structure)
        # ...
        pass

    # Initiate the Neuro-Symbolic Loop managed by the Coordinator.
    print(f"[runner][cycle={cycle_number}] Starting Agent-Supervised Optimization (ASO) Loop.")
    
    # The Coordinator handles the iterative process, consulting the Planner internally.
    optimization_results = coordinator.produce_candidates(
        cycle=cycle_number,
        experiment_id=experiment_id,
        n_candidates=active_budgets.screen_evals_per_cycle, # The budget
        template=active_cfg.boundary_template,
        initial_seeds=initial_seeds,
        initial_config=active_cfg
    )

    # --- 3. Post-Optimization Processing and Evaluation ---

    # The optimization_results contain candidates generated during the ASO loop.
    candidates_to_evaluate = optimization_results
    
    # (The rest of the _run_cycle handles final evaluation (if necessary), 
    # logging, promotion (S2), reporting, and budget feedback. This logic remains largely the same.)
    
    # ... (Evaluation, Logging, Reporting) ...
    
    # Placeholder return
    return None, None, None
```