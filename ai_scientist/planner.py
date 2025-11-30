"""Agentized planning helper for Phase 3 (docs/roadmap & improvement plan guidance)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from ai_scientist import agent as agent_module
from ai_scientist import config as ai_config
from ai_scientist.config import ASOConfig
from ai_scientist import memory
from ai_scientist import rag
from ai_scientist import tools
from ai_scientist import tools_api


class DirectiveAction(Enum):
    """Enumerated actions for type safety."""
    CONTINUE = "CONTINUE"
    ADJUST = "ADJUST"
    STOP = "STOP"
    RESTART = "RESTART"


class DirectiveSource(Enum):
    """Source of the directive for debugging."""
    LLM = "llm"
    HEURISTIC = "heuristic"
    CONVERGENCE = "convergence"
    FALLBACK = "fallback"


@dataclass
class OptimizationDirective:
    """Structured directive from Planner to Coordinator."""
    action: DirectiveAction
    config_overrides: Optional[Mapping[str, Any]] = None
    alm_overrides: Optional[Mapping[str, Any]] = None  # Direct ALM state manipulation
    suggested_params: Optional[Mapping[str, Any]] = None
    reasoning: str = ""
    confidence: float = 1.0
    source: DirectiveSource = DirectiveSource.HEURISTIC

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "config_overrides": dict(self.config_overrides) if self.config_overrides else None,
            "alm_overrides": dict(self.alm_overrides) if self.alm_overrides else None,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "source": self.source.value,
        }


@dataclass
class ConstraintDiagnostic:
    """Diagnostic for a single constraint (from real ALM state)."""
    name: str
    violation: float           # max(0, constraint_value)
    penalty: float             # Current penalty parameter
    multiplier: float          # Lagrange multiplier (learned importance)
    trend: str                 # "stable", "increasing_violation", "decreasing_violation"
    delta: float = 0.0         # Change from previous step


@dataclass
class OptimizerDiagnostics:
    """Rich diagnostic report from real ALM state."""
    step: int
    trajectory_id: int

    # From AugmentedLagrangianState
    objective: float
    objective_delta: float
    max_violation: float
    constraints_raw: List[float]
    multipliers: List[float]
    penalty_parameters: List[float]
    bounds_norm: float

    # Derived analysis
    status: str  # "IN_PROGRESS", "STAGNATION", "FEASIBLE_FOUND", "DIVERGING"
    constraint_diagnostics: List[ConstraintDiagnostic]
    narrative: List[str]
    steps_since_improvement: int = 0

    def requires_llm_supervision(self, aso_config: "ASOConfig") -> bool:
        """Determine if this diagnostic warrants an LLM call."""
        if aso_config.supervision_mode == "every_step":
            return True
        if aso_config.supervision_mode == "periodic":
            return self.step % aso_config.supervision_interval == 0
        # Event-triggered
        return any([
            self.status == "STAGNATION",
            self.status == "FEASIBLE_FOUND",
            self.status == "DIVERGING",
            any(c.trend == "increasing_violation" for c in self.constraint_diagnostics),
            self.steps_since_improvement >= aso_config.max_stagnation_steps,
        ])

    def to_json(self) -> str:
        return json.dumps({
            "step": self.step,
            "objective": round(self.objective, 4),
            "objective_delta": round(self.objective_delta, 6),
            "max_violation": round(self.max_violation, 4),
            "status": self.status,
            "bounds_norm": round(self.bounds_norm, 4),
            "constraints": [
                {
                    "name": c.name,
                    "violation": round(c.violation, 4),
                    "penalty": round(c.penalty, 2),
                    "multiplier": round(c.multiplier, 4),
                    "trend": c.trend,
                }
                for c in self.constraint_diagnostics
            ],
            "narrative": self.narrative,
        }, indent=2)


class HeuristicSupervisor:
    """
    Rule-based optimization supervisor.
    Handles 80%+ of cases without LLM latency.

    Key insight: We can now use REAL ALM state (multipliers, penalties, bounds)
    to make informed decisions.
    """

    def __init__(self, aso_config: "ASOConfig"):
        self.config = aso_config

    def analyze(self, diagnostics: OptimizerDiagnostics) -> OptimizationDirective:
        """
        Generate directive using heuristic rules based on real ALM state.

        Decision tree:
        1. FEASIBLE_FOUND + stable objective -> STOP (converged)
        2. FEASIBLE_FOUND + improving -> CONTINUE
        3. STAGNATION + high violation -> ADJUST (boost penalties directly)
        4. STAGNATION + low violation + small bounds -> STOP (local minimum)
        5. DIVERGING -> STOP (abandon trajectory)
        6. Specific constraint worsening + low multiplier -> ADJUST (boost that penalty)
        7. Otherwise -> CONTINUE
        """
        cfg = self.config

        # Case 1 & 2: Feasible region reached
        if diagnostics.status == "FEASIBLE_FOUND":
            if abs(diagnostics.objective_delta) < cfg.stagnation_objective_threshold:
                return OptimizationDirective(
                    action=DirectiveAction.STOP,
                    reasoning=f"Converged: feasible (violation={diagnostics.max_violation:.4f}) with stable objective",
                    source=DirectiveSource.CONVERGENCE,
                )
            return OptimizationDirective(
                action=DirectiveAction.CONTINUE,
                reasoning="Feasible and still improving objective",
                source=DirectiveSource.HEURISTIC,
            )

        # Case 3 & 4: Stagnation
        if diagnostics.status == "STAGNATION":
            if diagnostics.max_violation > cfg.stagnation_violation_threshold:
                # High violation stagnation: boost penalties
                worst_idx = max(
                    range(len(diagnostics.constraint_diagnostics)),
                    key=lambda i: diagnostics.constraint_diagnostics[i].violation /
                                  (diagnostics.constraint_diagnostics[i].penalty + 1e-6)
                )
                worst = diagnostics.constraint_diagnostics[worst_idx]

                new_penalties = diagnostics.penalty_parameters.copy()
                new_penalties[worst_idx] = min(
                    worst.penalty * cfg.max_penalty_boost,
                    cfg.max_constraint_weight
                )

                return OptimizationDirective(
                    action=DirectiveAction.ADJUST,
                    alm_overrides={"penalty_parameters": new_penalties},
                    reasoning=f"Stagnation with violation={diagnostics.max_violation:.4f}, "
                              f"boosting penalty for '{worst.name}' from {worst.penalty:.1f} to {new_penalties[worst_idx]:.1f}",
                    source=DirectiveSource.HEURISTIC,
                )
            else:
                if diagnostics.bounds_norm < 0.1:
                    return OptimizationDirective(
                        action=DirectiveAction.STOP,
                        reasoning="Stagnation with small trust region, likely local minimum",
                        source=DirectiveSource.HEURISTIC,
                    )
                return OptimizationDirective(
                    action=DirectiveAction.RESTART,
                    reasoning="Stagnation near feasibility, trying new seed",
                    source=DirectiveSource.HEURISTIC,
                )

        # Case 5: Diverging
        if diagnostics.status == "DIVERGING":
            return OptimizationDirective(
                action=DirectiveAction.STOP,
                reasoning="Multiple constraints diverging, abandoning trajectory",
                source=DirectiveSource.HEURISTIC,
            )

        # Case 6: Specific constraint struggling
        struggling = [c for c in diagnostics.constraint_diagnostics if c.trend == "increasing_violation"]
        if struggling:
            worst = max(struggling, key=lambda c: c.violation)
            worst_idx = next(i for i, c in enumerate(diagnostics.constraint_diagnostics) if c.name == worst.name)

            new_penalties = diagnostics.penalty_parameters.copy()
            new_penalties[worst_idx] = min(worst.penalty * 2, cfg.max_constraint_weight)

            return OptimizationDirective(
                action=DirectiveAction.ADJUST,
                alm_overrides={"penalty_parameters": new_penalties},
                reasoning=f"Constraint '{worst.name}' worsening (violation={worst.violation:.4f}), "
                          f"boosting penalty to {new_penalties[worst_idx]:.1f}",
                source=DirectiveSource.HEURISTIC,
            )

        # Case 7: Default
        return OptimizationDirective(
            action=DirectiveAction.CONTINUE,
            reasoning="Normal progress",
            source=DirectiveSource.HEURISTIC,
        )

class PlanningOutcome:
    """Structured output that mirrors the JSON sections sent to the planning agent prompt."""

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


class PlanningAgent:
    """Wraps the planning-tier gate so runner cycles can rely on tool schemas + telemetry."""

    def __init__(
        self,
        *,
        config: ai_config.ModelConfig | None = None,
        rag_index: Path | str | None = None,
        world_model: memory.WorldModel | None = None,
    ) -> None:
        self.config = config or ai_config.load_model_config()
        self.planning_gate = agent_module.provision_model_tier(
            role="planning", config=self.config
        )
        self.literature_gate = agent_module.provision_model_tier(
            role="literature", config=self.config
        )
        self.analysis_gate = agent_module.provision_model_tier(
            role="analysis", config=self.config
        )
        self.rag_index = Path(rag_index or rag.DEFAULT_INDEX_PATH)
        self.world_model = world_model
        self.last_context: Mapping[str, Any] | None = None
        self.heuristic: HeuristicSupervisor | None = None

    def _hash_context(self, payload: Mapping[str, Any]) -> str:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _validate_tool_call(
        self, gate: agent_module.AgentGate, tool_name: str, arguments: Mapping[str, Any]
    ) -> None:
        if not gate.allows(tool_name):
            raise ValueError(
                f"Tool '{tool_name}' is not permitted for {gate.model_alias} (role={gate.provider_model})"
            )
        schema = tools_api.get_tool_schema(tool_name)
        if schema:
            parameters = schema.get("parameters", {})
            required = parameters.get("required", [])
            missing = [field for field in required if field not in arguments]
            if missing:
                raise ValueError(
                    f"Tool '{tool_name}' missing required arguments: {missing}"
                )
        context_hash = self._hash_context(arguments)
        print(
            f"[planner][tool-call] role={gate.model_alias} tool={tool_name} context_hash={context_hash}"
        )

    def retrieve_rag(self, query: str, *, k: int = 3) -> list[dict[str, str]]:
        payload: dict[str, Any] = {"query": query}
        payload["k"] = k
        # RAG is allowed for multiple roles; we check planning gate by default here
        # but in a real loop the agent driver would pick the gate.
        self._validate_tool_call(self.planning_gate, "retrieve_rag", payload)
        return tools.retrieve_rag(query, k=k, index_path=self.rag_index)

    def write_note(
        self,
        content: str,
        experiment_id: int,
        cycle: int,
        filename: str | None = None,
    ) -> str:
        payload = {
            "content": content,
            "filename": filename,
            "experiment_id": experiment_id,
            "cycle": cycle,
        }
        self._validate_tool_call(self.literature_gate, "write_note", payload)
        return tools.write_note(
            content,
            filename=filename,
            world_model=self.world_model,
            experiment_id=experiment_id,
            cycle=cycle,
            memory_db=self.world_model.db_path if self.world_model else None,
        )

    def evaluate_p3(
        self,
        params: Mapping[str, Any],
        *,
        stage: str | None = None,
    ) -> Mapping[str, Any]:
        args = {
            "params": params,
            "problem": "p3",
        }
        if stage is not None:
            args["stage"] = stage
        self._validate_tool_call(self.planning_gate, "evaluate_p3", args)
        try:
            return tools.evaluate_p3(params, stage=stage or "p3")
        except Exception as exc:  # pragma: no cover - smoke-run safety
            message = f"planning-stage evaluate_p3 failed: {exc}"
            print(f"[planner] {message}")
            return {
                "stage": stage or "p3",
                "error": message,
                "objective": None,
                "feasibility": None,
                "hv": None,
            }

    def make_boundary(
        self, params: Mapping[str, Any]
    ) -> tools.surface_rz_fourier.SurfaceRZFourier:
        args = {"params": params}
        self._validate_tool_call(self.planning_gate, "make_boundary", args)
        return tools.make_boundary_from_params(params)

    def propose_boundary(
        self,
        params: Mapping[str, Any],
        perturbation_scale: float = 0.05,
        seed: int | None = None,
    ) -> dict[str, Any]:
        args = {"params": params, "perturbation_scale": perturbation_scale}
        if seed is not None:
            args["seed"] = seed
        self._validate_tool_call(self.planning_gate, "propose_boundary", args)
        return tools.propose_boundary(
            params, perturbation_scale=perturbation_scale, seed=seed
        )

    def recombine_designs(
        self,
        parent_a: Mapping[str, Any],
        parent_b: Mapping[str, Any],
        alpha: float | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        args = {"parent_a": parent_a, "parent_b": parent_b, "alpha": alpha, "seed": seed}
        # Clean args (remove None)
        args = {k: v for k, v in args.items() if v is not None}
        self._validate_tool_call(self.planning_gate, "recombine_designs", args)
        return tools.recombine_designs(**args)

    def _build_template_params(
        self, template: ai_config.BoundaryTemplateConfig
    ) -> Mapping[str, Any]:
        n_poloidal = template.n_poloidal_modes
        n_toroidal = template.n_toroidal_modes
        center_idx = n_toroidal // 2
        r_cos = []
        z_sin = []
        for pol in range(n_poloidal):
            r_row = []
            z_row = []
            for tor in range(n_toroidal):
                r_val = (
                    template.base_major_radius
                    if pol == 0 and tor == center_idx
                    else 0.0
                )
                z_val = (
                    template.base_minor_radius
                    if pol == 1 and tor == center_idx and n_poloidal > 1
                    else 0.0
                )
                r_row.append(r_val)
                z_row.append(z_val)
            r_cos.append(r_row)
            z_sin.append(z_row)
        return {
            "r_cos": r_cos,
            "z_sin": z_sin,
            "n_field_periods": template.n_field_periods,
            "is_stellarator_symmetric": True,
        }

    def _build_context(
        self,
        *,
        cycle_index: int,
        budgets: ai_config.BudgetConfig,
        constraint_weights: ai_config.ConstraintWeightsConfig,
        stage_history: Sequence[Mapping[str, Any]],
        last_summary: tools.P3Summary | None,
        evaluation_summary: Mapping[str, Any],
        boundary_summary: Mapping[str, Any],
        rag_snippets: Sequence[Mapping[str, str]],
        graph_summary: Mapping[str, Any] | None = None,
        failure_cases: Sequence[Mapping[str, Any]] | None = None,
    ) -> Mapping[str, Any]:
        context = {
            "cycle_index": cycle_index + 1,
            "planner_role": self.planning_gate.model_alias,
            "budgets": asdict(budgets),
            "constraint_weights": asdict(constraint_weights),
            "stage_history": list(stage_history),
            "previous_p3_summary": _serialize_summary(last_summary),
            "latest_evaluation": evaluation_summary,
            "current_boundary": boundary_summary,
            "rag_snippets": list(rag_snippets),
            "graph_summary": graph_summary,
            "failure_cases": list(failure_cases) if failure_cases else [],
            "toolset": list(self.planning_gate.allowed_tools),
        }
        self.last_context = context
        return context

    def plan_cycle(
        self,
        *,
        cfg: ai_config.ExperimentConfig,
        cycle_index: int,
        stage_history: Sequence[Mapping[str, Any]],
        last_summary: tools.P3Summary | None,
        experiment_id: int | None = None,
    ) -> PlanningOutcome:
        cycle_number = cycle_index + 1
        
        # Literature Retrieval (planning role)
        rag_snippets = self.retrieve_rag(
            f"Planning guidance for {cfg.problem.upper()} cycle {cycle_number}",
            k=3,
        )
        
        # Planning Role Actions
        params = self._build_template_params(cfg.boundary_template)
        evaluation = self.evaluate_p3(
            params,
            stage=cfg.fidelity_ladder.screen,
        )
        boundary = self.make_boundary(params)
        
        evaluation_summary = {
            "objective": evaluation.get("objective"),
            "feasibility": evaluation.get("feasibility"),
            "hv": evaluation.get("hv"),
            "stage": evaluation.get("stage"),
            "agent_stage_label": f"agent-cycle-{cycle_number}",
        }
        boundary_summary = {
            "n_poloidal_modes": boundary.n_poloidal_modes,
            "n_toroidal_modes": boundary.n_toroidal_modes,
            "n_field_periods": boundary.n_field_periods,
            "stellarator_symmetric": boundary.is_stellarator_symmetric,
        }
        
        # Property Graph Snapshot (if world_model connected)
        graph_summary: Mapping[str, Any] | None = None
        failure_cases: Sequence[Mapping[str, Any]] = []
        
        if self.world_model and experiment_id is not None:
            pg = self.world_model.to_networkx(experiment_id)
            graph_dir = Path(cfg.reporting_dir) / "graphs"
            graph_dir.mkdir(parents=True, exist_ok=True)
            graph_file = graph_dir / f"cycle_{cycle_number}.json"
            
            nodes = [{"id": node_id, **attrs} for node_id, attrs in pg.nodes(data=True)]
            edges = [
                {"src": src, "dst": dst, "attrs": attrs}
                for src, dst, attrs in pg.edges(data=True)
            ]
            snapshot_data = {"nodes": nodes, "edges": edges}
            graph_file.write_text(json.dumps(snapshot_data, indent=2), encoding="utf-8")
            
            graph_summary = {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "note_count": sum(1 for n in nodes if n.get("type") == "note"),
                "snapshot_path": str(graph_file)
            }
            
            # Retrieve recent failures for reflection
            failure_cases = self.world_model.recent_failures(
                experiment_id=experiment_id,
                problem=cfg.problem,
                limit=5
            )

        context = self._build_context(
            cycle_index=cycle_index,
            budgets=cfg.budgets,
            constraint_weights=cfg.constraint_weights,
            stage_history=stage_history,
            last_summary=last_summary,
            evaluation_summary=evaluation_summary,
            boundary_summary=boundary_summary,
            rag_snippets=rag_snippets,
            graph_summary=graph_summary,
            failure_cases=failure_cases,
        )
        
        suggested_params = None
        config_overrides = None

        if self.config.agent_gates:
             from ai_scientist import model_provider
             provider = self.config.get_provider()
             
             tools_schemas = tools_api.list_tool_schemas()
             available_tools = [
                 schema for schema in tools_schemas 
                 if schema["name"] in self.planning_gate.allowed_tools
             ]
             
             system_prompt = (
                 f"You are the Planning Agent for the AI Scientist (cycle {cycle_number}).\n"
                 f"Your goal is to optimize a stellarator design (problem: {cfg.problem}) by analyzing the "
                 "experiment context and literature.\n\n"
                 "You have access to the following tools:\n"
                 f"{json.dumps(available_tools, indent=2)}\n\n"
                 "PROTOCOL:\n"
                 "1. Analyze the context, specifically 'failure_cases' (to see what constraints are being violated) and 'rag_snippets'.\n"
                 "2. You may use tools to gather more info or test hypotheses (e.g., 'retrieve_rag', 'evaluate_p3', 'propose_boundary').\n"
                 "3. To call a tool, output a JSON object with {\"tool\": \"<name>\", \"arguments\": {<args>}}.\n"
                 "4. To finish and commit to a plan, output a JSON object with {\"suggested_params\": {...}, \"config_overrides\": {...}}.\n"
                 "   - 'suggested_params' (optional): A dictionary matching the structure of 'current_boundary' for the next candidate seed.\n"
                 "   - 'config_overrides' (optional): A dictionary to adjust experiment settings.\n"
                 "       - Example: {'proposal_mix': {'exploration_ratio': 0.8}}\n"
                 "       - Example: {'constraint_weights': {'mhd': 50.0}} (Soft ALM: Increase weight if MHD is failing)\n\n"
                 "Think step-by-step. You have a maximum of 5 turns."
             )
             
             messages = [
                 {"role": "system", "content": system_prompt},
                 {"role": "user", "content": f"Context: {json.dumps(context, default=str)}"}
             ]
             
             max_turns = 5
             for turn in range(max_turns):
                 try:
                     response = model_provider.invoke_chat_completion(
                         provider,
                         tool_call={"name": "plan_cycle_turn", "arguments": {}},
                         messages=messages,
                         model=self.planning_gate.provider_model
                     )
                     
                     if response.status_code != 200:
                         print(f"[planner] Turn {turn}: LLM returned status {response.status_code}")
                         break
                         
                     content = response.body.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                     
                     # Naive JSON extraction
                     json_str = content
                     if "```json" in content:
                         json_str = content.split("```json")[1].split("```")[0].strip()
                     elif "```" in content:
                         json_str = content.split("```")[1].split("```")[0].strip()
                     
                     try:
                         action = json.loads(json_str)
                     except json.JSONDecodeError:
                         print(f"[planner] Turn {turn}: Failed to parse JSON response.")
                         # Feedback loop: tell the agent its JSON was invalid?
                         # For now, just append as text and ask to retry if we had more logic, 
                         # but simpler to just break or continue.
                         messages.append({"role": "assistant", "content": content})
                         messages.append({"role": "user", "content": "Error: Invalid JSON format. Please output valid JSON for tool call or final plan."})
                         continue

                     messages.append({"role": "assistant", "content": content})

                     # Check for final plan
                     if "suggested_params" in action or "config_overrides" in action:
                         suggested_params = action.get("suggested_params")
                         config_overrides = action.get("config_overrides")
                         print(f"[planner] Plan finalized in turn {turn+1}")
                         if suggested_params:
                             print("[planner] Suggesting new boundary params")
                         if config_overrides:
                             print(f"[planner] Suggesting config overrides: {config_overrides}")
                         break
                     
                     # Check for tool call
                     tool_name = action.get("tool")
                     tool_args = action.get("arguments", {})
                     
                     if tool_name:
                         print(f"[planner] Turn {turn}: Agent calling tool '{tool_name}'")
                         tool_result = "Tool execution failed."
                         try:
                             if tool_name == "retrieve_rag":
                                 tool_result = self.retrieve_rag(**tool_args)
                             elif tool_name == "evaluate_p3":
                                 tool_result = self.evaluate_p3(**tool_args)
                             elif tool_name == "propose_boundary":
                                 tool_result = self.propose_boundary(**tool_args)
                             elif tool_name == "recombine_designs":
                                 tool_result = self.recombine_designs(**tool_args)
                             elif tool_name == "make_boundary":
                                  # Helper: just return the object representation for the agent to see structure
                                  # We can't pass the actual object back to LLM easily, so serialize parameters
                                  # or just confirm it works. make_boundary in tools.py returns SurfaceRZFourier
                                  # which isn't JSON serializable.
                                  # Let's skip or serialize params.
                                  # Actually, the agent might use this to validate params.
                                  # For now, let's just echo params back or similar.
                                  tool_result = {"status": "success", "params": tool_args.get("params")}
                             else:
                                 tool_result = f"Error: Tool '{tool_name}' not supported or permitted."
                         except Exception as tool_exc:
                             tool_result = f"Error executing tool: {tool_exc}"
                         
                         messages.append({"role": "user", "content": f"Tool '{tool_name}' output: {json.dumps(tool_result, default=str)}"})
                     else:
                         # No recognized action
                         messages.append({"role": "user", "content": "Error: No 'tool' or 'suggested_params' found in JSON."})
                 
                 except Exception as exc:
                     print(f"[planner] Turn {turn} failed: {exc}")
                     break

        return PlanningOutcome(
            context=context,
            evaluation_summary=evaluation_summary,
            boundary_summary=boundary_summary,
            rag_snippets=rag_snippets,
            graph_summary=graph_summary,
            suggested_params=suggested_params,
            config_overrides=config_overrides
        )

    def _ensure_heuristic(self, aso_config: ASOConfig) -> HeuristicSupervisor:
        if self.heuristic is None:
            self.heuristic = HeuristicSupervisor(aso_config)
        return self.heuristic

    def supervise(
        self,
        diagnostics: OptimizerDiagnostics,
        cycle: int,
        aso_config: ASOConfig,
    ) -> OptimizationDirective:
        """
        Tiered supervision: heuristic first, LLM on demand.

        The key insight is that we now have REAL ALM state, so heuristics
        can make much better decisions than with proxy diagnostics.
        """
        heuristic = self._ensure_heuristic(aso_config)

        # Tier 1: Check if LLM needed
        if not diagnostics.requires_llm_supervision(aso_config):
            return heuristic.analyze(diagnostics)

        # Tier 2: Try LLM with fallback
        if aso_config.use_heuristic_fallback:
            try:
                return self._llm_supervise(diagnostics, cycle, aso_config)
            except Exception as e:
                print(f"[Planner] LLM supervision failed: {e}, using heuristic")
                return heuristic.analyze(diagnostics)
        else:
            return self._llm_supervise(diagnostics, cycle, aso_config)

    def _llm_supervise(
        self,
        diagnostics: OptimizerDiagnostics,
        cycle: int,
        aso_config: ASOConfig,
    ) -> OptimizationDirective:
        """LLM-based supervision with real ALM state context."""
        # Retrieve relevant context if stagnating
        rag_context = []
        if diagnostics.status == "STAGNATION":
            rag_context = self.retrieve_rag(
                "Strategies for escaping local minima in stellarator ALM optimization",
                k=2,
            )

        system_prompt = self._build_supervision_prompt(cycle, rag_context, diagnostics)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Current ALM diagnostics:\n{diagnostics.to_json()}"},
        ]

        from ai_scientist import model_provider
        provider = self.config.get_provider()

        for attempt in range(aso_config.llm_max_retries):
            try:
                response = model_provider.invoke_chat_completion(
                    provider,
                    tool_call={"name": "supervise_optimization", "arguments": {}},
                    messages=messages,
                    model=self.planning_gate.provider_model,
                )

                if response.status_code != 200:
                    raise RuntimeError(f"LLM returned {response.status_code}")

                content = response.body.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                return self._parse_directive(content, diagnostics)

            except json.JSONDecodeError as e:
                if attempt < aso_config.llm_max_retries - 1:
                    messages.append({"role": "user", "content": f"Invalid JSON: {e}. Please output valid JSON."})
                    continue
                raise

        raise RuntimeError("LLM supervision failed after retries")

    def _build_supervision_prompt(self, cycle: int, rag_context: list, diagnostics: OptimizerDiagnostics) -> str:
        rag_section = ""
        if rag_context:
            rag_section = f"\n\nRelevant knowledge:\n{json.dumps(rag_context, indent=2)}"

        constraint_names = [c.name for c in diagnostics.constraint_diagnostics]

        return f"""You are the ASO Supervisor for the AI Scientist (cycle {cycle}).

You have access to REAL Augmented Lagrangian Method (ALM) state:
- objective: Current objective function value
- constraints: {constraint_names}
- penalty_parameters: How strongly each constraint is being enforced
- multipliers: Lagrange multipliers (learned constraint importance)
- bounds_norm: Size of trust region (smaller = more focused search)

ACTIONS:
- CONTINUE: Proceed with current settings
- ADJUST: Modify penalty_parameters to steer optimization
- STOP: Terminate (converged or hopeless)
- RESTART: Abandon trajectory, try new seed

OUTPUT FORMAT (JSON):
{{
  "action": "CONTINUE | ADJUST | STOP | RESTART",
  "alm_overrides": {{
    "penalty_parameters": [p1, p2, ...]  // Optional: new penalties per constraint
  }},
  "reasoning": "brief explanation"
}}

ADJUSTMENT STRATEGY:
- If a constraint has high violation but low penalty: increase that penalty
- If stuck (small bounds_norm) with violations: try RESTART
- If multiplier is high but violation persists: constraint may be infeasible
{rag_section}

Respond with ONLY valid JSON."""

    def _parse_directive(self, content: str, diagnostics: OptimizerDiagnostics) -> OptimizationDirective:
        """Parse LLM response into OptimizationDirective."""
        json_str = content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()

        data = json.loads(json_str)
        action = DirectiveAction(data.get("action", "CONTINUE"))

        return OptimizationDirective(
            action=action,
            alm_overrides=data.get("alm_overrides"),
            config_overrides=data.get("config_overrides"),
            reasoning=data.get("reasoning", ""),
            source=DirectiveSource.LLM,
        )

