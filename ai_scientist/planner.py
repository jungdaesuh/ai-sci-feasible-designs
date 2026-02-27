"""Agentized planning helper for Phase 3 (docs/roadmap & improvement plan guidance)."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence

import pydantic

from ai_scientist import agent as agent_module
from ai_scientist import config as ai_config
from ai_scientist import prompts
from ai_scientist import memory, model_provider, rag, tools, tools_api
from ai_scientist.constraints import get_constraint_names
from ai_scientist.config import ASOConfig

_DEFAULT_EXPERIENCE_INJECTION_PROBABILITY = 0.5
_DEFAULT_AGENT_SEED_CANDIDATE_COUNT = 3


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


class OptimizationDirective(pydantic.BaseModel):
    """Structured directive from Planner to Coordinator."""

    model_config = pydantic.ConfigDict(frozen=True)

    action: DirectiveAction
    config_overrides: Optional[Mapping[str, Any]] = None
    alm_overrides: Optional[Mapping[str, Any]] = None  # Direct ALM state manipulation
    suggested_params: Optional[Mapping[str, Any]] = None
    reasoning: str = ""
    override_reason: str | None = None
    confidence: float = 1.0
    source: DirectiveSource = DirectiveSource.HEURISTIC


class PlannerIntent(pydantic.BaseModel):
    """Structured cycle-level intent from planner to ASO (soft prior)."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    primary_constraint_order: list[str] = pydantic.Field(default_factory=list)
    target_move_family: str | None = None
    forbidden_moves: list[str] = pydantic.Field(default_factory=list)
    penalty_focus_indices: list[int] = pydantic.Field(default_factory=list)
    restart_policy: str | None = None
    confidence: float = pydantic.Field(default=0.5, ge=0.0, le=1.0)


class ConstraintDiagnostic(pydantic.BaseModel):
    """Diagnostic for a single constraint (from real ALM state)."""

    model_config = pydantic.ConfigDict(frozen=True)

    name: str
    violation: float  # max(0, constraint_value)
    penalty: float  # Current penalty parameter
    multiplier: float  # Lagrange multiplier (learned importance)
    trend: str  # "stable", "increasing_violation", "decreasing_violation"
    delta: float = 0.0  # Change from previous step


class OptimizerDiagnostics(pydantic.BaseModel):
    """Rich diagnostic report from real ALM state."""

    model_config = pydantic.ConfigDict(frozen=True)

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
        return any(
            [
                self.status == "STAGNATION",
                self.status == "FEASIBLE_FOUND",
                self.status == "DIVERGING",
                any(
                    c.trend == "increasing_violation"
                    for c in self.constraint_diagnostics
                ),
                self.steps_since_improvement >= aso_config.max_stagnation_steps,
            ]
        )

    def to_json(self) -> str:
        return json.dumps(
            {
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
            },
            indent=2,
        )


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
                    key=lambda i: diagnostics.constraint_diagnostics[i].violation
                    / (diagnostics.constraint_diagnostics[i].penalty + 1e-6),
                )
                worst = diagnostics.constraint_diagnostics[worst_idx]

                new_penalties = diagnostics.penalty_parameters.copy()
                new_penalties[worst_idx] = min(
                    worst.penalty * cfg.max_penalty_boost, cfg.max_constraint_weight
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
        struggling = [
            c
            for c in diagnostics.constraint_diagnostics
            if c.trend == "increasing_violation"
        ]
        if struggling:
            worst = max(struggling, key=lambda c: c.violation)
            worst_idx = next(
                i
                for i, c in enumerate(diagnostics.constraint_diagnostics)
                if c.name == worst.name
            )

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
        suggested_params_list: Sequence[Mapping[str, Any]] | None = None,
        config_overrides: Mapping[str, Any] | None = None,
        planner_intent: Mapping[str, Any] | None = None,
    ) -> None:
        self.context = context
        self.evaluation_summary = evaluation_summary
        self.boundary_summary = boundary_summary
        self.rag_snippets = rag_snippets
        self.graph_summary = graph_summary
        self.suggested_params = suggested_params
        self.suggested_params_list = (
            list(suggested_params_list) if suggested_params_list else None
        )
        self.config_overrides = config_overrides
        self.planner_intent = planner_intent


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


def _coerce_graph_attrs(attrs: Any) -> Mapping[str, Any]:
    if isinstance(attrs, Mapping):
        return attrs
    return {"value": attrs}


def _graph_nodes(graph: Any) -> list[tuple[str, Mapping[str, Any]]]:
    nodes_accessor = getattr(graph, "nodes", None)
    if callable(nodes_accessor):
        raw_nodes = nodes_accessor(data=True)
        if not isinstance(raw_nodes, Iterable):
            return []
        parsed_nodes: list[tuple[str, Mapping[str, Any]]] = []
        for item in raw_nodes:
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            node_id, attrs = item
            parsed_nodes.append((str(node_id), _coerce_graph_attrs(attrs)))
        return parsed_nodes
    if isinstance(nodes_accessor, Mapping):
        return [
            (str(node_id), _coerce_graph_attrs(attrs))
            for node_id, attrs in nodes_accessor.items()
        ]
    return []


def _graph_edges(graph: Any) -> list[tuple[str, str, Mapping[str, Any]]]:
    edges_accessor = getattr(graph, "edges", None)
    if callable(edges_accessor):
        raw_edges = edges_accessor(data=True)
        if not isinstance(raw_edges, Iterable):
            return []
        parsed_edges: list[tuple[str, str, Mapping[str, Any]]] = []
        for item in raw_edges:
            if not isinstance(item, tuple) or len(item) != 3:
                continue
            src, dst, attrs = item
            parsed_edges.append((str(src), str(dst), _coerce_graph_attrs(attrs)))
        return parsed_edges
    if isinstance(edges_accessor, Iterable):
        parsed_edges: list[tuple[str, str, Mapping[str, Any]]] = []
        for item in edges_accessor:
            if not isinstance(item, tuple) or len(item) != 3:
                continue
            src, dst, attrs = item
            parsed_edges.append((str(src), str(dst), _coerce_graph_attrs(attrs)))
        return parsed_edges
    return []


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

    def evaluate_p1(
        self,
        params: Mapping[str, Any],
        *,
        problem: str | None = None,
        stage: str | None = None,
    ) -> Mapping[str, Any]:
        del problem
        return self._evaluate_problem_tool(
            params=params,
            stage=stage,
            problem_key="p1",
            tool_name="evaluate_p1",
            default_stage="screen",
            evaluator=tools.evaluate_p1,
        )

    def evaluate_p2(
        self,
        params: Mapping[str, Any],
        *,
        problem: str | None = None,
        stage: str | None = None,
    ) -> Mapping[str, Any]:
        del problem
        return self._evaluate_problem_tool(
            params=params,
            stage=stage,
            problem_key="p2",
            tool_name="evaluate_p2",
            default_stage="p2",
            evaluator=tools.evaluate_p2,
        )

    def evaluate_p3(
        self,
        params: Mapping[str, Any],
        *,
        problem: str | None = None,
        stage: str | None = None,
    ) -> Mapping[str, Any]:
        del problem
        return self._evaluate_problem_tool(
            params=params,
            stage=stage,
            problem_key="p3",
            tool_name="evaluate_p3",
            default_stage="p3",
            evaluator=tools.evaluate_p3,
        )

    def _evaluate_problem_tool(
        self,
        *,
        params: Mapping[str, Any],
        stage: str | None,
        problem_key: str,
        tool_name: str,
        default_stage: str,
        evaluator: Callable[..., Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        args = {
            "params": params,
            "problem": problem_key,
        }
        if stage is not None:
            args["stage"] = stage
        self._validate_tool_call(self.planning_gate, tool_name, args)
        try:
            return evaluator(params, stage=stage or default_stage)
        except Exception as exc:  # pragma: no cover - smoke-run safety
            message = f"planning-stage {tool_name} failed: {exc}"
            print(f"[planner] {message}")
            return {
                "stage": stage or default_stage,
                "error": message,
                "objective": None,
                "feasibility": None,
                "gradient_proxy": None,
            }

    def evaluate_for_problem(
        self,
        problem: str,
        params: Mapping[str, Any],
        *,
        stage: str | None = None,
    ) -> Mapping[str, Any]:
        problem_key = str(problem or "p3").lower()
        evaluators: dict[str, Callable[..., Mapping[str, Any]]] = {
            "p1": self.evaluate_p1,
            "p2": self.evaluate_p2,
        }
        return evaluators.get(problem_key, self.evaluate_p3)(params, stage=stage)

    def _execute_planning_tool(
        self, tool_name: str, tool_args: Mapping[str, Any]
    ) -> Any:
        if tool_name == "make_boundary":
            # make_boundary returns a non-serializable object; return params echo.
            return {
                "status": "success",
                "params": tool_args.get("params"),
            }
        handlers: dict[str, Callable[..., Any]] = {
            "retrieve_rag": self.retrieve_rag,
            "evaluate_p1": self.evaluate_p1,
            "evaluate_p2": self.evaluate_p2,
            "evaluate_p3": self.evaluate_p3,
            "propose_boundary": self.propose_boundary,
            "recombine_designs": self.recombine_designs,
        }
        handler = handlers.get(tool_name)
        if handler is not None:
            return handler(**tool_args)
        return f"Error: Tool '{tool_name}' not supported or permitted."

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
        args = {
            "parent_a": parent_a,
            "parent_b": parent_b,
            "alpha": alpha,
            "seed": seed,
        }
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

    def _deterministic_roll(
        self,
        *,
        cycle_number: int,
        random_seed: int,
        experiment_id: int | None,
    ) -> float:
        payload = f"{cycle_number}:{random_seed}:{experiment_id or 0}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:8]
        return int(digest, 16) / 0xFFFFFFFF

    def _should_inject_experience(
        self,
        *,
        cycle_number: int,
        random_seed: int,
        experiment_id: int | None,
        probability: float = _DEFAULT_EXPERIENCE_INJECTION_PROBABILITY,
    ) -> bool:
        return self._deterministic_roll(
            cycle_number=cycle_number,
            random_seed=random_seed,
            experiment_id=experiment_id,
        ) < max(0.0, min(1.0, probability))

    def _build_experience_memo(
        self,
        *,
        recent_successes: Sequence[Mapping[str, Any]],
        recent_near_successes: Sequence[Mapping[str, Any]],
        recent_failures: Sequence[Mapping[str, Any]],
    ) -> str:
        def _fmt(entries: Sequence[Mapping[str, Any]], label: str) -> str:
            if not entries:
                return f"{label}: none"
            compact = []
            for item in entries[:2]:
                worst = item.get("worst_constraint") or "none"
                feasibility = item.get("feasibility")
                objective = item.get("objective")
                compact.append(
                    f"({label} feas={feasibility}, obj={objective}, worst={worst})"
                )
            return " ".join(compact)

        return " | ".join(
            [
                _fmt(recent_successes, "success"),
                _fmt(recent_near_successes, "near"),
                _fmt(recent_failures, "failure"),
            ]
        )

    def _build_planning_system_prompt(
        self,
        *,
        cfg: ai_config.ExperimentConfig,
        cycle_number: int,
        available_tools: Sequence[Mapping[str, Any]],
    ) -> str:
        constraint_names = get_constraint_names(cfg.problem, for_alm=True)
        max_penalty_index = len(constraint_names) - 1
        gate_prompt = str(
            getattr(self.planning_gate, "system_prompt", "") or ""
        ).strip()
        problem_prompt = prompts.build_problem_prompt(
            cfg.problem, cfg.fidelity_ladder.screen
        ).strip()
        contract_prompt = (
            "Repository contract:\n"
            "- Feasibility-first search: prioritize reducing constraint violations before objective tuning.\n"
            "- Bounded edits: each seed proposal should change a small subset of coefficients; avoid wholesale rewrites.\n"
            "- Reuse history: leverage recent_successes, recent_near_successes, and recent_failures when present.\n"
            "- Respect deterministic downstream selection; return multiple candidates only when they are diverse.\n"
        )
        protocol_prompt = (
            "PROTOCOL:\n"
            "1. Analyze cycle context, especially experience_memo, scratchpad_summary, feedback_adapter, and rag_snippets.\n"
            "2. You may call tools to test hypotheses (retrieve_rag/evaluate/propose/recombine).\n"
            '3. Tool call format: {"tool":"<name>","arguments":{...}}.\n'
            '4. Finalization format: {"suggested_params_list":[{...},{...}], "config_overrides":{...}, "planner_intent":{...}}.\n'
            "   - Prefer 3 candidates (k=3) and keep 1<=k<=3.\n"
            '   - Backward-compatible alternative accepted: {"suggested_params":{...}}.\n'
            "   - You MUST provide either valid suggested_params(_list) OR seed_fallback='template'.\n"
            "   - config_overrides is optional and must be a JSON object when present.\n"
            "   - planner_intent is optional and must be a JSON object when present.\n"
            "   - planner_intent schema keys: primary_constraint_order, target_move_family, forbidden_moves, penalty_focus_indices, restart_policy, confidence.\n"
            f"   - planner_intent.primary_constraint_order uses ALM names: {constraint_names}.\n"
            f"   - planner_intent.penalty_focus_indices must be integer indices in [0, {max_penalty_index}].\n"
            "5. seed_fallback='template' is allowed if you cannot produce valid coefficients.\n"
            "If no valid finalization payload is returned within turn budget, runtime falls back to template seed.\n"
            "Respond with JSON only."
        )
        sections = [
            f"You are the Planning Agent for the AI Scientist (cycle {cycle_number}).",
            gate_prompt,
            problem_prompt,
            contract_prompt,
            "Available tools:",
            json.dumps(list(available_tools), indent=2),
            protocol_prompt,
            "Think step-by-step. Max turns: 5.",
        ]
        return "\n\n".join(section for section in sections if section)

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
        success_cases: Sequence[Mapping[str, Any]] | None = None,
        near_success_cases: Sequence[Mapping[str, Any]] | None = None,
        feedback_adapter: Mapping[str, Any] | None = None,
        experience_memo: str | None = None,
        experience_injected: bool = True,
        experience_injection_probability: float = _DEFAULT_EXPERIENCE_INJECTION_PROBABILITY,
        scratchpad_summary: Mapping[str, Any] | None = None,
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
            "recent_failures": list(failure_cases) if failure_cases else [],
            "recent_successes": list(success_cases) if success_cases else [],
            "recent_near_successes": (
                list(near_success_cases) if near_success_cases else []
            ),
            "experience_memo": experience_memo or "",
            "experience_injected": experience_injected,
            "experience_injection_probability": experience_injection_probability,
            "scratchpad_summary": dict(scratchpad_summary or {}),
            "feedback_adapter": dict(feedback_adapter or {}),
            "toolset": list(self.planning_gate.allowed_tools),
        }
        self.last_context = context
        return context

    def _parse_planner_intent(
        self,
        raw_intent: Any,
        *,
        problem: str,
    ) -> PlannerIntent | None:
        if raw_intent is None:
            return None
        if not isinstance(raw_intent, Mapping):
            raise ValueError("planner_intent must be a JSON object.")

        intent = PlannerIntent.model_validate(raw_intent)
        allowed_constraint_names = get_constraint_names(problem, for_alm=True)
        allowed_constraint_set = set(allowed_constraint_names)

        if len(set(intent.primary_constraint_order)) != len(
            intent.primary_constraint_order
        ):
            raise ValueError("planner_intent.primary_constraint_order has duplicates.")

        unknown_constraints = [
            name
            for name in intent.primary_constraint_order
            if name not in allowed_constraint_set
        ]
        if unknown_constraints:
            raise ValueError(
                f"planner_intent.primary_constraint_order contains unknown constraints: {unknown_constraints}"
            )

        if len(set(intent.penalty_focus_indices)) != len(intent.penalty_focus_indices):
            raise ValueError("planner_intent.penalty_focus_indices has duplicates.")

        max_index = len(allowed_constraint_names) - 1
        out_of_range = [
            idx
            for idx in intent.penalty_focus_indices
            if idx < 0 or idx > max_index
        ]
        if out_of_range:
            raise ValueError(
                f"planner_intent.penalty_focus_indices out of range: {out_of_range}; expected 0..{max_index}"
            )

        return intent

    def _resolve_seeds_for_final_action(
        self,
        action: Mapping[str, Any],
        *,
        template_params: Mapping[str, Any],
    ) -> tuple[list[Mapping[str, Any]] | None, str | None]:
        def _validate_seed(seed_payload: Mapping[str, Any]) -> Mapping[str, Any]:
            self.make_boundary(seed_payload)
            return copy.deepcopy(dict(seed_payload))

        def _template_seed() -> list[Mapping[str, Any]]:
            seed = copy.deepcopy(dict(template_params))
            self.make_boundary(seed)
            return [seed]

        fallback_raw = action.get("seed_fallback")
        fallback = str(fallback_raw).strip().lower() if fallback_raw is not None else ""
        use_template_fallback = fallback == "template"

        raw_suggested_list = action.get("suggested_params_list")
        if raw_suggested_list is not None:
            if not isinstance(raw_suggested_list, list):
                if use_template_fallback:
                    return _template_seed(), fallback
                raise ValueError("suggested_params_list must be a JSON array.")
            if len(raw_suggested_list) == 0:
                if use_template_fallback:
                    return _template_seed(), fallback
                raise ValueError("suggested_params_list cannot be empty.")
            resolved_list: list[Mapping[str, Any]] = []
            for idx, raw_seed in enumerate(
                raw_suggested_list[:_DEFAULT_AGENT_SEED_CANDIDATE_COUNT]
            ):
                if not isinstance(raw_seed, Mapping):
                    if use_template_fallback:
                        return _template_seed(), fallback
                    raise ValueError(
                        f"suggested_params_list[{idx}] must be a JSON object."
                    )
                try:
                    resolved_list.append(_validate_seed(raw_seed))
                except Exception:
                    if use_template_fallback:
                        return _template_seed(), fallback
                    raise
            if resolved_list:
                return resolved_list, None

        raw_suggested = action.get("suggested_params")
        if raw_suggested is not None:
            if not isinstance(raw_suggested, Mapping):
                if use_template_fallback:
                    return _template_seed(), fallback
                raise ValueError("suggested_params must be a JSON object.")
            try:
                return [_validate_seed(raw_suggested)], None
            except Exception:
                if use_template_fallback:
                    return _template_seed(), fallback
                raise

        if use_template_fallback:
            return _template_seed(), fallback
        return None, None

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
        evaluation = self.evaluate_for_problem(
            cfg.problem,
            params,
            stage=cfg.fidelity_ladder.screen,
        )
        boundary = self.make_boundary(params)

        evaluation_summary = {
            "objective": evaluation.get("objective"),
            "feasibility": evaluation.get("feasibility"),
            "gradient_proxy": evaluation.get("gradient_proxy", evaluation.get("hv")),
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
        success_cases: Sequence[Mapping[str, Any]] = []
        near_success_cases: Sequence[Mapping[str, Any]] = []
        feedback_adapter: Mapping[str, Any] = {}
        experience_injected = self._should_inject_experience(
            cycle_number=cycle_number,
            random_seed=cfg.random_seed,
            experiment_id=experiment_id,
            probability=_DEFAULT_EXPERIENCE_INJECTION_PROBABILITY,
        )
        experience_memo = ""
        scratchpad_summary: Mapping[str, Any] | None = None

        if self.world_model and experiment_id is not None:
            pg = self.world_model.to_networkx(experiment_id)
            graph_dir = Path(cfg.reporting_dir) / "graphs"
            graph_dir.mkdir(parents=True, exist_ok=True)
            graph_file = graph_dir / f"cycle_{cycle_number}.json"

            nodes = [{"id": node_id, **attrs} for node_id, attrs in _graph_nodes(pg)]
            edges = [
                {"src": src, "dst": dst, "attrs": attrs}
                for src, dst, attrs in _graph_edges(pg)
            ]
            snapshot_data = {"nodes": nodes, "edges": edges}
            graph_file.write_text(json.dumps(snapshot_data, indent=2), encoding="utf-8")

            graph_summary = {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "note_count": sum(1 for n in nodes if n.get("type") == "note"),
                "snapshot_path": str(graph_file),
            }

            if hasattr(self.world_model, "recent_experience_pack"):
                experience_pack = self.world_model.recent_experience_pack(
                    experiment_id=experiment_id,
                    problem=cfg.problem,
                    limit_per_bucket=3,
                )
                feedback_adapter = dict(experience_pack.get("feedback_adapter", {}))
                if experience_injected:
                    success_cases = list(experience_pack.get("recent_successes", []))
                    near_success_cases = list(
                        experience_pack.get("recent_near_successes", [])
                    )
                    failure_cases = list(experience_pack.get("recent_failures", []))
                experience_memo = self._build_experience_memo(
                    recent_successes=success_cases,
                    recent_near_successes=near_success_cases,
                    recent_failures=failure_cases,
                )
            else:
                failure_cases = self.world_model.recent_failures(
                    experiment_id=experiment_id,
                    problem=cfg.problem,
                    limit=5,
                )
                experience_memo = self._build_experience_memo(
                    recent_successes=(),
                    recent_near_successes=(),
                    recent_failures=failure_cases,
                )
            if cycle_number > 1 and hasattr(self.world_model, "scratchpad_cycle_summary"):
                scratchpad_summary = self.world_model.scratchpad_cycle_summary(
                    experiment_id=experiment_id,
                    cycle=cycle_number - 1,
                    limit=20,
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
            success_cases=success_cases,
            near_success_cases=near_success_cases,
            feedback_adapter=feedback_adapter,
            experience_memo=experience_memo,
            experience_injected=experience_injected,
            experience_injection_probability=_DEFAULT_EXPERIENCE_INJECTION_PROBABILITY,
            scratchpad_summary=scratchpad_summary,
        )

        suggested_params = None
        suggested_params_list: list[Mapping[str, Any]] | None = None
        config_overrides = None
        planner_intent: PlannerIntent | None = None

        if self.config.agent_gates:
            provider = self.config.get_provider()

            tools_schemas = tools_api.list_tool_schemas()
            available_tools = [
                schema
                for schema in tools_schemas
                if schema["name"] in self.planning_gate.allowed_tools
            ]

            system_prompt = self._build_planning_system_prompt(
                cfg=cfg,
                cycle_number=cycle_number,
                available_tools=available_tools,
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context: {json.dumps(context, default=str)}",
                },
            ]

            max_turns = 5
            for turn in range(max_turns):
                try:
                    response = model_provider.invoke_chat_completion(
                        provider,
                        tool_call={"name": "plan_cycle_turn", "arguments": {}},
                        messages=messages,
                        model=self.planning_gate.provider_model,
                    )

                    if response.status_code != 200:
                        print(
                            f"[planner] Turn {turn}: LLM returned status {response.status_code}"
                        )
                        break

                    content = (
                        response.body.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "{}")
                    )

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
                        messages.append(
                            {
                                "role": "user",
                                "content": "Error: Invalid JSON format. Please output valid JSON for tool call or final plan.",
                            }
                        )
                        continue

                    messages.append({"role": "assistant", "content": content})

                    # Check for final plan
                    if (
                        "suggested_params_list" in action
                        or "suggested_params" in action
                        or "config_overrides" in action
                        or "planner_intent" in action
                        or "seed_fallback" in action
                    ):
                        raw_overrides = action.get("config_overrides")
                        if raw_overrides is not None and not isinstance(
                            raw_overrides, Mapping
                        ):
                            messages.append(
                                {
                                    "role": "user",
                                    "content": "Error: config_overrides must be a JSON object when provided.",
                                }
                            )
                            continue

                        try:
                            planner_intent = self._parse_planner_intent(
                                action.get("planner_intent"),
                                problem=cfg.problem,
                            )
                        except Exception as intent_exc:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": f"Error: Invalid planner_intent payload. {intent_exc}",
                                }
                            )
                            continue

                        try:
                            resolved_seeds, fallback_used = (
                                self._resolve_seeds_for_final_action(
                                    action,
                                    template_params=params,
                                )
                            )
                        except Exception as seed_exc:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": f"Error: Invalid seed payload. {seed_exc}",
                                }
                            )
                            continue

                        if resolved_seeds is None:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": "Error: Finalization requires valid suggested_params or seed_fallback ('template').",
                                }
                            )
                            continue

                        suggested_params_list = list(resolved_seeds)
                        suggested_params = dict(suggested_params_list[0])
                        config_overrides = raw_overrides
                        print(f"[planner] Plan finalized in turn {turn + 1}")
                        if fallback_used:
                            print(
                                f"[planner] Using explicit seed fallback path: {fallback_used}"
                            )
                        else:
                            print(
                                f"[planner] Suggesting {len(suggested_params_list)} boundary seed candidate(s)"
                            )
                        if config_overrides:
                            print(
                                f"[planner] Suggesting config overrides: {config_overrides}"
                            )
                        if planner_intent is not None:
                            print(
                                f"[planner] Suggesting planner intent: {planner_intent.model_dump(mode='json')}"
                            )
                        break

                    # Check for tool call
                    tool_name = action.get("tool")
                    tool_args = action.get("arguments", {})

                    if tool_name:
                        print(
                            f"[planner] Turn {turn}: Agent calling tool '{tool_name}'"
                        )
                        tool_result = "Tool execution failed."
                        try:
                            tool_result = self._execute_planning_tool(
                                tool_name, tool_args
                            )
                        except Exception as tool_exc:
                            tool_result = f"Error executing tool: {tool_exc}"

                        messages.append(
                            {
                                "role": "user",
                                "content": f"Tool '{tool_name}' output: {json.dumps(tool_result, default=str)}",
                            }
                        )
                    else:
                        # No recognized action
                        messages.append(
                            {
                                "role": "user",
                                "content": "Error: No supported action found. Provide a tool call or finalize with suggested_params/seed_fallback.",
                            }
                        )

                except Exception as exc:
                    print(f"[planner] Turn {turn} failed: {exc}")
                    break

        if self.config.agent_gates and suggested_params is None:
            fallback_seed = copy.deepcopy(dict(params))
            self.make_boundary(fallback_seed)
            suggested_params = fallback_seed
            suggested_params_list = [fallback_seed]
            print(
                "[planner] No valid seed returned; applying explicit fallback path: template."
            )

        return PlanningOutcome(
            context=context,
            evaluation_summary=evaluation_summary,
            boundary_summary=boundary_summary,
            rag_snippets=rag_snippets,
            graph_summary=graph_summary,
            suggested_params=suggested_params,
            suggested_params_list=suggested_params_list,
            config_overrides=config_overrides,
            planner_intent=(
                planner_intent.model_dump(mode="json")
                if planner_intent is not None
                else None
            ),
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
        planner_intent: Mapping[str, Any] | None = None,
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
                return self._llm_supervise(
                    diagnostics,
                    cycle,
                    aso_config,
                    planner_intent=planner_intent,
                )
            except Exception as e:
                print(f"[Planner] LLM supervision failed: {e}, using heuristic")
                return heuristic.analyze(diagnostics)
        else:
            return self._llm_supervise(
                diagnostics,
                cycle,
                aso_config,
                planner_intent=planner_intent,
            )

    def _llm_supervise(
        self,
        diagnostics: OptimizerDiagnostics,
        cycle: int,
        aso_config: ASOConfig,
        planner_intent: Mapping[str, Any] | None = None,
    ) -> OptimizationDirective:
        """LLM-based supervision with real ALM state context."""
        # Retrieve relevant context if stagnating
        rag_context = []
        if diagnostics.status == "STAGNATION":
            rag_context = self.retrieve_rag(
                "Strategies for escaping local minima in stellarator ALM optimization",
                k=2,
            )

        system_prompt = self._build_supervision_prompt(
            cycle,
            rag_context,
            diagnostics,
            planner_intent=planner_intent,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Current ALM diagnostics:\n{diagnostics.to_json()}",
            },
        ]

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

                content = (
                    response.body.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "{}")
                )
                return self._parse_directive(content, diagnostics)

            except json.JSONDecodeError as e:
                if attempt < aso_config.llm_max_retries - 1:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Invalid JSON: {e}. Please output valid JSON.",
                        }
                    )
                    continue
                raise

        raise RuntimeError("LLM supervision failed after retries")

    def _build_supervision_prompt(
        self,
        cycle: int,
        rag_context: list,
        diagnostics: OptimizerDiagnostics,
        planner_intent: Mapping[str, Any] | None = None,
    ) -> str:
        rag_section = ""
        if rag_context:
            rag_section = (
                f"\n\nRelevant knowledge:\n{json.dumps(rag_context, indent=2)}"
            )
        intent_section = "\nPlanner intent prior: none."
        if planner_intent:
            intent_section = (
                "\nPlanner intent prior (SOFT PRIOR):\n"
                f"{json.dumps(planner_intent, indent=2)}\n"
                "Treat planner intent as guidance. ALM diagnostics are hard truth."
            )

        constraint_rows = [
            {
                "index": idx,
                "name": c.name,
                "violation": round(c.violation, 6),
                "penalty": round(c.penalty, 6),
                "multiplier": round(c.multiplier, 6),
                "trend": c.trend,
                "delta": round(c.delta, 6),
            }
            for idx, c in enumerate(diagnostics.constraint_diagnostics)
        ]
        constraint_priority = sorted(
            constraint_rows, key=lambda row: float(row["violation"]), reverse=True
        )

        return f"""You are the ASO Supervisor for the AI Scientist (cycle {cycle}).

You have access to REAL Augmented Lagrangian Method (ALM) state:
- objective: Current objective function value
- constraints (with index mapping): {json.dumps(constraint_rows, indent=2)}
- constraint priority (highest violation first): {json.dumps(constraint_priority, indent=2)}
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
    "penalty_parameters": [p1, p2, ...]  // Optional: same length as constraints
  }},
  "reasoning": "brief explanation"
}}

POLICY (feasibility-first, bounded edits):
- Prioritize reducing max violation before objective optimization.
- ADJUST can modify at most 2 indices per call.
- Keep each adjusted penalty within [0.5x, 4.0x] of its current value.
- Do not reorder penalty array; index mapping is strict.
- Use STOP only when feasible and objective change is negligible, or trajectory is clearly hopeless.
- Use RESTART when stagnating with non-trivial violation and tight bounds.
- If multiplier is high while violation stays high, mark that constraint as likely bottleneck in reasoning.
- If planner intent conflicts with diagnostics, override it and include override_reason.
{intent_section}
{rag_section}

Respond with ONLY valid JSON."""

    def _parse_directive(
        self, content: str, diagnostics: OptimizerDiagnostics
    ) -> OptimizationDirective:
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
            override_reason=data.get("override_reason"),
            source=DirectiveSource.LLM,
        )
