"""Agentized planning helper for Phase 3 (docs/roadmap & improvement plan guidance)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai_scientist import agent as agent_module
from ai_scientist import config as ai_config
from ai_scientist import memory
from ai_scientist import rag
from ai_scientist import tools
from ai_scientist import tools_api


class PlanningOutcome:
    """Structured output that mirrors the JSON sections sent to the planning agent prompt."""

    def __init__(
        self,
        context: Mapping[str, Any],
        evaluation_summary: Mapping[str, Any],
        boundary_summary: Mapping[str, Any],
        rag_snippets: Sequence[Mapping[str, str]],
        graph_summary: Mapping[str, Any] | None = None,
    ) -> None:
        self.context = context
        self.evaluation_summary = evaluation_summary
        self.boundary_summary = boundary_summary
        self.rag_snippets = rag_snippets
        self.graph_summary = graph_summary


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
        stage_history: Sequence[Mapping[str, Any]],
        last_summary: tools.P3Summary | None,
        evaluation_summary: Mapping[str, Any],
        boundary_summary: Mapping[str, Any],
        rag_snippets: Sequence[Mapping[str, str]],
        graph_summary: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        context = {
            "cycle_index": cycle_index + 1,
            "planner_role": self.planning_gate.model_alias,
            "budgets": asdict(budgets),
            "stage_history": list(stage_history),
            "previous_p3_summary": _serialize_summary(last_summary),
            "latest_evaluation": evaluation_summary,
            "current_boundary": boundary_summary,
            "rag_snippets": list(rag_snippets),
            "graph_summary": graph_summary,
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
        if self.world_model and experiment_id is not None:
            pg = self.world_model.to_networkx(experiment_id)
            graph_dir = Path(cfg.reporting_dir) / "graphs"
            graph_dir.mkdir(parents=True, exist_ok=True)
            graph_file = graph_dir / f"cycle_{cycle_number}.json"
            
            nodes = [{"id": node, **attrs} for node, attrs in pg.nodes(data=True)]
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

        context = self._build_context(
            cycle_index=cycle_index,
            budgets=cfg.budgets,
            stage_history=stage_history,
            last_summary=last_summary,
            evaluation_summary=evaluation_summary,
            boundary_summary=boundary_summary,
            rag_snippets=rag_snippets,
            graph_summary=graph_summary,
        )
        
        return PlanningOutcome(
            context=context,
            evaluation_summary=evaluation_summary,
            boundary_summary=boundary_summary,
            rag_snippets=rag_snippets,
            graph_summary=graph_summary,
        )
