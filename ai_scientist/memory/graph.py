"""Graph structure for the AI Scientist world model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class PropertyGraph:
    nodes: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    edges: list[tuple[str, str, Mapping[str, Any]]] = field(default_factory=list)

    def add_node(self, node_id: str, **attrs: Any) -> None:
        self.nodes[node_id] = attrs

    def add_edge(self, src: str, dst: str, **attrs: Any) -> None:
        self.edges.append((src, dst, attrs))

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes
