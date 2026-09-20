"""Structural metrics of a skill graph, shared by gold graphs and run traces."""

from __future__ import annotations

from typing import Any, Dict

from mshab.skills.graph import SkillGraph


def graph_summary(skill_graph: SkillGraph) -> Dict[str, Any]:
    """The structural metrics the granularity experiment records."""

    subgraphs = skill_graph.subgraphs
    nodes = len(skill_graph.nodes)
    return {
        "subgoals": len(skill_graph.subgoal_graph.subgoals),
        "subgraphs": len(subgraphs),
        "nodes": nodes,
        "internal_edges": sum(len(subgraph.edges) for subgraph in subgraphs.values()),
        "cross_edges": len(skill_graph.cross_edges),
        "mean_nodes_per_subgraph": nodes / len(subgraphs) if subgraphs else 0.0,
    }
