"""Deterministic SVG rendering and artifact generation for the gold graphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
from xml.sax.saxutils import escape

from mshab.experiments.granularity.higher_layers.gold import (
    GOLD_GRAPH_SPECS,
    GoldGraph,
    build_gold_graph,
    gold_graph_document,
    graph_summary,
)
from mshab.experiments.granularity.lower_layers.library import split_contract_id
from mshab.experiments.granularity.paths import DEFAULT_GRAPH_DIR
from mshab.experiments.granularity.svg import text
from mshab.skills.graph import SkillNode
from mshab.skills.library import ContractLibrary


NODE_W = 230
NODE_H = 58
NODE_GAP = 40
LABEL_W = 330
ROW_H = 106
MARGIN = 30
HEADER_H = 104
FOOTER_H = 52
MIN_WIDTH = 1000

_MUTED = "#71809b"
_ROW_FILL = "#f6f3fc"
_ROW_STROKE = "#d6cdf3"
_NODE_STROKE = "#7552d6"
_ACHIEVER_STROKE = "#c6531a"
_ACHIEVER_FILL = "#fffaf2"


def _node_caption(node: SkillNode) -> str:
    _, contract_type, _ = split_contract_id(node.contract_id)
    return "{}({})".format(
        contract_type, ", ".join(str(value) for value in node.arguments.values())
    )


def gold_graph_svg(gold: GoldGraph) -> str:
    """One row per sub-goal, its skill nodes left to right, edges as arrows."""

    subgoal_graph = gold.subgoal_graph
    graph = gold.skill_graph
    order = subgoal_graph.execution_order()
    rows = [(subgoal_id, graph.subgraph_for_subgoal(subgoal_id)) for subgoal_id in order]
    max_nodes = max(len(subgraph.nodes) for _, subgraph in rows)
    width = max(
        MIN_WIDTH,
        2 * MARGIN + LABEL_W + max_nodes * NODE_W + (max_nodes - 1) * NODE_GAP + 20,
    )
    height = HEADER_H + len(rows) * ROW_H + FOOTER_H
    summary = graph_summary(graph)

    positions: Dict[str, Tuple[float, float]] = {}
    for row_index, (_, subgraph) in enumerate(rows):
        y = HEADER_H + row_index * ROW_H + (ROW_H - NODE_H) / 2
        for column, node_id in enumerate(subgraph.execution_order()):
            positions[node_id] = (MARGIN + LABEL_W + column * (NODE_W + NODE_GAP), y)

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        'viewBox="0 0 {w} {h}" role="img" aria-labelledby="title description">'.format(
            w=width, h=height
        ),
        "<title id=\"title\">{}</title>".format(escape(gold.spec.name)),
        "<desc id=\"description\">{}</desc>".format(escape(gold.spec.goal)),
        "<defs>",
        '<marker id="edge" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="{}"/></marker>'.format(_NODE_STROKE),
        '<marker id="cross" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="{}"/></marker>'.format(_ACHIEVER_STROKE),
        "</defs>",
        '<rect width="{}" height="{}" fill="#ffffff"/>'.format(width, height),
        text(MARGIN, 40, gold.spec.name, size=26, weight=700, fill="#6842cb"),
        text(MARGIN, 66, gold.spec.goal, size=14, fill=_MUTED),
        text(
            MARGIN,
            88,
            "{} sub-goals · {} skill nodes · {} internal edges · {} cross-subgraph "
            "edges · {:.1f} nodes per subgraph · task {} · granularity {}".format(
                summary["subgoals"],
                summary["nodes"],
                summary["internal_edges"],
                summary["cross_edges"],
                summary["mean_nodes_per_subgraph"],
                graph.task,
                gold.spec.granularity or "n/a",
            ),
            size=13,
            fill=_MUTED,
        ),
    ]

    for row_index, (subgoal_id, subgraph) in enumerate(rows):
        y = HEADER_H + row_index * ROW_H
        out.append(
            '<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" '
            'stroke="{stroke}" stroke-width="1.5"/>'.format(
                x=MARGIN, y=y + 6, w=width - 2 * MARGIN, h=ROW_H - 12,
                fill=_ROW_FILL, stroke=_ROW_STROKE,
            )
        )
        out.append(text(MARGIN + 18, y + ROW_H / 2 - 4, subgoal_id, size=15, weight=700))
        out.append(
            text(
                MARGIN + 18,
                y + ROW_H / 2 + 17,
                subgoal_graph.subgoals[subgoal_id].predicate,
                size=13,
                fill=_MUTED,
            )
        )
        achievers = {node.id for node in subgraph.achievers}
        for node_id in subgraph.execution_order():
            node = subgraph.nodes[node_id]
            x, node_y = positions[node_id]
            achiever = node_id in achievers
            out.append(
                '<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
                'stroke="{stroke}" stroke-width="{sw}"/>'.format(
                    x=x, y=node_y, w=NODE_W, h=NODE_H,
                    fill=_ACHIEVER_FILL if achiever else "#ffffff",
                    stroke=_ACHIEVER_STROKE if achiever else _NODE_STROKE,
                    sw=2.5 if achiever else 1.6,
                )
            )
            out.append(
                text(x + NODE_W / 2, node_y + 23, node_id, size=12, weight=700, anchor="middle")
            )
            out.append(
                text(
                    x + NODE_W / 2,
                    node_y + 43,
                    _node_caption(node),
                    size=12,
                    fill="#6245b1",
                    anchor="middle",
                )
            )
        for edge in subgraph.edges:
            sx, sy = positions[edge.source]
            tx, ty = positions[edge.target]
            out.append(
                '<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{stroke}" '
                'stroke-width="1.8" marker-end="url(#edge)"/>'.format(
                    x1=sx + NODE_W, y1=sy + NODE_H / 2, x2=tx, y2=ty + NODE_H / 2,
                    stroke=_NODE_STROKE,
                )
            )
            out.append(
                text(
                    (sx + NODE_W + tx) / 2,
                    sy + NODE_H / 2 - 7,
                    edge.relation.value.lower(),
                    size=10,
                    fill=_NODE_STROKE,
                    anchor="middle",
                )
            )

    for edge in graph.cross_edges:
        if edge.source_node is not None:
            sources = (edge.source_node,)
        else:
            sources = tuple(
                node.id for node in graph.subgraph_for_subgoal(edge.source_subgoal).achievers
            )
        if edge.target_node is not None:
            targets = (edge.target_node,)
        else:
            targets = tuple(
                node.id for node in graph.subgraph_for_subgoal(edge.target_subgoal).achievers
            )
        for source in sources:
            for target in targets:
                sx, sy = positions[source]
                tx, ty = positions[target]
                start_y = sy + NODE_H
                bend_y = start_y + (ROW_H - NODE_H) / 2
                out.append(
                    '<path d="M {x1} {y1} V {yb} H {x2} V {y2}" fill="none" '
                    'stroke="{stroke}" stroke-width="1.8" marker-end="url(#cross)"/>'.format(
                        x1=sx + NODE_W / 2, y1=start_y, yb=bend_y,
                        x2=tx + NODE_W / 2, y2=ty, stroke=_ACHIEVER_STROKE,
                    )
                )

    out.append(
        text(
            MARGIN,
            height - 20,
            "row = one sub-goal and its skill subgraph · box = one skill node, one "
            "contract call · orange border = achiever · purple arrow = relation inside "
            "a subgraph · orange arrow = cross-subgraph ENABLES (from any achiever)",
            size=12,
            fill=_MUTED,
        )
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def write_gold_graphs(
    library: ContractLibrary, directory: Path = DEFAULT_GRAPH_DIR
) -> List[str]:
    """Build every gold graph against ``library`` and write JSON plus SVG."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = []
    for spec in GOLD_GRAPH_SPECS:
        gold = build_gold_graph(spec.name, library)
        (directory / "{}.json".format(spec.name)).write_text(
            json.dumps(gold_graph_document(gold), indent=2, sort_keys=True) + "\n"
        )
        (directory / "{}.svg".format(spec.name)).write_text(gold_graph_svg(gold))
        names.append(spec.name)
    return names

