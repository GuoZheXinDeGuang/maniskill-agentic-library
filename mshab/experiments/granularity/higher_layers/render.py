"""Deterministic SVG rendering of two-layer skill graphs, and the gold-graph artifacts.

``skill_graph_svg`` draws any validated ``SubGoalGraph``/``SkillGraph`` pair
one row per sub-goal; ``gold_graph_svg`` is that rendering with a gold
graph's title lines.  The experiment's figures render a proposer's accepted
graphs through the same function, so a model's output and the gold reference
are drawn alike.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple
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
from mshab.skills.graph import SkillGraph, SkillNode, SubGoalGraph
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
_FLAG_STROKE = "#d64545"

_LEGEND = (
    "row = one sub-goal and its skill subgraph · box = one skill node, one "
    "contract call · orange border = achiever · purple arrow = relation inside "
    "a subgraph · orange arrow = cross-subgraph ENABLES (from any achiever)"
)
_FLAG_LEGEND = " · dashed red border = flagged node, tag above it"
# The thing acted on comes first in a caption, the destination last, whatever
# order a proposer's JSON had.
_ARGUMENT_ORDER = {"object": 0, "target": 0, "articulation": 0, "destination": 2}


def node_caption(node: SkillNode) -> str:
    """``type(arg, ...)``: the contract type and the grounded arguments."""

    _, contract_type, _ = split_contract_id(node.contract_id)
    arguments = sorted(node.arguments.items(), key=lambda item: _ARGUMENT_ORDER.get(item[0], 1))
    return "{}({})".format(contract_type, ", ".join(str(value) for _, value in arguments))


def graph_caption(graph: SkillGraph, granularity: Optional[str]) -> str:
    """The one-line structural summary under a graph's title."""

    summary = graph_summary(graph)
    return (
        "{} sub-goals · {} skill nodes · {} internal edges · {} cross-subgraph "
        "edges · {:.1f} nodes per subgraph · task {} · granularity {}".format(
            summary["subgoals"],
            summary["nodes"],
            summary["internal_edges"],
            summary["cross_edges"],
            summary["mean_nodes_per_subgraph"],
            graph.task,
            granularity or "n/a",
        )
    )


def skill_graph_svg(
    subgoal_graph: SubGoalGraph,
    graph: SkillGraph,
    *,
    title: str,
    subtitle: str,
    caption: str,
    flagged: Optional[Mapping[str, str]] = None,
) -> str:
    """One row per sub-goal, its skill nodes left to right, edges as arrows.

    ``flagged`` maps node ids to a short tag; those nodes get a dashed red
    border and the tag above them, which is how a figure marks the nodes of a
    proposal that have no counterpart in the gold graph.  A node id that a
    proposer prefixed with its sub-goal id (``subgoal.node``) is labelled
    without the prefix; the row already names the sub-goal.
    """

    flagged = dict(flagged or {})
    unknown = sorted(set(flagged) - set(graph.nodes))
    if unknown:
        raise KeyError("flagged nodes are not in the graph: {}".format(unknown))
    order = subgoal_graph.execution_order()
    rows = [(subgoal_id, graph.subgraph_for_subgoal(subgoal_id)) for subgoal_id in order]
    max_nodes = max(len(subgraph.nodes) for _, subgraph in rows)
    width = max(
        MIN_WIDTH,
        2 * MARGIN + LABEL_W + max_nodes * NODE_W + (max_nodes - 1) * NODE_GAP + 20,
    )
    height = HEADER_H + len(rows) * ROW_H + FOOTER_H

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
        "<title id=\"title\">{}</title>".format(escape(title)),
        "<desc id=\"description\">{}</desc>".format(escape(subtitle)),
        "<defs>",
        '<marker id="edge" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="{}"/></marker>'.format(_NODE_STROKE),
        '<marker id="cross" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="{}"/></marker>'.format(_ACHIEVER_STROKE),
        "</defs>",
        '<rect width="{}" height="{}" fill="#ffffff"/>'.format(width, height),
        text(MARGIN, 40, title, size=26, weight=700, fill="#6842cb"),
        text(MARGIN, 66, subtitle, size=14, fill=_MUTED),
        text(MARGIN, 88, caption, size=13, fill=_MUTED),
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
            flag = flagged.get(node_id)
            out.append(
                '<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
                'stroke="{stroke}" stroke-width="{sw}"{dash}/>'.format(
                    x=x, y=node_y, w=NODE_W, h=NODE_H,
                    fill=_ACHIEVER_FILL if achiever else "#ffffff",
                    stroke=_FLAG_STROKE if flag is not None
                    else _ACHIEVER_STROKE if achiever else _NODE_STROKE,
                    sw=2.5 if achiever or flag is not None else 1.6,
                    dash=' stroke-dasharray="7 5"' if flag is not None else "",
                )
            )
            if flag is not None:
                out.append(
                    text(
                        x + NODE_W - 4, node_y - 6, flag, size=10, weight=700,
                        fill=_FLAG_STROKE, anchor="end",
                    )
                )
            label = node_id[len(subgoal_id) + 1:] if node_id.startswith(subgoal_id + ".") else node_id
            out.append(
                text(x + NODE_W / 2, node_y + 23, label, size=12, weight=700, anchor="middle")
            )
            out.append(
                text(
                    x + NODE_W / 2,
                    node_y + 43,
                    node_caption(node),
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
            _LEGEND + (_FLAG_LEGEND if flagged else ""),
            size=12,
            fill=_MUTED,
        )
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def gold_graph_svg(gold: GoldGraph) -> str:
    """A gold graph drawn by :func:`skill_graph_svg`, titled with its name and goal."""

    return skill_graph_svg(
        gold.subgoal_graph,
        gold.skill_graph,
        title=gold.spec.name,
        subtitle=gold.spec.goal,
        caption=graph_caption(gold.skill_graph, gold.spec.granularity),
    )


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
