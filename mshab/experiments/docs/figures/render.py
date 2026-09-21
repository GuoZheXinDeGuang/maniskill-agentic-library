"""Figures of the higher-layers experiment.

    python -m mshab.experiments.docs.figures.render                  # render every figure from extracts/
    python -m mshab.experiments.docs.figures.render --extract RUN...  # refresh extracts/ from evaluation runs, then render

Three kinds of figure, all deterministic SVG next to this module:

- ``pipeline.svg``: the whole pipeline, from the request through the two
  proposer calls, the validator, and the controller loop to the trace.
- ``subgoal_chains.svg``: the Layer-1 chains side by side: the two gold
  SetTable decompositions and every DeepSeek decomposition in ``extracts/``.
- one ``deepseek_<granularity>_static.svg`` per static proposal, drawn by the
  gold-graph renderer with the nodes that have no gold counterpart flagged,
  and one ``deepseek_<scenario>_<granularity>_replans.svg`` per scenario run,
  one block per proposal with what the controller did with each node.

``extracts/`` holds compact extracts of evaluation traces (``static.json`` and
``scenario_*.json`` under ``$MSHAB_EXPS_DIR/planning/<run>/``): the accepted
responses, the decisions, and the evaluation numbers, without the contract
inventory every request repeats.  ``--extract`` rewrites them from run
directories, so the figures regenerate from the repository alone.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from mshab.experiments.granularity.higher_layers.gold import GoldGraph, build_gold_graph
from mshab.experiments.granularity.higher_layers.render import (
    graph_caption,
    skill_graph_svg,
)
from mshab.experiments.granularity.higher_layers.scenarios import SCENARIOS
from mshab.experiments.granularity.lower_layers.library import (
    EXPERIMENT_TASK,
    split_contract_id,
)
from mshab.experiments.granularity.paths import REPOSITORY_ROOT
from mshab.experiments.granularity.svg import text
from mshab.experiments.planning.documents import DecompositionResponse, SubgraphResponse
from mshab.experiments.planning.proposer import assemble_patch
from mshab.skills.graph import SkillGraph, SubGoalGraph


FIGURE_DIR = Path(__file__).resolve().parent
DATA_DIR = FIGURE_DIR / "extracts"
DATA_SCHEMA_VERSION = "mshab.experiment-figure-data.v1"

GRANULARITY_ORDER = {"coarse": 0, "free": 1, "fine": 2}

# The palette the experiment's other renderers use.
_TEXT = "#2f3a4d"
_MUTED = "#71809b"
_L1 = "#3484c5"
_L1_FILL = "#eef4fb"
_L2 = "#7552d6"
_L2_FILL = "#f6f3fc"
_CONTRACT = "#c6531a"
_CONTRACT_FILL = "#fffaf2"
_ENV = "#399447"
_ENV_FILL = "#eefaf0"
_GREY = "#94a3b8"
_GREY_FILL = "#f5f6f8"
_FLAG = "#d64545"
_FLAG_FILL = "#ffe9e9"
_WARN = "#d49b12"
_WARN_FILL = "#fff1ba"


# -- SVG primitives ---------------------------------------------------------------------


def _rect(
    x: float, y: float, w: float, h: float, *, fill: str, stroke: str,
    width: float = 1.5, rx: float = 10, dashed: bool = False,
) -> str:
    return (
        '<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        'stroke="{stroke}" stroke-width="{sw}"{dash}/>'
    ).format(
        x=x, y=y, w=w, h=h, rx=rx, fill=fill, stroke=stroke, sw=width,
        dash=' stroke-dasharray="7 5"' if dashed else "",
    )


def _arrow(path: str, *, stroke: str = _TEXT, dashed: bool = False, width: float = 1.8) -> str:
    return (
        '<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{w}" '
        'marker-end="url(#arrow-{marker})"{dash}/>'
    ).format(
        d=path, stroke=stroke, w=width, marker=stroke.lstrip("#"),
        dash=' stroke-dasharray="6 5"' if dashed else "",
    )


def _markers(colours: Iterable[str]) -> str:
    return "".join(
        '<marker id="arrow-{id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="{fill}"/></marker>'.format(
            id=colour.lstrip("#"), fill=colour
        )
        for colour in sorted(set(colours))
    )


def _document(width: float, height: float, title: str, description: str, body: Sequence[str],
              arrow_colours: Iterable[str] = ()) -> str:
    from xml.sax.saxutils import escape

    head = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        'viewBox="0 0 {w} {h}" role="img" aria-labelledby="title description">'.format(
            w=width, h=height
        ),
        '<title id="title">{}</title>'.format(escape(title)),
        '<desc id="description">{}</desc>'.format(escape(description)),
        "<defs>{}</defs>".format(_markers(arrow_colours)),
        '<rect width="{}" height="{}" fill="#ffffff"/>'.format(width, height),
    ]
    return "\n".join(head + list(body) + ["</svg>"]) + "\n"


def _lines(x: float, y: float, lines: Sequence[str], *, size: float = 12, step: float = 17,
           fill: str = _TEXT, weight: int = 400, anchor: str = "start") -> List[str]:
    return [
        text(x, y + index * step, line, size=size, fill=fill, weight=weight, anchor=anchor)
        for index, line in enumerate(lines)
    ]


def _wrap(value: str, width: int, max_lines: int) -> List[str]:
    lines = textwrap.wrap(value, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: width - 1].rstrip() + "…"
    return lines


def _short_type(contract_id: str) -> str:
    _, contract_type, _ = split_contract_id(contract_id)
    return {"navigate": "nav"}.get(contract_type, contract_type)


# -- data extraction --------------------------------------------------------------------


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _run_models(run_dir: Path) -> Tuple[Optional[str], List[str]]:
    """The model the run asked for, and the model names the API reported."""

    requested = None
    summary = run_dir / "summary.json"
    if summary.exists():
        requested = json.loads(summary.read_text()).get("proposer", {}).get("model")
    served: Counter = Counter()
    exchanges = run_dir / "exchanges.json"
    if exchanges.exists():
        for exchange in json.loads(exchanges.read_text()):
            if exchange.get("model"):
                served[exchange["model"]] += 1
    return requested, sorted(served)


def _proposal_record(proposal: Mapping[str, Any]) -> Dict[str, Any]:
    """The part of a validated proposal a figure needs: the accepted answers."""

    if "decomposition" not in proposal:
        return {
            "accepted": False,
            "attempt": proposal.get("attempt"),
            "rejections": proposal.get("rejections", []),
        }
    request = proposal["decomposition"]["request"]
    return {
        "accepted": True,
        "attempt": request["attempt"],
        "failure": request["failure"],
        "history": request["history"],
        "entities": [item["name"] for item in request["entities"]],
        "facts": list(request["facts"]),
        "rounds": len(proposal.get("rounds", [])),
        "decomposition": proposal["decomposition"]["response"],
        "subgraphs": [item["response"] for item in proposal["subgraphs"]],
    }


def extract_run(run_dir: Path, data_dir: Path = DATA_DIR) -> List[Path]:
    """Write the compact extracts of every trace under one evaluation run."""

    run_dir = Path(run_dir)
    requested, served = _run_models(run_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "schema_version": DATA_SCHEMA_VERSION,
        "run": run_dir.name,
        "model": requested,
        "served_models": served,
    }
    written: List[Path] = []
    for path in sorted(run_dir.rglob("static.json")):
        trace = json.loads(path.read_text())
        record = dict(common)
        record.update(
            {
                "kind": "static",
                "source": _relative(path),
                "task": trace["task"],
                "goal": trace["goal"],
                "granularity": trace["granularity"],
                "proposal": _proposal_record(trace),
                "evaluation": trace.get("evaluation", {}),
            }
        )
        target = data_dir / "static_{}.json".format(trace["granularity"])
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        written.append(target)
    for path in sorted(run_dir.rglob("scenario_*.json")):
        trace = json.loads(path.read_text())
        scenario = path.stem[len("scenario_"):]
        record = dict(common)
        record.update(
            {
                "kind": "scenario",
                "source": _relative(path),
                "scenario": scenario,
                "task": trace["task"],
                "goal": trace["goal"],
                "granularity": trace["granularity"],
                "status": trace["status"],
                "success": trace["success"],
                "goal_facts": trace["goal_facts"],
                "final_facts": trace["final_facts"],
                "achieved_predicates": trace["achieved_predicates"],
                "metrics": trace["metrics"],
                "proposals": [_proposal_record(item) for item in trace["proposals"]],
                "replans": trace["replans"],
                "decisions": [
                    {
                        key: item[key]
                        for key in (
                            "step", "node_id", "subgoal_id", "attempt", "outcome",
                            "failure_mode", "missing_effects", "missing_preconditions",
                            "note",
                        )
                    }
                    for item in trace["decisions"]
                ],
            }
        )
        target = data_dir / "scenario_{}_{}.json".format(scenario, trace["granularity"])
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        written.append(target)
    return written


def load_data(data_dir: Path = DATA_DIR) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Every static extract (coarse, free, fine order) and every scenario extract."""

    statics, scenarios = [], []
    for path in sorted(Path(data_dir).glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("schema_version") != DATA_SCHEMA_VERSION:
            raise ValueError("{} is not a figure data file".format(path))
        (statics if record["kind"] == "static" else scenarios).append(record)
    statics.sort(key=lambda item: GRANULARITY_ORDER.get(item["granularity"], 9))
    scenarios.sort(key=lambda item: (item["scenario"], item["granularity"]))
    return statics, scenarios


# -- rebuilding a proposal's graphs -----------------------------------------------------


def build_graphs(goal: str, proposal: Mapping[str, Any]) -> Tuple[SubGoalGraph, SkillGraph]:
    """Both layers from the accepted answers, through the assembler every proposer uses."""

    decomposition = DecompositionResponse.from_dict(proposal["decomposition"])
    subgraphs = [
        SubgraphResponse.from_dict(item, task=EXPERIMENT_TASK).subgraph
        for item in proposal["subgraphs"]
    ]
    patch = assemble_patch(EXPERIMENT_TASK, decomposition.subgoals, subgraphs)
    subgoal_graph = SubGoalGraph(goal)
    skill_graph = SkillGraph(EXPERIMENT_TASK, subgoal_graph=subgoal_graph)
    patch.apply(subgoal_graph, skill_graph)
    skill_graph.validate()
    return subgoal_graph, skill_graph


def _role(node) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    _, contract_type, _ = split_contract_id(node.contract_id)
    return contract_type, tuple(sorted((key, str(value)) for key, value in node.arguments.items()))


def gold_by_predicate(gold: GoldGraph) -> Dict[str, List[Tuple[str, Tuple[Tuple[str, str], ...]]]]:
    """Predicate -> the roles (contract type, arguments) of its gold subgraph."""

    result = {}
    for subgoal_id, subgoal in gold.subgoal_graph.subgoals.items():
        subgraph = gold.skill_graph.subgraph_for_subgoal(subgoal_id)
        result[subgoal.predicate] = [_role(node) for node in subgraph.nodes.values()]
    return result


def flag_extra_nodes(
    subgoal_graph: SubGoalGraph, skill_graph: SkillGraph, gold: GoldGraph
) -> Dict[str, str]:
    """Nodes whose role the gold subgraph of the same predicate does not contain."""

    reference = gold_by_predicate(gold)
    flagged: Dict[str, str] = {}
    for subgoal_id, subgoal in subgoal_graph.subgoals.items():
        subgraph = skill_graph.subgraph_for_subgoal(subgoal_id)
        if subgoal.predicate not in reference:
            for node_id in subgraph.nodes:
                flagged[node_id] = "predicate not in {}".format(gold.spec.name)
            continue
        remaining = list(reference[subgoal.predicate])
        for node_id in subgraph.execution_order():
            role = _role(subgraph.nodes[node_id])
            if role in remaining:
                remaining.remove(role)
            else:
                flagged[node_id] = "not in {}".format(gold.spec.name)
    return flagged


def reference_for(granularity: str) -> Optional[str]:
    """The gold graph a proposal at ``granularity`` is drawn against; none for ``free``."""

    return {"coarse": "set_table_coarse", "fine": "set_table_fine"}.get(granularity)


# -- figure 1: the pipeline -------------------------------------------------------------


def _card(x: float, y: float, w: float, h: float, title: str, lines: Sequence[str], *,
          stroke: str, fill: str = "#ffffff", title_fill: Optional[str] = None,
          size: float = 12) -> List[str]:
    out = [_rect(x, y, w, h, fill=fill, stroke=stroke, width=2)]
    out.append(text(x + 14, y + 24, title, size=14, weight=700, fill=title_fill or stroke))
    out.extend(_lines(x + 14, y + 46, lines, size=size, step=size + 5))
    return out


def pipeline_svg() -> str:
    """The whole pipeline: proposal, execution loop, fixed lower layers, references."""

    W, H = 1560, 1010
    body: List[str] = [
        text(30, 40, "Higher-layers experiment: the pipeline", size=26, weight=700, fill="#6842cb"),
        text(
            30, 66,
            "request → proposer (two calls) → validator → controller loop over an environment → "
            "trace and metrics · the lower layers and the gold graphs stay fixed · stages 1–6 of "
            "higher-layers-plan.md",
            size=14, fill=_MUTED,
        ),
    ]

    # Band 1: proposal.
    body.append(_rect(30, 90, 1500, 290, fill=_L1_FILL, stroke=_L1, width=1.5, rx=18))
    body.append(text(48, 114, "Layers 1–2 · proposal", size=15, weight=700, fill=_L1))
    body.append(text(240, 114, "simulator-independent · mshab/experiments/planning/", size=12, fill=_MUTED))
    body.extend(_card(
        50, 130, 250, 220, "Request",
        [
            "DecompositionRequest / SubgraphRequest",
            "goal text",
            "entities: objects, storages, receptacle",
            "facts: present(...), closed(...), gripper_empty(), ...",
            "contracts: 5 × Contract.as_dict()",
            "granularity: free | coarse | fine",
            "history + failure   (on a replan)",
            "rejections            (on a retry)",
        ],
        stroke=_GREY, title_fill=_TEXT,
    ))
    body.append(_rect(340, 130, 330, 220, fill="#ffffff", stroke=_L2, width=2))
    body.append(text(354, 154, "Proposer  (GraphProposer) · stages 3, 5, 6", size=14, weight=700, fill=_L2))
    body.append(_rect(354, 168, 302, 54, fill=_L2_FILL, stroke=_L2, width=1.2, rx=8))
    body.extend(_lines(366, 189, ["call 1  decompose(request)", "→ ordered sub-goals, one predicate each (Layer 1)"], size=12, step=16))
    body.append(_rect(354, 232, 302, 54, fill=_L2_FILL, stroke=_L2, width=1.2, rx=8))
    body.extend(_lines(366, 253, ["call 2  plan_subgraph(sub-goal), once per sub-goal", "→ one SkillSubgraph of contract calls (Layer 2)"], size=12, step=16))
    body.extend(_lines(354, 306, [
        "DeepSeekProposer: one chat completion per call, JSON",
        "ScriptedProposer: tables cut from the gold graphs",
        "SetTableRuleProposer: rules, for the MS-HAB rollout",
    ], size=11, step=14, fill=_MUTED))
    body.extend(_card(
        710, 160, 200, 134, "assemble_patch",
        [
            "Layer 1: the sequence with",
            "consecutive dependencies",
            "Layer 2: the subgraphs + one",
            "ENABLES cross edge into each",
            "root of the next sub-goal",
            "→ one SkillGraphPatch",
        ],
        stroke=_L2, title_fill=_L2, size=11,
    ))
    body.extend(_card(
        950, 130, 300, 220, "ProposalValidator · retries = 2",
        [
            "strict from_dict of every answer",
            "ground every node against the library",
            "apply the patch to fresh graphs, validate",
            "achiever effects ⊇ sub-goal predicate",
            "SkillPlanner.plan() over the whole graph",
            "accepted → ValidatedProposal (+ rounds)",
            "rejected → Rejection(stage, sub-goal, msg)",
        ],
        stroke=_CONTRACT, title_fill=_CONTRACT,
    ))
    body.extend(_card(
        1290, 130, 220, 220, "Accepted graphs",
        [
            "Layer 1  SubGoalGraph",
            "  ordered chain of sub-goals",
            "Layer 2  SkillGraph",
            "  one subgraph per sub-goal,",
            "  each node = one contract call",
            "nominal plan (SkillPlanner)",
            "every request, response, and",
            "rejection, verbatim",
        ],
        stroke=_L1, title_fill=_L1,
    ))
    body.append(_arrow("M 300 240 H 338"))
    body.append(_arrow("M 670 227 H 708"))
    body.append(_arrow("M 910 227 H 948"))
    body.append(_arrow("M 1250 240 H 1288"))
    body.append(_arrow("M 1100 350 V 368 H 505 V 352", stroke=_CONTRACT, dashed=True))
    body.append(text(802, 364, "rejections: the same request asked again (≤ 2 retries); a sub-goal rejection repeats only that subgraph call", size=11, fill=_CONTRACT, anchor="middle"))

    # Between the bands: the accepted graphs go down to the controller, replans go back up.
    body.append(_arrow("M 1400 350 V 406 H 400 V 458", stroke=_L1))
    body.append(text(900, 402, "accepted proposal: both layers and the plan", size=11, fill=_L1, anchor="middle"))
    body.append(_arrow("M 120 420 V 394 H 320 V 240 H 338", stroke=_ENV, dashed=True))
    body.append(text(126, 389, "Layer-1 replan (≤ 2 per run)", size=11, fill=_ENV))

    # Band 2: execution.
    body.append(_rect(30, 420, 1500, 290, fill=_ENV_FILL, stroke=_ENV, width=1.5, rx=18))
    body.append(text(48, 444, "controller loop · execute → observe → re-decide", size=15, weight=700, fill=_ENV))
    body.append(text(470, 444, "stage 4 on the symbolic environment, stage 6 on MS-HAB · the loop itself never changes", size=12, fill=_MUTED))
    body.extend(_card(
        50, 460, 400, 222, "TaskController",
        [
            "1  absorb facts: an untouched sub-goal whose predicate",
            "    already holds is achieved, its nodes are skipped",
            "2  SkillPlanner.decide(completed, failed) → one node:",
            "    prerequisites, the achiever, then its follow-ups",
            "3  SkillRuntime.execute_node: admission (preconditions,",
            "    invariants) → policy run → verify the effects",
            "4  success → completed · failure → retry, ≤ 2 attempts",
            "5  NoViableCandidate → Failure(sub-goal, node, mode) → replan",
        ],
        stroke=_ENV, title_fill=_ENV, size=11,
    ))
    body.extend(_card(
        490, 460, 310, 222, "SkillRuntime + PolicyExecutor",
        [
            "select_policy(contract, grounded arguments):",
            "  the target's own checkpoint first, then all",
            "SymbolicPolicyExecutor  (stages 4–5)",
            "  add the effects, retract the deletes, or a",
            "  ScriptedFailure keyed by type and target",
            "CheckpointPolicyExecutor  (stage 6)",
            "  point_at(plan subtask), run SAC / PPO until",
            "  MS-HAB's subtask checker passes",
        ],
        stroke=_L2, title_fill=_L2, size=11,
    ))
    body.extend(_card(
        840, 460, 330, 222, "EnvironmentAdapter",
        [
            "SymbolicEnvironmentAdapter  (stages 4–5)",
            "  a set of predicates + two world rules:",
            "  holding(x) retracts gripper_empty() and at(x,·);",
            "  reaching y retracts every other reachable(·)",
            "RolloutEnvironmentAdapter / SkillRollout-v0  (stage 6)",
            "  facts = MS-HAB's own checkers: holding = grasp,",
            "  at = inside the goal, reachable = navigation",
            "  success, collision_safe = force limit",
        ],
        stroke=_ENV, title_fill=_ENV, size=11,
    ))
    body.extend(_card(
        1210, 460, 300, 222, "RunResult trace",
        [
            "proposals: every round verbatim",
            "decisions: node, attempt, outcome,",
            "  facts added and removed",
            "replans: the failure that caused each",
            "metrics: executions, retries, skipped,",
            "  redundant, replanning_span,",
            "  goal_facts_achieved, recovery_success",
            "status: success · goal_not_reached ·",
            "  proposal_rejected · replans_exhausted",
        ],
        stroke=_GREY, title_fill=_TEXT, size=11,
    ))
    body.append(_arrow("M 450 545 H 488"))
    body.append(text(469, 536, "execute", size=10, fill=_MUTED, anchor="middle"))
    body.append(_arrow("M 800 545 H 838"))
    body.append(text(819, 536, "step", size=10, fill=_MUTED, anchor="middle"))
    body.append(_arrow("M 1005 682 V 700 H 250 V 684", stroke=_ENV))
    body.append(text(628, 696, "snapshot: the current facts", size=10, fill=_ENV, anchor="middle"))
    body.append(_arrow("M 1170 600 H 1208"))
    body.append(text(1189, 591, "record", size=10, fill=_MUTED, anchor="middle"))

    # Band 3: fixed lower layers and references.
    body.append(_rect(30, 730, 1500, 230, fill=_GREY_FILL, stroke=_GREY, width=1.5, rx=18))
    body.append(text(48, 754, "fixed: Layers 3–4 and the references", size=15, weight=700, fill=_TEXT))
    body.extend(_card(
        50, 770, 720, 170, "ContractLibrary · stage 1  (mshab/experiments/granularity/lower_layers/)",
        [
            "Layer 3   five generic contracts: navigate · pick · place · open · close, ids mshab.granularity.<type>.all;",
            "              each with parameters, preconditions, effects, deletes, invariants, and a horizon",
            "Layer 4   53 checkpoint policies from the MS-HAB download, each bound to the contract of its type;",
            "              task_families=('set_table',) binds the 11 SetTable checkpoints for the rollout;",
            "              one SymbolicPolicy per contract for the symbolic environment",
            "a proposer only ever names a contract and its arguments; which policy runs is the library's choice",
        ],
        stroke=_CONTRACT, title_fill=_CONTRACT, size=11,
    ))
    body.extend(_card(
        810, 770, 700, 170, "Gold graphs and measurement · stages 2, 5  (higher_layers/, evaluate.py)",
        [
            "gold graphs: hand-authored, validated, committed as JSON + SVG under granularity/graphs/",
            "    set_table_coarse  2 sub-goals / 16 nodes · set_table_fine  16 / 16",
            "the ScriptedProposer answers from them; a model's decomposition and subgraphs are compared with them",
            "    (exact match, order similarity, predicate precision / recall, node and edge Jaccard, role P / R)",
            "scenarios with injected failures, identical for every proposer: nominal, pick_fails_once, pick_exhausted,",
            "    object_dropped, object_already_delivered, storage_already_open, object_elsewhere",
        ],
        stroke=_L1, title_fill=_L1, size=11,
    ))
    body.append(_arrow("M 640 770 V 684", stroke=_CONTRACT))
    body.append(text(648, 726, "contracts, policies", size=10, fill=_CONTRACT))
    body.append(_arrow("M 175 770 V 720 H 40 V 240 H 48", stroke=_CONTRACT, dashed=True))
    body.append(text(185, 726, "Contract.as_dict() records on every request", size=10, fill=_CONTRACT))

    body.append(text(
        30, H - 22,
        "blue = proposal (simulator-independent) · green = execution · grey = fixed lower layers and references · "
        "solid arrow = data flow · dashed arrow = a loop back (retry, replan) · every stage is exercised on CPU by the "
        "test suite; only stage 6 needs the simulator",
        size=12, fill=_MUTED,
    ))
    return _document(
        W, H,
        "Higher-layers experiment: the pipeline",
        "From the request through the proposer, the validator, and the controller loop to the trace.",
        body,
        arrow_colours=(_TEXT, _CONTRACT, _L1, _ENV),
    )


# -- figure 2: the sub-goal chains side by side ------------------------------------------


def _predicate_style(predicate: str) -> Tuple[str, str]:
    name = predicate.split("(", 1)[0]
    return {
        "at": (_CONTRACT_FILL, _CONTRACT),
        "holding": (_L2_FILL, _L2),
        "reachable": (_L1_FILL, _L1),
        "open": (_ENV_FILL, _ENV),
        "closed": (_ENV_FILL, _ENV),
    }.get(name, ("#ffffff", _GREY))


class ChainColumn:
    """One decomposition to draw: sub-goals in order with their node counts."""

    def __init__(self, title: str, subtitle: str, source: str, note: str,
                 subgoals: Sequence[Tuple[str, str, int]]) -> None:
        self.title = title
        self.subtitle = subtitle
        self.source = source
        self.note = note
        self.subgoals = list(subgoals)  # (id, predicate, node count)


def _gold_column(name: str) -> ChainColumn:
    gold = build_gold_graph(name)
    subgoals = [
        (
            subgoal_id,
            gold.subgoal_graph.subgoals[subgoal_id].predicate,
            len(gold.skill_graph.subgraph_for_subgoal(subgoal_id).nodes),
        )
        for subgoal_id in gold.subgoal_graph.execution_order()
    ]
    return ChainColumn(
        "gold · {}".format(gold.spec.granularity),
        "hand-authored reference",
        "granularity/graphs/{}.json".format(name),
        "",
        subgoals,
    )


def _proposal_subgoals(proposal: Mapping[str, Any]) -> List[Tuple[str, str, int]]:
    counts = {
        item["subgraph"]["subgoal_id"]: len(item["subgraph"]["nodes"])
        for item in proposal["subgraphs"]
    }
    return [
        (item["id"], item["predicate"], counts.get(item["id"], 0))
        for item in proposal["decomposition"]["subgoals"]
    ]


def _entity_order_note(entities: Sequence[str]) -> str:
    return "entities alphabetical" if list(entities) == sorted(entities) else "entities in plan order"


def _agreement_note(record: Mapping[str, Any]) -> str:
    """``coarse: exact · fine: order 0.12``, from the sweep's decomposition agreement."""

    agreement = record.get("evaluation", {}).get("decomposition", {}).get("agreement", {})
    parts = []
    for granularity in ("coarse", "fine"):
        name = reference_for(granularity)
        if name in agreement:
            item = agreement[name]
            parts.append("{}: {}".format(
                granularity,
                "exact" if item["exact"] else "order {:.2f}".format(item["order_similarity"]),
            ))
    return "agreement " + " · ".join(parts) if parts else ""


def gold_predicates() -> Set[str]:
    """Every sub-goal predicate of the two gold graphs: the vocabulary a chain is checked against."""

    return {
        subgoal.predicate
        for granularity in ("coarse", "fine")
        for subgoal in build_gold_graph(reference_for(granularity)).subgoal_graph.subgoals.values()
    }


def _off_gold(subgoals: Sequence[Tuple[str, str, int]], vocabulary: Set[str]) -> Tuple[int, int]:
    """How many of a column's predicates fall outside the gold vocabulary, and how many there are."""

    return sum(1 for _, predicate, _ in subgoals if predicate not in vocabulary), len(subgoals)


def _static_column(record: Mapping[str, Any]) -> ChainColumn:
    proposal = record["proposal"]
    return ChainColumn(
        "DeepSeek · {}".format(record["granularity"]),
        "static proposal",
        "run {}".format(record["run"]),
        "; ".join(filter(None, [_entity_order_note(proposal["entities"]), _agreement_note(record)])),
        _proposal_subgoals(proposal),
    )


def _scenario_column(record: Mapping[str, Any], vocabulary: Set[str]) -> ChainColumn:
    proposal = record["proposals"][0]
    subgoals = _proposal_subgoals(proposal)
    off, total = _off_gold(subgoals, vocabulary)
    return ChainColumn(
        "DeepSeek · {}".format(record["granularity"]),
        "initial proposal of the {} run".format(record["scenario"]),
        "run {}".format(record["run"]),
        "{}; {} of {} predicates outside the gold vocabulary".format(
            _entity_order_note(proposal["entities"]), off, total
        ),
        subgoals,
    )


def subgoal_chains_svg(statics: Sequence[Mapping[str, Any]],
                       scenarios: Sequence[Mapping[str, Any]]) -> str:
    """Every Layer-1 chain as a column, one row per skill node: a sub-goal spans its subgraph."""

    coarse = build_gold_graph(reference_for("coarse"))
    vocabulary = gold_predicates()
    columns = [_gold_column(reference_for("coarse")), _gold_column(reference_for("fine"))]
    columns += [_static_column(record) for record in statics]
    scenario_columns = [
        _scenario_column(record, vocabulary)
        for record in scenarios
        if record["proposals"] and record["proposals"][0].get("accepted")
    ]
    columns += scenario_columns
    off = sum(_off_gold(column.subgoals, vocabulary)[0] for column in scenario_columns + [
        _static_column(record) for record in statics
    ])
    total = sum(_off_gold(column.subgoals, vocabulary)[1] for column in scenario_columns + [
        _static_column(record) for record in statics
    ])

    rows = max(sum(nodes for _, _, nodes in column.subgoals) for column in columns)
    LEFT, LABEL_W, COL_W, COL_GAP = 30, 250, 238, 18
    HEADER, ROW_H, FOOTER = 204, 34, 70
    # Wide enough for the header and legend lines even with the two gold columns alone.
    width = max(1460, LEFT * 2 + LABEL_W + len(columns) * (COL_W + COL_GAP))
    height = HEADER + rows * ROW_H + FOOTER

    body: List[str] = [
        text(LEFT, 40, "Layer-1 sub-goal chains: gold references and DeepSeek decompositions", size=26, weight=700, fill="#6842cb"),
        text(LEFT, 66, "SetTable, two objects out of two storages · each column is one decomposition, top to bottom in execution order · "
             "a sub-goal spans one row per skill node of its subgraph", size=14, fill=_MUTED),
        text(LEFT, 88, "the goal does not say which storage holds which object; the gold graphs assume the official episode "
             "(bowl in the kitchen counter drawer, apple in the fridge);", size=13, fill=_MUTED),
        text(LEFT, 106, "a dashed red box is a predicate outside the gold vocabulary ({} of {} in the DeepSeek columns)".format(off, total),
             size=13, fill=_MUTED),
    ]

    # Segment bands: the gold coarse sub-goals, one band each, as many rows as nodes.
    y = HEADER
    for index, subgoal_id in enumerate(coarse.subgoal_graph.execution_order()):
        nodes = len(coarse.skill_graph.subgraph_for_subgoal(subgoal_id).nodes)
        band_h = nodes * ROW_H
        body.append(_rect(LEFT, y, width - 2 * LEFT, band_h, fill=_GREY_FILL if index % 2 else "#ffffff", stroke="none", rx=0))
        body.append(text(LEFT + 12, y + band_h / 2 - 4, subgoal_id, size=13, weight=700))
        body.append(text(LEFT + 12, y + band_h / 2 + 14, coarse.subgoal_graph.subgoals[subgoal_id].predicate, size=11, fill=_MUTED))
        y += band_h

    for column_index, column in enumerate(columns):
        x = LEFT + LABEL_W + column_index * (COL_W + COL_GAP)
        body.append(text(x, HEADER - 74, column.title, size=13, weight=700))
        body.append(text(x, HEADER - 58, column.subtitle, size=11, weight=700, fill=_MUTED))
        body.extend(_lines(x, HEADER - 43, _wrap(column.source, 42, 1) + _wrap(column.note, 42, 2), size=10, step=13, fill=_MUTED))
        y = HEADER
        for subgoal_id, predicate, nodes in column.subgoals:
            span = max(nodes, 1)
            h = span * ROW_H - 6
            fill, stroke = _predicate_style(predicate)
            matched = predicate in vocabulary
            body.append(_rect(x, y + 3, COL_W, h, fill=fill, stroke=stroke if matched else _FLAG, width=1.4 if matched else 2, rx=8, dashed=not matched))
            if span >= 3:
                body.append(text(x + 10, y + 25, subgoal_id, size=12, weight=700))
                body.append(text(x + 10, y + 43, predicate, size=11, fill=_TEXT))
                body.append(text(x + 10, y + 61, "{} skill node{}".format(nodes, "" if nodes == 1 else "s"), size=10, fill=_MUTED))
            else:
                body.append(text(x + 10, y + 3 + h / 2 + 4, predicate, size=11, fill=_TEXT))
                if nodes != 1:
                    body.append(text(x + COL_W - 8, y + 3 + h / 2 + 4, "n={}".format(nodes), size=10, weight=700, fill=_L2, anchor="end"))
            if not matched:
                body.append(text(x + COL_W - 8, y + 17, "≠ gold", size=10, weight=700, fill=_FLAG, anchor="end"))
            y += span * ROW_H

    body.append(text(
        LEFT, height - 38,
        "box = one sub-goal (Layer 1) · fill: orange at(object, receptacle), purple holding(object), blue reachable(x), green open(x) / closed(x) · "
        "rows = skill nodes in the sub-goal's subgraph",
        size=12, fill=_MUTED,
    ))
    body.append(text(
        LEFT, height - 20,
        "dashed red = a predicate that is in neither gold graph; the sub-goal ids are the proposer's own",
        size=12, fill=_MUTED,
    ))
    return _document(
        width, height,
        "Layer-1 sub-goal chains",
        "Gold SetTable decompositions next to every DeepSeek decomposition, one row per skill node.",
        body,
    )


# -- figure 3: a static proposal's node graph --------------------------------------------


def static_proposal_svg(record: Mapping[str, Any]) -> str:
    proposal = record["proposal"]
    subgoal_graph, skill_graph = build_graphs(record["goal"], proposal)
    reference = reference_for(record["granularity"])
    flagged: Dict[str, str] = {}
    note = "no gold graph at this granularity; compared with none"
    if reference is not None:
        flagged = flag_extra_nodes(subgoal_graph, skill_graph, build_gold_graph(reference))
        note = "{} node{} without a counterpart in {}".format(
            len(flagged), "" if len(flagged) == 1 else "s", reference
        ) if flagged else "every node has its counterpart in {}".format(reference)
    served = ", ".join(record.get("served_models") or []) or "n/a"
    return skill_graph_svg(
        subgoal_graph,
        skill_graph,
        title="DeepSeek · {} · static proposal".format(record["granularity"]),
        subtitle="{} · run {} · model {} (API reported {}) · {} round{}".format(
            record["goal"], record["run"], record.get("model") or "n/a", served,
            proposal.get("rounds", 1), "" if proposal.get("rounds", 1) == 1 else "s",
        ),
        caption="{} · {}".format(graph_caption(skill_graph, record["granularity"]), note),
        flagged=flagged,
    )


# -- figure 4: the proposals of one scenario run ------------------------------------------


_SHORT_FAILURE = {
    "precondition_violated": "precondition",
    "scripted_failure": "scripted",
    "object_dropped": "dropped",
    "no_matching_subtask": "no subtask",
    "invariant_violated": "invariant",
}


def _short_failure(mode: Optional[str]) -> str:
    if not mode:
        return ""
    short = _SHORT_FAILURE.get(mode, mode)
    return short if len(short) <= 13 else short[:12] + "…"


_OUTCOME_STYLE = {
    "success": ("#dff4df", _ENV),
    "failed": (_FLAG_FILL, _FLAG),
    "admission_failed": (_WARN_FILL, _WARN),
    "skipped": ("#eeeeee", _GREY),
}


def _decisions_by_proposal(record: Mapping[str, Any]) -> List[Dict[str, List[Dict[str, Any]]]]:
    """Per proposal, node id -> the decisions taken while that proposal was active."""

    boundaries = [item["after_step"] for item in record["replans"]]
    buckets: List[Dict[str, List[Dict[str, Any]]]] = [dict() for _ in record["proposals"]]
    for decision in record["decisions"]:
        index = sum(1 for boundary in boundaries if decision["step"] > boundary)
        index = min(index, len(buckets) - 1)
        buckets[index].setdefault(decision["node_id"], []).append(decision)
    return buckets


def _subgoal_state(node_ids: Sequence[str], achievers: Sequence[str],
                   decisions: Mapping[str, List[Dict[str, Any]]]) -> Tuple[str, str, str]:
    """(label, fill, stroke) of one sub-goal from what its nodes did."""

    touched = [item for node_id in node_ids for item in decisions.get(node_id, [])]
    if not touched:
        return "not reached", "#ffffff", _GREY
    if any(item["outcome"] == "skipped" for item in touched):
        return "skipped: already true", "#eeeeee", _GREY
    if any(item["outcome"] == "success" for node_id in achievers for item in decisions.get(node_id, [])):
        return "achieved", "#dff4df", _ENV
    if any(item["outcome"] in ("failed", "admission_failed") for item in touched):
        return "failed → replan", _FLAG_FILL, _FLAG
    return "interrupted", _WARN_FILL, _WARN


def _failure_text(failure: Optional[Mapping[str, Any]]) -> str:
    if not failure:
        return ""
    missing = list(failure.get("missing_effects", [])) + list(failure.get("missing_preconditions", []))
    return "{} {}{}".format(
        failure["node_id"].split(".")[-1], failure["failure_mode"],
        " · missing {}".format(", ".join(missing)) if missing else "",
    )


def scenario_replans_svg(record: Mapping[str, Any]) -> str:
    scenario = SCENARIOS.get(record["scenario"])
    goal_facts = list(record["goal_facts"])
    metrics = record["metrics"]
    buckets = _decisions_by_proposal(record)

    BOX_W, BOX_GAP, LEFT = 272, 14, 30
    BLOCK_H, HEADER, FOOTER = 190, 118, 92
    proposals = record["proposals"]
    max_subgoals = max((len(p["decomposition"]["subgoals"]) for p in proposals if p.get("accepted")), default=1)
    width = max(1000, LEFT * 2 + max_subgoals * (BOX_W + BOX_GAP) + 200)
    height = HEADER + len(proposals) * BLOCK_H + FOOTER
    served = ", ".join(record.get("served_models") or []) or "n/a"

    body: List[str] = [
        text(LEFT, 40, "DeepSeek · scenario {} · {}".format(record["scenario"], record["granularity"]), size=26, weight=700, fill="#6842cb"),
        text(LEFT, 66, scenario.description if scenario else record["goal"], size=14, fill=_MUTED),
        text(
            LEFT, 88,
            "status {} · goal facts {} / {} · {} executions ({} failed, {} admission failure{}, {} retr{}) · {} replan{} · "
            "run {} · model {} (API reported {})".format(
                record["status"], metrics.get("goal_facts_achieved"), metrics.get("goal_facts"),
                metrics.get("node_executions"), metrics.get("failed_executions"),
                metrics.get("admission_failures"), "" if metrics.get("admission_failures") == 1 else "s",
                metrics.get("retries"), "y" if metrics.get("retries") == 1 else "ies",
                metrics.get("replans"), "" if metrics.get("replans") == 1 else "s",
                record["run"], record.get("model") or "n/a", served,
            ),
            size=13, fill=_MUTED,
        ),
    ]

    for index, proposal in enumerate(proposals):
        y = HEADER + index * BLOCK_H
        if index:
            body.append(_arrow("M {x} {y1} V {y2}".format(x=LEFT + 60, y1=y - 14, y2=y + 2), stroke=_ENV, dashed=True))
        if not proposal.get("accepted"):
            body.append(text(LEFT, y + 20, "proposal {} · rejected".format(index), size=14, weight=700, fill=_FLAG))
            body.extend(_lines(LEFT, y + 40, [str(item) for item in proposal.get("rejections", [])][:3], size=11, step=14, fill=_MUTED))
            continue
        heading = "proposal {} · initial plan (attempt 0)".format(index) if not proposal.get("failure") else (
            "proposal {} · replan after step {} · triggered by {}".format(
                index, record["replans"][index - 1]["after_step"] if index - 1 < len(record["replans"]) else "?",
                _failure_text(proposal["failure"]),
            )
        )
        body.append(text(LEFT, y + 20, heading, size=14, weight=700))
        body.append(text(LEFT, y + 38, "request: {} · achieved so far: {} · facts: {}".format(
            _entity_order_note(proposal["entities"]),
            ", ".join(proposal["history"]["achieved_subgoals"]) or "none",
            ", ".join(fact for fact in proposal["facts"] if not fact.startswith("present(") and fact not in ("collision_safe()", "gripper_empty()")) or "only the initial ones",
        ), size=11, fill=_MUTED))
        rationale = _wrap("rationale: " + proposal["decomposition"].get("rationale", ""), 190, 2)
        body.extend(_lines(LEFT, y + 54, rationale, size=11, step=14, fill=_MUTED))

        subgoal_graph, skill_graph = build_graphs(record["goal"], proposal)
        decisions = buckets[index]
        for column, subgoal_id in enumerate(subgoal_graph.execution_order()):
            subgraph = skill_graph.subgraph_for_subgoal(subgoal_id)
            predicate = subgoal_graph.subgoals[subgoal_id].predicate
            node_ids = subgraph.execution_order()
            achievers = [node.id for node in subgraph.achievers]
            label, fill, stroke = _subgoal_state(node_ids, achievers, decisions)
            x = LEFT + column * (BOX_W + BOX_GAP)
            box_y = y + 86
            on_target = predicate in goal_facts
            body.append(_rect(x, box_y, BOX_W, 90, fill=fill, stroke=stroke if on_target else _FLAG, width=1.6, rx=9, dashed=not on_target))
            body.append(text(x + 10, box_y + 18, subgoal_id, size=11, weight=700))
            body.append(text(x + 10, box_y + 34, predicate, size=10.5, fill=_TEXT))
            body.append(text(x + BOX_W - 8, box_y + 18, label, size=10, weight=700, fill=stroke, anchor="end"))
            if not on_target:
                body.append(text(x + BOX_W - 8, box_y + 34, "≠ goal fact", size=10, weight=700, fill=_FLAG, anchor="end"))
            # The node strip: one small box per skill node, coloured by what the controller did.
            strip_w = (BOX_W - 20 - 6 * (len(node_ids) - 1)) / max(len(node_ids), 1)
            for slot, node_id in enumerate(node_ids):
                node = subgraph.nodes[node_id]
                taken = decisions.get(node_id, [])
                outcome = taken[-1]["outcome"] if taken else None
                node_fill, node_stroke = _OUTCOME_STYLE.get(outcome, ("#ffffff", _GREY))
                nx = x + 10 + slot * (strip_w + 6)
                body.append(_rect(nx, box_y + 48, strip_w, 30, fill=node_fill, stroke=node_stroke, width=1.3, rx=6,
                                  dashed=outcome == "skipped"))
                caption = _short_type(node.contract_id)
                if len(taken) > 1:
                    caption += " ×{}".format(len(taken))
                body.append(text(nx + strip_w / 2, box_y + 62, caption, size=10, weight=700, fill=node_stroke, anchor="middle"))
                detail = ""
                if taken:
                    failed = [item for item in taken if item["outcome"] in ("failed", "admission_failed")]
                    detail = _short_failure(failed[-1]["failure_mode"]) if failed else "step {}".format(taken[-1]["step"])
                body.append(text(nx + strip_w / 2, box_y + 73, detail, size=8.5, fill=_MUTED, anchor="middle"))

    final_state = [
        fact for fact in record["final_facts"] if fact.startswith(("at(", "open(", "closed("))
    ]
    body.append(text(LEFT, height - 60, "goal facts:  " + " · ".join(goal_facts), size=11, fill=_MUTED))
    body.append(text(LEFT, height - 44, "final facts: " + (" · ".join(final_state) or "no object delivered"), size=11, fill=_MUTED))
    body.append(text(
        LEFT, height - 20,
        "box = one sub-goal of that proposal, in order · small boxes = its skill nodes with the controller's last outcome: "
        "green success, red failed, yellow admission refused, grey dashed skipped, white never run · ×k = attempts · "
        "dashed red border = destination not among the goal facts",
        size=11, fill=_MUTED,
    ))
    return _document(
        width, height,
        "DeepSeek proposals in scenario {}".format(record["scenario"]),
        "Every proposal of the run, the failure that triggered each replan, and what the controller did with every node.",
        body,
        arrow_colours=(_ENV,),
    )


# -- writing everything -----------------------------------------------------------------


def write_figures(directory: Path = FIGURE_DIR, data_dir: Path = DATA_DIR) -> List[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    statics, scenarios = load_data(data_dir)
    outputs: List[Tuple[str, str]] = [
        ("pipeline.svg", pipeline_svg()),
        ("subgoal_chains.svg", subgoal_chains_svg(statics, scenarios)),
    ]
    for record in statics:
        outputs.append(("deepseek_{}_static.svg".format(record["granularity"]), static_proposal_svg(record)))
    for record in scenarios:
        outputs.append((
            "deepseek_{}_{}_replans.svg".format(record["scenario"], record["granularity"]),
            scenario_replans_svg(record),
        ))
    written = []
    for name, svg in outputs:
        path = directory / name
        path.write_text(svg)
        written.append(path)
    return written


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Render the higher-layers experiment figures.")
    parser.add_argument(
        "--extract", nargs="*", type=Path, default=None, metavar="RUN_DIR",
        help="evaluation run directories ($MSHAB_EXPS_DIR/planning/<run>) to extract into extracts/ first",
    )
    parser.add_argument("--output", type=Path, default=FIGURE_DIR)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    if args.extract:
        for run_dir in args.extract:
            for path in extract_run(run_dir, args.data):
                print("extracted {}".format(path))
    for path in write_figures(args.output, args.data):
        print("wrote {}".format(path))


if __name__ == "__main__":
    main()
