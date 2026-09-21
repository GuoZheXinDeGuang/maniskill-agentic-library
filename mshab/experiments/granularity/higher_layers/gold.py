"""Registry, validation, and persistence of the gold graphs.

A *gold graph* is a hand-authored, validated Layer-1/2 skill graph that
exists only for this experiment: it is the answer the scripted proposer
returns, and the reference a real model's output is compared against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from mshab.experiments.granularity.higher_layers.builders import (
    SET_TABLE_GOAL,
    SetTableGraphBuilder,
)
from mshab.experiments.granularity.lower_layers.library import EXPERIMENT_TASK
from mshab.experiments.granularity.paths import DEFAULT_GRAPH_DIR
from mshab.experiments.planning.metrics import graph_summary
from mshab.skills import schema
from mshab.skills.extension import SkillGraphBuilder, SkillGraphPatch
from mshab.skills.graph import SkillGraph, SubGoalGraph
from mshab.skills.library import ContractLibrary
from mshab.skills.plan import SkillPlan, SkillPlanner
from mshab.skills.runtime import SkillGrounder


SCHEMA_VERSION = "mshab.granularity-gold-graph.v1"


@dataclass(frozen=True)
class GoldGraphSpec:
    """How one gold graph is built: its builder, goal, and context."""

    name: str
    source_task: str
    granularity: Optional[str]
    goal: str
    builder: SkillGraphBuilder
    context: Mapping[str, Any] = field(default_factory=dict)


GOLD_GRAPH_SPECS: Tuple[GoldGraphSpec, ...] = (
    GoldGraphSpec(
        "set_table_coarse", "set_table", "coarse", SET_TABLE_GOAL,
        SetTableGraphBuilder("coarse"),
    ),
    GoldGraphSpec(
        "set_table_fine", "set_table", "fine", SET_TABLE_GOAL,
        SetTableGraphBuilder("fine"),
    ),
)
GOLD_GRAPHS: Mapping[str, GoldGraphSpec] = {spec.name: spec for spec in GOLD_GRAPH_SPECS}


@dataclass(frozen=True)
class GoldGraph:
    """A built gold graph: the patch that produced it, both layers, its plan."""

    spec: GoldGraphSpec
    patch: SkillGraphPatch
    subgoal_graph: SubGoalGraph
    skill_graph: SkillGraph
    plan: SkillPlan


def validate_gold_graph(
    subgoal_graph: SubGoalGraph,
    skill_graph: SkillGraph,
    library: Optional[ContractLibrary] = None,
) -> SkillPlan:
    """The three checks every gold graph passes: structure, grounding, a plan."""

    skill_graph.validate()
    if library is not None:
        SkillGrounder(library).grounded_skills(skill_graph)
    return SkillPlanner(subgoal_graph, skill_graph).plan()


def build_gold_graph(
    name: str, library: Optional[ContractLibrary] = None
) -> GoldGraph:
    try:
        spec = GOLD_GRAPHS[name]
    except KeyError as exc:
        raise KeyError(
            "unknown gold graph {!r}; available={}".format(name, sorted(GOLD_GRAPHS))
        ) from exc
    patch = spec.builder.propose(spec.goal, EXPERIMENT_TASK, spec.context)
    subgoal_graph = SubGoalGraph(spec.goal)
    skill_graph = SkillGraph(EXPERIMENT_TASK, subgoal_graph=subgoal_graph)
    patch.apply(subgoal_graph, skill_graph, library=library)
    plan = validate_gold_graph(subgoal_graph, skill_graph, library)
    return GoldGraph(spec, patch, subgoal_graph, skill_graph, plan)


def gold_graph_document(gold: GoldGraph) -> Dict[str, Any]:
    """The committed JSON form: metadata, summary, nominal plan, and the patch."""

    return {
        "schema_version": SCHEMA_VERSION,
        "name": gold.spec.name,
        "task": EXPERIMENT_TASK,
        "source_task": gold.spec.source_task,
        "granularity": gold.spec.granularity,
        "goal": gold.spec.goal,
        "context": json.loads(json.dumps(dict(gold.spec.context))),
        "summary": graph_summary(gold.skill_graph),
        "nominal_plan": gold.plan.as_dict(),
        "patch": gold.patch.as_dict(),
    }


def load_gold_graph(
    path: Path, library: Optional[ContractLibrary] = None
) -> Tuple[SubGoalGraph, SkillGraph]:
    """Rebuild and revalidate both layers from a committed document.

    Only the ``patch`` is authoritative; the summary and nominal plan are
    derived views and are checked against the rebuilt graph.
    """

    where = "gold_graph"
    document = schema.require_mapping(json.loads(Path(path).read_text()), where=where)
    schema.require_keys(
        document,
        where=where,
        required=(
            "schema_version",
            "name",
            "task",
            "source_task",
            "granularity",
            "goal",
            "context",
            "summary",
            "nominal_plan",
            "patch",
        ),
    )
    schema.require_schema_version(document, where=where, expected=SCHEMA_VERSION)
    task = schema.require_identifier(document, "task", where=where)
    goal = schema.require_str(document, "goal", where=where)
    patch = SkillGraphPatch.from_dict(document["patch"], task=task)
    subgoal_graph = SubGoalGraph(goal)
    skill_graph = SkillGraph(task, subgoal_graph=subgoal_graph)
    patch.apply(subgoal_graph, skill_graph, library=library)
    plan = validate_gold_graph(subgoal_graph, skill_graph, library)
    declared = SkillPlan.from_dict(document["nominal_plan"])
    if declared.order != plan.order:
        raise schema.SchemaError("{} nominal_plan is stale".format(where))
    if document["summary"] != graph_summary(skill_graph):
        raise schema.SchemaError("{} summary is stale".format(where))
    return subgoal_graph, skill_graph


def gold_graph_path(name: str, directory: Path = DEFAULT_GRAPH_DIR) -> Path:
    return Path(directory) / "{}.json".format(name)
