"""Proposers, the graph assembler, and the scripted pseudo proposer.

A :class:`GraphProposer` answers the two calls of the boundary.  It is also a
``SkillGraphBuilder``: ``propose()`` runs both calls and hands the result to
:func:`assemble_patch`, so the existing ``build()`` path keeps working.
"""

from __future__ import annotations

import json
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from mshab.experiments.planning.documents import (
    DecompositionRequest,
    DecompositionResponse,
    Neighbours,
    PlanningContext,
    SubgraphRequest,
    SubgraphResponse,
)
from mshab.skills import schema
from mshab.skills.extension import SkillGraphBuilder, SkillGraphPatch
from mshab.skills.graph import (
    CrossSubgraphEdge,
    SkillRelation,
    SkillSubgraph,
    SubGoal,
    SubGoalDependency,
)


def subgraph_requests(
    task: str,
    goal: str,
    context: PlanningContext,
    decomposition: DecompositionResponse,
) -> Tuple[SubgraphRequest, ...]:
    """One independent request per sub-goal, with its neighbours as context."""

    subgoals = decomposition.subgoals
    requests = []
    for index, subgoal in enumerate(subgoals):
        previous = subgoals[index - 1].predicate if index > 0 else None
        following = subgoals[index + 1].predicate if index + 1 < len(subgoals) else None
        requests.append(
            SubgraphRequest(
                task=task,
                goal=goal,
                subgoal=subgoal,
                contracts=context.contracts,
                entities=context.entities,
                facts=context.facts,
                neighbours=Neighbours(previous, following),
                images=context.images,
            )
        )
    return tuple(requests)


def root_nodes(subgraph: SkillSubgraph) -> Tuple[str, ...]:
    """Nodes with no causal prerequisite inside their subgraph, in execution order."""

    roots = tuple(
        node_id
        for node_id in subgraph.execution_order()
        if not subgraph.prerequisite_groups(node_id)
    )
    if not roots:
        raise ValueError(
            "skill subgraph {!r} has no root node".format(subgraph.subgoal_id)
        )
    return roots


def assemble_patch(
    task: str,
    subgoals: Sequence[SubGoal],
    subgraphs: Sequence[SkillSubgraph],
) -> SkillGraphPatch:
    """Combine an ordered sub-goal sequence and its subgraphs into one patch.

    Layer 1 is the ordered sequence with consecutive dependencies.  Layer 2 is
    the subgraphs plus one sub-goal-level ``ENABLES`` cross edge from every
    sub-goal to each root node of the next one.  The source endpoint is left
    open, so any achiever of the earlier sub-goal enables the later one; this
    is the shape ``SkillGraph.validate()`` requires of every Layer-1
    dependency.  The assembler never invents or reorders sub-goals.  The
    hand-authored gold graphs and every proposer go through this one function.
    """

    subgoals = tuple(subgoals)
    owners = [subgraph.subgoal_id for subgraph in subgraphs]
    expected = [subgoal.id for subgoal in subgoals]
    if owners != expected:
        raise ValueError(
            "subgraphs {} do not match the decomposition {}".format(owners, expected)
        )
    for subgraph in subgraphs:
        if subgraph.task != task:
            raise ValueError(
                "subgraph {!r} belongs to task {!r}, not {!r}".format(
                    subgraph.subgoal_id, subgraph.task, task
                )
            )
    dependencies = tuple(
        SubGoalDependency(earlier.id, later.id)
        for earlier, later in zip(subgoals, subgoals[1:])
    )
    cross_edges = []
    for earlier, later, subgraph in zip(subgoals, subgoals[1:], subgraphs[1:]):
        for root in root_nodes(subgraph):
            cross_edges.append(
                CrossSubgraphEdge(earlier.id, later.id, None, root, SkillRelation.ENABLES)
            )
    return SkillGraphPatch(
        subgoals=tuple(subgoals),
        subgoal_dependencies=dependencies,
        skill_subgraphs=tuple(subgraphs),
        cross_edges=tuple(cross_edges),
    )


class GraphProposer(SkillGraphBuilder):
    """Anything that answers the two planning calls.

    ``propose()`` makes a proposer usable wherever a ``SkillGraphBuilder`` is:
    the ``context`` is a :class:`PlanningContext` or its ``as_dict()`` form.
    Errors propagate; :class:`~mshab.experiments.planning.validator.ProposalValidator`
    is the path that turns them into structured rejections.
    """

    @abstractmethod
    def decompose(self, request: DecompositionRequest) -> DecompositionResponse:
        """Call 1: the goal as an ordered sub-goal sequence."""

    @abstractmethod
    def plan_subgraph(self, request: SubgraphRequest) -> SubgraphResponse:
        """Call 2: one skill subgraph for one sub-goal, planned independently."""

    def propose(
        self, goal: str, task: str, context: Mapping[str, Any]
    ) -> SkillGraphPatch:
        planning = (
            context
            if isinstance(context, PlanningContext)
            else PlanningContext.from_dict(context)
        )
        decomposition = self.decompose(DecompositionRequest(task, goal, planning))
        subgraphs = [
            self.plan_subgraph(request).subgraph
            for request in subgraph_requests(task, goal, planning, decomposition)
        ]
        return assemble_patch(task, decomposition.subgoals, subgraphs)


class ProposerUnavailable(RuntimeError):
    """The proposer could not be reached at all: a transport or credential
    failure, not a bad answer.  The validator does not turn it into a
    rejection; it propagates to whoever runs the experiment."""


class UnscriptedRequest(KeyError):
    """The scripted proposer has no canned answer for this request."""


DecompositionKey = Tuple[str, str, str, int, Optional[str]]
SubgraphKey = Tuple[str, str, str]
SCRIPT_SCHEMA_VERSION = "mshab.planning-script.v1"


class ScriptedProposer(GraphProposer):
    """The pseudo proposer: canned answers keyed by request fingerprint.

    A decomposition is keyed by ``(task, goal, granularity, attempt, failed
    sub-goal id)``; a subgraph by ``(task, sub-goal id, predicate)``.  An
    unknown fingerprint raises instead of answering with a default, so a test
    cannot pass on an answer nobody wrote.  Tables are scripted in code
    (``mshab.experiments.granularity.higher_layers.scenarios`` cuts them out
    of the gold graphs) or loaded from JSON.  A retry of a request (its
    ``rejections`` filled in) keeps its fingerprint and gets the same answer.
    """

    def __init__(
        self,
        decompositions: Mapping[DecompositionKey, DecompositionResponse] = (),
        subgraphs: Mapping[SubgraphKey, SubgraphResponse] = (),
    ) -> None:
        self._decompositions: Dict[DecompositionKey, DecompositionResponse] = {}
        self._subgraphs: Dict[SubgraphKey, SubgraphResponse] = {}
        for key, response in dict(decompositions).items():
            self.script_decomposition(key, response)
        for key, response in dict(subgraphs).items():
            self.script_subgraph(key, response)

    # -- scripting ----------------------------------------------------------

    def script_decomposition(
        self, key: DecompositionKey, response: DecompositionResponse
    ) -> None:
        if not isinstance(response, DecompositionResponse):
            raise TypeError("a scripted decomposition must be a DecompositionResponse")
        self._decompositions[tuple(key)] = response

    def script_subgraph(self, key: SubgraphKey, response: SubgraphResponse) -> None:
        if not isinstance(response, SubgraphResponse):
            raise TypeError("a scripted subgraph must be a SubgraphResponse")
        key = tuple(key)
        if key[1] != response.subgraph.subgoal_id:
            raise ValueError(
                "scripted subgraph for sub-goal {!r} is owned by {!r}".format(
                    key[1], response.subgraph.subgoal_id
                )
            )
        self._subgraphs[key] = response

    @property
    def decomposition_keys(self) -> Tuple[DecompositionKey, ...]:
        return tuple(sorted(self._decompositions, key=repr))

    @property
    def subgraph_keys(self) -> Tuple[SubgraphKey, ...]:
        return tuple(sorted(self._subgraphs))

    def plan_subgraph_by_key(self, key: SubgraphKey) -> SubgraphResponse:
        return self._subgraphs[tuple(key)]

    # -- the two calls ------------------------------------------------------

    def decompose(self, request: DecompositionRequest) -> DecompositionResponse:
        key = request.fingerprint()
        try:
            return self._decompositions[key]
        except KeyError:
            raise UnscriptedRequest(
                "no scripted decomposition for {} (task, goal, granularity, attempt, "
                "failed sub-goal); scripted={}".format(key, list(self.decomposition_keys))
            ) from None

    def plan_subgraph(self, request: SubgraphRequest) -> SubgraphResponse:
        key = request.fingerprint()
        try:
            return self._subgraphs[key]
        except KeyError:
            raise UnscriptedRequest(
                "no scripted subgraph for {} (task, sub-goal id, predicate); "
                "scripted={}".format(key, list(self.subgraph_keys))
            ) from None

    # -- JSON -----------------------------------------------------------------

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCRIPT_SCHEMA_VERSION,
            "decompositions": [
                {
                    "task": key[0],
                    "goal": key[1],
                    "granularity": key[2],
                    "attempt": key[3],
                    "failed_subgoal": key[4],
                    "response": self._decompositions[key].as_dict(),
                }
                for key in self.decomposition_keys
            ],
            "subgraphs": [
                {
                    "task": key[0],
                    "subgoal_id": key[1],
                    "predicate": key[2],
                    "response": self._subgraphs[key].as_dict(),
                }
                for key in self.subgraph_keys
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScriptedProposer":
        where = "planning_script"
        payload = schema.require_mapping(payload, where=where)
        schema.require_schema_version(payload, where=where, expected=SCRIPT_SCHEMA_VERSION)
        schema.require_keys(
            payload, where=where, required=("schema_version", "decompositions", "subgraphs")
        )
        proposer = cls()
        for index, raw in enumerate(schema.require_sequence(payload, "decompositions", where=where)):
            entry_where = "{}.decompositions[{}]".format(where, index)
            entry = schema.require_mapping(raw, where=entry_where)
            schema.require_keys(
                entry,
                where=entry_where,
                required=("task", "goal", "granularity", "attempt", "failed_subgoal", "response"),
            )
            key = (
                schema.require_identifier(entry, "task", where=entry_where),
                schema.require_str(entry, "goal", where=entry_where),
                schema.require_str(entry, "granularity", where=entry_where),
                schema.require_non_negative_int(entry, "attempt", where=entry_where),
                schema.optional_identifier(entry, "failed_subgoal", where=entry_where),
            )
            proposer.script_decomposition(key, DecompositionResponse.from_dict(entry["response"]))
        for index, raw in enumerate(schema.require_sequence(payload, "subgraphs", where=where)):
            entry_where = "{}.subgraphs[{}]".format(where, index)
            entry = schema.require_mapping(raw, where=entry_where)
            schema.require_keys(
                entry, where=entry_where, required=("task", "subgoal_id", "predicate", "response")
            )
            task = schema.require_identifier(entry, "task", where=entry_where)
            key = (
                task,
                schema.require_identifier(entry, "subgoal_id", where=entry_where),
                schema.require_str(entry, "predicate", where=entry_where),
            )
            proposer.script_subgraph(key, SubgraphResponse.from_dict(entry["response"], task=task))
        return proposer

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> "ScriptedProposer":
        return cls.from_dict(json.loads(Path(path).read_text()))
