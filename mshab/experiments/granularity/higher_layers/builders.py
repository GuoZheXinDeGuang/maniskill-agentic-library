"""Hand-authored gold graphs over the five generic granularity contracts.

Every skill node is one contract call, which is one MS-HAB atomic subtask
(navigate, pick, place, open, close): that is the only unit a policy can
execute and the only unit the environment verifies.  With one generic
contract per type there is exactly one candidate node per role, so these
graphs contain no ``FALLBACK_TO`` chains.  What changes between the coarse
and fine SetTable variants is only which sub-goal owns each node.

A builder decides two things, an ordered sub-goal sequence and one subgraph
per sub-goal, and hands both to ``assemble_patch``: the same function every
proposer's answers go through.  The gold graphs are therefore reproducible
by a proposer by construction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mshab.experiments.granularity.lower_layers.library import (
    EXPERIMENT_TASK,
    contract_id,
)
from mshab.experiments.planning.proposer import assemble_patch
from mshab.skills.extension import SkillGraphBuilder, SkillGraphPatch
from mshab.skills.graph import SkillNode, SkillRelation, SkillSubgraph, SubGoal


# The two Layer-1 granularities the gold graphs are authored at.  The proposer
# boundary (``mshab.experiments.planning.GRANULARITIES``) adds ``free`` for a
# model that chooses its own decomposition; no gold graph exists for that.
GOLD_GRANULARITIES = ("coarse", "fine")

# The official SetTable episode: the bowl from the kitchen counter drawer, then
# the apple from the fridge, both onto the dining table.  A segment is
# ``(label, object, source)``; the label names the segment's sub-goals and
# nodes.  The names are symbolic; the rollout maps them to scene entities.
DEFAULT_SET_TABLE_SEGMENTS: Tuple[Tuple[str, str, str], ...] = (
    ("bowl", "024_bowl", "kitchen_counter"),
    ("apple", "013_apple", "fridge"),
)
DEFAULT_SET_TABLE_DESTINATION = "dining_table"

# The goal names the results only.  It does not say which storage holds which
# object: a proposer that opens the wrong one has to notice and plan around it.
SET_TABLE_GOAL = "Set the table with the bowl and apple, then close their storage."

# The four steps of one segment, in the official order.  Each step is two
# atomic subtasks: a navigation and the manipulation it prepares.
SEGMENT_STEPS = ("open", "pick", "place", "close")
_STEP_ROLES = {
    "open": ("navigate_to_source", "open"),
    "pick": ("navigate_to_object", "pick"),
    "place": ("navigate_to_destination", "place"),
    "close": ("navigate_back_to_source", "close"),
}
SEGMENT_ROLES = tuple(role for step in SEGMENT_STEPS for role in _STEP_ROLES[step])
# The fine sub-goal of each role: one verifiable state transition per node.
FINE_KINDS = {
    "navigate_to_source": "source_reachable",
    "open": "source_open",
    "navigate_to_object": "reachable",
    "pick": "holding",
    "navigate_to_destination": "destination_reachable",
    "place": "placed",
    "navigate_back_to_source": "source_reachable_again",
    "close": "source_closed",
}
COARSE_KINDS = ("delivered", "storage_closed")
SUBGOAL_KINDS = COARSE_KINDS + tuple(FINE_KINDS[role] for role in SEGMENT_ROLES)


def coarse_subgoal_id(label: str) -> str:
    """The one sub-goal of a segment in the coarse graph: ``at(object, destination)``.

    Its subgraph is the whole segment: opening the storage and fetching the
    object prepare the place node, the achiever; closing the storage again
    follows it as the achiever's follow-up.
    """

    return "{}_delivered".format(label)


def closing_subgoal_id(label: str) -> str:
    """The coarse sub-goal of a segment that has nothing left but its closing.

    A replan uses it when an object was given up or already delivered while
    its storage still stands open: ``closed(source)`` is then the result.
    """

    return "{}_storage_closed".format(label)


def fine_subgoal_ids(label: str) -> Tuple[str, ...]:
    """The eight sub-goals of a segment in the fine graph, in execution order."""

    return tuple("{}_{}".format(label, FINE_KINDS[role]) for role in SEGMENT_ROLES)


def segment_subgoal(
    subgoal_id: str, labels: Sequence[str]
) -> Optional[Tuple[str, str]]:
    """``(label, kind)`` of a sub-goal id one of the builders wrote, or ``None``.

    The inverse of :func:`coarse_subgoal_id`, :func:`closing_subgoal_id`, and
    :func:`fine_subgoal_ids` for the given segment labels: ``bowl_holding``
    is ``("bowl", "holding")``.  An id a real model invented gives ``None``.
    """

    for label in labels:
        for kind in SUBGOAL_KINDS:
            if subgoal_id == "{}_{}".format(label, kind):
                return label, kind
    return None


def segment_node_ids(label: str) -> Dict[str, str]:
    """Role -> node id of one segment; the same ids in both granularities."""

    return {
        "navigate_to_source": "navigate_to_{}_source".format(label),
        "open": "open_{}_source".format(label),
        "navigate_to_object": "navigate_to_{}".format(label),
        "pick": "pick_{}".format(label),
        "navigate_to_destination": "navigate_{}_to_destination".format(label),
        "place": "place_{}".format(label),
        "navigate_back_to_source": "navigate_back_to_{}_source".format(label),
        "close": "close_{}_source".format(label),
    }


def navigate_node(node_id: str, target: str, achieves: Sequence[str] = ()) -> SkillNode:
    return SkillNode(node_id, contract_id("navigate"), {"target": target}, tuple(achieves))


def pick_node(node_id: str, obj: str, achieves: Sequence[str] = ()) -> SkillNode:
    return SkillNode(node_id, contract_id("pick"), {"object": obj}, tuple(achieves))


def place_node(
    node_id: str, obj: str, destination: str, achieves: Sequence[str] = ()
) -> SkillNode:
    return SkillNode(
        node_id,
        contract_id("place"),
        {"object": obj, "destination": destination},
        tuple(achieves),
    )


def open_node(node_id: str, articulation: str, achieves: Sequence[str] = ()) -> SkillNode:
    return SkillNode(
        node_id, contract_id("open"), {"articulation": articulation}, tuple(achieves)
    )


def close_node(node_id: str, articulation: str, achieves: Sequence[str] = ()) -> SkillNode:
    return SkillNode(
        node_id, contract_id("close"), {"articulation": articulation}, tuple(achieves)
    )


def role_node(
    role: str,
    node_id: str,
    obj: str,
    source: str,
    destination: str,
    achieves: Sequence[str] = (),
) -> SkillNode:
    """The skill node of one segment role with its grounded arguments."""

    if role in ("navigate_to_source", "navigate_back_to_source"):
        return navigate_node(node_id, source, achieves)
    if role == "navigate_to_object":
        return navigate_node(node_id, obj, achieves)
    if role == "navigate_to_destination":
        return navigate_node(node_id, destination, achieves)
    if role == "open":
        return open_node(node_id, source, achieves)
    if role == "pick":
        return pick_node(node_id, obj, achieves)
    if role == "place":
        return place_node(node_id, obj, destination, achieves)
    if role == "close":
        return close_node(node_id, source, achieves)
    raise ValueError("unknown segment role {!r}".format(role))


def role_predicate(role: str, obj: str, source: str, destination: str) -> str:
    """The state transition one segment role establishes."""

    if role in ("navigate_to_source", "navigate_back_to_source"):
        return "reachable({})".format(source)
    if role == "navigate_to_object":
        return "reachable({})".format(obj)
    if role == "navigate_to_destination":
        return "reachable({})".format(destination)
    if role == "open":
        return "open({})".format(source)
    if role == "pick":
        return "holding({})".format(obj)
    if role == "place":
        return "at({},{})".format(obj, destination)
    if role == "close":
        return "closed({})".format(source)
    raise ValueError("unknown segment role {!r}".format(role))


def _require_task(builder: str, task: str) -> None:
    if task != EXPERIMENT_TASK:
        raise ValueError(
            "{} builds graphs over the {!r} contracts, not task {!r}".format(
                builder, EXPERIMENT_TASK, task
            )
        )


def chain_subgraph(
    subgoal_id: str, task: str, nodes: Sequence[SkillNode]
) -> SkillSubgraph:
    """A subgraph whose nodes run in the given order, joined by ``ENABLES``."""

    subgraph = SkillSubgraph(subgoal_id, task)
    for node in nodes:
        subgraph.add_node(node)
    for source, target in zip(nodes, nodes[1:]):
        subgraph.relate(source.id, target.id, SkillRelation.ENABLES)
    return subgraph


def require_segments(value: Any) -> Tuple[Tuple[str, str, str], ...]:
    """``(label, object, source)`` triples with distinct labels and objects."""

    items = tuple(tuple(item) for item in value)
    if not items:
        raise ValueError("segments must not be empty")
    for item in items:
        if len(item) != 3 or not all(isinstance(field, str) and field for field in item):
            raise ValueError(
                "segments entries must be (label, object, source) non-empty strings, "
                "got {!r}".format(item)
            )
    labels = [label for label, _, _ in items]
    if len(set(labels)) != len(labels):
        raise ValueError("segment labels must be distinct, got {!r}".format(labels))
    objects = [obj for _, obj, _ in items]
    if len(set(objects)) != len(objects):
        raise ValueError("segment objects must be distinct, got {!r}".format(objects))
    return items


def require_steps(
    value: Any, labels: Sequence[str]
) -> Dict[str, Tuple[str, ...]]:
    """The steps each segment still needs, in the official order.

    A segment absent from ``value`` needs every step.  ``open`` is only ever
    done to ``pick``, and ``pick`` only to ``place``; ``close`` stands alone
    for a storage that must be shut with nothing left to fetch from it.
    """

    steps = {label: SEGMENT_STEPS for label in labels}
    for label, chosen in dict(value or {}).items():
        if label not in steps:
            raise ValueError(
                "steps name segment {!r}; segments are {}".format(label, list(labels))
            )
        chosen = tuple(chosen)
        unknown = [step for step in chosen if step not in SEGMENT_STEPS]
        if unknown:
            raise ValueError(
                "unknown steps {} for segment {!r}; steps are {}".format(
                    unknown, label, list(SEGMENT_STEPS)
                )
            )
        ordered = tuple(step for step in SEGMENT_STEPS if step in chosen)
        if not ordered:
            raise ValueError("segment {!r} needs at least one step".format(label))
        if "open" in ordered and "pick" not in ordered:
            raise ValueError("segment {!r} opens its storage without picking".format(label))
        if "pick" in ordered and "place" not in ordered:
            raise ValueError("segment {!r} picks its object without placing".format(label))
        steps[label] = ordered
    return steps


class SetTableGraphBuilder(SkillGraphBuilder):
    """SetTable at one of two Layer-1 granularities over identical nodes.

    ``coarse`` owns each segment by one result sub-goal ``at(object,
    destination)``: opening the storage and fetching the object prepare the
    place node, closing the storage again follows it.  ``fine`` gives each
    node its own sub-goal: ``reachable(source) -> open(source) ->
    reachable(object) -> holding(object) -> reachable(destination) ->
    at(object, destination) -> reachable(source) -> closed(source)``.  Node
    ids, contracts, arguments, and the nominal execution order are the same
    in both variants.

    The context names the ``segments`` (``(label, object, source)`` triples),
    the ``destination``, and optionally the ``steps`` each segment still
    needs (a subset of ``open``, ``pick``, ``place``, ``close``).  A proposer
    that replans an episode passes the steps its facts leave open, so a
    storage that already stands open is not opened again and a held object is
    not picked again; a segment with only ``close`` left becomes the sub-goal
    ``closed(source)``.
    """

    def __init__(self, granularity: str) -> None:
        if granularity not in GOLD_GRANULARITIES:
            raise ValueError(
                "granularity must be one of {}, got {!r}".format(
                    GOLD_GRANULARITIES, granularity
                )
            )
        self.granularity = granularity

    def propose(
        self, goal: str, task: str, context: Mapping[str, Any]
    ) -> SkillGraphPatch:
        _require_task(type(self).__name__, task)
        segments = require_segments(context.get("segments", DEFAULT_SET_TABLE_SEGMENTS))
        destination = str(context.get("destination", DEFAULT_SET_TABLE_DESTINATION))
        if not destination:
            raise ValueError("destination must be a non-empty entity name")
        steps = require_steps(context.get("steps"), [label for label, _, _ in segments])
        subgoals: List[SubGoal] = []
        subgraphs: List[SkillSubgraph] = []
        for label, obj, source in segments:
            roles = [role for step in steps[label] for role in _STEP_ROLES[step]]
            ids = segment_node_ids(label)
            if self.granularity == "coarse":
                if "place" in steps[label]:
                    subgoal_id = coarse_subgoal_id(label)
                    achiever = "place"
                else:
                    subgoal_id = closing_subgoal_id(label)
                    achiever = "close"
                predicate = role_predicate(achiever, obj, source, destination)
                subgoals.append(SubGoal(subgoal_id, predicate))
                subgraphs.append(
                    chain_subgraph(
                        subgoal_id,
                        task,
                        [
                            role_node(
                                role,
                                ids[role],
                                obj,
                                source,
                                destination,
                                (subgoal_id,) if role == achiever else (),
                            )
                            for role in roles
                        ],
                    )
                )
                continue
            for role in roles:
                subgoal_id = "{}_{}".format(label, FINE_KINDS[role])
                subgoals.append(
                    SubGoal(subgoal_id, role_predicate(role, obj, source, destination))
                )
                subgraphs.append(
                    chain_subgraph(
                        subgoal_id,
                        task,
                        (role_node(role, ids[role], obj, source, destination, (subgoal_id,)),),
                    )
                )
        return assemble_patch(task, subgoals, subgraphs)
