"""Hand-authored gold graphs over the five generic granularity contracts.

Every skill node is one contract call, which is one MS-HAB atomic subtask
(navigate, pick, place, open, close): that is the only unit a policy can
execute and the only unit the environment verifies.  With one generic
contract per type there is exactly one candidate node per role, so these
graphs contain no ``FALLBACK_TO`` chains.  What changes between the coarse
and fine TidyHouse variants is only which sub-goal owns each node.

A builder decides two things, an ordered sub-goal sequence and one subgraph
per sub-goal, and hands both to ``assemble_patch``: the same function every
proposer's answers go through.  The gold graphs are therefore reproducible
by a proposer by construction.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

from mshab.experiments.granularity.lower_layers.library import (
    EXPERIMENT_TASK,
    contract_id,
)
from mshab.experiments.planning.proposer import assemble_patch
from mshab.skills.extension import SkillGraphBuilder, SkillGraphPatch
from mshab.skills.graph import SkillNode, SkillRelation, SkillSubgraph, SubGoal


GRANULARITIES = ("coarse", "fine")

# Five transfers of distinct household objects to distinct receptacles.  The
# receptacle names are symbolic; an environment adapter maps them to scene
# entities later.
DEFAULT_TIDY_HOUSE_TRANSFERS: Tuple[Tuple[str, str], ...] = (
    ("002_master_chef_can", "kitchen_counter"),
    ("003_cracker_box", "dining_table"),
    ("004_sugar_box", "coffee_table"),
    ("005_tomato_soup_can", "tv_stand"),
    ("007_tuna_fish_can", "bedside_table"),
)

# The official SetTable order: bowl from the kitchen counter, then apple
# from the fridge.
DEFAULT_SET_TABLE_SEGMENTS: Tuple[Tuple[str, str, str], ...] = (
    ("bowl", "024_bowl", "kitchen_counter"),
    ("apple", "013_apple", "fridge"),
)

TIDY_HOUSE_GOAL = "Tidy the house: move every object to its target receptacle."
SET_TABLE_GOAL = "Set the table with the bowl and apple, then close their storage."


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


def _require_task(builder: str, task: str) -> None:
    if task != EXPERIMENT_TASK:
        raise ValueError(
            "{} builds graphs over the {!r} contracts, not task {!r}".format(
                builder, EXPERIMENT_TASK, task
            )
        )


def _chain_subgraph(
    subgoal_id: str, task: str, nodes: Sequence[SkillNode]
) -> SkillSubgraph:
    """A subgraph whose nodes run in the given order, joined by ``ENABLES``."""

    subgraph = SkillSubgraph(subgoal_id, task)
    for node in nodes:
        subgraph.add_node(node)
    for source, target in zip(nodes, nodes[1:]):
        subgraph.relate(source.id, target.id, SkillRelation.ENABLES)
    return subgraph


def _require_pairs(value: Any, name: str, width: int) -> Tuple[Tuple[str, ...], ...]:
    items = tuple(tuple(item) for item in value)
    if not items:
        raise ValueError("{} must not be empty".format(name))
    for item in items:
        if len(item) != width or not all(
            isinstance(field, str) and field for field in item
        ):
            raise ValueError(
                "{} entries must be {} non-empty strings, got {!r}".format(
                    name, width, item
                )
            )
    return items


class TidyHouseGraphBuilder(SkillGraphBuilder):
    """TidyHouse at one of two Layer-1 granularities over identical nodes.

    ``coarse`` owns each object's four nodes by one result sub-goal
    ``at(object, destination)``.  ``fine`` gives each node its own sub-goal:
    ``reachable(object) -> holding(object) -> reachable(destination) ->
    at(object, destination)``.  Node ids, contracts, arguments, and the
    nominal execution order are the same in both variants.
    """

    def __init__(self, granularity: str) -> None:
        if granularity not in GRANULARITIES:
            raise ValueError(
                "granularity must be one of {}, got {!r}".format(
                    GRANULARITIES, granularity
                )
            )
        self.granularity = granularity

    def propose(
        self, goal: str, task: str, context: Mapping[str, Any]
    ) -> SkillGraphPatch:
        _require_task(type(self).__name__, task)
        transfers = _require_pairs(
            context.get("transfers", DEFAULT_TIDY_HOUSE_TRANSFERS), "transfers", 2
        )
        subgoals: List[SubGoal] = []
        subgraphs: List[SkillSubgraph] = []
        for index, (obj, destination) in enumerate(transfers, start=1):
            navigate_object = "navigate_to_object_{}".format(index)
            pick = "pick_object_{}".format(index)
            navigate_destination = "navigate_to_destination_{}".format(index)
            place = "place_object_{}".format(index)
            delivered = "at({},{})".format(obj, destination)

            if self.granularity == "coarse":
                subgoal_id = "object_{}_delivered".format(index)
                subgoals.append(SubGoal(subgoal_id, delivered))
                subgraphs.append(
                    _chain_subgraph(
                        subgoal_id,
                        task,
                        (
                            navigate_node(navigate_object, obj),
                            pick_node(pick, obj),
                            navigate_node(navigate_destination, destination),
                            place_node(place, obj, destination, (subgoal_id,)),
                        ),
                    )
                )
                continue

            reachable = "object_{}_reachable".format(index)
            holding = "object_{}_holding".format(index)
            destination_reachable = "destination_{}_reachable".format(index)
            placed = "object_{}_placed".format(index)
            for subgoal_id, predicate, node in (
                (reachable, "reachable({})".format(obj), navigate_node(navigate_object, obj, (reachable,))),
                (holding, "holding({})".format(obj), pick_node(pick, obj, (holding,))),
                (
                    destination_reachable,
                    "reachable({})".format(destination),
                    navigate_node(navigate_destination, destination, (destination_reachable,)),
                ),
                (placed, delivered, place_node(place, obj, destination, (placed,))),
            ):
                subgoals.append(SubGoal(subgoal_id, predicate))
                subgraphs.append(_chain_subgraph(subgoal_id, task, (node,)))
        return assemble_patch(task, subgoals, subgraphs)


class SetTableGenericGraphBuilder(SkillGraphBuilder):
    """The official SetTable order expressed with the generic contracts.

    Same eight sub-goals and node roles as the packaged
    ``SetTableGraphBuilder``, but one node per role: the specialized and
    generic candidate pairs of the packaged graph collapse into a single
    node on the ``all`` contract, and policy choice moves to Layer 4.
    """

    def propose(
        self, goal: str, task: str, context: Mapping[str, Any]
    ) -> SkillGraphPatch:
        _require_task(type(self).__name__, task)
        destination = str(context.get("destination", "dining_table"))
        segments = _require_pairs(
            context.get("segments", DEFAULT_SET_TABLE_SEGMENTS), "segments", 3
        )
        subgoals: List[SubGoal] = []
        subgraphs: List[SkillSubgraph] = []
        for label, obj, source in segments:
            open_subgoal = "{}_source_open".format(label)
            retrieved_subgoal = "{}_retrieved".format(label)
            placed_subgoal = "{}_placed".format(label)
            closed_subgoal = "{}_source_closed".format(label)
            for subgoal_id, predicate, nodes in (
                (
                    open_subgoal,
                    "open({})".format(source),
                    (
                        navigate_node("navigate_to_{}_source".format(label), source),
                        open_node("open_{}_source".format(label), source, (open_subgoal,)),
                    ),
                ),
                (
                    retrieved_subgoal,
                    "holding({})".format(obj),
                    (
                        navigate_node("navigate_to_{}".format(label), obj),
                        pick_node("pick_{}".format(label), obj, (retrieved_subgoal,)),
                    ),
                ),
                (
                    placed_subgoal,
                    "at({},{})".format(obj, destination),
                    (
                        navigate_node("navigate_{}_to_destination".format(label), destination),
                        place_node("place_{}".format(label), obj, destination, (placed_subgoal,)),
                    ),
                ),
                (
                    closed_subgoal,
                    "closed({})".format(source),
                    (
                        navigate_node("navigate_back_to_{}_source".format(label), source),
                        close_node("close_{}_source".format(label), source, (closed_subgoal,)),
                    ),
                ),
            ):
                subgoals.append(SubGoal(subgoal_id, predicate))
                subgraphs.append(_chain_subgraph(subgoal_id, task, nodes))
        return assemble_patch(task, subgoals, subgraphs)
