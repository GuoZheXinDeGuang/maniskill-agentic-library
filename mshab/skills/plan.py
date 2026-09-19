"""Selecting one execution path out of the candidate graph.

The Layer-2 graph is a *candidate* structure: a sub-goal subgraph may hold several
achievers, and ``SkillGraph.execution_order()`` is therefore only a
partial order over candidates -- never a plan.  Something has to choose.  That
chooser is a rule-based planner here and a trained decision model later; both
answer the same question, one node at a time:

    graph + completed + failed  ->  the next SkillNode

Keeping the interface incremental (rather than "emit the whole sequence") is
what lets a future model re-decide after every execution, and what lets a
failed primary candidate hand off to its ``FALLBACK_TO`` successor.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Set, Tuple

from mshab.skills import schema
from mshab.skills.graph import (
    SubGoalGraph,
    SkillSubgraph,
    SkillGraph,
    SkillNode,
    SkillRelation,
)


class NoViableCandidate(RuntimeError):
    """Every achiever for a sub-goal has been exhausted.

    The sub-goal as a whole has failed.  Recovery is a Layer-1 replan of the
    remaining goal, not another candidate from this graph.
    """


@dataclass(frozen=True)
class SkillPlan:
    """One resolved execution path: exactly one achiever per sub-goal."""

    selections: Mapping[str, str]
    order: Tuple[str, ...]

    def __post_init__(self) -> None:
        selections = dict(self.selections)
        order = tuple(self.order)
        if len(order) != len(set(order)):
            raise ValueError("a skill plan cannot execute the same node twice")
        missing = sorted(set(selections.values()) - set(order))
        if missing:
            raise ValueError(
                "selected achievers missing from skill-plan order {}".format(missing)
            )
        object.__setattr__(self, "selections", MappingProxyType(selections))
        object.__setattr__(self, "order", order)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkillPlan":
        where = "skill_plan"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload, where=where, required=("selections", "order")
        )
        raw_selections = schema.require_mapping(
            payload["selections"], where="{}.selections".format(where)
        )
        selections = {}
        for subgoal_id, node_id in raw_selections.items():
            schema.require_identifier({"goal": subgoal_id}, "goal", where=where)
            selections[subgoal_id] = schema.require_identifier(
                {"node": node_id}, "node", where=where
            )
        order = schema.require_str_tuple(payload, "order", where=where)
        for node_id in order:
            schema.require_identifier({"node": node_id}, "node", where=where)
        return cls(selections=selections, order=order)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selections": dict(sorted(self.selections.items())),
            "order": list(self.order),
        }


class SkillPlanner:
    """Rule-based decision model over a candidate skill graph.

    Candidate choice is read from the graph rather than hard-coded: within a
    sub-goal subgraph the achiever that is the *source* of a ``FALLBACK_TO`` chain
    is the primary, and each ``FALLBACK_TO`` target is the next candidate to try
    once its predecessor is reported failed.  Instrumental prerequisites follow
    the same rule along their own chains.
    """

    def __init__(
        self,
        subgoals: SubGoalGraph,
        graph: SkillGraph,
    ) -> None:
        if graph.subgoal_graph is not None and graph.subgoal_graph is not subgoals:
            raise ValueError("graph is bound to a different sub-goal graph")
        graph.validate()
        self.subgoals = subgoals
        self.graph = graph

    def decide(
        self,
        completed: Iterable[str] = (),
        failed: Iterable[str] = (),
    ) -> Optional[SkillNode]:
        """Return exactly one next skill node, or ``None`` when nothing remains."""

        completed_ids = self._known(completed, "completed")
        failed_ids = self._known(failed, "failed")
        # ``execution_order()`` on the sub-goal graph is only a deterministic
        # iteration order.  Layer 2 may impose additional cross-subgraph
        # requirements, so a node is selectable only when the composition
        # graph says that all of its (possibly disjunctive) prerequisite
        # groups are satisfied.  Without this gate a valid relation such as
        # ``subgoal_b ENABLES subgoal_a`` was ignored whenever Layer 1 left the two
        # subgoals unordered.
        ready_ids = {
            node.id for node in self.graph.ready_nodes(completed=completed_ids)
        }
        for subgoal_id in self.subgoals.execution_order():
            if subgoal_id not in self.graph.subgraphs:
                continue
            subgraph = self.graph.subgraph_for_subgoal(subgoal_id)
            selected = self.select_achiever(subgraph, failed_ids)
            wanted = self._wanted(subgraph, selected.id, completed_ids, failed_ids)
            for node_id in subgraph.execution_order():
                if (
                    node_id not in wanted
                    or node_id in completed_ids
                    or node_id not in ready_ids
                ):
                    continue
                if node_id in failed_ids:  # unreachable: _wanted never picks a failed node
                    raise NoViableCandidate(
                        "node {!r} failed and was not replaced".format(node_id)
                    )
                return subgraph.nodes[node_id]
        return None

    def plan(self, failed: Iterable[str] = ()) -> SkillPlan:
        """Roll the incremental decisions forward into a complete path."""

        failed_ids = tuple(failed)
        completed: List[str] = []
        while True:
            node = self.decide(completed, failed_ids)
            if node is None:
                break
            completed.append(node.id)
        selections = {}
        for subgoal_id in self.graph.subgraphs:
            chosen = [
                node_id
                for node_id in completed
                if subgoal_id in self.graph.nodes[node_id].achieves
            ]
            if len(chosen) != 1:
                raise NoViableCandidate(
                    "sub-goal {!r} selected {} achievers".format(subgoal_id, len(chosen))
                )
            selections[subgoal_id] = chosen[0]
        return SkillPlan(selections=selections, order=tuple(completed))

    def select_achiever(
        self,
        subgraph: SkillSubgraph,
        failed: Iterable[str] = (),
    ) -> SkillNode:
        """The first achiever in this sub-goal's fallback chain that has not failed."""

        failed_ids = set(failed)
        chain = self.fallback_chain(subgraph)
        for node in chain:
            if node.id not in failed_ids:
                return node
        raise NoViableCandidate(
            "every achiever for sub-goal {!r} failed: {}".format(
                subgraph.subgoal_id, [node.id for node in chain]
            )
        )

    @staticmethod
    def fallback_chain(subgraph: SkillSubgraph) -> Tuple[SkillNode, ...]:
        """Achievers ordered primary-first along ``FALLBACK_TO`` edges."""

        achievers = {node.id: node for node in subgraph.achievers}
        if not achievers:
            raise ValueError(
                "sub-goal skill subgraph {!r} has no achiever".format(subgraph.subgoal_id)
            )
        chains = [chain for chain in subgraph.fallback_chains() if chain[0] in achievers]
        if not chains:
            if len(achievers) > 1:
                raise ValueError(
                    "sub-goal {!r} has {} achievers but no FALLBACK_TO order".format(
                        subgraph.subgoal_id, len(achievers)
                    )
                )
            return tuple(achievers.values())
        if len(chains) != 1:
            raise ValueError(
                "sub-goal {!r} does not have one unambiguous primary achiever".format(
                    subgraph.subgoal_id
                )
            )
        unreachable = sorted(set(achievers) - set(chains[0]))
        if unreachable:
            raise ValueError(
                "achievers {} in sub-goal {!r} are not on the fallback chain".format(
                    unreachable, subgraph.subgoal_id
                )
            )
        return tuple(achievers[node_id] for node_id in chains[0])

    @staticmethod
    def _wanted(
        subgraph: SkillSubgraph,
        achiever_id: str,
        completed: Set[str],
        failed: Set[str],
    ) -> Set[str]:
        """The achiever plus one runnable member of every prerequisite chain.

        Prerequisites arrive as fallback chains.  The member to run is one that
        already completed, else the first member that has not failed; when
        every member failed the sub-goal as a whole has failed.
        """

        wanted = {achiever_id}
        frontier = [achiever_id]
        while frontier:
            current = frontier.pop()
            for group in subgraph.prerequisite_groups(current):
                chosen = SkillPlanner._resolve(subgraph, group, completed, failed)
                if chosen not in wanted:
                    wanted.add(chosen)
                    frontier.append(chosen)
        return wanted

    @staticmethod
    def _resolve(
        subgraph: SkillSubgraph,
        group: FrozenSet[str],
        completed: Set[str],
        failed: Set[str],
    ) -> str:
        chain = subgraph.fallback_chain_of(next(iter(group)))
        for node_id in chain:
            if node_id in completed:
                return node_id
        for node_id in chain:
            if node_id not in failed:
                return node_id
        raise NoViableCandidate(
            "node {!r} and its fallbacks {} all failed in sub-goal {!r}".format(
                chain[0], list(chain[1:]), subgraph.subgoal_id
            )
        )

    def _known(self, node_ids: Iterable[str], label: str) -> Set[str]:
        values = set(node_ids)
        unknown = sorted(values - set(self.graph.nodes))
        if unknown:
            raise KeyError("unknown {} nodes {}".format(label, unknown))
        return values
