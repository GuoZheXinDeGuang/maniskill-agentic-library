"""Task-conditioned goal and skill-composition graphs.

The graph is the only owner of skill-to-skill composition semantics.  There is
no ``AlternativeSkill`` or ``SequentialSkill`` class: alternatives, fallback,
requirements, and enabling order are relations between grounded skill nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from mshab.skills.model import SkillInvocation


class SkillRelation(str, Enum):
    """Closed vocabulary for edges in a candidate skill-composition graph."""

    IS_A = "is_a"
    ENABLES = "enables"
    REQUIRES = "requires"
    ALTERNATIVE_TO = "alternative_to"
    FALLBACK_TO = "fallback_to"


@dataclass(frozen=True)
class FunctionalGoal:
    """One desired world-state predicate from a task instruction."""

    id: str
    predicate: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.predicate:
            raise ValueError("functional goal id and predicate must be non-empty")


@dataclass(frozen=True)
class GoalDependency:
    """The source goal must be achieved before the target goal."""

    source: str
    target: str

    def __post_init__(self) -> None:
        if not self.source or not self.target or self.source == self.target:
            raise ValueError("goal dependency needs two distinct non-empty ids")


class FunctionalGoalGraph:
    """Layer 1: task instruction decomposed into ordered functional goals."""

    def __init__(self, instruction: str) -> None:
        if not instruction:
            raise ValueError("task instruction must be non-empty")
        self.instruction = instruction
        self._goals: Dict[str, FunctionalGoal] = {}
        self._dependencies: List[GoalDependency] = []

    @property
    def goals(self) -> Mapping[str, FunctionalGoal]:
        return dict(self._goals)

    @property
    def dependencies(self) -> Tuple[GoalDependency, ...]:
        return tuple(self._dependencies)

    def add_goal(self, goal: FunctionalGoal) -> None:
        if goal.id in self._goals:
            raise ValueError("duplicate functional goal {!r}".format(goal.id))
        self._goals[goal.id] = goal

    def add_dependency(self, source: str, target: str) -> None:
        self._require_goal(source)
        self._require_goal(target)
        dependency = GoalDependency(source, target)
        if dependency in self._dependencies:
            raise ValueError("duplicate goal dependency {} -> {}".format(source, target))
        self._dependencies.append(dependency)
        try:
            self.execution_order()  # reject a cycle at insertion time
        except ValueError:
            self._dependencies.pop()
            raise

    def achieved(self, goal_id: str, facts: Iterable[str]) -> bool:
        return self._require_goal(goal_id).predicate in set(facts)

    def ready_goals(self, facts: Iterable[str]) -> Tuple[FunctionalGoal, ...]:
        fact_set = set(facts)
        ready = []
        for goal in self._goals.values():
            if goal.predicate in fact_set:
                continue
            prerequisites = {
                dependency.source
                for dependency in self._dependencies
                if dependency.target == goal.id
            }
            if all(self._goals[item].predicate in fact_set for item in prerequisites):
                ready.append(goal)
        return tuple(sorted(ready, key=lambda item: item.id))

    def execution_order(self) -> Tuple[str, ...]:
        edges = [(item.source, item.target) for item in self._dependencies]
        return _topological_order(self._goals, edges, "functional goal")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "goals": [
                {
                    "id": goal.id,
                    "predicate": goal.predicate,
                    "description": goal.description,
                }
                for goal in sorted(self._goals.values(), key=lambda item: item.id)
            ],
            "dependencies": [
                {"source": item.source, "target": item.target}
                for item in self._dependencies
            ],
        }

    def _require_goal(self, goal_id: str) -> FunctionalGoal:
        try:
            return self._goals[goal_id]
        except KeyError as exc:
            raise KeyError("unknown functional goal {!r}".format(goal_id)) from exc


@dataclass(frozen=True)
class SkillNode:
    """One grounded skill call and the functional goals it may achieve."""

    id: str
    invocation: SkillInvocation
    achieves: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("skill node id must be non-empty")
        object.__setattr__(self, "achieves", tuple(self.achieves))


@dataclass(frozen=True)
class SkillEdge:
    """One typed skill-to-skill relation.

    Direction semantics:

    - ``IS_A``: source is a specialization of target;
    - ``ENABLES``: source completion enables target;
    - ``REQUIRES``: source requires target to complete first;
    - ``ALTERNATIVE_TO``: symmetric candidate equivalence;
    - ``FALLBACK_TO``: try target after source fails.
    """

    source: str
    target: str
    relation: SkillRelation

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", SkillRelation(self.relation))
        if not self.source or not self.target or self.source == self.target:
            raise ValueError("skill edge needs two distinct non-empty node ids")


class SkillCompositionGraph:
    """Layer 2: candidates and all composition semantics for one task.

    Graph nodes are grounded :class:`SkillInvocation` objects. Pairwise
    ``ALTERNATIVE_TO`` edges form an undirected connected alternative group;
    ``FALLBACK_TO`` remains directed because retry priority matters.
    """

    def __init__(
        self,
        task: str,
        goal_graph: Optional[FunctionalGoalGraph] = None,
    ) -> None:
        if not task:
            raise ValueError("composition graph task must be non-empty")
        self.task = task
        self.goal_graph = goal_graph
        self._nodes: Dict[str, SkillNode] = {}
        self._edges: List[SkillEdge] = []

    @property
    def nodes(self) -> Mapping[str, SkillNode]:
        return dict(self._nodes)

    @property
    def edges(self) -> Tuple[SkillEdge, ...]:
        return tuple(self._edges)

    def add_node(self, node: SkillNode) -> None:
        if node.id in self._nodes:
            raise ValueError("duplicate skill node {!r}".format(node.id))
        if node.invocation.skill.task != self.task:
            raise ValueError(
                "node {} belongs to task {!r}, graph belongs to {!r}".format(
                    node.id, node.invocation.skill.task, self.task
                )
            )
        if self.goal_graph is not None:
            unknown_goals = sorted(set(node.achieves) - set(self.goal_graph.goals))
            if unknown_goals:
                raise ValueError(
                    "node {} achieves unknown functional goals {}".format(
                        node.id, unknown_goals
                    )
                )
        self._nodes[node.id] = node

    def relate(self, source: str, target: str, relation: SkillRelation) -> None:
        self._require_node(source)
        self._require_node(target)
        relation = SkillRelation(relation)
        edge = SkillEdge(source, target, relation)
        if edge in self._edges:
            raise ValueError(
                "duplicate skill relation {} -{}-> {}".format(
                    source, relation.value, target
                )
            )
        if relation == SkillRelation.ALTERNATIVE_TO:
            reverse = SkillEdge(target, source, relation)
            if reverse in self._edges:
                raise ValueError(
                    "ALTERNATIVE_TO is symmetric; reverse edge already exists"
                )
        self._edges.append(edge)
        try:
            self.execution_order()  # reject causal cycles at insertion time
        except ValueError:
            self._edges.pop()
            raise

    def alternatives(self, node_id: str) -> Tuple[SkillNode, ...]:
        """The complete ALTERNATIVE_TO connected component, excluding self."""

        self._require_node(node_id)
        adjacency: Dict[str, Set[str]] = {item: set() for item in self._nodes}
        for edge in self._edges:
            if edge.relation == SkillRelation.ALTERNATIVE_TO:
                adjacency[edge.source].add(edge.target)
                adjacency[edge.target].add(edge.source)
        seen = {node_id}
        frontier = [node_id]
        while frontier:
            current = frontier.pop()
            for candidate in adjacency[current] - seen:
                seen.add(candidate)
                frontier.append(candidate)
        seen.remove(node_id)
        return tuple(self._nodes[item] for item in sorted(seen))

    def fallbacks(self, node_id: str) -> Tuple[SkillNode, ...]:
        self._require_node(node_id)
        return tuple(
            self._nodes[edge.target]
            for edge in self._edges
            if edge.source == node_id and edge.relation == SkillRelation.FALLBACK_TO
        )

    def candidates_for_goal(self, goal_id: str) -> Tuple[SkillNode, ...]:
        """Candidate skill nodes connected to one goal by ACHIEVED_BY."""

        if self.goal_graph is None:
            raise RuntimeError("composition graph has no functional goal graph")
        if goal_id not in self.goal_graph.goals:
            raise KeyError("unknown functional goal {!r}".format(goal_id))
        return tuple(
            node
            for node in sorted(self._nodes.values(), key=lambda item: item.id)
            if goal_id in node.achieves
        )

    def uncovered_goals(self) -> Tuple[FunctionalGoal, ...]:
        """Functional goals for which the graph currently has no candidate."""

        if self.goal_graph is None:
            return ()
        covered = {goal_id for node in self._nodes.values() for goal_id in node.achieves}
        return tuple(
            goal
            for goal in sorted(self.goal_graph.goals.values(), key=lambda item: item.id)
            if goal.id not in covered
        )

    def prerequisites(self, node_id: str) -> Tuple[SkillNode, ...]:
        """Causal prerequisites derived from REQUIRES and ENABLES edges."""

        self._require_node(node_id)
        required = set()
        for edge in self._edges:
            if edge.relation == SkillRelation.REQUIRES and edge.source == node_id:
                required.add(edge.target)
            elif edge.relation == SkillRelation.ENABLES and edge.target == node_id:
                required.add(edge.source)
        return tuple(self._nodes[item] for item in sorted(required))

    def ready_nodes(
        self,
        facts: Iterable[str],
        completed: Iterable[str] = (),
    ) -> Tuple[SkillNode, ...]:
        fact_set = set(facts)
        completed_set = set(completed)
        unknown_completed = sorted(completed_set - set(self._nodes))
        if unknown_completed:
            raise KeyError("unknown completed nodes {}".format(unknown_completed))
        ready = []
        for node in self._nodes.values():
            if node.id in completed_set or not node.invocation.skill.ready:
                continue
            prerequisite_ids = {item.id for item in self.prerequisites(node.id)}
            if not prerequisite_ids.issubset(completed_set):
                continue
            if node.invocation.contract.can_start(fact_set):
                ready.append(node)
        return tuple(sorted(ready, key=lambda item: item.id))

    def execution_order(self) -> Tuple[str, ...]:
        causal_edges = []
        for edge in self._edges:
            if edge.relation == SkillRelation.ENABLES:
                causal_edges.append((edge.source, edge.target))
            elif edge.relation == SkillRelation.REQUIRES:
                causal_edges.append((edge.target, edge.source))
        return _topological_order(self._nodes, causal_edges, "skill composition")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "goal_graph": self.goal_graph.as_dict() if self.goal_graph else None,
            "nodes": [
                {
                    "id": node.id,
                    "invocation": node.invocation.id,
                    "skill_id": node.invocation.skill.id,
                    "arguments": dict(node.invocation.arguments),
                    "backend_key": node.invocation.backend_key,
                    "achieves": list(node.achieves),
                }
                for node in sorted(self._nodes.values(), key=lambda item: item.id)
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation.value,
                }
                for edge in self._edges
            ],
        }

    def _require_node(self, node_id: str) -> SkillNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise KeyError("unknown skill node {!r}".format(node_id)) from exc


def _topological_order(
    nodes: Mapping[str, Any],
    edges: Sequence[Tuple[str, str]],
    label: str,
) -> Tuple[str, ...]:
    """Stable Kahn order; raise when directed causal edges contain a cycle."""

    indegree = {node_id: 0 for node_id in nodes}
    outgoing: Dict[str, Set[str]] = {node_id: set() for node_id in nodes}
    for source, target in edges:
        if target not in outgoing[source]:
            outgoing[source].add(target)
            indegree[target] += 1
    frontier = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    order = []
    while frontier:
        node_id = frontier.pop(0)
        order.append(node_id)
        for target in sorted(outgoing[node_id]):
            indegree[target] -= 1
            if indegree[target] == 0:
                frontier.append(target)
                frontier.sort()
    if len(order) != len(nodes):
        cyclic = sorted(node_id for node_id, degree in indegree.items() if degree)
        raise ValueError("{} graph has a causal cycle: {}".format(label, cyclic))
    return tuple(order)
