"""Task-conditioned goal and skill-composition graphs.

The graph is the only owner of skill-to-skill composition semantics.  There is
no ``AlternativeSkill`` or ``SequentialSkill`` class: alternatives, fallback,
requirements, and enabling order are relations between semantic skill nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from mshab.skills import schema


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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FunctionalGoal":
        where = "functional_goal"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            required=("id", "predicate"),
            optional=("description",),
        )
        return cls(
            id=schema.require_identifier(payload, "id", where=where),
            predicate=schema.require_str(payload, "predicate", where=where),
            description=schema.optional_str(payload, "description", where=where),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "predicate": self.predicate,
            "description": self.description,
        }


@dataclass(frozen=True)
class GoalDependency:
    """The source goal must be achieved before the target goal."""

    source: str
    target: str

    def __post_init__(self) -> None:
        if not self.source or not self.target or self.source == self.target:
            raise ValueError("goal dependency needs two distinct non-empty ids")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalDependency":
        where = "goal_dependency"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(payload, where=where, required=("source", "target"))
        return cls(
            source=schema.require_identifier(payload, "source", where=where),
            target=schema.require_identifier(payload, "target", where=where),
        )

    def as_dict(self) -> Dict[str, str]:
        return {"source": self.source, "target": self.target}


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

    def ready_goals(
        self,
        facts: Iterable[str],
        completed: Iterable[str] = (),
    ) -> Tuple[FunctionalGoal, ...]:
        fact_set = set(facts)
        completed_set = set(completed)
        unknown_completed = sorted(completed_set - set(self._goals))
        if unknown_completed:
            raise KeyError("unknown completed goals {}".format(unknown_completed))
        ready = []
        for goal in self._goals.values():
            if goal.id in completed_set or goal.predicate in fact_set:
                continue
            prerequisites = {
                dependency.source
                for dependency in self._dependencies
                if dependency.target == goal.id
            }
            if all(
                item in completed_set or self._goals[item].predicate in fact_set
                for item in prerequisites
            ):
                ready.append(goal)
        return tuple(sorted(ready, key=lambda item: item.id))

    def execution_order(self) -> Tuple[str, ...]:
        edges = [(item.source, item.target) for item in self._dependencies]
        return _topological_order(self._goals, edges, "functional goal")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "goals": [
                goal.as_dict()
                for goal in sorted(self._goals.values(), key=lambda item: item.id)
            ],
            "dependencies": [item.as_dict() for item in self._dependencies],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FunctionalGoalGraph":
        where = "functional_goal_graph"
        payload = schema.require_mapping(payload, where=where)
        schema.require_schema_version(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            required=("instruction",),
            optional=("goals", "dependencies", "schema_version", "execution_order"),
        )
        graph = cls(schema.require_str(payload, "instruction", where=where))
        for item in schema.require_sequence(payload, "goals", where=where):
            graph.add_goal(FunctionalGoal.from_dict(item))
        for item in schema.require_sequence(payload, "dependencies", where=where):
            dependency = GoalDependency.from_dict(item)
            graph.add_dependency(dependency.source, dependency.target)
        return graph

    def _adopt(self, other: "FunctionalGoalGraph") -> None:
        """Replace this graph's contents with a validated staging copy.

        Used to commit a :class:`~mshab.skills.extension.SkillGraphPatch` in one
        step so a rejected patch can never leave the live graph half-modified.
        """

        self.instruction = other.instruction
        self._goals = dict(other._goals)
        self._dependencies = list(other._dependencies)

    def _require_goal(self, goal_id: str) -> FunctionalGoal:
        try:
            return self._goals[goal_id]
        except KeyError as exc:
            raise KeyError("unknown functional goal {!r}".format(goal_id)) from exc


@dataclass(frozen=True)
class SkillNode:
    """One scene-independent semantic skill call.

    Layer 2 stores only a stable library id plus symbolic arguments.  It does
    not own a bound contract, checkpoint, simulator object, or backend choice;
    those are resolved by Layer 3/4 after an environment is selected.
    """

    id: str
    skill_id: str
    arguments: Mapping[str, Any]
    achieves: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or not self.skill_id:
            raise ValueError("skill node id and skill_id must be non-empty")
        arguments = dict(self.arguments)
        for name, value in arguments.items():
            if not isinstance(value, schema.SCALAR_TYPES):
                raise TypeError(
                    "skill node {} argument {!r} must be a scalar, got {}".format(
                        self.id, name, type(value).__name__
                    )
                )
        object.__setattr__(self, "arguments", MappingProxyType(arguments))
        object.__setattr__(self, "achieves", tuple(self.achieves))

    def __hash__(self) -> int:
        # dataclass(frozen=True) derives __hash__ from the field tuple, which
        # contains an unhashable mappingproxy.  Hash the scalar contents instead
        # so nodes may be used in sets and as dict keys.
        return hash(
            (self.id, self.skill_id, tuple(sorted(self.arguments.items())), self.achieves)
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkillNode":
        where = "skill_node"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            required=("id", "skill_id"),
            optional=("arguments", "achieves"),
        )
        achieves = schema.require_str_tuple(payload, "achieves", where=where)
        for item in achieves:
            schema.require_identifier({"achieves": item}, "achieves", where=where)
        return cls(
            id=schema.require_identifier(payload, "id", where=where),
            skill_id=schema.require_skill_id(payload, "skill_id", where=where),
            arguments=schema.require_arguments(payload, "arguments", where=where),
            achieves=achieves,
        )

    def as_dict(self) -> Dict[str, Any]:
        return _node_dict(self)


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
        object.__setattr__(self, "relation", _require_relation(self.relation))
        if not self.source or not self.target or self.source == self.target:
            raise ValueError("skill edge needs two distinct non-empty node ids")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkillEdge":
        where = "skill_edge"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload, where=where, required=("source", "target", "relation")
        )
        return cls(
            source=schema.require_identifier(payload, "source", where=where),
            target=schema.require_identifier(payload, "target", where=where),
            relation=_require_relation(
                schema.require_str(payload, "relation", where=where)
            ),
        )

    def as_dict(self) -> Dict[str, str]:
        return _edge_dict(self)


class GoalSkillSubgraph:
    """Layer-2 implementation subgraph owned by one functional goal.

    A subgraph contains both instrumental nodes (for example, navigation) and
    one or more achiever nodes.  An achiever may only claim the functional goal
    owned by this subgraph.  This makes the Layer-1-to-Layer-2 boundary explicit
    instead of reconstructing it later from a flat collection of nodes.
    """

    def __init__(self, goal_id: str, task: str) -> None:
        if not goal_id or not task:
            raise ValueError("goal_id and task must be non-empty")
        self._goal_id = goal_id
        self._task = task
        self._nodes: Dict[str, SkillNode] = {}
        self._edges: List[SkillEdge] = []
        self._sealed = False

    def __setattr__(self, name: str, value: Any) -> None:
        """Freeze the whole object once a composition graph has registered it.

        ``_seal`` used to guard only ``add_node``/``relate``, which left the
        aggregate handing out a live object whose ``goal_id`` could be
        reassigned -- silently breaking single-goal ownership, the achiever
        invariant, and global node-id uniqueness.
        """

        if getattr(self, "_sealed", False):
            raise RuntimeError(
                "goal skill subgraph {!r} is registered and sealed; build a "
                "replacement with GoalSkillSubgraphExtension instead".format(
                    self._goal_id
                )
            )
        object.__setattr__(self, name, value)

    @property
    def goal_id(self) -> str:
        return self._goal_id

    @property
    def task(self) -> str:
        return self._task

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def nodes(self) -> Mapping[str, SkillNode]:
        return dict(self._nodes)

    @property
    def edges(self) -> Tuple[SkillEdge, ...]:
        return tuple(self._edges)

    @property
    def achievers(self) -> Tuple[SkillNode, ...]:
        return tuple(
            node
            for node in sorted(self._nodes.values(), key=lambda item: item.id)
            if self.goal_id in node.achieves
        )

    def add_node(self, node: SkillNode) -> None:
        self._ensure_mutable()
        if node.id in self._nodes:
            raise ValueError("duplicate skill node {!r}".format(node.id))
        expected_prefix = "mshab.{}.".format(self.task)
        if not node.skill_id.startswith(expected_prefix):
            raise ValueError(
                "node {} skill {!r} is outside task namespace {!r}".format(
                    node.id, node.skill_id, self.task
                )
            )
        foreign_goals = sorted(set(node.achieves) - {self.goal_id})
        if foreign_goals:
            raise ValueError(
                "node {} is inside goal subgraph {!r} but achieves {}".format(
                    node.id, self.goal_id, foreign_goals
                )
            )
        self._nodes[node.id] = node

    def relate(self, source: str, target: str, relation: SkillRelation) -> None:
        self._ensure_mutable()
        self._require_node(source)
        self._require_node(target)
        edge = SkillEdge(source, target, _require_relation(relation))
        _append_checked_edge(self._edges, edge)
        try:
            self.execution_order()
        except ValueError:
            self._edges.pop()
            raise

    def execution_order(self) -> Tuple[str, ...]:
        return _topological_order(
            self._nodes, _causal_pairs(self._edges), "goal skill subgraph"
        )

    def validate(self) -> None:
        if not self._nodes:
            raise ValueError(
                "goal skill subgraph {!r} must contain at least one node".format(
                    self.goal_id
                )
            )
        if not self.achievers:
            raise ValueError(
                "goal skill subgraph {!r} has no achiever node".format(self.goal_id)
            )
        self.execution_order()

    def _seal(self) -> None:
        """Transfer structural ownership to a SkillCompositionGraph."""

        self.validate()
        object.__setattr__(self, "_sealed", True)

    def unsealed_copy(self) -> "GoalSkillSubgraph":
        """A mutable clone: the only supported way to revise a sealed subgraph."""

        clone = GoalSkillSubgraph(self._goal_id, self._task)
        for node in self._nodes.values():
            clone.add_node(node)
        for edge in self._edges:
            clone.relate(edge.source, edge.target, edge.relation)
        return clone

    def _ensure_mutable(self) -> None:
        if self._sealed:
            raise RuntimeError(
                "goal skill subgraph {!r} is registered and sealed".format(
                    self.goal_id
                )
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "nodes": [
                _node_dict(node)
                for node in sorted(
                    self._nodes.values(), key=lambda item: item.id
                )
            ],
            "edges": [_edge_dict(edge) for edge in self._edges],
            "achiever_nodes": [node.id for node in self.achievers],
            "execution_order": list(self.execution_order()),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, task: str
    ) -> "GoalSkillSubgraph":
        where = "goal_skill_subgraph"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            required=("goal_id",),
            # achiever_nodes/execution_order are derived views in as_dict();
            # they are accepted on input but recomputed, never trusted.
            optional=("nodes", "edges", "achiever_nodes", "execution_order"),
        )
        subgraph = cls(
            schema.require_identifier(payload, "goal_id", where=where), task
        )
        for item in schema.require_sequence(payload, "nodes", where=where):
            subgraph.add_node(SkillNode.from_dict(item))
        for item in schema.require_sequence(payload, "edges", where=where):
            edge = SkillEdge.from_dict(item)
            subgraph.relate(edge.source, edge.target, edge.relation)
        return subgraph

    def _require_node(self, node_id: str) -> SkillNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise KeyError(
                "unknown skill node {!r} in goal subgraph {!r}".format(
                    node_id, self.goal_id
                )
            ) from exc


@dataclass(frozen=True)
class SkillSubgraphRelation:
    """A typed skill relation crossing two functional-goal subgraphs.

    An endpoint is either a specific node id or ``None``, which means *any
    achiever of that goal*.  Goal-level endpoints are what make alternative
    candidates usable: pinning a downstream ``ENABLES`` to one specific
    achiever silently makes that candidate mandatory, so a task that recovers
    through a fallback can never reach the next functional goal.
    """

    source_goal: str
    target_goal: str
    source_node: Optional[str] = None
    target_node: Optional[str] = None
    relation: SkillRelation = SkillRelation.ENABLES

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", _require_relation(self.relation))
        if not self.source_goal or not self.target_goal:
            raise ValueError("subgraph relation goals must be non-empty")
        if self.source_goal == self.target_goal:
            raise ValueError("subgraph relation must connect two distinct goals")
        for name in ("source_node", "target_node"):
            value = getattr(self, name)
            if value is not None and not value:
                raise ValueError("subgraph relation {} must be non-empty".format(name))
        if (
            self.source_node is not None
            and self.target_node is not None
            and self.source_node == self.target_node
        ):
            raise ValueError("subgraph relation needs two distinct node ids")
        if self.is_goal_level and self.relation not in _CAUSAL_RELATIONS:
            raise ValueError(
                "a goal-level endpoint is only meaningful for {}; got {}".format(
                    sorted(item.value for item in _CAUSAL_RELATIONS),
                    self.relation.value,
                )
            )

    @property
    def is_goal_level(self) -> bool:
        return self.source_node is None or self.target_node is None

    def endpoints(
        self, subgraphs: Mapping[str, "GoalSkillSubgraph"]
    ) -> Tuple[Tuple[str, str], ...]:
        """Resolve to concrete ``(source_node, target_node)`` pairs."""

        sources = self._resolve(self.source_goal, self.source_node, subgraphs)
        targets = self._resolve(self.target_goal, self.target_node, subgraphs)
        return tuple(
            (source, target)
            for source in sources
            for target in targets
            if source != target
        )

    def edges(
        self, subgraphs: Mapping[str, "GoalSkillSubgraph"]
    ) -> Tuple[SkillEdge, ...]:
        return tuple(
            SkillEdge(source, target, self.relation)
            for source, target in self.endpoints(subgraphs)
        )

    @staticmethod
    def _resolve(
        goal_id: str,
        node_id: Optional[str],
        subgraphs: Mapping[str, "GoalSkillSubgraph"],
    ) -> Tuple[str, ...]:
        try:
            subgraph = subgraphs[goal_id]
        except KeyError as exc:
            raise KeyError(
                "unknown goal skill subgraph {!r}".format(goal_id)
            ) from exc
        if node_id is not None:
            if node_id not in subgraph.nodes:
                raise ValueError(
                    "node {!r} is not owned by subgraph {!r}".format(node_id, goal_id)
                )
            return (node_id,)
        achievers = tuple(node.id for node in subgraph.achievers)
        if not achievers:
            raise ValueError(
                "goal-level endpoint needs an achiever in subgraph {!r}".format(goal_id)
            )
        return achievers

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkillSubgraphRelation":
        where = "subgraph_relation"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            required=("source_goal", "target_goal"),
            optional=("source_node", "target_node", "relation"),
        )
        return cls(
            source_goal=schema.require_identifier(payload, "source_goal", where=where),
            target_goal=schema.require_identifier(payload, "target_goal", where=where),
            source_node=schema.optional_identifier(payload, "source_node", where=where),
            target_node=schema.optional_identifier(payload, "target_node", where=where),
            relation=_require_relation(
                payload.get("relation", SkillRelation.ENABLES.value)
            ),
        )

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            "source_goal": self.source_goal,
            "target_goal": self.target_goal,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "relation": self.relation.value,
        }


class SkillCompositionGraph:
    """Layer 2: one goal-owned subgraph per Layer-1 functional goal.

    The composition graph is an aggregate root.  Each
    :class:`GoalSkillSubgraph` owns its nodes and internal relations, while this
    object owns relations crossing goal boundaries.  ``nodes`` and ``edges``
    expose read-only flattened views for planners and legacy query code.
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
        self._subgraphs: Dict[str, GoalSkillSubgraph] = {}
        self._subgraph_relations: List[SkillSubgraphRelation] = []

    @property
    def subgraphs(self) -> Mapping[str, GoalSkillSubgraph]:
        return dict(self._subgraphs)

    @property
    def subgraph_relations(self) -> Tuple[SkillSubgraphRelation, ...]:
        return tuple(self._subgraph_relations)

    @property
    def nodes(self) -> Mapping[str, SkillNode]:
        return {
            node_id: node
            for subgraph in self._subgraphs.values()
            for node_id, node in subgraph.nodes.items()
        }

    @property
    def edges(self) -> Tuple[SkillEdge, ...]:
        """Flattened read-only view; goal-level relations are expanded here.

        A goal-level relation yields one edge per achiever of its open
        endpoint.  That is the correct *partial order over candidates*: every
        achiever precedes the downstream node, and exactly one of them will be
        chosen by :class:`~mshab.skills.plan.SkillPlanner`.
        """

        internal = tuple(
            edge for subgraph in self._subgraphs.values() for edge in subgraph.edges
        )
        cross = tuple(
            edge
            for item in self._subgraph_relations
            for edge in item.edges(self._subgraphs)
        )
        return internal + cross

    def add_subgraph(self, subgraph: GoalSkillSubgraph) -> None:
        if subgraph.task != self.task:
            raise ValueError("subgraph task must match composition graph task")
        if subgraph.goal_id in self._subgraphs:
            raise ValueError(
                "duplicate goal skill subgraph {!r}".format(subgraph.goal_id)
            )
        if (
            self.goal_graph is not None
            and subgraph.goal_id not in self.goal_graph.goals
        ):
            raise ValueError(
                "subgraph references unknown functional goal {!r}".format(
                    subgraph.goal_id
                )
            )
        duplicate_nodes = sorted(set(subgraph.nodes) & set(self.nodes))
        if duplicate_nodes:
            raise ValueError("duplicate skill nodes {}".format(duplicate_nodes))
        subgraph._seal()
        self._subgraphs[subgraph.goal_id] = subgraph

    def replace_subgraph(self, subgraph: GoalSkillSubgraph) -> None:
        """Swap in a revised implementation for an already-registered goal.

        This is the commit half of
        :class:`~mshab.skills.extension.GoalSkillSubgraphExtension`: a sealed
        subgraph is never edited in place, it is replaced by a validated
        successor.  The replacement must keep every node that a cross-goal
        relation still points at, and must not collide with another goal's ids.
        """

        goal_id = subgraph.goal_id
        current = self._require_subgraph(goal_id)
        if subgraph is current:
            raise ValueError("replacement must be a distinct subgraph object")
        if subgraph.task != self.task:
            raise ValueError("subgraph task must match composition graph task")
        foreign = {
            node_id
            for other_id, other in self._subgraphs.items()
            if other_id != goal_id
            for node_id in other.nodes
        }
        duplicate_nodes = sorted(set(subgraph.nodes) & foreign)
        if duplicate_nodes:
            raise ValueError("duplicate skill nodes {}".format(duplicate_nodes))
        subgraph._seal()
        self._subgraphs[goal_id] = subgraph
        try:
            self._check_relation_integrity()
            self.execution_order()
        except (KeyError, ValueError):
            self._subgraphs[goal_id] = current
            raise

    def add_node(self, node: SkillNode, goal_id: Optional[str] = None) -> None:
        """Compatibility helper; explicit ``add_subgraph`` is preferred.

        A node can only be inserted this way when its owning goal is explicit
        (or uniquely declared by ``node.achieves``).  Instrumental nodes should
        be added directly to a :class:`GoalSkillSubgraph`.
        """

        owner = goal_id
        if owner is None and len(node.achieves) == 1:
            owner = node.achieves[0]
        if owner is None:
            raise ValueError("goal_id is required for an instrumental skill node")
        subgraph = self._subgraphs.get(owner)
        if subgraph is None:
            subgraph = GoalSkillSubgraph(owner, self.task)
            subgraph.add_node(node)
            self.add_subgraph(subgraph)
        else:
            raise RuntimeError(
                "subgraph {!r} is already registered; assemble it before "
                "calling add_subgraph()".format(owner)
            )

    def relate(self, source: str, target: str, relation: SkillRelation) -> None:
        source_goal = self.owner_of(source)
        target_goal = self.owner_of(target)
        if source_goal == target_goal:
            raise RuntimeError(
                "internal relation belongs to sealed subgraph {!r}; add it "
                "before calling add_subgraph()".format(source_goal)
            )
        self.relate_subgraphs(source_goal, target_goal, source, target, relation)

    def relate_subgraphs(
        self,
        source_goal: str,
        target_goal: str,
        source_node: Optional[str] = None,
        target_node: Optional[str] = None,
        relation: SkillRelation = SkillRelation.ENABLES,
    ) -> None:
        """Relate two goal subgraphs; ``None`` endpoints mean *any achiever*."""

        self._require_subgraph(source_goal)
        self._require_subgraph(target_goal)
        item = SkillSubgraphRelation(
            source_goal, target_goal, source_node, target_node, relation
        )
        # Resolving here verifies both endpoint owners before anything is stored.
        new_edges = item.edges(self._subgraphs)
        existing = [
            edge
            for previous in self._subgraph_relations
            for edge in previous.edges(self._subgraphs)
        ]
        for edge in new_edges:
            _check_duplicate_edge(existing, edge)
            existing.append(edge)
        self._subgraph_relations.append(item)
        try:
            self.execution_order()  # reject causal cycles at insertion time
        except ValueError:
            self._subgraph_relations.pop()
            raise

    def owner_of(self, node_id: str) -> str:
        owners = [
            goal_id
            for goal_id, subgraph in self._subgraphs.items()
            if node_id in subgraph.nodes
        ]
        if not owners:
            raise KeyError("unknown skill node {!r}".format(node_id))
        return owners[0]

    def subgraph_for_goal(self, goal_id: str) -> GoalSkillSubgraph:
        """Return the Layer-2 implementation owned by a Layer-1 goal."""

        if self.goal_graph is not None and goal_id not in self.goal_graph.goals:
            raise KeyError("unknown functional goal {!r}".format(goal_id))
        return self._require_subgraph(goal_id)

    def alternatives(self, node_id: str) -> Tuple[SkillNode, ...]:
        """The complete ALTERNATIVE_TO connected component, excluding self."""

        self._require_node(node_id)
        nodes = self.nodes
        adjacency: Dict[str, Set[str]] = {item: set() for item in nodes}
        for edge in self.edges:
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
        return tuple(nodes[item] for item in sorted(seen))

    def fallbacks(self, node_id: str) -> Tuple[SkillNode, ...]:
        self._require_node(node_id)
        nodes = self.nodes
        return tuple(
            nodes[edge.target]
            for edge in self.edges
            if edge.source == node_id and edge.relation == SkillRelation.FALLBACK_TO
        )

    def candidates_for_goal(self, goal_id: str) -> Tuple[SkillNode, ...]:
        """Candidate skill nodes connected to one goal by ACHIEVED_BY."""

        if self.goal_graph is None:
            raise RuntimeError("composition graph has no functional goal graph")
        if goal_id not in self.goal_graph.goals:
            raise KeyError("unknown functional goal {!r}".format(goal_id))
        subgraph = self._subgraphs.get(goal_id)
        return subgraph.achievers if subgraph is not None else ()

    def uncovered_goals(self) -> Tuple[FunctionalGoal, ...]:
        """Functional goals for which the graph currently has no candidate."""

        if self.goal_graph is None:
            return ()
        return tuple(
            goal
            for goal in sorted(self.goal_graph.goals.values(), key=lambda item: item.id)
            if goal.id not in self._subgraphs or not self._subgraphs[goal.id].achievers
        )

    def prerequisite_groups(self, node_id: str) -> Tuple[FrozenSet[str], ...]:
        """Causal prerequisites as disjunctive groups.

        Each group must be satisfied by *at least one* completed member.  A
        node-level relation yields a singleton group; a goal-level relation
        yields one group holding every achiever of that goal, so recovering
        through a fallback still satisfies the downstream dependency.
        """

        self._require_node(node_id)
        groups: List[FrozenSet[str]] = []
        for subgraph in self._subgraphs.values():
            for edge in subgraph.edges:
                group = _causal_requirement(edge.source, edge.target, edge.relation, node_id)
                if group is not None:
                    groups.append(frozenset(group))
        for item in self._subgraph_relations:
            if item.relation not in _CAUSAL_RELATIONS:
                continue
            sources, targets = self._relation_sides(item)
            if item.relation == SkillRelation.ENABLES and node_id in targets:
                groups.append(frozenset(sources))
            elif item.relation == SkillRelation.REQUIRES and node_id in sources:
                groups.append(frozenset(targets))
        return tuple(sorted(groups, key=lambda group: sorted(group)))

    def prerequisites(self, node_id: str) -> Tuple[SkillNode, ...]:
        """Every node appearing in any prerequisite group.

        This is the conjunctive flattening, kept for display and for the common
        case where each group is a singleton.  Scheduling must use
        :meth:`prerequisite_groups`, which preserves the disjunction.
        """

        nodes = self.nodes
        required = {item for group in self.prerequisite_groups(node_id) for item in group}
        return tuple(nodes[item] for item in sorted(required))

    def ready_nodes(self, completed: Iterable[str] = ()) -> Tuple[SkillNode, ...]:
        """Dependency-ready nodes, without consulting an environment.

        Contract admission belongs to :class:`SkillRuntime`, because facts and
        backend readiness are Layer-3/4 environment-specific concerns.
        """

        completed_set = set(completed)
        nodes = self.nodes
        unknown_completed = sorted(completed_set - set(nodes))
        if unknown_completed:
            raise KeyError("unknown completed nodes {}".format(unknown_completed))
        ready = []
        for node in nodes.values():
            if node.id in completed_set:
                continue
            if all(
                group & completed_set
                for group in self.prerequisite_groups(node.id)
            ):
                ready.append(node)
        return tuple(sorted(ready, key=lambda item: item.id))

    def _relation_sides(
        self, item: SkillSubgraphRelation
    ) -> Tuple[FrozenSet[str], FrozenSet[str]]:
        pairs = item.endpoints(self._subgraphs)
        return (
            frozenset(source for source, _ in pairs),
            frozenset(target for _, target in pairs),
        )

    def execution_order(self) -> Tuple[str, ...]:
        return _topological_order(
            self.nodes, _causal_pairs(self.edges), "skill composition"
        )

    def validate(self) -> None:
        """Whole-aggregate invariants that no single subgraph can check alone."""

        for subgraph in self._subgraphs.values():
            subgraph.validate()
        self._check_relation_integrity()
        self.execution_order()
        if self.goal_graph is None:
            return
        uncovered = [goal.id for goal in self.uncovered_goals()]
        if uncovered:
            raise ValueError("functional goals without an achiever {}".format(uncovered))
        connected = {
            (item.source_goal, item.target_goal)
            for item in self._subgraph_relations
            if item.relation == SkillRelation.ENABLES
        }
        missing = sorted(
            "{} -> {}".format(dependency.source, dependency.target)
            for dependency in self.goal_graph.dependencies
            if (dependency.source, dependency.target) not in connected
        )
        if missing:
            raise ValueError(
                "goal dependencies without a Layer-2 ENABLES relation {}".format(missing)
            )

    def _check_relation_integrity(self) -> None:
        """Every cross-goal relation still resolves to owned, existing nodes."""

        seen: List[SkillEdge] = []
        for item in self._subgraph_relations:
            for edge in item.edges(self._subgraphs):
                _check_duplicate_edge(seen, edge)
                seen.append(edge)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "goal_graph": self.goal_graph.as_dict() if self.goal_graph else None,
            "subgraphs": [
                self._subgraphs[goal_id].as_dict()
                for goal_id in self._ordered_goal_ids()
            ],
            "subgraph_relations": [
                item.as_dict() for item in self._subgraph_relations
            ],
            # Derived flattened views, retained for generic planners.  The
            # subgraphs above remain the single source of truth: these are
            # recomputed on load, never trusted as input.
            "_derived": True,
            "nodes": [
                _node_dict(node)
                for node in sorted(self.nodes.values(), key=lambda item: item.id)
            ],
            "edges": [_edge_dict(edge) for edge in self.edges],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        goal_graph: Optional[FunctionalGoalGraph] = None,
    ) -> "SkillCompositionGraph":
        where = "skill_composition_graph"
        payload = schema.require_mapping(payload, where=where)
        schema.require_schema_version(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            required=("task",),
            optional=(
                "goal_graph",
                "subgraphs",
                "subgraph_relations",
                "nodes",
                "edges",
                "execution_order",
                "candidate_partial_order",
                "schema_version",
                "_derived",
            ),
        )
        task = schema.require_identifier(payload, "task", where=where)
        if goal_graph is None and isinstance(payload.get("goal_graph"), Mapping):
            goal_graph = FunctionalGoalGraph.from_dict(payload["goal_graph"])
        graph = cls(task, goal_graph=goal_graph)
        for item in schema.require_sequence(payload, "subgraphs", where=where):
            graph.add_subgraph(GoalSkillSubgraph.from_dict(item, task=task))
        for item in schema.require_sequence(payload, "subgraph_relations", where=where):
            relation = SkillSubgraphRelation.from_dict(item)
            graph.relate_subgraphs(
                relation.source_goal,
                relation.target_goal,
                relation.source_node,
                relation.target_node,
                relation.relation,
            )
        return graph

    def _adopt(self, other: "SkillCompositionGraph") -> None:
        """Replace this aggregate's contents with a validated staging copy."""

        self._subgraphs = dict(other._subgraphs)
        self._subgraph_relations = list(other._subgraph_relations)

    def _require_node(self, node_id: str) -> SkillNode:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise KeyError("unknown skill node {!r}".format(node_id)) from exc

    def _require_subgraph(self, goal_id: str) -> GoalSkillSubgraph:
        try:
            return self._subgraphs[goal_id]
        except KeyError as exc:
            raise KeyError(
                "unknown goal skill subgraph {!r}".format(goal_id)
            ) from exc

    def _ordered_goal_ids(self) -> Tuple[str, ...]:
        if self.goal_graph is None:
            return tuple(sorted(self._subgraphs))
        ordered = [
            goal_id
            for goal_id in self.goal_graph.execution_order()
            if goal_id in self._subgraphs
        ]
        ordered.extend(sorted(set(self._subgraphs) - set(ordered)))
        return tuple(ordered)


def _node_dict(node: SkillNode) -> Dict[str, Any]:
    return {
        "id": node.id,
        "skill_id": node.skill_id,
        "arguments": dict(node.arguments),
        "achieves": list(node.achieves),
    }


def _edge_dict(edge: SkillEdge) -> Dict[str, str]:
    return {
        "source": edge.source,
        "target": edge.target,
        "relation": edge.relation.value,
    }


#: Relations that impose an execution ordering.
_CAUSAL_RELATIONS = frozenset({SkillRelation.ENABLES, SkillRelation.REQUIRES})


def _require_relation(value: Any) -> SkillRelation:
    try:
        return SkillRelation(value)
    except ValueError as exc:
        raise ValueError(
            "unknown skill relation {!r}; expected one of {}".format(
                value, sorted(item.value for item in SkillRelation)
            )
        ) from exc


def _check_duplicate_edge(edges: Sequence[SkillEdge], edge: SkillEdge) -> None:
    if edge in edges:
        raise ValueError(
            "duplicate skill relation {} -{}-> {}".format(
                edge.source, edge.relation.value, edge.target
            )
        )
    if edge.relation == SkillRelation.ALTERNATIVE_TO:
        reverse = SkillEdge(edge.target, edge.source, edge.relation)
        if reverse in edges:
            raise ValueError(
                "ALTERNATIVE_TO is symmetric; reverse edge already exists"
            )


def _append_checked_edge(edges: List[SkillEdge], edge: SkillEdge) -> None:
    _check_duplicate_edge(edges, edge)
    edges.append(edge)


def _causal_requirement(
    source: str, target: str, relation: SkillRelation, node_id: str
) -> Optional[Tuple[str, ...]]:
    """The prerequisite one node-level causal edge imposes on ``node_id``."""

    if relation == SkillRelation.ENABLES and target == node_id:
        return (source,)
    if relation == SkillRelation.REQUIRES and source == node_id:
        return (target,)
    return None


def _causal_pairs(edges: Iterable[SkillEdge]) -> List[Tuple[str, str]]:
    pairs = []
    for edge in edges:
        if edge.relation == SkillRelation.ENABLES:
            pairs.append((edge.source, edge.target))
        elif edge.relation == SkillRelation.REQUIRES:
            pairs.append((edge.target, edge.source))
    return pairs


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
