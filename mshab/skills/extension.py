"""Validated extension interface for manual and future VLM graph builders.

Three kinds of change reach Layer 1 and Layer 2, and all of them travel as one
declarative :class:`SkillGraphPatch`:

1. a brand-new functional goal together with its implementation subgraph;
2. a relation connecting that subgraph to subgraphs that already exist;
3. a new candidate node inside a subgraph that is already registered.

The third case is what :class:`GoalSkillSubgraphExtension` exists for.  A
registered subgraph is sealed, so it is never edited in place: the extension
builds a validated successor and the aggregate swaps it in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from mshab.skills import schema
from mshab.skills.graph import (
    FunctionalGoal,
    FunctionalGoalGraph,
    GoalSkillSubgraph,
    GoalDependency,
    SkillCompositionGraph,
    SkillEdge,
    SkillNode,
    SkillSubgraphRelation,
)
from mshab.skills.library import SkillLibrary


@dataclass(frozen=True)
class GoalSkillSubgraphExtension:
    """Add candidates or internal relations to an already-registered goal.

    The extension never mutates the sealed subgraph.  :meth:`rebuild` returns an
    unsealed successor which the composition graph revalidates and installs, so
    a rejected extension leaves the live graph untouched.
    """

    goal_id: str
    nodes: Tuple[SkillNode, ...] = ()
    edges: Tuple[SkillEdge, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        if not self.goal_id:
            raise ValueError("subgraph extension goal_id must be non-empty")
        if not self.nodes and not self.edges:
            raise ValueError("subgraph extension must add a node or a relation")
        for node in self.nodes:
            if not isinstance(node, SkillNode):
                raise TypeError(
                    "subgraph extension nodes must be SkillNode, got {}".format(
                        type(node).__name__
                    )
                )
        for edge in self.edges:
            if not isinstance(edge, SkillEdge):
                raise TypeError(
                    "subgraph extension edges must be SkillEdge, got {}".format(
                        type(edge).__name__
                    )
                )

    def rebuild(self, current: GoalSkillSubgraph) -> GoalSkillSubgraph:
        if current.goal_id != self.goal_id:
            raise ValueError(
                "extension targets goal {!r}, not {!r}".format(
                    self.goal_id, current.goal_id
                )
            )
        successor = current.unsealed_copy()
        for node in self.nodes:
            successor.add_node(node)
        for edge in self.edges:
            successor.relate(edge.source, edge.target, edge.relation)
        successor.validate()
        return successor

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalSkillSubgraphExtension":
        where = "subgraph_extension"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload, where=where, required=("goal_id",), optional=("nodes", "edges")
        )
        return cls(
            goal_id=schema.require_identifier(payload, "goal_id", where=where),
            nodes=tuple(
                SkillNode.from_dict(item)
                for item in schema.require_sequence(payload, "nodes", where=where)
            ),
            edges=tuple(
                SkillEdge.from_dict(item)
                for item in schema.require_sequence(payload, "edges", where=where)
            ),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "nodes": [node.as_dict() for node in self.nodes],
            "edges": [edge.as_dict() for edge in self.edges],
        }


@dataclass(frozen=True)
class SkillGraphPatch:
    """A declarative, reviewable addition to Layer 1 and Layer 2.

    Manual code and a future VLM emit this same object.  Applying the patch goes
    through the existing duplicate, namespace, unknown-goal, achiever, and cycle
    checks instead of letting a proposer mutate graph internals.
    """

    goals: Tuple[FunctionalGoal, ...] = ()
    goal_dependencies: Tuple[GoalDependency, ...] = ()
    skill_subgraphs: Tuple[GoalSkillSubgraph, ...] = ()
    subgraph_relations: Tuple[SkillSubgraphRelation, ...] = ()
    subgraph_extensions: Tuple[GoalSkillSubgraphExtension, ...] = ()

    _FIELD_TYPES = {
        "goals": FunctionalGoal,
        "goal_dependencies": GoalDependency,
        "skill_subgraphs": GoalSkillSubgraph,
        "subgraph_relations": SkillSubgraphRelation,
        "subgraph_extensions": GoalSkillSubgraphExtension,
    }

    def __post_init__(self) -> None:
        for name, expected in self._FIELD_TYPES.items():
            items = tuple(getattr(self, name))
            for item in items:
                # Without this, a duck-typed object walks straight into the
                # graph and can behave differently on the staging and commit
                # passes -- exactly the trust boundary a VLM sits behind.
                if not isinstance(item, expected):
                    raise TypeError(
                        "patch.{} must contain {} instances, got {}".format(
                            name, expected.__name__, type(item).__name__
                        )
                    )
            object.__setattr__(self, name, items)

    def with_extension(
        self,
        goal_id: str,
        nodes: Iterable[SkillNode] = (),
        edges: Iterable[SkillEdge] = (),
    ) -> "SkillGraphPatch":
        """Return a new patch that also extends an existing goal subgraph."""

        extension = GoalSkillSubgraphExtension(
            goal_id=goal_id, nodes=tuple(nodes), edges=tuple(edges)
        )
        return SkillGraphPatch(
            goals=self.goals,
            goal_dependencies=self.goal_dependencies,
            skill_subgraphs=self.skill_subgraphs,
            subgraph_relations=self.subgraph_relations,
            subgraph_extensions=self.subgraph_extensions + (extension,),
        )

    def apply(
        self,
        goal_graph: FunctionalGoalGraph,
        skill_graph: SkillCompositionGraph,
        library: Optional[SkillLibrary] = None,
    ) -> None:
        """Atomically apply this patch, or leave both graphs untouched.

        Everything is built and validated on a staging copy; the live objects
        are only touched by the final two adopt calls, which cannot fail.  The
        earlier design replayed the mutations on the live graph after a dry
        run, so any divergence between the two passes left a rejected patch
        half-applied.
        """

        if skill_graph.goal_graph is not goal_graph:
            raise ValueError("skill_graph must reference the supplied goal_graph")
        staging_goals, staging_skills = _staging_copy(goal_graph, skill_graph)
        self._apply_to(staging_goals, staging_skills, library)
        staging_skills.validate()
        goal_graph._adopt(staging_goals)
        skill_graph._adopt(staging_skills)

    def _apply_to(
        self,
        goal_graph: FunctionalGoalGraph,
        skill_graph: SkillCompositionGraph,
        library: Optional[SkillLibrary],
    ) -> None:
        if library is not None:
            self.validate_skill_ids(library)
        for goal in self.goals:
            goal_graph.add_goal(goal)
        for dependency in self.goal_dependencies:
            goal_graph.add_dependency(dependency.source, dependency.target)
        for subgraph in self.skill_subgraphs:
            skill_graph.add_subgraph(subgraph.unsealed_copy())
        for extension in self.subgraph_extensions:
            current = skill_graph.subgraph_for_goal(extension.goal_id)
            skill_graph.replace_subgraph(extension.rebuild(current))
        for relation in self.subgraph_relations:
            skill_graph.relate_subgraphs(
                relation.source_goal,
                relation.target_goal,
                relation.source_node,
                relation.target_node,
                relation.relation,
            )

    def validate_skill_ids(self, library: SkillLibrary) -> None:
        """Reject a patch that references a skill the library does not have."""

        referenced = {
            node.skill_id
            for subgraph in self.skill_subgraphs
            for node in subgraph.nodes.values()
        }
        referenced |= {
            node.skill_id
            for extension in self.subgraph_extensions
            for node in extension.nodes
        }
        missing = sorted(referenced - {skill.id for skill in library.find()})
        if missing:
            raise KeyError(
                "graph patch references unregistered skills {}".format(missing)
            )

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, task: str
    ) -> "SkillGraphPatch":
        """Parse an untrusted patch document under a strict schema.

        ``task`` is supplied by the caller rather than read from the payload:
        a proposer must not be able to redirect a patch into another task's
        namespace.
        """

        where = "skill_graph_patch"
        payload = schema.require_mapping(payload, where=where)
        schema.require_schema_version(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            optional=(
                "goals",
                "goal_dependencies",
                "skill_subgraphs",
                "subgraph_relations",
                "subgraph_extensions",
                "schema_version",
            ),
        )
        return cls(
            goals=tuple(
                FunctionalGoal.from_dict(item)
                for item in schema.require_sequence(payload, "goals", where=where)
            ),
            goal_dependencies=tuple(
                GoalDependency.from_dict(item)
                for item in schema.require_sequence(
                    payload, "goal_dependencies", where=where
                )
            ),
            skill_subgraphs=tuple(
                GoalSkillSubgraph.from_dict(item, task=task)
                for item in schema.require_sequence(
                    payload, "skill_subgraphs", where=where
                )
            ),
            subgraph_relations=tuple(
                SkillSubgraphRelation.from_dict(item)
                for item in schema.require_sequence(
                    payload, "subgraph_relations", where=where
                )
            ),
            subgraph_extensions=tuple(
                GoalSkillSubgraphExtension.from_dict(item)
                for item in schema.require_sequence(
                    payload, "subgraph_extensions", where=where
                )
            ),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": schema.SCHEMA_VERSION,
            "goals": [goal.as_dict() for goal in self.goals],
            "goal_dependencies": [
                item.as_dict() for item in self.goal_dependencies
            ],
            "skill_subgraphs": [
                subgraph.as_dict() for subgraph in self.skill_subgraphs
            ],
            "subgraph_relations": [
                relation.as_dict() for relation in self.subgraph_relations
            ],
            "subgraph_extensions": [
                extension.as_dict() for extension in self.subgraph_extensions
            ],
        }


class SkillGraphBuilder(ABC):
    """Extension point shared by hand-written and future VLM builders."""

    @abstractmethod
    def propose(
        self,
        instruction: str,
        task: str,
        context: Mapping[str, Any],
    ) -> SkillGraphPatch:
        """Return a patch; do not mutate graph internals directly."""

    def build(
        self,
        instruction: str,
        task: str,
        context: Optional[Mapping[str, Any]] = None,
        library: Optional[SkillLibrary] = None,
    ) -> Tuple[FunctionalGoalGraph, SkillCompositionGraph]:
        goal_graph = FunctionalGoalGraph(instruction)
        skill_graph = SkillCompositionGraph(task, goal_graph=goal_graph)
        patch = self.propose(instruction, task, context or {})
        patch.apply(goal_graph, skill_graph, library=library)
        return goal_graph, skill_graph


def _staging_copy(
    goal_graph: FunctionalGoalGraph,
    skill_graph: SkillCompositionGraph,
) -> Tuple[FunctionalGoalGraph, SkillCompositionGraph]:
    staging_goals = FunctionalGoalGraph(goal_graph.instruction)
    staging_skills = SkillCompositionGraph(skill_graph.task, goal_graph=staging_goals)
    for goal in goal_graph.goals.values():
        staging_goals.add_goal(goal)
    for dependency in goal_graph.dependencies:
        staging_goals.add_dependency(dependency.source, dependency.target)
    for subgraph in skill_graph.subgraphs.values():
        staging_skills.add_subgraph(subgraph.unsealed_copy())
    for relation in skill_graph.subgraph_relations:
        staging_skills.relate_subgraphs(
            relation.source_goal,
            relation.target_goal,
            relation.source_node,
            relation.target_node,
            relation.relation,
        )
    return staging_goals, staging_skills
