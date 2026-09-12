"""First hand-authored, scene-independent MS-HAB skill graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mshab.skills.extension import SkillGraphBuilder, SkillGraphPatch
from mshab.skills.graph import (
    FunctionalGoal,
    FunctionalGoalGraph,
    GoalSkillSubgraph,
    GoalDependency,
    SkillCompositionGraph,
    SkillNode,
    SkillRelation,
    SkillSubgraphRelation,
)
from mshab.skills.library import SkillLibrary
from mshab.skills.model import (
    CheckpointBackend,
    CloseSkill,
    NavigateSkill,
    OpenSkill,
    PickSkill,
    PlaceSkill,
)
from mshab.skills.runtime import SkillGrounder


class SetTableAppleGraphBuilder(SkillGraphBuilder):
    """Manual v1 graph using the downloaded SetTable apple checkpoints.

    The graph uses symbolic category names only.  It can be built before a
    scene is loaded; an environment adapter later maps these names to the
    concrete actor, articulation, and goal objects for one episode.
    """

    def propose(
        self,
        instruction: str,
        task: str,
        context: Mapping[str, Any],
    ) -> SkillGraphPatch:
        if task != "set_table":
            raise ValueError("SetTableAppleGraphBuilder requires task='set_table'")
        object_name = str(context.get("object", "013_apple"))
        source = str(context.get("source", "fridge"))
        destination = str(context.get("destination", "dining_table"))
        segment = _build_segment_subgraphs(
            task=task,
            label="object",
            object_name=object_name,
            source=source,
            destination=destination,
            goal_ids=(
                "source_open",
                "object_retrieved",
                "object_placed",
                "source_closed",
            ),
        )
        return SkillGraphPatch(
            goals=segment.goals,
            goal_dependencies=segment.dependencies,
            skill_subgraphs=segment.subgraphs,
            subgraph_relations=segment.relations,
        )


class SetTableGraphBuilder(SkillGraphBuilder):
    """Manual v1 of the complete official SetTable semantic graph.

    The official task plan handles the bowl from the kitchen counter first,
    followed by the apple from the fridge.  Each object uses the eight-step
    Navigate/Open/Navigate/Pick/Navigate/Place/Navigate/Close pattern.
    """

    _SEGMENTS = (
        ("bowl", "024_bowl", "kitchen_counter"),
        ("apple", "013_apple", "fridge"),
    )

    def propose(
        self,
        instruction: str,
        task: str,
        context: Mapping[str, Any],
    ) -> SkillGraphPatch:
        if task != "set_table":
            raise ValueError("SetTableGraphBuilder requires task='set_table'")
        destination = str(context.get("destination", "dining_table"))
        goals = []
        dependencies = []
        subgraphs = []
        relations = []
        previous_close_goal = None

        for label, object_name, source in self._SEGMENTS:
            open_goal = "{}_source_open".format(label)
            retrieved_goal = "{}_retrieved".format(label)
            placed_goal = "{}_placed".format(label)
            close_goal = "{}_source_closed".format(label)
            segment = _build_segment_subgraphs(
                task=task,
                label=label,
                object_name=object_name,
                source=source,
                destination=destination,
                goal_ids=(open_goal, retrieved_goal, placed_goal, close_goal),
            )
            goals.extend(segment.goals)
            dependencies.extend(segment.dependencies)
            subgraphs.extend(segment.subgraphs)
            relations.extend(segment.relations)
            if previous_close_goal is not None:
                dependencies.append(GoalDependency(previous_close_goal, open_goal))
            if previous_close_goal is not None:
                relations.append(
                    SkillSubgraphRelation(
                        previous_close_goal,
                        open_goal,
                        None,
                        segment.entry_node,
                        SkillRelation.ENABLES,
                    )
                )
            previous_close_goal = close_goal

        return SkillGraphPatch(
            goals=tuple(goals),
            goal_dependencies=tuple(dependencies),
            skill_subgraphs=tuple(subgraphs),
            subgraph_relations=tuple(relations),
        )


@dataclass(frozen=True)
class _SegmentSubgraphs:
    goals: tuple
    dependencies: tuple
    subgraphs: tuple
    relations: tuple
    entry_node: str
    exit_node: str


def _build_segment_subgraphs(
    task: str,
    label: str,
    object_name: str,
    source: str,
    destination: str,
    goal_ids: tuple,
) -> _SegmentSubgraphs:
    """Create four goal-owned subgraphs for one SetTable object segment."""

    open_goal, retrieved_goal, placed_goal, close_goal = goal_ids
    prefix = "mshab.{}.".format(task)
    navigate_source = "navigate_to_{}_source".format(label)
    open_source = "open_{}_source".format(label)
    navigate_object = "navigate_to_{}".format(label)
    pick_specialized = "pick_{}_specialized".format(label)
    pick_generic = "pick_{}_generic".format(label)
    navigate_destination = "navigate_{}_to_destination".format(label)
    place_specialized = "place_{}_specialized".format(label)
    place_generic = "place_{}_generic".format(label)
    navigate_back = "navigate_back_to_{}_source".format(label)
    close_source = "close_{}_source".format(label)

    open_subgraph = GoalSkillSubgraph(open_goal, task)
    open_subgraph.add_node(
        SkillNode(navigate_source, prefix + "navigate.all", {"goal": source})
    )
    open_subgraph.add_node(
        SkillNode(open_source, prefix + "open." + source, {}, (open_goal,))
    )
    open_subgraph.relate(navigate_source, open_source, SkillRelation.ENABLES)

    retrieve_subgraph = GoalSkillSubgraph(retrieved_goal, task)
    retrieve_subgraph.add_node(
        SkillNode(navigate_object, prefix + "navigate.all", {"goal": object_name})
    )
    retrieve_subgraph.add_node(
        SkillNode(
            pick_specialized,
            prefix + "pick." + object_name,
            {},
            (retrieved_goal,),
        )
    )
    retrieve_subgraph.add_node(
        SkillNode(
            pick_generic,
            prefix + "pick.all",
            {"object": object_name},
            (retrieved_goal,),
        )
    )
    retrieve_subgraph.relate(
        navigate_object, pick_specialized, SkillRelation.ENABLES
    )
    retrieve_subgraph.relate(pick_generic, navigate_object, SkillRelation.REQUIRES)
    retrieve_subgraph.relate(pick_specialized, pick_generic, SkillRelation.IS_A)
    retrieve_subgraph.relate(
        pick_specialized, pick_generic, SkillRelation.ALTERNATIVE_TO
    )
    retrieve_subgraph.relate(
        pick_specialized, pick_generic, SkillRelation.FALLBACK_TO
    )

    place_subgraph = GoalSkillSubgraph(placed_goal, task)
    place_subgraph.add_node(
        SkillNode(
            navigate_destination,
            prefix + "navigate.all",
            {"goal": destination},
        )
    )
    place_subgraph.add_node(
        SkillNode(
            place_specialized,
            prefix + "place." + object_name,
            {"destination": destination},
            (placed_goal,),
        )
    )
    place_subgraph.add_node(
        SkillNode(
            place_generic,
            prefix + "place.all",
            {"object": object_name, "destination": destination},
            (placed_goal,),
        )
    )
    place_subgraph.relate(
        navigate_destination, place_specialized, SkillRelation.ENABLES
    )
    place_subgraph.relate(
        place_generic, navigate_destination, SkillRelation.REQUIRES
    )
    place_subgraph.relate(place_specialized, place_generic, SkillRelation.IS_A)
    place_subgraph.relate(
        place_specialized, place_generic, SkillRelation.ALTERNATIVE_TO
    )
    place_subgraph.relate(
        place_specialized, place_generic, SkillRelation.FALLBACK_TO
    )

    close_subgraph = GoalSkillSubgraph(close_goal, task)
    close_subgraph.add_node(
        SkillNode(navigate_back, prefix + "navigate.all", {"goal": source})
    )
    close_subgraph.add_node(
        SkillNode(close_source, prefix + "close." + source, {}, (close_goal,))
    )
    close_subgraph.relate(navigate_back, close_source, SkillRelation.ENABLES)

    goals = (
        FunctionalGoal(open_goal, "open({})".format(source)),
        FunctionalGoal(retrieved_goal, "holding({})".format(object_name)),
        FunctionalGoal(placed_goal, "at({},{})".format(object_name, destination)),
        FunctionalGoal(close_goal, "closed({})".format(source)),
    )
    dependencies = (
        GoalDependency(open_goal, retrieved_goal),
        GoalDependency(retrieved_goal, placed_goal),
        GoalDependency(placed_goal, close_goal),
    )
    # Cross-goal edges leave the source endpoint open (``None`` = any achiever
    # of that goal).  Pinning them to the specialized candidate would make the
    # generic fallback a dead end: recovering through it would never enable the
    # next functional goal.
    relations = (
        SkillSubgraphRelation(
            open_goal,
            retrieved_goal,
            None,
            navigate_object,
            SkillRelation.ENABLES,
        ),
        SkillSubgraphRelation(
            retrieved_goal,
            placed_goal,
            None,
            navigate_destination,
            SkillRelation.ENABLES,
        ),
        SkillSubgraphRelation(
            placed_goal,
            close_goal,
            None,
            navigate_back,
            SkillRelation.ENABLES,
        ),
    )
    return _SegmentSubgraphs(
        goals,
        dependencies,
        (open_subgraph, retrieve_subgraph, place_subgraph, close_subgraph),
        relations,
        navigate_source,
        close_source,
    )


_APPLE_INSTRUCTION = (
    "Retrieve the apple, close the fridge, and set the apple on the table."
)
_SET_TABLE_INSTRUCTION = (
    "Set the table with the bowl and apple, then close their storage."
)


# Canonical SetTable skill inventory.  This is a manifest, not filesystem
# discovery: the semantic library and its contracts must be constructible on a
# clean clone even before the external policy artifacts have been downloaded.
_SET_TABLE_ATOMIC_SKILLS = (
    (NavigateSkill, "navigate", "all"),
    (OpenSkill, "open", "fridge"),
    (OpenSkill, "open", "kitchen_counter"),
    (PickSkill, "pick", "013_apple"),
    (PickSkill, "pick", "024_bowl"),
    (PickSkill, "pick", "all"),
    (PlaceSkill, "place", "013_apple"),
    (PlaceSkill, "place", "024_bowl"),
    (PlaceSkill, "place", "all"),
    (CloseSkill, "close", "fridge"),
    (CloseSkill, "close", "kitchen_counter"),
)


def build_set_table_library(checkpoint_root: Path) -> SkillLibrary:
    """Build the canonical 11-skill inventory around an optional RL layout.

    The returned skills always exist and therefore always expose their
    contracts.  Each checkpoint backend computes ``ready/missing/partial`` at
    runtime from ``checkpoint_root``; artifact availability never changes the
    canonical Layer-1/2 graph or prevents documentation from being generated.
    """

    root = Path(checkpoint_root)
    library = SkillLibrary()
    for skill_class, raw_type, target in _SET_TABLE_ATOMIC_SKILLS:
        skill = skill_class(task="set_table", target=target)
        leaf = root / "rl" / "set_table" / raw_type / target
        skill.add_backend(
            CheckpointBackend(
                key="rl",
                family="rl",
                checkpoint_path=leaf / "policy.pt",
                config_path=leaf / "config.yml",
                policy_type="rl_all_obj" if target == "all" else "rl_per_obj",
            )
        )
        library.register(skill)
    return library


@dataclass(frozen=True)
class StarterSkillStack:
    """Concrete handles for all four layers of the starter example."""

    goal_graph: FunctionalGoalGraph
    skill_graph: SkillCompositionGraph
    library: SkillLibrary
    grounder: SkillGrounder

    @property
    def contracts(self):
        """Layer-3 bound contracts keyed by Layer-2 node id."""

        return self.grounder.contracts(self.skill_graph)


def build_set_table_apple_graph(
    instruction: str = _APPLE_INSTRUCTION,
    **context: Any
):
    """Build scene-independent Layer 1 and Layer 2 only."""

    return SetTableAppleGraphBuilder().build(
        instruction=instruction,
        task="set_table",
        context=context,
    )


def build_set_table_starter(checkpoint_root: Path, **context: Any) -> StarterSkillStack:
    """Build the four-layer starter stack from the canonical skill manifest."""

    library = build_set_table_library(Path(checkpoint_root))
    goals, graph = SetTableAppleGraphBuilder().build(
        instruction=_APPLE_INSTRUCTION,
        task="set_table",
        context=context,
        library=library,
    )
    grounder = SkillGrounder(library)
    # Eagerly bind every Layer-3 contract so an invalid manual graph fails now.
    grounder.contracts(graph)
    return StarterSkillStack(goals, graph, library, grounder)


def build_set_table_graph(
    instruction: str = _SET_TABLE_INSTRUCTION,
    **context: Any
):
    """Build complete scene-independent SetTable Layer 1 and Layer 2 graphs."""

    return SetTableGraphBuilder().build(
        instruction=instruction,
        task="set_table",
        context=context,
    )


def build_set_table_stack(
    checkpoint_root: Path, **context: Any
) -> StarterSkillStack:
    """Build all four layers of the complete manual SetTable graph."""

    library = build_set_table_library(Path(checkpoint_root))
    # One patch, validated against the library and then applied: validating a
    # second, independently proposed patch only worked because propose() is
    # deterministic.
    goals, graph = SetTableGraphBuilder().build(
        instruction=_SET_TABLE_INSTRUCTION,
        task="set_table",
        context=context,
        library=library,
    )
    grounder = SkillGrounder(library)
    grounder.contracts(graph)
    return StarterSkillStack(goals, graph, library, grounder)
