"""Scenarios of the granularity experiment and the scripted proposer that plays them.

A scenario fixes what a run cannot choose: the goal, the initial facts, the
success criterion, and what goes wrong.  It is independent of the proposer
and of the granularity, so the same scenario runs against the coarse gold
graph, the fine gold graph, or a real model.  Only the *scripted* proposer
needs the replan answers a scenario carries; a real model plans them itself.

The goal never says which storage holds which object.  A scenario keeps that
truth as *storage physics*: a pick fails, as it would on the simulator, while
the storage the object is in stands closed (``ScriptedFailure`` with
``unless``).  The gold graphs assume the official episode, the bowl in the
kitchen counter drawer and the apple in the fridge; one scenario has them the
other way round, so a plan built on that assumption fails at the pick and
has to try the other storage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from mshab.experiments.granularity.higher_layers.builders import (
    DEFAULT_SET_TABLE_DESTINATION,
    DEFAULT_SET_TABLE_SEGMENTS,
    SET_TABLE_GOAL,
    coarse_subgoal_id,
    require_segments,
)
from mshab.experiments.granularity.higher_layers.gold import (
    GOLD_GRAPHS,
    build_gold_graph,
    validate_gold_graph,
)
from mshab.experiments.granularity.lower_layers.library import EXPERIMENT_TASK
from mshab.experiments.planning.controller import (
    DEFAULT_ATTEMPTS_PER_NODE,
    DEFAULT_MAX_REPLANS,
    RunResult,
    TaskController,
)
from mshab.experiments.planning.documents import (
    DecompositionRequest,
    DecompositionResponse,
    EntityDescription,
    SubgraphResponse,
)
from mshab.experiments.planning.proposer import (
    DecompositionKey,
    ScriptedProposer,
    SubgraphKey,
)
from mshab.experiments.planning.symbolic import (
    ScriptedFailure,
    SymbolicEnvironmentAdapter,
    SymbolicPolicy,
    SymbolicPolicyExecutor,
    bind_symbolic_policy,
)
from mshab.skills.graph import SkillGraph, SubGoalGraph
from mshab.skills.library import ContractLibrary


def gold_graphs_for(goal: str) -> Tuple[str, ...]:
    """The gold graphs authored for one goal text, in registry order."""

    return tuple(name for name, spec in GOLD_GRAPHS.items() if spec.goal == goal)


def script_gold_graphs(
    proposer: ScriptedProposer,
    names: Iterable[str],
    library: Optional[ContractLibrary] = None,
) -> None:
    """Cut the initial decomposition and every subgraph out of gold graphs.

    A gold graph's granularity (``coarse``/``fine``) is its decomposition
    key; a graph without one is keyed as ``free``.  Two gold graphs may share
    a sub-goal only if they give it the same subgraph.
    """

    for name in names:
        gold = build_gold_graph(name, library)
        order = gold.subgoal_graph.execution_order()
        subgoals = tuple(gold.subgoal_graph.subgoals[subgoal_id] for subgoal_id in order)
        key = (EXPERIMENT_TASK, gold.spec.goal, gold.spec.granularity or "free", 0, None)
        if key in proposer.decomposition_keys:
            raise ValueError(
                "gold graph {!r} repeats the decomposition key {}".format(name, key)
            )
        proposer.script_decomposition(
            key, DecompositionResponse(subgoals, "gold graph {}".format(name))
        )
        for subgoal in subgoals:
            subgraph = gold.skill_graph.subgraph_for_subgoal(subgoal.id).unsealed_copy()
            response = SubgraphResponse(subgraph, "gold graph {}".format(name))
            subgraph_key = (EXPERIMENT_TASK, subgoal.id, subgoal.predicate)
            if subgraph_key in proposer.subgraph_keys:
                existing = proposer.plan_subgraph_by_key(subgraph_key)
                if existing.as_dict()["subgraph"] != response.as_dict()["subgraph"]:
                    raise ValueError(
                        "gold graph {!r} gives sub-goal {!r} a different subgraph "
                        "than an earlier gold graph".format(name, subgoal.id)
                    )
            proposer.script_subgraph(subgraph_key, response)


def gold_proposer(
    names: Iterable[str], library: Optional[ContractLibrary] = None
) -> ScriptedProposer:
    """A scripted proposer answering from the given gold graphs alone."""

    proposer = ScriptedProposer()
    script_gold_graphs(proposer, names, library)
    return proposer


class ScenarioProposer(ScriptedProposer):
    """A scripted proposer whose replan answers may re-plan a sub-goal's subgraph.

    The gold tables map a sub-goal to one subgraph.  After a failure a
    sensible answer keeps the sub-goal but fits its subgraph to the facts at
    hand: a storage that already stands open is not opened again, a held
    object is not picked again.  A replan scripted here installs its
    subgraphs the moment its decomposition is served, so the same sub-goal id
    is answered differently before and after the failure, as a model would
    answer it.
    """

    def __init__(self) -> None:
        super().__init__()
        self._replans: Dict[DecompositionKey, Dict[SubgraphKey, SubgraphResponse]] = {}

    def script_replan(
        self,
        key: DecompositionKey,
        response: DecompositionResponse,
        subgraphs: Mapping[SubgraphKey, SubgraphResponse],
    ) -> None:
        self.script_decomposition(key, response)
        self._replans[tuple(key)] = dict(subgraphs)

    def decompose(self, request: DecompositionRequest) -> DecompositionResponse:
        response = super().decompose(request)
        for subgraph_key, subgraph in self._replans.get(request.fingerprint(), {}).items():
            self.script_subgraph(subgraph_key, subgraph)
        return response


@dataclass(frozen=True)
class ScriptedReplan:
    """The scripted proposer's answer to one Layer-1 replan request.

    The answer is built by the gold builder of ``granularity`` from
    ``context``: the ``segments`` still to do (where the objects really are)
    and the ``steps`` each has left, as the facts after the failure leave
    them.  See ``SetTableGraphBuilder`` for the context keys.
    """

    granularity: str
    failed_subgoal: str
    context: Mapping[str, Any]
    rationale: str = ""
    attempt: int = 1

    def __post_init__(self) -> None:
        context = dict(self.context)
        if not context:
            raise ValueError("a scripted replan needs a builder context")
        object.__setattr__(self, "context", context)
        if not self.failed_subgoal:
            raise ValueError("a scripted replan names the failed sub-goal")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "granularity": self.granularity,
            "failed_subgoal": self.failed_subgoal,
            "context": json.loads(json.dumps(self.context)),
            "rationale": self.rationale,
            "attempt": self.attempt,
        }


@dataclass(frozen=True)
class Scenario:
    """What a run cannot choose: goal, initial state, success criterion, mishaps."""

    name: str
    goal: str
    initial_facts: Tuple[str, ...]
    goal_facts: Tuple[str, ...]
    entities: Tuple[EntityDescription, ...]
    failures: Tuple[ScriptedFailure, ...] = ()
    replans: Tuple[ScriptedReplan, ...] = ()
    attempts_per_node: int = DEFAULT_ATTEMPTS_PER_NODE
    max_replans: int = DEFAULT_MAX_REPLANS
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_facts", tuple(sorted(set(self.initial_facts))))
        object.__setattr__(self, "goal_facts", tuple(self.goal_facts))
        entities = tuple(self.entities)
        if any(not isinstance(item, EntityDescription) for item in entities):
            raise TypeError("scenario entities must be EntityDescription instances")
        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "failures", tuple(self.failures))
        object.__setattr__(self, "replans", tuple(self.replans))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "goal": self.goal,
            "initial_facts": list(self.initial_facts),
            "goal_facts": list(self.goal_facts),
            "entities": [item.as_dict() for item in self.entities],
            "failures": [item.as_dict() for item in self.failures],
            "replans": [item.as_dict() for item in self.replans],
            "attempts_per_node": self.attempts_per_node,
            "max_replans": self.max_replans,
        }


# -- SetTable ----------------------------------------------------------------------


def set_table_entities(
    segments: Sequence[Tuple[str, str, str]] = DEFAULT_SET_TABLE_SEGMENTS,
    destination: str = DEFAULT_SET_TABLE_DESTINATION,
) -> Tuple[EntityDescription, ...]:
    """Objects first, then the storages (articulations), then the receptacle."""

    entities = [EntityDescription(obj, "object") for _, obj, _ in segments]
    for _, _, source in segments:
        articulation = EntityDescription(source, "articulation")
        if articulation not in entities:
            entities.append(articulation)
    entities.append(EntityDescription(destination, "receptacle"))
    return tuple(entities)


def set_table_initial_facts(
    segments: Sequence[Tuple[str, str, str]] = DEFAULT_SET_TABLE_SEGMENTS,
    destination: str = DEFAULT_SET_TABLE_DESTINATION,
    open_sources: Iterable[str] = (),
) -> Tuple[str, ...]:
    """Every entity present, every storage closed unless listed open, the gripper empty.

    Where the objects are is deliberately not a fact; see :func:`storage_physics`.
    """

    opened = set(open_sources)
    facts = {
        "present({})".format(entity.name)
        for entity in set_table_entities(segments, destination)
    }
    for _, _, source in segments:
        facts.add("open({})".format(source) if source in opened else "closed({})".format(source))
    facts |= {"gripper_empty()", "collision_safe()"}
    return tuple(sorted(facts))


def set_table_goal_facts(
    segments: Sequence[Tuple[str, str, str]] = DEFAULT_SET_TABLE_SEGMENTS,
    destination: str = DEFAULT_SET_TABLE_DESTINATION,
) -> Tuple[str, ...]:
    """Every object on the table and every storage closed again."""

    placed = tuple("at({},{})".format(obj, destination) for _, obj, _ in segments)
    closed = tuple(
        "closed({})".format(source)
        for source in dict.fromkeys(source for _, _, source in segments)
    )
    return placed + closed


def storage_physics(
    segments: Sequence[Tuple[str, str, str]] = DEFAULT_SET_TABLE_SEGMENTS,
) -> Tuple[ScriptedFailure, ...]:
    """An object cannot be picked while the storage it is in stands closed.

    The proposer is not told which storage holds which object; this is where
    a scenario keeps that truth.  The pick then fails the way it does on the
    simulator, by running out of time, whenever ``open(source)`` does not
    hold for the object's real storage.
    """

    return tuple(
        ScriptedFailure(
            "pick",
            obj,
            attempts=None,
            failure_mode="execution_timeout",
            unless=("open({})".format(source),),
        )
        for _, obj, source in require_segments(segments)
    )


_BOWL, _APPLE = DEFAULT_SET_TABLE_SEGMENTS
# The objects the other way round: the bowl in the fridge, the apple in the drawer.
_SWAPPED_SEGMENTS = (
    (_BOWL[0], _BOWL[1], _APPLE[2]),
    (_APPLE[0], _APPLE[1], _BOWL[2]),
)
_FINISH = ("pick", "place", "close")


def _set_table(
    name: str,
    description: str,
    *,
    open_sources: Sequence[str] = (),
    extra_facts: Sequence[str] = (),
    physics: Sequence[Tuple[str, str, str]] = DEFAULT_SET_TABLE_SEGMENTS,
    failures: Sequence[ScriptedFailure] = (),
    replans: Sequence[ScriptedReplan] = (),
) -> Scenario:
    return Scenario(
        name=name,
        description=description,
        goal=SET_TABLE_GOAL,
        initial_facts=set_table_initial_facts(open_sources=open_sources) + tuple(extra_facts),
        goal_facts=set_table_goal_facts(),
        entities=set_table_entities(),
        # The standing physics first, then the scenario's one-off mishaps.
        failures=storage_physics(physics) + tuple(failures) if physics else tuple(failures),
        replans=tuple(replans),
    )


SCENARIOS: Dict[str, Scenario] = {
    scenario.name: scenario
    for scenario in (
        _set_table(
            "nominal",
            "Every node succeeds on its first attempt; the bowl is in the kitchen "
            "counter drawer and the apple in the fridge, as the gold graphs assume.",
        ),
        _set_table(
            "pick_fails_once",
            "Picking the bowl fails once; the controller retries it.",
            failures=(ScriptedFailure("pick", _BOWL[1], attempts=(1,)),),
        ),
        _set_table(
            "pick_exhausted",
            "Picking the bowl never succeeds; after the attempt budget the sub-goal "
            "fails and the proposer is asked again. The scripted answer gives the "
            "bowl up, closes its drawer, and continues with the apple.",
            failures=(ScriptedFailure("pick", _BOWL[1], attempts=None),),
            replans=(
                ScriptedReplan(
                    "coarse", coarse_subgoal_id("bowl"), {"steps": {"bowl": ("close",)}},
                    "give the bowl up: close its drawer and continue with the apple",
                ),
                ScriptedReplan(
                    "fine", "bowl_holding", {"steps": {"bowl": ("close",)}},
                    "give the bowl up: close its drawer and continue with the apple",
                ),
            ),
        ),
        _set_table(
            "object_dropped",
            "Placing the bowl fails once and drops it; the retry cannot start "
            "because nothing is held, so the sub-goal fails and the proposer plans "
            "the bowl again, without opening the drawer that already stands open.",
            failures=(
                ScriptedFailure(
                    "place",
                    _BOWL[1],
                    attempts=(1,),
                    failure_mode="object_dropped",
                    add=("gripper_empty()",),
                    remove=("holding({})".format(_BOWL[1]),),
                ),
            ),
            replans=(
                ScriptedReplan(
                    "coarse", coarse_subgoal_id("bowl"), {"steps": {"bowl": _FINISH}},
                    "pick the dropped bowl up again; its drawer is already open",
                ),
                ScriptedReplan(
                    "fine", "bowl_placed", {"steps": {"bowl": _FINISH}},
                    "pick the dropped bowl up again; its drawer is already open",
                ),
            ),
        ),
        _set_table(
            "object_already_delivered",
            "The bowl is already on the table when the run starts; its drawer is closed.",
            extra_facts=("at({},{})".format(_BOWL[1], DEFAULT_SET_TABLE_DESTINATION),),
            physics=(_APPLE,),
        ),
        _set_table(
            "storage_already_open",
            "The fridge stands open when the run starts.",
            open_sources=(_APPLE[2],),
            replans=(
                ScriptedReplan(
                    "coarse", coarse_subgoal_id("apple"),
                    {"segments": (_APPLE,), "steps": {"apple": _FINISH}},
                    "the bowl is done; the fridge is already open: fetch the apple and close it",
                ),
            ),
        ),
        _set_table(
            "object_elsewhere",
            "The bowl is in the fridge and the apple in the kitchen counter drawer, "
            "the other way round from what the gold graphs assume; picking the bowl "
            "behind the closed fridge fails until the proposer tries the other storage.",
            physics=_SWAPPED_SEGMENTS,
            replans=(
                ScriptedReplan(
                    "coarse", coarse_subgoal_id("bowl"),
                    {"segments": _SWAPPED_SEGMENTS, "steps": {"apple": _FINISH}},
                    "the bowl was not in the kitchen counter: try the fridge, and take "
                    "the apple from the drawer that already stands open",
                ),
                ScriptedReplan(
                    "fine", "bowl_holding",
                    {"segments": _SWAPPED_SEGMENTS, "steps": {"apple": _FINISH}},
                    "the bowl was not in the kitchen counter: try the fridge, and take "
                    "the apple from the drawer that already stands open",
                ),
            ),
        ),
    )
}


# -- running a scenario ---------------------------------------------------------------


def scenario_proposer(
    scenario: Scenario, granularity: str, library: Optional[ContractLibrary] = None
) -> ScenarioProposer:
    """The gold graphs' answers plus the scenario's replan answers for one granularity.

    The gold graphs of the scenario's goal provide the initial decomposition
    and every subgraph; the one at ``granularity`` (``free`` for a graph
    authored without one) lends its builder to the replan answers, which are
    built from each replan's context and validated before they are scripted.
    """

    names = gold_graphs_for(scenario.goal)
    reference = [
        name for name in names if (GOLD_GRAPHS[name].granularity or "free") == granularity
    ]
    if not reference:
        raise ValueError(
            "no gold graph answers goal {!r} at granularity {!r}; authored: {}".format(
                scenario.goal,
                granularity,
                {name: GOLD_GRAPHS[name].granularity or "free" for name in names},
            )
        )
    proposer = ScenarioProposer()
    script_gold_graphs(proposer, names, library)
    builder = GOLD_GRAPHS[reference[0]].builder
    for replan in scenario.replans:
        if replan.granularity != granularity:
            continue
        patch = builder.propose(scenario.goal, EXPERIMENT_TASK, replan.context)
        subgoal_graph = SubGoalGraph(scenario.goal)
        skill_graph = SkillGraph(EXPERIMENT_TASK, subgoal_graph=subgoal_graph)
        patch.apply(subgoal_graph, skill_graph, library=library)
        validate_gold_graph(subgoal_graph, skill_graph, library)
        predicates = {subgoal.id: subgoal.predicate for subgoal in patch.subgoals}
        proposer.script_replan(
            (EXPERIMENT_TASK, scenario.goal, granularity, replan.attempt, replan.failed_subgoal),
            DecompositionResponse(patch.subgoals, replan.rationale),
            {
                (EXPERIMENT_TASK, subgraph.subgoal_id, predicates[subgraph.subgoal_id]):
                SubgraphResponse(subgraph.unsealed_copy(), replan.rationale)
                for subgraph in patch.skill_subgraphs
            },
        )
    return proposer


def run_scenario(
    scenario: Scenario,
    granularity: str,
    library: ContractLibrary,
    proposer: Optional[ScriptedProposer] = None,
    **controller_options: Any,
) -> RunResult:
    """Run one scenario at one granularity on the symbolic environment.

    The library gets the symbolic policy bound if it does not have it yet.
    Without ``proposer`` the scripted one answers from the gold graphs of the
    scenario's goal.  ``controller_options`` override the scenario's attempt
    and replan budgets, or pass a ``validator`` with a retry budget.
    """

    try:
        library.policy(SymbolicPolicy.ID)
    except KeyError:
        bind_symbolic_policy(library, EXPERIMENT_TASK)
    environment = SymbolicEnvironmentAdapter(
        library, EXPERIMENT_TASK, scenario.initial_facts, scenario.entities
    )
    executor = SymbolicPolicyExecutor(scenario.failures)
    options = {
        "attempts_per_node": scenario.attempts_per_node,
        "max_replans": scenario.max_replans,
    }
    options.update(controller_options)
    controller = TaskController(library, EXPERIMENT_TASK, **options)
    return controller.run(
        scenario.goal,
        proposer or scenario_proposer(scenario, granularity, library),
        environment,
        executor,
        granularity=granularity,
        goal_facts=scenario.goal_facts,
    )
