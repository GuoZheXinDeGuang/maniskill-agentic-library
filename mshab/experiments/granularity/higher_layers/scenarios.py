"""Scenarios of the granularity experiment and the scripted proposer that plays them.

A scenario fixes what a run cannot choose: the goal, the initial facts, the
success criterion, and what goes wrong.  It is independent of the proposer
and of the granularity, so the same scenario runs against the coarse gold
graph, the fine gold graph, or a real model.  Only the *scripted* proposer
needs the replan answers a scenario carries; a real model plans them itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from mshab.experiments.granularity.higher_layers.builders import (
    DEFAULT_TIDY_HOUSE_TRANSFERS,
    TIDY_HOUSE_GOAL,
)
from mshab.experiments.granularity.higher_layers.gold import build_gold_graph
from mshab.experiments.granularity.lower_layers.library import EXPERIMENT_TASK
from mshab.experiments.planning.controller import RunResult, TaskController
from mshab.experiments.planning.documents import (
    DecompositionResponse,
    SubgraphResponse,
)
from mshab.experiments.planning.proposer import ScriptedProposer
from mshab.experiments.planning.symbolic import (
    ScriptedFailure,
    SymbolicEnvironmentAdapter,
    SymbolicPolicy,
    SymbolicPolicyExecutor,
    bind_symbolic_policy,
)
from mshab.skills.library import ContractLibrary


GOLD_GRAPHS_BY_GRANULARITY = {"coarse": "tidy_house_coarse", "fine": "tidy_house_fine"}


def gold_proposer(
    names: Iterable[str], library: Optional[ContractLibrary] = None
) -> ScriptedProposer:
    """Cut the initial decomposition and every subgraph out of gold graphs.

    A gold graph's granularity (``coarse``/``fine``) is its decomposition
    key; a graph without one is keyed as ``free``.  Two gold graphs may share
    a sub-goal only if they give it the same subgraph.
    """

    proposer = ScriptedProposer()
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
    return proposer


@dataclass(frozen=True)
class ScriptedReplan:
    """The scripted proposer's answer to one Layer-1 replan request."""

    granularity: str
    failed_subgoal: str
    subgoal_ids: Tuple[str, ...]
    rationale: str = ""
    attempt: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "subgoal_ids", tuple(self.subgoal_ids))
        if not self.subgoal_ids:
            raise ValueError("a scripted replan must list at least one sub-goal")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "granularity": self.granularity,
            "failed_subgoal": self.failed_subgoal,
            "subgoal_ids": list(self.subgoal_ids),
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
    entities: Tuple[Tuple[str, str], ...]
    failures: Tuple[ScriptedFailure, ...] = ()
    replans: Tuple[ScriptedReplan, ...] = ()
    attempts_per_node: int = 2
    max_replans: int = 2
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_facts", tuple(sorted(set(self.initial_facts))))
        object.__setattr__(self, "goal_facts", tuple(self.goal_facts))
        object.__setattr__(self, "entities", tuple(tuple(item) for item in self.entities))
        object.__setattr__(self, "failures", tuple(self.failures))
        object.__setattr__(self, "replans", tuple(self.replans))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "goal": self.goal,
            "initial_facts": list(self.initial_facts),
            "goal_facts": list(self.goal_facts),
            "entities": [list(item) for item in self.entities],
            "failures": [item.as_dict() for item in self.failures],
            "replans": [item.as_dict() for item in self.replans],
            "attempts_per_node": self.attempts_per_node,
            "max_replans": self.max_replans,
        }


# -- TidyHouse ---------------------------------------------------------------------


def coarse_ids(indices: Iterable[int]) -> Tuple[str, ...]:
    return tuple("object_{}_delivered".format(index) for index in indices)


def fine_ids(indices: Iterable[int]) -> Tuple[str, ...]:
    ids = []
    for index in indices:
        ids.extend(
            (
                "object_{}_reachable".format(index),
                "object_{}_holding".format(index),
                "destination_{}_reachable".format(index),
                "object_{}_placed".format(index),
            )
        )
    return tuple(ids)


def tidy_house_entities(
    transfers: Sequence[Tuple[str, str]] = DEFAULT_TIDY_HOUSE_TRANSFERS,
) -> Tuple[Tuple[str, str], ...]:
    entities = [(obj, "object") for obj, _ in transfers]
    for _, destination in transfers:
        if (destination, "receptacle") not in entities:
            entities.append((destination, "receptacle"))
    return tuple(entities)


def tidy_house_initial_facts(
    transfers: Sequence[Tuple[str, str]] = DEFAULT_TIDY_HOUSE_TRANSFERS,
) -> Tuple[str, ...]:
    facts = {"present({})".format(name) for name, _ in tidy_house_entities(transfers)}
    facts |= {"gripper_empty()", "collision_safe()"}
    return tuple(sorted(facts))


def tidy_house_goal_facts(
    transfers: Sequence[Tuple[str, str]] = DEFAULT_TIDY_HOUSE_TRANSFERS,
) -> Tuple[str, ...]:
    return tuple("at({},{})".format(obj, destination) for obj, destination in transfers)


def _tidy_house(
    name: str,
    description: str,
    *,
    extra_facts: Sequence[str] = (),
    failures: Sequence[ScriptedFailure] = (),
    replans: Sequence[ScriptedReplan] = (),
) -> Scenario:
    return Scenario(
        name=name,
        description=description,
        goal=TIDY_HOUSE_GOAL,
        initial_facts=tidy_house_initial_facts() + tuple(extra_facts),
        goal_facts=tidy_house_goal_facts(),
        entities=tidy_house_entities(),
        failures=tuple(failures),
        replans=tuple(replans),
    )


_SECOND = DEFAULT_TIDY_HOUSE_TRANSFERS[1]  # ("003_cracker_box", "dining_table")
_FIRST = DEFAULT_TIDY_HOUSE_TRANSFERS[0]  # ("002_master_chef_can", "kitchen_counter")

SCENARIOS: Dict[str, Scenario] = {
    scenario.name: scenario
    for scenario in (
        _tidy_house("nominal", "Every node succeeds on its first attempt."),
        _tidy_house(
            "pick_fails_once",
            "Picking the second object fails once; the controller retries it.",
            failures=(ScriptedFailure("pick", _SECOND[0], attempts=(1,)),),
        ),
        _tidy_house(
            "pick_exhausted",
            "Picking the second object never succeeds; after the attempt budget "
            "the sub-goal fails and the proposer is asked again. The scripted "
            "answer gives the object up.",
            failures=(ScriptedFailure("pick", _SECOND[0], attempts=None),),
            replans=(
                ScriptedReplan(
                    "coarse", "object_2_delivered", coarse_ids((3, 4, 5)),
                    "skip the object whose pick keeps failing",
                ),
                ScriptedReplan(
                    "fine", "object_2_holding", fine_ids((3, 4, 5)),
                    "skip the object whose pick keeps failing",
                ),
            ),
        ),
        _tidy_house(
            "object_dropped",
            "Placing the second object fails once and drops it; the retry cannot "
            "start because nothing is held, so the sub-goal fails and the "
            "proposer plans the transfer again.",
            failures=(
                ScriptedFailure(
                    "place",
                    _SECOND[0],
                    attempts=(1,),
                    failure_mode="object_dropped",
                    add=("gripper_empty()",),
                    remove=("holding({})".format(_SECOND[0]),),
                ),
            ),
            replans=(
                ScriptedReplan(
                    "coarse", "object_2_delivered", coarse_ids((2, 3, 4, 5)),
                    "pick the dropped object up again and continue",
                ),
                ScriptedReplan(
                    "fine", "object_2_placed", fine_ids((2, 3, 4, 5)),
                    "pick the dropped object up again and continue",
                ),
            ),
        ),
        _tidy_house(
            "object_already_delivered",
            "The first object is already at its destination when the run starts.",
            extra_facts=("at({},{})".format(*_FIRST),),
        ),
    )
}


# -- running a scenario ---------------------------------------------------------------


def scenario_proposer(
    scenario: Scenario, granularity: str, library: Optional[ContractLibrary] = None
) -> ScriptedProposer:
    """The gold graphs' answers plus the scenario's replan answers for one granularity."""

    if granularity not in GOLD_GRAPHS_BY_GRANULARITY:
        raise ValueError(
            "scenarios are scripted for granularities {}, not {!r}".format(
                sorted(GOLD_GRAPHS_BY_GRANULARITY), granularity
            )
        )
    proposer = gold_proposer(GOLD_GRAPHS_BY_GRANULARITY.values(), library)
    gold = build_gold_graph(GOLD_GRAPHS_BY_GRANULARITY[granularity], library)
    for replan in scenario.replans:
        if replan.granularity != granularity:
            continue
        subgoals = tuple(gold.subgoal_graph.subgoals[subgoal_id] for subgoal_id in replan.subgoal_ids)
        proposer.script_decomposition(
            (EXPERIMENT_TASK, scenario.goal, granularity, replan.attempt, replan.failed_subgoal),
            DecompositionResponse(subgoals, replan.rationale),
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
    ``controller_options`` override the scenario's attempt and replan budgets.
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
