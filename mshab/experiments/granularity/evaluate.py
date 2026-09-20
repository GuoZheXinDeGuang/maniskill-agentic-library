"""Evaluate a proposer on the granularity sweep: the stage-5 experiment.

    python -m mshab.experiments.granularity.evaluate --samples 3
    python -m mshab.experiments.granularity.evaluate --proposer scripted   # offline dry run

Per goal, per granularity (``free``, ``coarse``, ``fine``), per sample, the
evaluation first asks the proposer for a plan under the goal's nominal facts
and records how the validator received it (accepted at once, accepted after
retries, or rejected), the decomposition's size and predicate vocabulary,
and its agreement with the gold sequence and with the gold subgraphs of the
same predicates.  Then it runs the goal's scenarios through the controller
with the same proposer and the same injected failures and records the run
metrics, replanning span included.  Every rejection of every round, static
or inside a run, is tallied by validator rule, which shows which rule the
prompt explains badly.  Traces go to the output directory; ``summary.json``
aggregates them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
)

from mshab.experiments.granularity.higher_layers.builders import (
    SET_TABLE_GOAL,
    TIDY_HOUSE_GOAL,
)
from mshab.experiments.granularity.higher_layers.gold import (
    GOLD_GRAPHS,
    GoldGraph,
    build_gold_graph,
)
from mshab.experiments.granularity.higher_layers.scenarios import (
    SCENARIOS,
    Scenario,
    gold_graphs_for,
    gold_proposer,
    run_scenario,
    scenario_proposer,
)
from mshab.experiments.granularity.lower_layers.library import (
    EXPERIMENT_TASK,
    build_granularity_library,
    split_contract_id,
)
from mshab.experiments.granularity.paths import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_EVALUATION_DIR,
)
from mshab.experiments.planning.deepseek import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS,
    DeepSeekProposer,
)
from mshab.experiments.planning.documents import GRANULARITIES, PlanningContext
from mshab.experiments.planning.proposer import GraphProposer, ProposerUnavailable
from mshab.experiments.planning.symbolic import SymbolicPolicy, bind_symbolic_policy
from mshab.experiments.planning.validator import (
    ProposalRejected,
    ProposalRound,
    ProposalValidator,
)
from mshab.skills.graph import SkillGraph, SkillNode, SkillSubgraph, SubGoal, SubGoalGraph
from mshab.skills.library import ContractLibrary


EVALUATION_SCHEMA_VERSION = "mshab.granularity-evaluation.v1"
VALIDITY = ("accepted_first_try", "accepted_after_retries", "rejected")
#: The plan's retry budget for the real model: a refused call is asked again
#: at most twice, with the rejections as a further user turn.
DEFAULT_MODEL_RETRIES = 2

#: Builds the proposer for one static proposal (``scenario`` is ``None``) or
#: one scenario run.  The real model is one object for everything; the
#: scripted one needs the scenario's replan answers.
ProposerFactory = Callable[["EvaluationGoal", str, Optional[Scenario]], GraphProposer]


# -- what is evaluated ------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationGoal:
    """One goal of the sweep: its text, its gold references, its scenarios."""

    name: str
    goal: str
    references: Mapping[str, str]
    scenarios: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "references", dict(self.references))
        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        if not self.scenarios:
            raise ValueError("evaluation goal {!r} has no scenario".format(self.name))
        unknown = sorted(set(self.scenarios) - set(SCENARIOS))
        if unknown:
            raise KeyError("unknown scenarios {}".format(unknown))

    def reference_names(self, granularity: str) -> Tuple[str, ...]:
        """The gold graphs a plan at ``granularity`` is compared with.

        The graph authored at that granularity when there is one; otherwise
        every reference of the goal (``free`` has no gold graph of its own).
        """

        if granularity in self.references:
            return (self.references[granularity],)
        return tuple(self.references.values())

    @property
    def nominal(self) -> Scenario:
        """The scenario whose initial facts and entities the static proposal uses."""

        return SCENARIOS[self.scenarios[0]]


def _references_for(goal: str) -> Dict[str, str]:
    return {
        (GOLD_GRAPHS[name].granularity or "free"): name for name in gold_graphs_for(goal)
    }


def _scenarios_for(goal: str) -> Tuple[str, ...]:
    return tuple(name for name, scenario in SCENARIOS.items() if scenario.goal == goal)


EVALUATION_GOALS: Mapping[str, EvaluationGoal] = {
    "tidy_house": EvaluationGoal(
        "tidy_house",
        TIDY_HOUSE_GOAL,
        _references_for(TIDY_HOUSE_GOAL),
        _scenarios_for(TIDY_HOUSE_GOAL),
    ),
    "set_table": EvaluationGoal(
        "set_table",
        SET_TABLE_GOAL,
        _references_for(SET_TABLE_GOAL),
        _scenarios_for(SET_TABLE_GOAL),
    ),
}


# -- agreement with the gold graphs --------------------------------------------------


Role = Tuple[str, Tuple[Tuple[str, Any], ...]]


def node_role(node: SkillNode) -> Role:
    """``(contract type, sorted arguments)``: what a node does, whatever it is called.

    A proposer chooses its own node ids, so nodes are compared by role.
    """

    _, contract_type, _ = split_contract_id(node.contract_id)
    return (contract_type, tuple(sorted(node.arguments.items())))


def role_label(role: Role) -> str:
    return "{}({})".format(role[0], ",".join(str(value) for _, value in role[1]))


def predicate_name(predicate: str) -> str:
    """``at`` of ``at(x,y)``: the vocabulary a decomposition draws on."""

    head, _, _ = predicate.partition("(")
    return head or predicate


def _lcs_length(first: Sequence[str], second: Sequence[str]) -> int:
    previous = [0] * (len(second) + 1)
    for item in first:
        current = [0]
        for index, other in enumerate(second, start=1):
            if item == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


def order_similarity(first: Sequence[str], second: Sequence[str]) -> float:
    """Longest common subsequence over the longer length: 1.0 for identical sequences."""

    if not first and not second:
        return 1.0
    return _lcs_length(first, second) / max(len(first), len(second))


def _jaccard(first: Set[Any], second: Set[Any]) -> float:
    if not first and not second:
        return 1.0
    return len(first & second) / len(first | second)


def _ratio(part: float, whole: float) -> float:
    return part / whole if whole else 0.0


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _stats(values: Iterable[float]) -> Dict[str, float]:
    items = list(values)
    if not items:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": sum(items) / len(items), "min": min(items), "max": max(items)}


def decomposition_agreement(
    subgoals: Sequence[SubGoal], gold: SubGoalGraph
) -> Dict[str, Any]:
    """How a proposed sub-goal sequence relates to a gold one, by predicate.

    Sub-goal ids are the proposer's own, so only predicates are compared:
    exact sequence match, order similarity, and set precision and recall.
    """

    predicates = [subgoal.predicate for subgoal in subgoals]
    gold_predicates = [gold.subgoals[subgoal_id].predicate for subgoal_id in gold.execution_order()]
    common = set(predicates) & set(gold_predicates)
    return {
        "exact": predicates == gold_predicates,
        "order_similarity": order_similarity(predicates, gold_predicates),
        "precision": _ratio(len(common), len(set(predicates))),
        "recall": _ratio(len(common), len(set(gold_predicates))),
        "subgoals": len(predicates),
        "gold_subgoals": len(gold_predicates),
    }


def _roles(subgraph: SkillSubgraph) -> Set[Role]:
    return {node_role(node) for node in subgraph.nodes.values()}


def _edge_roles(subgraph: SkillSubgraph) -> Set[Tuple[Role, Role, str]]:
    nodes = subgraph.nodes
    return {
        (node_role(nodes[edge.source]), node_role(nodes[edge.target]), edge.relation.value)
        for edge in subgraph.edges
    }


def subgraph_agreement(
    subgoal_graph: SubGoalGraph, skill_graph: SkillGraph, gold: GoldGraph
) -> Dict[str, Any]:
    """Node and edge agreement with the gold subgraph of the same predicate.

    Nodes are matched by role and edges by the roles they join.  The whole
    graph's role set is compared too: it is the same 20 atomic subtasks at
    every TidyHouse granularity, so a difference there is not a granularity
    effect but a planning one.
    """

    gold_by_predicate: Dict[str, str] = {}
    for gold_id in gold.subgoal_graph.execution_order():
        gold_by_predicate.setdefault(gold.subgoal_graph.subgoals[gold_id].predicate, gold_id)
    per_subgoal: List[Dict[str, Any]] = []
    unmatched: List[str] = []
    proposed: Set[str] = set()
    for subgoal_id in subgoal_graph.execution_order():
        predicate = subgoal_graph.subgoals[subgoal_id].predicate
        proposed.add(predicate)
        gold_id = gold_by_predicate.get(predicate)
        if gold_id is None:
            unmatched.append(predicate)
            continue
        ours = skill_graph.subgraph_for_subgoal(subgoal_id)
        theirs = gold.skill_graph.subgraph_for_subgoal(gold_id)
        our_roles, gold_roles = _roles(ours), _roles(theirs)
        our_edges, gold_edges = _edge_roles(ours), _edge_roles(theirs)
        per_subgoal.append(
            {
                "predicate": predicate,
                "nodes_match": our_roles == gold_roles,
                "edges_match": our_edges == gold_edges,
                "node_jaccard": _jaccard(our_roles, gold_roles),
                "edge_jaccard": _jaccard(our_edges, gold_edges),
                "nodes": len(our_roles),
                "gold_nodes": len(gold_roles),
            }
        )
    all_roles = {node_role(node) for node in skill_graph.nodes.values()}
    gold_all = {node_role(node) for node in gold.skill_graph.nodes.values()}
    return {
        "matched": len(per_subgoal),
        "unmatched_predicates": unmatched,
        "missing_predicates": [
            predicate for predicate in gold_by_predicate if predicate not in proposed
        ],
        "nodes_match_rate": _mean(item["nodes_match"] for item in per_subgoal),
        "edges_match_rate": _mean(item["edges_match"] for item in per_subgoal),
        "node_jaccard": _mean(item["node_jaccard"] for item in per_subgoal),
        "edge_jaccard": _mean(item["edge_jaccard"] for item in per_subgoal),
        "graph_role_precision": _ratio(len(all_roles & gold_all), len(all_roles)),
        "graph_role_recall": _ratio(len(all_roles & gold_all), len(gold_all)),
        "per_subgoal": per_subgoal,
    }


# -- rejections by rule ------------------------------------------------------------------


_QUOTED = re.compile(r"'[^']*'")
_BRACKETED = re.compile(r"\[[^\[\]]*\]")


def rejection_rule(message: str) -> str:
    """A rejection message with its identifiers and lists blanked: the rule that fired."""

    rule = _QUOTED.sub("'...'", message)
    rule = _BRACKETED.sub("[...]", rule)
    return rule[:160]


class RejectionTally:
    """How often each validator rule refused an answer, with one example each."""

    def __init__(self) -> None:
        self._counts: Counter = Counter()
        self._examples: Dict[Tuple[str, str], str] = {}

    def add(self, stage: str, message: str) -> None:
        key = (stage, rejection_rule(message))
        self._counts[key] += 1
        self._examples.setdefault(key, message)

    def add_rounds(self, rounds: Iterable[ProposalRound]) -> None:
        for item in rounds:
            for rejection in item.rejections:
                self.add(rejection.stage, rejection.message)

    def add_records(self, proposals: Iterable[Mapping[str, Any]]) -> None:
        """The controller's proposal records, whose rounds are already dicts."""

        for record in proposals:
            for item in record.get("rounds", ()):
                for rejection in item.get("rejections", ()):
                    self.add(rejection["stage"], rejection["message"])

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    def as_list(self) -> List[Dict[str, Any]]:
        rows = sorted(self._counts.items(), key=lambda item: (-item[1], item[0]))
        return [
            {"stage": stage, "rule": rule, "count": count, "example": self._examples[(stage, rule)]}
            for (stage, rule), count in rows
        ]


def validity_of(rounds: int, accepted: bool) -> str:
    if not accepted:
        return "rejected"
    return "accepted_first_try" if rounds <= 1 else "accepted_after_retries"


def _calls(rounds: Iterable[ProposalRound]) -> int:
    return sum(len(item.calls) for item in rounds)


# -- the sweep -----------------------------------------------------------------------------


class Evaluation:
    """The sweep over goals, granularities, and samples for one proposer."""

    def __init__(
        self,
        library: ContractLibrary,
        make_proposer: ProposerFactory,
        *,
        retries: int = DEFAULT_MODEL_RETRIES,
        output: Optional[Path] = None,
        description: Optional[Mapping[str, Any]] = None,
        log: Callable[[str], None] = lambda line: None,
    ) -> None:
        self.library = library
        self.make_proposer = make_proposer
        self.retries = retries
        self.output = None if output is None else Path(output)
        self.description = dict(description or {})
        self.log = log
        self.tally = RejectionTally()
        self.samples: List[Dict[str, Any]] = []
        try:
            library.policy(SymbolicPolicy.ID)
        except KeyError:
            bind_symbolic_policy(library, EXPERIMENT_TASK)

    def run(
        self,
        goals: Iterable[str],
        granularities: Iterable[str],
        samples: int,
        *,
        scenarios: Optional[Iterable[str]] = None,
        static: bool = True,
        run_scenarios: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate every (goal, granularity) ``samples`` times; return the summary."""

        if samples < 1:
            raise ValueError("samples must be at least 1")
        scenario_names = None if scenarios is None else tuple(scenarios)
        for goal_name in goals:
            try:
                goal = EVALUATION_GOALS[goal_name]
            except KeyError as exc:
                raise KeyError(
                    "unknown evaluation goal {!r}; available={}".format(
                        goal_name, sorted(EVALUATION_GOALS)
                    )
                ) from exc
            for granularity in granularities:
                if granularity not in GRANULARITIES:
                    raise ValueError(
                        "granularity must be one of {}, got {!r}".format(GRANULARITIES, granularity)
                    )
                for index in range(samples):
                    self.samples.append(
                        self.evaluate_sample(
                            goal,
                            granularity,
                            index,
                            scenarios=scenario_names,
                            static=static,
                            run_scenarios=run_scenarios,
                        )
                    )
        return self.summary()

    def evaluate_sample(
        self,
        goal: EvaluationGoal,
        granularity: str,
        index: int,
        *,
        scenarios: Optional[Sequence[str]] = None,
        static: bool = True,
        run_scenarios: bool = True,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "goal": goal.name,
            "granularity": granularity,
            "sample": index,
            "static": None,
            "scenarios": {},
        }
        directory = (
            None
            if self.output is None
            else self.output / goal.name / granularity / "sample_{:02d}".format(index)
        )
        label = "[{} {} #{}]".format(goal.name, granularity, index)
        if static:
            self.log("{} static proposal".format(label))
            record["static"] = self._static(goal, granularity, directory)
            self.log("{} -> {}".format(label, record["static"].get("validity", record["static"].get("error"))))
        if run_scenarios:
            names = goal.scenarios if scenarios is None else scenarios
            for name in names:
                if name not in SCENARIOS:
                    raise KeyError(
                        "unknown scenario {!r}; available={}".format(name, sorted(SCENARIOS))
                    )
                if name not in goal.scenarios:
                    continue
                self.log("{} scenario {}".format(label, name))
                record["scenarios"][name] = self._scenario(goal, granularity, SCENARIOS[name], directory)
                self.log(
                    "{} -> {}".format(
                        label, record["scenarios"][name].get("status", record["scenarios"][name].get("error"))
                    )
                )
        return record

    # -- one static proposal ----------------------------------------------------------------

    def _static(
        self, goal: EvaluationGoal, granularity: str, directory: Optional[Path]
    ) -> Dict[str, Any]:
        proposer = self.make_proposer(goal, granularity, None)
        validator = ProposalValidator(self.library, EXPERIMENT_TASK, retries=self.retries)
        nominal = goal.nominal
        context = PlanningContext.initial(
            self.library,
            EXPERIMENT_TASK,
            entities=nominal.entities,
            facts=nominal.initial_facts,
            granularity=granularity,
        )
        started = time.monotonic()
        try:
            proposal = validator.plan(proposer, goal.goal, context)
        except ProposalRejected as exc:
            self.tally.add_rounds(exc.rounds)
            self._write(
                directory,
                "static.json",
                {
                    "accepted": False,
                    "goal": goal.goal,
                    "granularity": granularity,
                    "context": context.as_dict(),
                    "rejections": exc.as_dict(),
                    "rounds": [item.as_dict() for item in exc.rounds],
                },
            )
            return {
                "validity": "rejected",
                "rounds": len(exc.rounds),
                "calls": _calls(exc.rounds),
                "elapsed": round(time.monotonic() - started, 3),
                "rejections": exc.as_dict(),
                "decomposition": None,
                "subgraphs": {},
            }
        except ProposerUnavailable as exc:
            return {"error": str(exc)}
        elapsed = time.monotonic() - started
        self.tally.add_rounds(proposal.rounds)
        subgoals = proposal.decomposition_response.subgoals
        references = {
            name: build_gold_graph(name, self.library) for name in goal.reference_names(granularity)
        }
        result = {
            "validity": validity_of(len(proposal.rounds), True),
            "rounds": len(proposal.rounds),
            "calls": _calls(proposal.rounds),
            "elapsed": round(elapsed, 3),
            "rejections": [],
            "decomposition": {
                "subgoals": len(subgoals),
                "predicates": [subgoal.predicate for subgoal in subgoals],
                "vocabulary": dict(
                    Counter(predicate_name(subgoal.predicate) for subgoal in subgoals)
                ),
                "agreement": {
                    name: decomposition_agreement(subgoals, gold.subgoal_graph)
                    for name, gold in references.items()
                },
            },
            "subgraphs": {
                name: subgraph_agreement(proposal.subgoal_graph, proposal.skill_graph, gold)
                for name, gold in references.items()
            },
        }
        trace = proposal.as_dict()
        trace["accepted"] = True
        trace["granularity"] = granularity
        trace["evaluation"] = {
            "decomposition": result["decomposition"],
            "subgraphs": result["subgraphs"],
        }
        self._write(directory, "static.json", trace)
        return result

    # -- one scenario run -------------------------------------------------------------------

    def _scenario(
        self,
        goal: EvaluationGoal,
        granularity: str,
        scenario: Scenario,
        directory: Optional[Path],
    ) -> Dict[str, Any]:
        proposer = self.make_proposer(goal, granularity, scenario)
        validator = ProposalValidator(self.library, EXPERIMENT_TASK, retries=self.retries)
        started = time.monotonic()
        try:
            run = run_scenario(
                scenario, granularity, self.library, proposer=proposer, validator=validator
            )
        except ProposerUnavailable as exc:
            return {"error": str(exc)}
        self.tally.add_records(run.proposals)
        self._write(directory, "scenario_{}.json".format(scenario.name), run.as_dict())
        return {
            # .value, not the member: this record is both json.dumps'd and
            # interpolated into the progress log, and "{}".format() on a str
            # enum prints "Status.SUCCESS".
            "status": run.status.value,
            "success": run.success,
            "elapsed": round(time.monotonic() - started, 3),
            "proposals": len(run.proposals),
            "metrics": dict(run.metrics),
        }

    def _write(self, directory: Optional[Path], name: str, document: Mapping[str, Any]) -> None:
        if directory is None:
            return
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    # -- aggregation --------------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for record in self.samples:
            groups.setdefault((record["goal"], record["granularity"]), []).append(record)
        goals: Dict[str, Dict[str, Any]] = {}
        for (goal, granularity), records in groups.items():
            goals.setdefault(goal, {})[granularity] = aggregate_samples(records)
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "proposer": dict(self.description),
            "retries": self.retries,
            "samples": len(self.samples),
            "goals": goals,
            "rejections": self.tally.as_list(),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2, sort_keys=True) + "\n")


def aggregate_samples(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The per-(goal, granularity) block of the summary."""

    statics = [
        record["static"]
        for record in records
        if record.get("static") and "error" not in record["static"]
    ]
    errors = sum(
        1 for record in records if record.get("static") and "error" in record["static"]
    ) + sum(
        1 for record in records for run in record["scenarios"].values() if "error" in run
    )
    validity = Counter(item["validity"] for item in statics)
    decompositions = [item["decomposition"] for item in statics if item["decomposition"]]
    vocabulary: Counter = Counter()
    for item in decompositions:
        vocabulary.update(item["vocabulary"])
    agreement: Dict[str, Dict[str, float]] = {}
    for name in sorted({name for item in decompositions for name in item["agreement"]}):
        items = [item["agreement"][name] for item in decompositions if name in item["agreement"]]
        agreement[name] = {
            "exact_rate": _mean(item["exact"] for item in items),
            "order_similarity": _mean(item["order_similarity"] for item in items),
            "precision": _mean(item["precision"] for item in items),
            "recall": _mean(item["recall"] for item in items),
            "gold_subgoals": items[0]["gold_subgoals"],
        }
    subgraphs: Dict[str, Dict[str, float]] = {}
    for name in sorted({name for item in statics for name in item["subgraphs"]}):
        items = [item["subgraphs"][name] for item in statics if name in item["subgraphs"]]
        subgraphs[name] = {
            key: _mean(item[key] for item in items)
            for key in (
                "matched",
                "nodes_match_rate",
                "edges_match_rate",
                "node_jaccard",
                "edge_jaccard",
                "graph_role_precision",
                "graph_role_recall",
            )
        }
    scenarios: Dict[str, Dict[str, Any]] = {}
    for name in sorted({name for record in records for name in record["scenarios"]}):
        runs = [
            record["scenarios"][name]
            for record in records
            if name in record["scenarios"] and "error" not in record["scenarios"][name]
        ]
        keys = sorted(
            {
                key
                for run in runs
                for key, value in run["metrics"].items()
                if isinstance(value, (int, float))
            }
        )
        scenarios[name] = {
            "runs": len(runs),
            "success_rate": _mean(run["success"] for run in runs),
            "statuses": dict(Counter(run["status"] for run in runs)),
            "metrics": {
                key: _mean(run["metrics"][key] for run in runs if key in run["metrics"])
                for key in keys
            },
        }
    return {
        "samples": len(records),
        "validity": {name: validity.get(name, 0) for name in VALIDITY},
        "rounds": _mean(item["rounds"] for item in statics),
        "calls": _mean(item["calls"] for item in statics),
        "decomposition": {
            "subgoals": _stats(item["subgoals"] for item in decompositions),
            "vocabulary": dict(vocabulary),
            "agreement": agreement,
        },
        "subgraphs": subgraphs,
        "scenarios": scenarios,
        "errors": errors,
    }


# -- proposers -----------------------------------------------------------------------------


def scripted_factory(library: ContractLibrary) -> ProposerFactory:
    """The gold graphs as the proposer: the offline dry run of the whole sweep."""

    def make(goal: EvaluationGoal, granularity: str, scenario: Optional[Scenario]) -> GraphProposer:
        if scenario is None:
            return gold_proposer(gold_graphs_for(goal.goal), library)
        return scenario_proposer(scenario, granularity, library)

    return make


def fixed_factory(proposer: GraphProposer) -> ProposerFactory:
    """One proposer object for everything: the real model."""

    return lambda goal, granularity, scenario: proposer


# -- command line -------------------------------------------------------------------------


def print_summary(summary: Mapping[str, Any], out: Optional[TextIO] = None) -> None:
    # Resolved at call time so a redirected stdout is honoured.
    out = sys.stdout if out is None else out
    proposer = summary.get("proposer", {})
    out.write(
        "proposer: {}   retries: {}   samples: {}\n".format(
            ", ".join("{}={}".format(key, value) for key, value in sorted(proposer.items())),
            summary.get("retries"),
            summary.get("samples"),
        )
    )
    for goal, blocks in summary["goals"].items():
        for granularity, block in blocks.items():
            validity = block["validity"]
            out.write(
                "{:<12} {:<7} n={:<3} first={} retried={} rejected={} rounds={:.2f} "
                "calls={:.1f} subgoals={:.1f}\n".format(
                    goal,
                    granularity,
                    block["samples"],
                    validity["accepted_first_try"],
                    validity["accepted_after_retries"],
                    validity["rejected"],
                    block["rounds"],
                    block["calls"],
                    block["decomposition"]["subgoals"]["mean"],
                )
            )
            for name, item in block["decomposition"]["agreement"].items():
                out.write(
                    "    vs {:<20} exact={:.2f} order={:.2f} precision={:.2f} recall={:.2f}".format(
                        name,
                        item["exact_rate"],
                        item["order_similarity"],
                        item["precision"],
                        item["recall"],
                    )
                )
                subgraph = block["subgraphs"].get(name)
                if subgraph is not None:
                    out.write(
                        "  nodes={:.2f} edges={:.2f} roles p/r={:.2f}/{:.2f}".format(
                            subgraph["nodes_match_rate"],
                            subgraph["edges_match_rate"],
                            subgraph["graph_role_precision"],
                            subgraph["graph_role_recall"],
                        )
                    )
                out.write("\n")
            for name, item in block["scenarios"].items():
                metrics = item["metrics"]
                out.write(
                    "    {:<26} success={:.2f} executions={:.1f} retries={:.1f} replans={:.1f} "
                    "span={:.1f} proposal_retries={:.1f}\n".format(
                        name,
                        item["success_rate"],
                        metrics.get("node_executions", 0.0),
                        metrics.get("retries", 0.0),
                        metrics.get("replans", 0.0),
                        metrics.get("replanning_span", 0.0),
                        metrics.get("proposal_retries", 0.0),
                    )
                )
    if summary["rejections"]:
        out.write("most frequent rejections:\n")
        for row in summary["rejections"][:10]:
            out.write("    {:>4}  [{}] {}\n".format(row["count"], row["stage"], row["rule"]))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a proposer on the granularity sweep (stage 5)."
    )
    parser.add_argument(
        "--proposer",
        choices=("deepseek", "scripted"),
        default="deepseek",
        help="the real model, or the gold graphs as an offline dry run",
    )
    parser.add_argument("--model", default=DEEPSEEK_DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEEPSEEK_BASE_URL)
    parser.add_argument(
        "--temperature", type=float, default=None, help="server default when omitted"
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--no-json-mode", action="store_true", help="do not request response_format json_object"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_MODEL_RETRIES,
        help="how often a refused call is asked again with its rejections",
    )
    parser.add_argument("--goals", nargs="+", choices=sorted(EVALUATION_GOALS), default=list(EVALUATION_GOALS))
    parser.add_argument("--granularities", nargs="+", choices=GRANULARITIES, default=list(GRANULARITIES))
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="scenario names to run; default: every scenario of the goal",
    )
    parser.add_argument("--no-scenarios", action="store_true", help="static proposals only")
    parser.add_argument("--no-static", action="store_true", help="scenario runs only")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="trace directory; default: {}/<proposer>-<timestamp>".format(DEFAULT_EVALUATION_DIR),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="mshab_checkpoints root; used to construct Policy objects only",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    library = build_granularity_library(Path(args.checkpoint_root))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    proposer: Optional[DeepSeekProposer] = None
    if args.proposer == "deepseek":
        try:
            proposer = DeepSeekProposer(
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                json_mode=not args.no_json_mode,
            )
        except ProposerUnavailable as exc:
            sys.exit(str(exc))
        make_proposer = fixed_factory(proposer)
        description = proposer.describe()
        label = args.model
    else:
        make_proposer = scripted_factory(library)
        description = {"proposer": "ScriptedProposer", "source": "gold graphs and scenario replans"}
        label = "scripted"
    output = args.output if args.output is not None else DEFAULT_EVALUATION_DIR / "{}-{}".format(label, stamp)
    evaluation = Evaluation(
        library,
        make_proposer,
        retries=args.retries,
        output=output,
        description=description,
        log=lambda line: print(line, flush=True),
    )
    for goal_name in args.goals:
        goal = EVALUATION_GOALS[goal_name]
        # The scripted proposer only answers granularities a gold graph was
        # authored at; the real model is asked every one.
        granularities = [
            item
            for item in args.granularities
            if args.proposer == "deepseek" or item in goal.references
        ]
        evaluation.run(
            [goal_name],
            granularities,
            args.samples,
            scenarios=args.scenarios,
            static=not args.no_static,
            run_scenarios=not args.no_scenarios,
        )
    summary = evaluation.summary()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if proposer is not None:
        proposer.save_exchanges(output / "exchanges.json")
    print_summary(summary)
    print("wrote {}".format(output))
    return summary


if __name__ == "__main__":
    main()
