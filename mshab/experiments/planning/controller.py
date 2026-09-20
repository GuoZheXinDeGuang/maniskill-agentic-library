"""The online loop: propose, decide one node, execute, observe, re-decide.

``TaskController.run`` is the milestone the skill-library guide calls
``execute -> observe -> re-decide``.  It asks a proposer for both upper
layers, then repeats: absorb the facts, let ``SkillPlanner`` pick one node,
run it through ``SkillRuntime``, record the evidence.  A node that fails is
retried within its attempt budget, then marked failed; when the planner has
no candidate left the sub-goal has failed and the proposer is asked again
with the failure, the history, and the current facts.

Sub-goal achievement follows the guide: a sub-goal counts as achieved when
its predicate holds at the moment it becomes the next one in line, and that
stays recorded even if a later step retracts the predicate.  The skill nodes
of a sub-goal achieved that way are skipped.  After a replan achievement is
derived from the current facts again; the earlier history stays in the trace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from mshab.experiments.planning.documents import (
    Failure,
    History,
    PlanningContext,
    entity_descriptions,
)
from mshab.experiments.planning.metrics import graph_summary
from mshab.experiments.planning.proposer import GraphProposer
from mshab.experiments.planning.validator import (
    ProposalRejected,
    ProposalValidator,
    Rejection,
    ValidatedProposal,
    error_message,
)
from mshab.skills.environment import EnvironmentAdapter
from mshab.skills.library import ContractLibrary
from mshab.skills.plan import NoViableCandidate, SkillPlanner
from mshab.skills.runtime import ContractViolation, PolicyExecutor, SkillRuntime


OUTCOMES = ("success", "failed", "admission_failed", "skipped")
STATUSES = ("success", "goal_not_reached", "proposal_rejected", "replans_exhausted")
DEFAULT_ATTEMPTS_PER_NODE = 2
DEFAULT_MAX_REPLANS = 2


@dataclass(frozen=True)
class Decision:
    """One node the controller acted on: an execution attempt or a skip."""

    step: int
    node_id: str
    subgoal_id: str
    contract_id: str
    attempt: int
    outcome: str
    policy_id: Optional[str] = None
    failure_mode: Optional[str] = None
    missing_effects: Tuple[str, ...] = ()
    missing_preconditions: Tuple[str, ...] = ()
    facts_added: Tuple[str, ...] = ()
    facts_removed: Tuple[str, ...] = ()
    redundant: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError("unknown decision outcome {!r}".format(self.outcome))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "node_id": self.node_id,
            "subgoal_id": self.subgoal_id,
            "contract_id": self.contract_id,
            "attempt": self.attempt,
            "outcome": self.outcome,
            "policy_id": self.policy_id,
            "failure_mode": self.failure_mode,
            "missing_effects": list(self.missing_effects),
            "missing_preconditions": list(self.missing_preconditions),
            "facts_added": list(self.facts_added),
            "facts_removed": list(self.facts_removed),
            "redundant": self.redundant,
            "note": self.note,
        }


@dataclass(frozen=True)
class Replan:
    """One Layer-1 replan: why it was asked for and what came back."""

    after_step: int
    failure: Failure
    accepted: bool
    rejections: Tuple[Rejection, ...] = ()
    subgoals: int = 0
    span: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "after_step": self.after_step,
            "failure": self.failure.as_dict(),
            "accepted": self.accepted,
            "rejections": [item.as_dict() for item in self.rejections],
            "subgoals": self.subgoals,
            "span": self.span,
        }


@dataclass(frozen=True)
class RunResult:
    """Everything one run said and did, plus the metrics computed from it."""

    task: str
    goal: str
    granularity: str
    status: str
    success: bool
    proposals: Tuple[Dict[str, Any], ...]
    decisions: Tuple[Decision, ...]
    replans: Tuple[Replan, ...]
    initial_facts: Tuple[str, ...]
    final_facts: Tuple[str, ...]
    goal_facts: Tuple[str, ...]
    achieved_predicates: Tuple[str, ...]
    metrics: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "goal": self.goal,
            "granularity": self.granularity,
            "status": self.status,
            "success": self.success,
            "metrics": dict(self.metrics),
            "proposals": [dict(item) for item in self.proposals],
            "decisions": [item.as_dict() for item in self.decisions],
            "replans": [item.as_dict() for item in self.replans],
            "initial_facts": list(self.initial_facts),
            "final_facts": list(self.final_facts),
            "goal_facts": list(self.goal_facts),
            "achieved_predicates": list(self.achieved_predicates),
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")


class _ActiveGraph:
    """The graph currently being executed and the controller's state on it."""

    def __init__(self, proposal: ValidatedProposal) -> None:
        self.proposal = proposal
        self.subgoal_graph = proposal.subgoal_graph
        self.skill_graph = proposal.skill_graph
        self.planner = SkillPlanner(proposal.subgoal_graph, proposal.skill_graph)
        self.completed: Set[str] = set()
        self.executed: List[str] = []
        self.failed: Set[str] = set()
        self.achieved: List[str] = []
        self.attempts: Dict[str, int] = {}


class TaskController:
    """Run one goal to completion, replanning through the proposer when needed."""

    def __init__(
        self,
        library: ContractLibrary,
        task: str,
        *,
        attempts_per_node: int = DEFAULT_ATTEMPTS_PER_NODE,
        max_replans: int = DEFAULT_MAX_REPLANS,
        policy_id: Optional[str] = None,
        validator: Optional[ProposalValidator] = None,
    ) -> None:
        if attempts_per_node < 1:
            raise ValueError("attempts_per_node must be at least 1")
        if max_replans < 0:
            raise ValueError("max_replans cannot be negative")
        self.library = library
        self.task = task
        self.attempts_per_node = attempts_per_node
        self.max_replans = max_replans
        self.policy_id = policy_id
        self.validator = validator or ProposalValidator(library, task)

    def run(
        self,
        goal: str,
        proposer: GraphProposer,
        environment: EnvironmentAdapter,
        executor: PolicyExecutor,
        *,
        granularity: str = "free",
        goal_facts: Sequence[str] = (),
    ) -> RunResult:
        """Reset the environment, plan, and execute until done or stuck.

        ``goal_facts`` is the success criterion, judged on the final facts
        independently of how the proposer decomposed the goal.  Without it
        success means every sub-goal of the final graph was achieved.
        """

        goal_facts = tuple(goal_facts)
        snapshot = environment.reset()
        initial_facts = tuple(sorted(snapshot.facts))
        runtime = SkillRuntime(self.library, environment)
        context = PlanningContext.initial(
            self.library,
            self.task,
            entities=entity_descriptions(environment.description),
            facts=snapshot.facts,
            granularity=granularity,
        )
        proposals: List[Dict[str, Any]] = []
        decisions: List[Decision] = []
        replans: List[Replan] = []
        achieved_predicates: List[str] = []
        status: Optional[str] = None
        initial_summary: Dict[str, Any] = {}
        step = 0

        proposal, _ = self._propose(proposer, goal, context, proposals)
        if proposal is None:
            status = "proposal_rejected"
            active = None
        else:
            active = _ActiveGraph(proposal)
            initial_summary = graph_summary(proposal.skill_graph)
        last_failure: Optional[Failure] = None

        while active is not None:
            facts = environment.snapshot().facts
            step = self._absorb(active, facts, decisions, step, achieved_predicates)
            try:
                node = active.planner.decide(active.completed, active.failed)
            except NoViableCandidate as exc:
                if len([item for item in replans if item.accepted]) >= self.max_replans:
                    status = "replans_exhausted"
                    break
                if last_failure is None:
                    pending = [
                        subgoal_id
                        for subgoal_id in active.subgoal_graph.execution_order()
                        if subgoal_id not in active.achieved
                    ]
                    last_failure = Failure(
                        subgoal_id=pending[0] if pending else active.subgoal_graph.execution_order()[-1],
                        failure_mode="no_viable_candidate",
                    )
                history = History(
                    tuple(active.achieved),
                    tuple(active.executed),
                    tuple(sorted(active.failed)),
                )
                context = context.replan(last_failure, history, facts)
                proposal, rejections = self._propose(proposer, goal, context, proposals)
                if proposal is None:
                    replans.append(Replan(step, last_failure, False, rejections))
                    status = "proposal_rejected"
                    break
                span = sum(
                    1
                    for subgoal_id in proposal.subgoal_graph.execution_order()
                    if not proposal.subgoal_graph.achieved(subgoal_id, facts)
                )
                replans.append(
                    Replan(
                        step,
                        last_failure,
                        True,
                        (),
                        len(proposal.subgoal_graph.subgoals),
                        span,
                    )
                )
                active = _ActiveGraph(proposal)
                last_failure = None
                continue
            if node is None:
                break

            step += 1
            attempt = active.attempts.get(node.id, 0) + 1
            active.attempts[node.id] = attempt
            owner = active.skill_graph.owner_of(node.id)
            grounded = runtime.grounder.ground(node)
            before = facts
            redundant = set(grounded.effects) <= before
            try:
                result = runtime.execute_node(
                    active.skill_graph, node.id, executor, self.policy_id
                )
            except ContractViolation as exc:
                after = environment.snapshot().facts
                admission = exc.phase == "admission"
                missing = (
                    exc.missing_preconditions + exc.missing_invariants if admission else ()
                )
                decision = Decision(
                    step=step,
                    node_id=node.id,
                    subgoal_id=owner,
                    contract_id=node.contract_id,
                    attempt=attempt,
                    outcome="admission_failed" if admission else "failed",
                    failure_mode="precondition_violated" if admission else "invariant_violated",
                    missing_preconditions=missing,
                    facts_added=tuple(sorted(after - before)),
                    facts_removed=tuple(sorted(before - after)),
                    redundant=redundant,
                    note=error_message(exc),
                )
            else:
                after = result.after.facts
                missing_effects = tuple(result.missing_effects) + tuple(
                    predicate
                    for predicate in result.missing_verification
                    if predicate not in result.missing_effects
                )
                decision = Decision(
                    step=step,
                    node_id=node.id,
                    subgoal_id=owner,
                    contract_id=node.contract_id,
                    attempt=attempt,
                    outcome="success" if result.success else "failed",
                    policy_id=result.policy_id,
                    failure_mode=result.failure_mode,
                    missing_effects=missing_effects,
                    facts_added=tuple(sorted(after - before)),
                    facts_removed=tuple(sorted(before - after)),
                    redundant=redundant,
                    note=", ".join(
                        "unretracted {}".format(item) for item in result.unretracted_deletes
                    ),
                )
            decisions.append(decision)
            if decision.outcome == "success":
                active.completed.add(node.id)
                active.executed.append(node.id)
            elif attempt >= self.attempts_per_node:
                active.failed.add(node.id)
                last_failure = Failure(
                    subgoal_id=owner,
                    node_id=node.id,
                    failure_mode=decision.failure_mode,
                    missing_effects=decision.missing_effects,
                    missing_preconditions=decision.missing_preconditions,
                )

        final_facts = environment.snapshot().facts
        if goal_facts:
            success = set(goal_facts) <= final_facts
        elif status is None and active is not None:
            success = set(active.achieved) == set(active.subgoal_graph.subgoals)
        else:
            success = False
        if status is None:
            status = "success" if success else "goal_not_reached"
        metrics = run_metrics(
            initial_summary,
            decisions,
            replans,
            success,
            achieved_predicates,
            goal_facts,
            final_facts,
            proposals,
        )
        return RunResult(
            task=self.task,
            goal=goal,
            granularity=granularity,
            status=status,
            success=success,
            proposals=tuple(proposals),
            decisions=tuple(decisions),
            replans=tuple(replans),
            initial_facts=initial_facts,
            final_facts=tuple(sorted(final_facts)),
            goal_facts=goal_facts,
            achieved_predicates=tuple(achieved_predicates),
            metrics=metrics,
        )

    # -- helpers ------------------------------------------------------------

    def _propose(
        self,
        proposer: GraphProposer,
        goal: str,
        context: PlanningContext,
        proposals: List[Dict[str, Any]],
    ) -> Tuple[Optional[ValidatedProposal], Tuple[Rejection, ...]]:
        try:
            proposal = self.validator.plan(proposer, goal, context)
        except ProposalRejected as exc:
            proposals.append(
                {
                    "accepted": False,
                    "attempt": context.attempt,
                    "context": context.as_dict(),
                    "rejections": exc.as_dict(),
                    "rounds": [item.as_dict() for item in exc.rounds],
                }
            )
            return None, exc.rejections
        record = proposal.as_dict()
        record["accepted"] = True
        record["attempt"] = context.attempt
        record["summary"] = graph_summary(proposal.skill_graph)
        proposals.append(record)
        return proposal, ()

    @staticmethod
    def _absorb(
        active: _ActiveGraph,
        facts: frozenset,
        decisions: List[Decision],
        step: int,
        achieved_predicates: List[str],
    ) -> int:
        """Mark the sub-goals next in line whose predicates already hold.

        Only a sub-goal whose predecessors are all achieved is examined, so a
        transient predicate that happens to hold early is not mistaken for a
        later sub-goal being done.  The nodes of a sub-goal achieved this way
        are recorded as skipped and count as completed for the planner.
        """

        changed = True
        while changed:
            changed = False
            for subgoal_id in active.subgoal_graph.execution_order():
                if subgoal_id in active.achieved:
                    continue
                if not active.subgoal_graph.predecessors(subgoal_id) <= set(active.achieved):
                    continue
                if not active.subgoal_graph.achieved(subgoal_id, facts):
                    continue
                active.achieved.append(subgoal_id)
                predicate = active.subgoal_graph.subgoals[subgoal_id].predicate
                if predicate not in achieved_predicates:
                    achieved_predicates.append(predicate)
                subgraph = active.skill_graph.subgraph_for_subgoal(subgoal_id)
                for node_id in subgraph.execution_order():
                    if node_id in active.completed or node_id in active.failed:
                        continue
                    active.completed.add(node_id)
                    step += 1
                    decisions.append(
                        Decision(
                            step=step,
                            node_id=node_id,
                            subgoal_id=subgoal_id,
                            contract_id=subgraph.nodes[node_id].contract_id,
                            attempt=0,
                            outcome="skipped",
                            note="sub-goal already achieved: {}".format(predicate),
                        )
                    )
                changed = True
        return step


def run_metrics(
    initial_summary: Dict[str, Any],
    decisions: Sequence[Decision],
    replans: Sequence[Replan],
    success: bool,
    achieved_predicates: Sequence[str],
    goal_facts: Sequence[str],
    final_facts: frozenset,
    proposals: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """The per-run measurements the granularity experiment compares.

    ``proposals`` are the controller's proposal records; their ``rounds``
    give the cost of talking to the proposer: rejected rounds that were
    retried, and proposer calls made in total.
    """

    executions = [item for item in decisions if item.outcome != "skipped"]
    successes = [item for item in executions if item.outcome == "success"]
    failures = [item for item in executions if item.outcome != "success"]
    accepted = [item for item in replans if item.accepted]
    rounds = [record.get("rounds", ()) for record in proposals]
    return {
        "subgoals": initial_summary.get("subgoals", 0),
        "subgraphs": initial_summary.get("subgraphs", 0),
        "nodes": initial_summary.get("nodes", 0),
        "mean_nodes_per_subgraph": initial_summary.get("mean_nodes_per_subgraph", 0.0),
        "success": success,
        "goal_facts": len(goal_facts),
        "goal_facts_achieved": sum(fact in final_facts for fact in goal_facts),
        "achieved_subgoals": len(achieved_predicates),
        "node_executions": len(executions),
        "successful_executions": len(successes),
        "failed_executions": len(failures),
        "admission_failures": sum(item.outcome == "admission_failed" for item in executions),
        "retries": sum(item.attempt > 1 for item in executions),
        "skipped_nodes": sum(item.outcome == "skipped" for item in decisions),
        "redundant_executions": sum(item.redundant for item in successes),
        "replans": len(accepted),
        "replanning_span": sum(item.span for item in accepted),
        "recovery_success": bool(success and (failures or accepted)),
        "proposal_retries": sum(max(len(items) - 1, 0) for items in rounds),
        "proposer_calls": sum(len(item["calls"]) for items in rounds for item in items),
    }
