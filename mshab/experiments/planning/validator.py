"""Turn a proposer's answers into validated graphs, or into named rejections.

The validator sits between any proposer and the graph.  It never touches a
live graph: both layers are built fresh from the proposer's answers, so a
rejected plan leaves whatever the controller currently holds byte identical.
Rejections name the stage and, where possible, the sub-goal, because they are
what a real model is shown when asked to try again.

A *round* is one pass over the two calls.  With ``retries`` the validator asks
again after a rejected round.  A rejection that names a sub-goal repeats that
sub-goal's subgraph call, with the rejections attached to the request, and
keeps every answer that was not refused; a rejection that names no sub-goal
(the decomposition itself, or the assembled graph) repeats the round from the
decomposition call.  Every round is recorded, so the trace holds each request,
answer, and rejection of the conversation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from mshab.experiments.planning.documents import (
    STAGES,
    DecompositionRequest,
    DecompositionResponse,
    PlanningContext,
    Rejection,
    SubgraphRequest,
    SubgraphResponse,
)
from mshab.experiments.planning.proposer import (
    GraphProposer,
    assemble_patch,
    subgraph_requests,
)
from mshab.skills.extension import SkillGraphPatch
from mshab.skills.graph import SkillGraph, SubGoal, SubGoalGraph
from mshab.skills.library import ContractLibrary
from mshab.skills.plan import NoViableCandidate, SkillPlan, SkillPlanner
from mshab.skills.runtime import SkillGrounder


__all__ = [
    "DEFAULT_RETRIES",
    "STAGES",
    "ProposalRejected",
    "ProposalRound",
    "ProposalValidator",
    "Rejection",
    "RoundCall",
    "ValidatedProposal",
    "error_message",
]

_PROPOSER_ERRORS = (ValueError, TypeError, KeyError)
DEFAULT_RETRIES = 0

SubgraphAnswer = Tuple[SubgraphRequest, SubgraphResponse]


@dataclass(frozen=True)
class RoundCall:
    """One proposer call: the request as it was asked, and the answer or the error.

    ``response`` is what the proposer returned when it returned a document at
    all; ``error`` is set when the call raised or the document was refused,
    so a parsed but rejected answer carries both.
    """

    call: str
    request: Union[DecompositionRequest, SubgraphRequest]
    response: Optional[Union[DecompositionResponse, SubgraphResponse]] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.call not in ("decompose", "plan_subgraph"):
            raise ValueError("unknown proposer call {!r}".format(self.call))

    @property
    def subgoal_id(self) -> Optional[str]:
        if isinstance(self.request, SubgraphRequest):
            return self.request.subgoal.id
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "call": self.call,
            "subgoal_id": self.subgoal_id,
            "request": self.request.as_dict(),
            "response": None if self.response is None else self.response.as_dict(),
            "error": self.error,
        }


@dataclass(frozen=True)
class ProposalRound:
    """One pass over the two calls: what was asked, what was kept, what was refused."""

    index: int
    calls: Tuple[RoundCall, ...]
    decomposition_reused: bool = False
    reused_subgoals: Tuple[str, ...] = ()
    rejections: Tuple[Rejection, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "calls", tuple(self.calls))
        object.__setattr__(self, "reused_subgoals", tuple(self.reused_subgoals))
        object.__setattr__(self, "rejections", tuple(self.rejections))

    @property
    def accepted(self) -> bool:
        return not self.rejections

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "decomposition_reused": self.decomposition_reused,
            "reused_subgoals": list(self.reused_subgoals),
            "calls": [call.as_dict() for call in self.calls],
            "rejections": [item.as_dict() for item in self.rejections],
        }


class ProposalRejected(RuntimeError):
    """The proposer's answers did not produce a valid, plannable graph.

    ``rejections`` are those of the last round; ``rounds`` is every round
    that was tried, for the trace.
    """

    def __init__(
        self, rejections: Sequence[Rejection], rounds: Sequence[ProposalRound] = ()
    ) -> None:
        self.rejections = tuple(rejections)
        self.rounds = tuple(rounds)
        super().__init__(
            "plan rejected: "
            + "; ".join(
                "[{}{}] {}".format(
                    item.stage,
                    "" if item.subgoal_id is None else " " + item.subgoal_id,
                    item.message,
                )
                for item in self.rejections
            )
        )

    def as_dict(self) -> List[Dict[str, Any]]:
        return [item.as_dict() for item in self.rejections]


@dataclass(frozen=True)
class ValidatedProposal:
    """Both layers built from a proposer's answers, plus everything that was said.

    The request and response fields are the accepted answers; when a round was
    retried they may come from different rounds.  ``rounds`` is the complete
    conversation, rejected rounds included.
    """

    task: str
    goal: str
    decomposition_request: DecompositionRequest
    decomposition_response: DecompositionResponse
    subgraph_requests: Tuple[SubgraphRequest, ...]
    subgraph_responses: Tuple[SubgraphResponse, ...]
    patch: SkillGraphPatch
    subgoal_graph: SubGoalGraph
    skill_graph: SkillGraph
    plan: SkillPlan
    rounds: Tuple[ProposalRound, ...] = ()

    @property
    def retries(self) -> int:
        """Rounds that were refused before this proposal was accepted."""

        return max(len(self.rounds) - 1, 0)

    def as_dict(self) -> Dict[str, Any]:
        """The trace record: requests and responses verbatim, the plan, every round."""

        return {
            "task": self.task,
            "goal": self.goal,
            "decomposition": {
                "request": self.decomposition_request.as_dict(),
                "response": self.decomposition_response.as_dict(),
            },
            "subgraphs": [
                {"request": request.as_dict(), "response": response.as_dict()}
                for request, response in zip(self.subgraph_requests, self.subgraph_responses)
            ],
            "nominal_plan": self.plan.as_dict(),
            "rounds": [item.as_dict() for item in self.rounds],
        }


def error_message(exc: BaseException) -> str:
    """The exception's own message, without the type name: what a trace records."""

    if exc.args and isinstance(exc.args[0], str):
        return exc.args[0]
    return str(exc)


@dataclass(frozen=True)
class _Carry:
    """The answers a rejected round did not refuse, kept for the next round."""

    decomposition_request: DecompositionRequest
    decomposition: DecompositionResponse
    subgraphs: Mapping[str, SubgraphAnswer]


class ProposalValidator:
    """Ask a proposer for both layers and accept only what the graph accepts."""

    def __init__(
        self, library: ContractLibrary, task: str, *, retries: int = DEFAULT_RETRIES
    ) -> None:
        if not task:
            raise ValueError("validator task must be non-empty")
        if retries < 0:
            raise ValueError("retries cannot be negative")
        self.library = library
        self.task = task
        self.retries = retries
        self.grounder = SkillGrounder(library)

    def plan(
        self, proposer: GraphProposer, goal: str, context: PlanningContext
    ) -> ValidatedProposal:
        """Run call 1, then call 2 per sub-goal, then assemble and plan.

        Rejections in call 1 stop the round.  Rejections in call 2 are
        collected across every sub-goal before the round stops, so a model
        sees all of them at once.  With ``retries`` a rejected round is
        followed by another that repeats only the refused calls, or the whole
        round when the refusal named no sub-goal.  Once the retries are used
        up :class:`ProposalRejected` is raised; nothing is returned half built.
        """

        rounds: List[ProposalRound] = []
        carry: Optional[_Carry] = None
        for index in range(self.retries + 1):
            previous = rounds[-1].rejections if rounds else ()
            record, proposal, carry = self._round(
                proposer, goal, context, index, previous, carry
            )
            rounds.append(record)
            if proposal is not None:
                return replace(proposal, rounds=tuple(rounds))
        raise ProposalRejected(rounds[-1].rejections, rounds)

    # -- one round --------------------------------------------------------------

    def _round(
        self,
        proposer: GraphProposer,
        goal: str,
        context: PlanningContext,
        index: int,
        previous: Tuple[Rejection, ...],
        carry: Optional[_Carry],
    ) -> Tuple[ProposalRound, Optional[ValidatedProposal], Optional[_Carry]]:
        calls: List[RoundCall] = []
        if carry is None:
            request = DecompositionRequest(self.task, goal, context, rejections=previous)
            decomposition, error = self._decompose(proposer, request)
            calls.append(RoundCall("decompose", request, decomposition, error))
            if error is not None or decomposition is None:
                rejections = (Rejection("decomposition", None, error or "no answer"),)
                return ProposalRound(index, calls, rejections=rejections), None, None
            kept: Mapping[str, SubgraphAnswer] = {}
            reused = False
        else:
            request = carry.decomposition_request
            decomposition = carry.decomposition
            kept = carry.subgraphs
            reused = True

        # Node ids of the kept answers are taken first, so a new answer that
        # collides with one of them is the answer that is refused.
        owners: Dict[str, str] = {
            node_id: subgoal_id
            for subgoal_id, (_, response) in kept.items()
            for node_id in response.subgraph.nodes
        }
        answers: Dict[str, SubgraphAnswer] = {}
        reused_subgoals: List[str] = []
        rejections: List[Rejection] = []
        for subgraph_request in subgraph_requests(self.task, goal, context, decomposition):
            subgoal = subgraph_request.subgoal
            if subgoal.id in kept:
                answers[subgoal.id] = kept[subgoal.id]
                reused_subgoals.append(subgoal.id)
                continue
            if reused:
                subgraph_request = subgraph_request.with_rejections(
                    [item for item in previous if item.subgoal_id == subgoal.id]
                )
            response, error = self._plan_subgraph(proposer, subgraph_request, owners)
            calls.append(RoundCall("plan_subgraph", subgraph_request, response, error))
            if error is not None or response is None:
                rejections.append(Rejection("subgraph", subgoal.id, error or "no answer"))
                continue
            answers[subgoal.id] = (subgraph_request, response)
        if rejections:
            record = ProposalRound(index, calls, reused, reused_subgoals, rejections)
            return record, None, _Carry(request, decomposition, dict(answers))

        ordered = [answers[subgoal.id] for subgoal in decomposition.subgoals]
        responses = tuple(response for _, response in ordered)
        rejections, built = self._build(goal, decomposition, responses)
        if rejections:
            record = ProposalRound(index, calls, reused, reused_subgoals, rejections)
            if all(item.subgoal_id is not None for item in rejections):
                refused = {item.subgoal_id for item in rejections}
                remaining = {
                    subgoal_id: answer
                    for subgoal_id, answer in answers.items()
                    if subgoal_id not in refused
                }
                return record, None, _Carry(request, decomposition, remaining)
            return record, None, None

        patch, subgoal_graph, skill_graph, plan = built
        proposal = ValidatedProposal(
            task=self.task,
            goal=goal,
            decomposition_request=request,
            decomposition_response=decomposition,
            subgraph_requests=tuple(item for item, _ in ordered),
            subgraph_responses=responses,
            patch=patch,
            subgoal_graph=subgoal_graph,
            skill_graph=skill_graph,
            plan=plan,
        )
        return ProposalRound(index, calls, reused, reused_subgoals), proposal, None

    # -- the calls ----------------------------------------------------------------

    def _decompose(
        self, proposer: GraphProposer, request: DecompositionRequest
    ) -> Tuple[Optional[DecompositionResponse], Optional[str]]:
        response: Optional[DecompositionResponse] = None
        try:
            answer = proposer.decompose(request)
            if not isinstance(answer, DecompositionResponse):
                raise TypeError(
                    "decompose() must return a DecompositionResponse, got {}".format(
                        type(answer).__name__
                    )
                )
            response = answer
            # The sequence must at least form a Layer-1 chain.
            SubGoalGraph.from_sequence(request.goal, response.subgoals)
        except _PROPOSER_ERRORS as exc:
            return response, error_message(exc)
        return response, None

    def _plan_subgraph(
        self,
        proposer: GraphProposer,
        request: SubgraphRequest,
        owners: Dict[str, str],
    ) -> Tuple[Optional[SubgraphResponse], Optional[str]]:
        response: Optional[SubgraphResponse] = None
        try:
            answer = proposer.plan_subgraph(request)
            if not isinstance(answer, SubgraphResponse):
                raise TypeError(
                    "plan_subgraph() must return a SubgraphResponse, got {}".format(
                        type(answer).__name__
                    )
                )
            response = answer
            self._check_subgraph(response, request.subgoal, owners)
        except _PROPOSER_ERRORS as exc:
            return response, error_message(exc)
        return response, None

    def _check_subgraph(
        self, response: SubgraphResponse, subgoal: SubGoal, owners: Dict[str, str]
    ) -> None:
        """Owner, task, structure, grounding, node-id uniqueness, and the predicate.

        ``owners`` is only extended once every check passed, so a refused
        answer never reserves node ids.
        """

        subgraph = response.subgraph
        if subgraph.subgoal_id != subgoal.id:
            raise ValueError(
                "subgraph is for sub-goal {!r}, not {!r}".format(
                    subgraph.subgoal_id, subgoal.id
                )
            )
        if subgraph.task != self.task:
            raise ValueError(
                "subgraph belongs to task {!r}, not {!r}".format(subgraph.task, self.task)
            )
        subgraph.validate()
        grounded = {}
        for node_id in subgraph.execution_order():
            owner = owners.get(node_id)
            if owner is not None:
                raise ValueError(
                    "node id {!r} is already used by sub-goal {!r}".format(node_id, owner)
                )
            # Every node must name a library contract and bind to it.
            grounded[node_id] = self.grounder.ground(subgraph.nodes[node_id])
        for achiever in subgraph.achievers:
            effects = grounded[achiever.id].effects
            if subgoal.predicate not in effects:
                raise ValueError(
                    "achiever {!r} does not establish the sub-goal predicate {!r}; "
                    "its grounded effects are {}".format(
                        achiever.id, subgoal.predicate, list(effects)
                    )
                )
        for node_id in grounded:
            owners[node_id] = subgoal.id

    # -- assembly, graph, plan ------------------------------------------------------

    def _build(
        self,
        goal: str,
        decomposition: DecompositionResponse,
        responses: Sequence[SubgraphResponse],
    ) -> Tuple[Tuple[Rejection, ...], Optional[Tuple[SkillGraphPatch, SubGoalGraph, SkillGraph, SkillPlan]]]:
        try:
            patch = assemble_patch(
                self.task,
                decomposition.subgoals,
                [response.subgraph for response in responses],
            )
        except _PROPOSER_ERRORS as exc:
            return (Rejection("assembly", None, error_message(exc)),), None

        subgoal_graph = SubGoalGraph(goal)
        skill_graph = SkillGraph(self.task, subgoal_graph=subgoal_graph)
        try:
            patch.apply(subgoal_graph, skill_graph, library=self.library)
            skill_graph.validate()
        except (_PROPOSER_ERRORS + (RuntimeError,)) as exc:
            return (Rejection("graph", None, error_message(exc)),), None

        # A sub-goal whose achievers have no primary cannot be planned; that
        # is the sub-goal's fault and is attributed to it, so a retry repeats
        # only its call.
        rejections = []
        for subgoal in decomposition.subgoals:
            try:
                SkillPlanner.fallback_chain(skill_graph.subgraph_for_subgoal(subgoal.id))
            except ValueError as exc:
                rejections.append(Rejection("plan", subgoal.id, error_message(exc)))
        if rejections:
            return tuple(rejections), None
        try:
            plan = SkillPlanner(subgoal_graph, skill_graph).plan()
        except (NoViableCandidate, ValueError, KeyError) as exc:
            return (Rejection("plan", None, error_message(exc)),), None
        return (), (patch, subgoal_graph, skill_graph, plan)
