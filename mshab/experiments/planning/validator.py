"""Turn a proposer's answers into validated graphs, or into named rejections.

The validator sits between any proposer and the graph.  It never touches a
live graph: both layers are built fresh from the proposer's answers, so a
rejected plan leaves whatever the controller currently holds byte identical.
Rejections name the stage and, where possible, the sub-goal, because they are
what a real model will be shown when asked to try again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from mshab.experiments.planning.documents import (
    DecompositionRequest,
    DecompositionResponse,
    PlanningContext,
    SubgraphRequest,
    SubgraphResponse,
)
from mshab.experiments.planning.proposer import (
    GraphProposer,
    assemble_patch,
    subgraph_requests,
)
from mshab.skills.extension import SkillGraphPatch
from mshab.skills.graph import SkillGraph, SkillNode, SubGoalGraph
from mshab.skills.library import ContractLibrary
from mshab.skills.plan import NoViableCandidate, SkillPlan, SkillPlanner
from mshab.skills.runtime import SkillGrounder


STAGES = ("decomposition", "subgraph", "assembly", "graph", "plan")
_PROPOSER_ERRORS = (ValueError, TypeError, KeyError)


@dataclass(frozen=True)
class Rejection:
    """One reason a plan was refused, attributed to a stage and a sub-goal."""

    stage: str
    subgoal_id: Optional[str]
    message: str

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError("unknown rejection stage {!r}".format(self.stage))

    def as_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "subgoal_id": self.subgoal_id, "message": self.message}


class ProposalRejected(RuntimeError):
    """The proposer's answers did not produce a valid, plannable graph."""

    def __init__(self, rejections: Tuple[Rejection, ...]) -> None:
        self.rejections = tuple(rejections)
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
    """Both layers built from a proposer's answers, plus everything that was said."""

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

    def as_dict(self) -> Dict[str, Any]:
        """The trace record: requests and responses verbatim, plus the plan."""

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
        }


def _message(exc: BaseException) -> str:
    if exc.args and isinstance(exc.args[0], str):
        return exc.args[0]
    return str(exc)


class ProposalValidator:
    """Ask a proposer for both layers and accept only what the graph accepts."""

    def __init__(self, library: ContractLibrary, task: str) -> None:
        if not task:
            raise ValueError("validator task must be non-empty")
        self.library = library
        self.task = task
        self.grounder = SkillGrounder(library)

    def plan(
        self, proposer: GraphProposer, goal: str, context: PlanningContext
    ) -> ValidatedProposal:
        """Run call 1, then call 2 per sub-goal, then assemble and plan.

        Rejections in call 1 stop the round.  Rejections in call 2 are
        collected across every sub-goal before the round stops, so a model
        sees all of them at once.  A rejected round raises
        :class:`ProposalRejected`; nothing is returned half built.
        """

        request = DecompositionRequest(self.task, goal, context)
        try:
            decomposition = proposer.decompose(request)
            if not isinstance(decomposition, DecompositionResponse):
                raise TypeError(
                    "decompose() must return a DecompositionResponse, got {}".format(
                        type(decomposition).__name__
                    )
                )
            # The sequence must at least form a Layer-1 chain.
            SubGoalGraph.from_sequence(goal, decomposition.subgoals)
        except _PROPOSER_ERRORS as exc:
            raise ProposalRejected((Rejection("decomposition", None, _message(exc)),))

        requests = subgraph_requests(self.task, goal, context, decomposition)
        responses: List[SubgraphResponse] = []
        rejections: List[Rejection] = []
        owners: Dict[str, str] = {}
        for subgraph_request in requests:
            subgoal = subgraph_request.subgoal
            try:
                response = proposer.plan_subgraph(subgraph_request)
                if not isinstance(response, SubgraphResponse):
                    raise TypeError(
                        "plan_subgraph() must return a SubgraphResponse, got {}".format(
                            type(response).__name__
                        )
                    )
                self._check_subgraph(response, subgoal.id, owners)
            except _PROPOSER_ERRORS as exc:
                rejections.append(Rejection("subgraph", subgoal.id, _message(exc)))
                continue
            responses.append(response)
        if rejections:
            raise ProposalRejected(tuple(rejections))

        try:
            patch = assemble_patch(
                self.task,
                decomposition.subgoals,
                [response.subgraph for response in responses],
            )
        except _PROPOSER_ERRORS as exc:
            raise ProposalRejected((Rejection("assembly", None, _message(exc)),))

        subgoal_graph = SubGoalGraph(goal)
        skill_graph = SkillGraph(self.task, subgoal_graph=subgoal_graph)
        try:
            patch.apply(subgoal_graph, skill_graph, library=self.library)
            skill_graph.validate()
        except (_PROPOSER_ERRORS + (RuntimeError,)) as exc:
            raise ProposalRejected((Rejection("graph", None, _message(exc)),))

        try:
            plan = SkillPlanner(subgoal_graph, skill_graph).plan()
        except (NoViableCandidate, ValueError, KeyError) as exc:
            raise ProposalRejected((Rejection("plan", None, _message(exc)),))

        return ValidatedProposal(
            task=self.task,
            goal=goal,
            decomposition_request=request,
            decomposition_response=decomposition,
            subgraph_requests=requests,
            subgraph_responses=tuple(responses),
            patch=patch,
            subgoal_graph=subgoal_graph,
            skill_graph=skill_graph,
            plan=plan,
        )

    def _check_subgraph(
        self, response: SubgraphResponse, subgoal_id: str, owners: Dict[str, str]
    ) -> None:
        subgraph = response.subgraph
        if subgraph.subgoal_id != subgoal_id:
            raise ValueError(
                "subgraph is for sub-goal {!r}, not {!r}".format(
                    subgraph.subgoal_id, subgoal_id
                )
            )
        if subgraph.task != self.task:
            raise ValueError(
                "subgraph belongs to task {!r}, not {!r}".format(subgraph.task, self.task)
            )
        subgraph.validate()
        for node_id in subgraph.execution_order():
            owner = owners.get(node_id)
            if owner is not None:
                raise ValueError(
                    "node id {!r} is already used by sub-goal {!r}".format(node_id, owner)
                )
            self._ground(subgraph.nodes[node_id])
            owners[node_id] = subgoal_id

    def _ground(self, node: SkillNode) -> None:
        """Every node must name a library contract and bind to it."""

        self.grounder.ground(node)
