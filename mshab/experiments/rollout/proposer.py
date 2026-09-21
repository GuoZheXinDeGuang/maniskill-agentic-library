"""A rule-based pseudo model for one TidyHouse episode.

The scripted proposer of stages 3 to 5 answers from tables cut out of the gold
graphs and cannot answer a replan it was not given.  On a real episode the
objects, receptacles, and failures are not known in advance, so this proposer
plays the pseudo model's role with rules over the request instead of tables:

- ``decompose``: the transfers whose object is not yet at its destination, in
  plan order with a held object first, at the requested granularity (``coarse``
  or ``fine``, the two the gold graphs are authored at).  A transfer whose
  sub-goal failed is kept, so the next graph retries it, until it has caused
  ``give_up_after`` replans; then it is dropped.  A transfer whose object is
  in the gripper is never dropped: nothing else can be picked until it is
  placed, and there is no contract for putting an object down anywhere.
- ``plan_subgraph``: the gold builder's subgraph for that sub-goal.  A coarse
  transfer whose object is already held gets only ``navigate -> place``,
  because a pick could not be admitted with the gripper full.

It exists to validate the MS-HAB adapter and executor with a proposer that is
always right about the scene and never wrong about the vocabulary.  The real
model is :class:`~mshab.experiments.planning.deepseek.DeepSeekProposer`.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence

from mshab.experiments.granularity.higher_layers.builders import (
    GOLD_GRANULARITIES,
    TidyHouseGraphBuilder,
    chain_subgraph,
    coarse_subgoal_id,
    transfer_index,
)
from mshab.experiments.granularity.lower_layers.library import (
    EXPERIMENT_TASK,
    contract_id,
)
from mshab.experiments.planning.documents import (
    DecompositionRequest,
    DecompositionResponse,
    SubgraphRequest,
    SubgraphResponse,
)
from mshab.experiments.planning.proposer import GraphProposer
from mshab.experiments.rollout.episode import TidyHouseEpisode, Transfer


DEFAULT_GIVE_UP_AFTER = 2


class TidyHouseRuleProposer(GraphProposer):
    """Rules over the request in place of a model; see the module docstring."""

    def __init__(
        self,
        episode: TidyHouseEpisode,
        *,
        give_up_after: int = DEFAULT_GIVE_UP_AFTER,
        task: str = EXPERIMENT_TASK,
    ) -> None:
        if give_up_after < 1:
            raise ValueError("give_up_after must be at least 1")
        self.episode = episode
        self.give_up_after = give_up_after
        self.task = task
        #: transfer index -> replans its failures have caused
        self.failures: Counter = Counter()

    def describe(self) -> Dict[str, Any]:
        return {
            "proposer": type(self).__name__,
            "give_up_after": self.give_up_after,
            "plan_index": self.episode.plan_index,
        }

    # -- call 1 -----------------------------------------------------------------

    def decompose(self, request: DecompositionRequest) -> DecompositionResponse:
        self._require_task(request.task)
        granularity = request.context.granularity
        if granularity not in GOLD_GRANULARITIES:
            raise ValueError(
                "the rule proposer decomposes at {}, not {!r}".format(
                    GOLD_GRANULARITIES, granularity
                )
            )
        facts = set(request.context.facts)
        notes: List[str] = []
        failure = request.context.failure
        if failure is not None:
            index = transfer_index(failure.subgoal_id)
            if index is not None and 1 <= index <= len(self.episode.transfers):
                self.failures[index] += 1
                notes.append(
                    "transfer {} failed {} time(s) ({})".format(
                        index, self.failures[index], failure.failure_mode or "unknown"
                    )
                )
        remaining = [
            item
            for item in self.episode.transfers
            if item.delivered not in facts
            and (self.failures[item.index] < self.give_up_after or item.held in facts)
        ]
        given_up = [
            item.index
            for item in self.episode.transfers
            if item.delivered not in facts
            and self.failures[item.index] >= self.give_up_after
            and item.held not in facts
        ]
        kept_in_hand = [
            item.index
            for item in remaining
            if self.failures[item.index] >= self.give_up_after and item.held in facts
        ]
        if kept_in_hand:
            notes.append(
                "transfer(s) {} kept although given up: the object is in the gripper".format(
                    kept_in_hand
                )
            )
        if given_up:
            notes.append("gave up transfer(s) {}".format(given_up))
        if not remaining:
            raise ValueError(
                "nothing left to plan: every transfer is delivered or given up{}".format(
                    "; " + "; ".join(notes) if notes else ""
                )
            )
        # Deliver what is in the gripper first; the rest keeps the plan order.
        remaining.sort(key=lambda item: (item.held not in facts, item.index))
        patch = TidyHouseGraphBuilder(granularity).propose(
            request.goal, request.task, self._context(remaining)
        )
        rationale = "rule: {} transfer(s) not yet delivered at {} granularity".format(
            len(remaining), granularity
        )
        if notes:
            rationale += "; " + "; ".join(notes)
        return DecompositionResponse(patch.subgoals, rationale)

    # -- call 2 -----------------------------------------------------------------

    def plan_subgraph(self, request: SubgraphRequest) -> SubgraphResponse:
        self._require_task(request.task)
        subgoal = request.subgoal
        index = transfer_index(subgoal.id)
        if index is None or not 1 <= index <= len(self.episode.transfers):
            raise ValueError(
                "sub-goal {!r} is not one the rule proposer decomposes into".format(subgoal.id)
            )
        transfer = self.episode.transfers[index - 1]
        granularity = "coarse" if subgoal.id == coarse_subgoal_id(index) else "fine"
        patch = TidyHouseGraphBuilder(granularity).propose(
            request.goal, request.task, self._context([transfer])
        )
        declared = {item.id: item.predicate for item in patch.subgoals}
        if declared.get(subgoal.id) != subgoal.predicate:
            raise ValueError(
                "sub-goal {!r} should read {!r}, not {!r}".format(
                    subgoal.id, declared.get(subgoal.id), subgoal.predicate
                )
            )
        subgraph = next(
            item for item in patch.skill_subgraphs if item.subgoal_id == subgoal.id
        )
        rationale = "rule: the gold {} subgraph of transfer {}".format(granularity, index)
        if granularity == "coarse" and transfer.held in request.facts:
            nodes = [subgraph.nodes[node_id] for node_id in subgraph.execution_order()]
            kept = [
                node
                for node in nodes
                if node.contract_id != contract_id("pick")
                and not (
                    node.contract_id == contract_id("navigate")
                    and node.arguments.get("target") == transfer.object
                )
            ]
            subgraph = chain_subgraph(subgoal.id, request.task, kept)
            rationale = "rule: {} is already held, so navigate to {} and place".format(
                transfer.object, transfer.destination
            )
        return SubgraphResponse(subgraph, rationale)

    # -- helpers ----------------------------------------------------------------

    def _require_task(self, task: str) -> None:
        if task != self.task:
            raise ValueError(
                "the rule proposer answers for task {!r}, not {!r}".format(self.task, task)
            )

    @staticmethod
    def _context(transfers: Sequence[Transfer]) -> Dict[str, Any]:
        return {
            "transfers": tuple(item.pair for item in transfers),
            "transfer_indices": tuple(item.index for item in transfers),
        }
