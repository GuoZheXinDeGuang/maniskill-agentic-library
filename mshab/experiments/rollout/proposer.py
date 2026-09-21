"""A rule-based pseudo model for one SetTable episode.

The scripted proposer of stages 3 to 5 answers from tables cut out of the gold
graphs and cannot answer a replan it was not given.  On a real episode the
objects, storages, and failures are not known in advance, so this proposer
plays the pseudo model's role with rules over the request instead of tables:

- ``decompose``: one segment per object not yet on the table, in plan order
  with a held object first, at the requested granularity (``coarse`` or
  ``fine``, the two the gold graphs are authored at).  The steps a segment
  still needs are read off the facts: the storage is opened only while it is
  closed, the object picked only while it is not held, and the storage closed
  again whenever it was or will be open.  A segment whose sub-goal failed is
  kept, so the next graph retries it, until it has caused ``give_up_after``
  replans; then its object is given up and only the closing of its storage
  remains.  A segment whose object is in the gripper is never given up:
  nothing else can be picked until it is placed, and there is no contract for
  putting an object down anywhere.
- ``plan_subgraph``: the gold builder's subgraph for that sub-goal, from the
  same steps the facts leave open.

Unlike a real model the rules know the true scene, which storage holds which
object included, so they never open the wrong one.  They exist to validate
the MS-HAB adapter and executor with a proposer that is always right about
the scene and never wrong about the vocabulary.  The real model is
:class:`~mshab.experiments.planning.deepseek.DeepSeekProposer`.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from mshab.experiments.granularity.higher_layers.builders import (
    COARSE_KINDS,
    GOLD_GRANULARITIES,
    SetTableGraphBuilder,
    segment_subgoal,
)
from mshab.experiments.granularity.lower_layers.library import EXPERIMENT_TASK
from mshab.experiments.planning.documents import (
    DecompositionRequest,
    DecompositionResponse,
    SubgraphRequest,
    SubgraphResponse,
)
from mshab.experiments.planning.proposer import GraphProposer
from mshab.experiments.rollout.episode import Segment, SetTableEpisode


DEFAULT_GIVE_UP_AFTER = 2


def remaining_steps(
    segment: Segment, facts: Iterable[str], *, given_up: bool = False
) -> Tuple[str, ...]:
    """The steps of one segment the facts leave open, in the official order.

    Nothing is fetched for a given-up object, but its storage is still
    closed when it stands open.
    """

    facts = set(facts)
    steps: List[str] = []
    if segment.delivered not in facts and not given_up:
        if segment.held not in facts:
            if segment.source_open not in facts:
                steps.append("open")
            steps.append("pick")
        steps.append("place")
    if "open" in steps or segment.source_closed not in facts:
        steps.append("close")
    return tuple(steps)


class SetTableRuleProposer(GraphProposer):
    """Rules over the request in place of a model; see the module docstring."""

    def __init__(
        self,
        episode: SetTableEpisode,
        *,
        give_up_after: int = DEFAULT_GIVE_UP_AFTER,
        task: str = EXPERIMENT_TASK,
    ) -> None:
        if give_up_after < 1:
            raise ValueError("give_up_after must be at least 1")
        self.episode = episode
        self.give_up_after = give_up_after
        self.task = task
        #: segment label -> replans its failures have caused
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
            parsed = segment_subgoal(failure.subgoal_id, self.episode.labels)
            if parsed is not None:
                label = parsed[0]
                self.failures[label] += 1
                notes.append(
                    "segment {} failed {} time(s) ({})".format(
                        label, self.failures[label], failure.failure_mode or "unknown"
                    )
                )
        remaining = self._remaining(facts, notes)
        if not remaining:
            raise ValueError(
                "nothing left to plan: every object is on the table or given up and "
                "every storage is closed{}".format("; " + "; ".join(notes) if notes else "")
            )
        patch = SetTableGraphBuilder(granularity).propose(
            request.goal, request.task, self._context(remaining)
        )
        rationale = "rule: {} segment(s) with work left at {} granularity".format(
            len(remaining), granularity
        )
        if notes:
            rationale += "; " + "; ".join(notes)
        return DecompositionResponse(patch.subgoals, rationale)

    # -- call 2 -----------------------------------------------------------------

    def plan_subgraph(self, request: SubgraphRequest) -> SubgraphResponse:
        self._require_task(request.task)
        subgoal = request.subgoal
        parsed = segment_subgoal(subgoal.id, self.episode.labels)
        if parsed is None:
            raise ValueError(
                "sub-goal {!r} is not one the rule proposer decomposes into".format(subgoal.id)
            )
        label, kind = parsed
        segment = self.episode.segment_by_label(label)
        granularity = "coarse" if kind in COARSE_KINDS else "fine"
        facts = set(request.facts)
        steps = remaining_steps(
            segment, facts, given_up=self._given_up(segment, facts)
        )
        if not steps:
            raise ValueError(
                "sub-goal {!r} asks for segment {} although its facts leave nothing to "
                "do".format(subgoal.id, label)
            )
        patch = SetTableGraphBuilder(granularity).propose(
            request.goal, request.task, self._context([(segment, steps)])
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
        rationale = "rule: the gold {} subgraph of segment {} for steps {}".format(
            granularity, label, list(steps)
        )
        return SubgraphResponse(subgraph, rationale)

    # -- helpers ----------------------------------------------------------------

    def _require_task(self, task: str) -> None:
        if task != self.task:
            raise ValueError(
                "the rule proposer answers for task {!r}, not {!r}".format(self.task, task)
            )

    def _given_up(self, segment: Segment, facts: Iterable[str]) -> bool:
        return (
            self.failures[segment.label] >= self.give_up_after
            and segment.held not in set(facts)
        )

    def _remaining(
        self, facts: Iterable[str], notes: List[str]
    ) -> List[Tuple[Segment, Tuple[str, ...]]]:
        facts = set(facts)
        remaining = []
        given_up = []
        kept_in_hand = []
        for segment in self.episode.segments:
            gave_up = self._given_up(segment, facts)
            if gave_up:
                given_up.append(segment.label)
            elif self.failures[segment.label] >= self.give_up_after:
                kept_in_hand.append(segment.label)
            steps = remaining_steps(segment, facts, given_up=gave_up)
            if steps:
                remaining.append((segment, steps))
        if kept_in_hand:
            notes.append(
                "segment(s) {} kept although given up: the object is in the gripper".format(
                    kept_in_hand
                )
            )
        if given_up:
            notes.append("gave up the object(s) of segment(s) {}".format(given_up))
        # Deliver what is in the gripper first; the rest keeps the plan order.
        remaining.sort(key=lambda item: (item[0].held not in facts, item[0].index))
        return remaining

    @staticmethod
    def _context(remaining: Sequence[Tuple[Segment, Tuple[str, ...]]]) -> Dict[str, Any]:
        destinations = {segment.destination for segment, _ in remaining}
        if len(destinations) != 1:
            raise ValueError(
                "the gold builder plans one destination, the segments left go to {}".format(
                    sorted(destinations)
                )
            )
        return {
            "segments": tuple(segment.triple for segment, _ in remaining),
            "destination": destinations.pop(),
            "steps": {segment.label: steps for segment, steps in remaining},
        }
