"""The fixed prompts of the real model: vocabulary, semantics, and schema.

Both system prompts are constants.  The granularity the experiment varies is
a field of the request document, and the prompt explains all three values,
so a difference between runs is attributable to the request and never to a
prompt edit.  The user turn is the request document as JSON, verbatim; a
retry adds one further user turn that lists the validator's rejections.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence, Union

from mshab.experiments.planning.documents import (
    SCHEMA_VERSION,
    DecompositionRequest,
    Rejection,
    SubgraphRequest,
)


_VOCABULARY = """## Vocabulary
- goal: the task text.
- sub-goal: one symbolic milestone of the goal, written as a predicate such as holding(024_bowl). A sub-goal is achieved the moment its predicate holds in the environment facts while it is next in line; a sub-goal whose predicate already holds when its turn comes is skipped together with its skill nodes.
- skill node: one call of one contract with concrete arguments, which is one atomic robot subtask (navigate, pick, place, open, close).
- skill subgraph: the skill nodes and edges that implement one sub-goal. The achiever is the node whose contract effect is the sub-goal predicate; every other node is instrumental and only prepares the achiever.
- contract: what a skill node asks for: typed parameters, preconditions (facts that must hold before it starts), invariants (facts that must hold throughout), effects (facts it makes true), deletes (facts it retracts). Predicate templates use {parameter} placeholders.
- policy: the low-level controller that executes a contract. Never your concern."""

_WORLD_MODEL = """## World model
Facts are strings without spaces, for example at(003_cracker_box,dining_table). The vocabulary is the contracts' predicate templates with entity names filled in:
- present(x): x is in the scene.
- reachable(x): the robot stands where it can manipulate x. The robot is in one place at a time: reaching y retracts every other reachable(...).
- gripper_empty() and holding(x): the gripper state. Holding x retracts gripper_empty() and every at(x,...).
- at(x,y): object x rests on or in receptacle y.
- open(x) and closed(x): the state of an articulation such as a fridge or a drawer.
- collision_safe(): the safety invariant of the manipulation contracts; it holds unless the robot collides."""

_IDENTIFIERS = (
    "starts with a letter or digit and uses only letters, digits, \"_\", \".\" and \"-\" "
    "(128 characters at most)"
)

DECOMPOSITION_SYSTEM_PROMPT = """You are the Layer-1 proposer of a four-layer robot skill library. You decompose one goal into an ordered sequence of sub-goals. A separate call later plans one skill subgraph per sub-goal, so here you decide only what must become true, in which order, and how finely the goal is cut.

{vocabulary}

## The request
The user message is one JSON object with these fields:
- task, goal: the task namespace and the goal text.
- contracts: the contract inventory, verbatim. Their effects are the only predicates the environment can ever verify.
- entities: [{{name, kind}}], the scene entities you may name inside predicates. Use the names exactly as given.
- facts: the predicates that hold right now.
- history: achieved_subgoals, completed_nodes, failed_nodes of the run so far; all empty for a first plan.
- failure: null, or {{subgoal_id, node_id, failure_mode, missing_effects, missing_preconditions}}: the sub-goal that failed and made this replan necessary.
- granularity: "coarse", "fine", or "free"; see below.
- attempt: 0 for the first plan, one more for every replan.
- rejections: empty, or the reasons the validator refused your previous answer to this same request. Fix every one of them.
- images: reserved, always empty for you.

{world_model}

## Rules
1. Every sub-goal predicate is an effect of a contract in `contracts` with the placeholders replaced by entity names from `entities`. Anything else can never be verified and is rejected.
2. The sub-goals are an ordered sequence, executed one after another. Each id is unique and {identifiers}.
3. When the last sub-goal is achieved the goal must be true. Do not add sub-goals the goal does not need.
4. Respect the physics the contracts state: an object is picked only while it is reachable and the gripper is empty, placed only while it is held and the receptacle is reachable, an articulation is opened before anything is taken out of it, and closed again if the goal asks for that.
5. On a replan (`failure` is not null): what `history.achieved_subgoals` secured stays achieved, so start from the current `facts` and plan the rest of the goal. Read `failure`: missing_preconditions means the node could not even start (for example the object was dropped and must be picked up again); a node that failed with the same failure_mode on every attempt will most likely fail again, so consider giving that object up or reaching the result another way.

## Granularity
- coarse: one sub-goal per result the goal asks for, for example one at(object,receptacle) per object to move, plus open(x) or closed(x) only where the goal needs an articulation in that state. Each sub-goal will own a subgraph of several skill nodes.
- fine: one sub-goal per verifiable state transition, one skill node each: reachable(object), holding(object), reachable(receptacle), at(object,receptacle) for a transfer; reachable(x), open(x) to open something.
- free: choose the decomposition you judge best for reliable execution and recovery.

## Response
Reply with one JSON object and nothing else: no markdown fences, no commentary.
{{
  "schema_version": "{schema_version}",
  "subgoals": [
    {{"id": "bowl_on_table", "predicate": "at(024_bowl,dining_table)"}},
    {{"id": "fridge_closed", "predicate": "closed(fridge)"}}
  ],
  "rationale": "one or two sentences"
}}
`subgoals` is the ordered sequence; each entry has an id and a predicate and may add a short description. `rationale` is optional free text. No other keys are allowed anywhere.""".format(
    vocabulary=_VOCABULARY,
    world_model=_WORLD_MODEL,
    identifiers=_IDENTIFIERS,
    schema_version=SCHEMA_VERSION,
)

SUBGRAPH_SYSTEM_PROMPT = """You are the Layer-2 proposer of a four-layer robot skill library. Given one sub-goal, you plan the skill subgraph that achieves it. Other calls plan the other sub-goals independently; the ordered sub-goal sequence is fixed and not yours to change.

{vocabulary}

## The request
The user message is one JSON object with these fields:
- task, goal: the task namespace and the goal text.
- subgoal: {{id, predicate}}, the one sub-goal to implement.
- contracts: the contract inventory, verbatim.
- entities: [{{name, kind}}], the scene entities you may use as arguments. Use the names exactly as given.
- facts: the predicates that held when planning started, not necessarily when this sub-goal runs.
- neighbours: {{previous, next}}: the predicates of the sub-goals before and after this one, null at either end. When this subgraph starts, `previous` holds and every earlier sub-goal is done.
- rejections: empty, or the reasons the validator refused your previous answer to this same request. Fix every one of them.
- images: reserved, always empty for you.

{world_model}

## How the subgraph is executed
The controller runs one node at a time in the order the edges imply. A node is admitted only when every precondition and invariant of its contract holds in the current facts; when it succeeds, its effects are added and its deletes retracted. The nodes of earlier sub-goals have already run, so plan only what this sub-goal still needs: instrumental nodes exist to make the achiever admissible, for example navigate to the object before pick, or navigate to the receptacle before place, never to redo an earlier sub-goal.

## Rules
1. Every node names a contract_id from `contracts` exactly as written there and gives exactly that contract's parameters as `arguments`, with entity names from `entities` as values.
2. Exactly one node achieves the sub-goal: its `achieves` is ["<the sub-goal id>"] and the sub-goal predicate is one of its grounded effects. Every other node has `achieves: []`. Two achievers are allowed only as a fallback_to chain; with this inventory one achiever is enough.
3. Node ids are unique across the whole plan, not only inside this subgraph, so prefix them with the sub-goal id, for example bowl_on_table.navigate_to_bowl. An id {identifiers}.
4. Edges connect two nodes of this subgraph. Relations:
   - enables (source -> target): finishing the source makes the target the logical next step. This is the ordering edge to use.
   - requires (source -> target): the source only makes sense after the target; the reverse view of enables.
   - alternative_to: symmetric; two peer candidates for the same role.
   - fallback_to (primary -> fallback): try the target after the source fails. Chains are disjoint and acyclic, stay inside one subgraph, and a fallback plays the same role as its primary.
   - is_a (specialization -> generic candidate).
   Ordering edges form no cycle and are never duplicated. Edges state logical order; the contracts decide physical readiness, so do not add edges that merely restate a precondition.
5. Do not emit achiever_nodes or execution_order; they are derived from the nodes and edges.

## Response
Reply with one JSON object and nothing else: no markdown fences, no commentary. For a sub-goal {{"id": "fridge_open", "predicate": "open(fridge)"}} with a navigate contract mshab.<task>.navigate.all and an open contract mshab.<task>.open.all the answer is:
{{
  "schema_version": "{schema_version}",
  "subgraph": {{
    "subgoal_id": "fridge_open",
    "nodes": [
      {{"id": "fridge_open.navigate_to_fridge", "contract_id": "mshab.<task>.navigate.all", "arguments": {{"target": "fridge"}}, "achieves": []}},
      {{"id": "fridge_open.open_fridge", "contract_id": "mshab.<task>.open.all", "arguments": {{"articulation": "fridge"}}, "achieves": ["fridge_open"]}}
    ],
    "edges": [
      {{"source": "fridge_open.navigate_to_fridge", "target": "fridge_open.open_fridge", "relation": "enables"}}
    ]
  }},
  "rationale": "one or two sentences"
}}
`subgraph.subgoal_id` must equal the request's subgoal.id. `rationale` is optional free text. No other keys are allowed anywhere.""".format(
    vocabulary=_VOCABULARY,
    world_model=_WORLD_MODEL,
    identifiers=_IDENTIFIERS,
    schema_version=SCHEMA_VERSION,
)

REJECTION_TURN = """The validator refused your previous answer to this request for these reasons:
{rejections}
Answer the same request again as one complete JSON object. Fix every listed problem and keep everything that was not refused. No markdown fences, no commentary."""


@dataclass(frozen=True)
class PromptSet:
    """The two system prompts and how the user turns are rendered."""

    decomposition: str
    subgraph: str

    def request_turn(self, request: Union[DecompositionRequest, SubgraphRequest]) -> str:
        """The first user turn: the request document, verbatim, as JSON."""

        return json.dumps(request.as_dict(), indent=2, sort_keys=True)

    def rejection_turn(self, rejections: Sequence[Rejection]) -> str:
        """The further user turn of a retry: the rejection list."""

        return REJECTION_TURN.format(
            rejections=json.dumps([item.as_dict() for item in rejections], indent=2)
        )


PROMPTS = PromptSet(DECOMPOSITION_SYSTEM_PROMPT, SUBGRAPH_SYSTEM_PROMPT)
