# Layers 1 and 2: sub-goal graph and skill graph

The simulator-independent half of the library: sub-goal decomposition, one
skill subgraph per sub-goal, what the edges mean, how one next node is chosen,
and what happens when a node or a whole sub-goal fails.

Part of the [MS-HAB skill library guide](../README.md).

## Layer 1: sub-goal graph

`SubGoalGraph` holds the goal text and the sub-goals it decomposes into. It
represents what must happen, independently of how a particular scene or robot
achieves it. The goal -> sub-goal decomposition produces an **ordered**
sequence; `from_sequence()` stores that order as a chain of dependencies. The
incremental `add_subgoal`/`add_dependency` API remains for hand-authored
graphs and patches, which may express a partial order.

| Object/API | Meaning |
| --- | --- |
| `goal` | The original task text |
| `subgoals` | Sub-goal id to `SubGoal` mapping |
| `dependencies` | Coarse required ordering between sub-goals; Layer 2 must mirror each one with an `ENABLES` cross-subgraph edge and may add finer ordering |
| `add_subgoal(subgoal)` | Add one symbolic predicate sub-goal |
| `add_dependency(source, target)` | Add ordering and reject cycles |
| `from_sequence(goal, subgoals)` | Build the graph from an ordered sub-goal sequence; dependencies are the consecutive pairs |
| `ready_subgoals(facts, completed)` | Sub-goals whose predecessors have been achieved |
| `predecessors(subgoal_id)` | The sub-goals that dependencies order before this one |
| `execution_order()` | Stable topological order; for a sequence-built graph it is the generating sequence |

Sub-goals may be transient. The task runner should pass already completed
sub-goal ids to `ready_subgoals(facts, completed=...)`, so closing the fridge
does not erase the history that it was opened successfully.

```python
from mshab.skills import SubGoal, SubGoalGraph

subgoals = SubGoalGraph.from_sequence(
    "Retrieve the apple",  # the goal
    (
        SubGoal("reachable", "reachable(013_apple)"),
        SubGoal("retrieved", "holding(013_apple)"),
    ),
)
assert subgoals.execution_order() == ("reachable", "retrieved")
```

Predicates at this layer are canonical symbolic vocabulary. Concrete actor
handles, tensor indices, poses, and scene ids belong in the environment
adapter, not in this graph.

## Layer 2: one skill subgraph per sub-goal

A `SkillNode` is a request to run one contract with symbolic arguments, not an
executable invocation.

```python
from mshab.skills import (
    SkillSubgraph,
    SkillGraph,
    SkillNode,
    SkillRelation,
)

graph = SkillGraph("set_table", subgoal_graph=subgoals)

reachable = SkillSubgraph(subgoal_id="reachable", task="set_table")
reachable.add_node(
    SkillNode(
        id="navigate_to_apple",
        contract_id="mshab.set_table.navigate.all",
        arguments={"target": "013_apple"},
        achieves=("reachable",),
    )
)

retrieved = SkillSubgraph(subgoal_id="retrieved", task="set_table")
retrieved.add_node(
    SkillNode(
        id="pick_apple",
        contract_id="mshab.set_table.pick.013_apple",
        arguments={},
        achieves=("retrieved",),
    )
)

graph.add_subgraph(reachable)
graph.add_subgraph(retrieved)

# The two nodes have different owners, so this becomes a cross-subgraph edge.
graph.relate("navigate_to_apple", "pick_apple", SkillRelation.ENABLES)
```

| Object/API | Important attributes | Responsibility |
| --- | --- | --- |
| `SkillGraph` | `task`, `subgoal_graph`, `subgraphs`, `cross_edges` | Own the complete Layer-2 aggregate and cross-sub-goal relations |
| `SkillSubgraph` | `subgoal_id`, `task`, `nodes`, `edges`, `achievers` | Own the complete candidate implementation for one sub-goal |
| `SkillNode` | `id`, `contract_id`, `arguments`, `achieves` | Refer to one contract with symbolic arguments |
| `SkillEdge` | `source`, `target`, `relation` | Relate two skill nodes inside the same sub-goal subgraph |
| `CrossSubgraphEdge` | `source_subgoal`, `target_subgoal`, `source_node`, `target_node`, `relation` | Relate skill nodes belonging to two different sub-goal subgraphs |

Important invariants are enforced by methods rather than convention:

- a subgraph belongs to exactly one `subgoal_id`;
- every registered subgraph has at least one achiever;
- a node inside a subgraph may only claim that subgraph's sub-goal;
- registering a subgraph seals it: every later attribute assignment, not just
  `add_node`/`relate`, is rejected;
- node ids are unique across the aggregate;
- a cross-subgraph edge verifies both node owners;
- causal cycles and duplicate relations are rejected;
- every Layer-1 dependency is mirrored by an `ENABLES` cross-subgraph edge
  (`validate()`); Layer 2 may add ordering that Layer 1 leaves open;
- `FALLBACK_TO` edges form disjoint chains inside one subgraph: a node
  declares at most one fallback and is the fallback of at most one node,
  chains never cycle, a fallback plays the same role as its primary (achiever
  for achiever, instrumental for instrumental), no causal edge joins two
  members of one chain, and a cross-subgraph edge cannot be a fallback;
- a cross-sub-goal relation may name any achiever of its source sub-goal, so
  a fallback candidate satisfies downstream dependencies.

| Relation | Direction | Meaning |
| --- | --- | --- |
| `IS_A` | specialization -> generic candidate | Semantic specialization |
| `ENABLES` | first -> next | Completing the source makes the target the logical next step |
| `REQUIRES` | consumer -> prerequisite | The consumer only makes sense after the target |
| `ALTERNATIVE_TO` | symmetric | Peer candidates for the same role |
| `FALLBACK_TO` | primary -> fallback | Try target after source fails. Chains are disjoint, acyclic, role-preserving, and stay inside one subgraph |

### Edges are about semantics; contracts are about physics

Every relation above states that an order is *logically* sensible for the
task: the counter is opened before the bowl is picked, the bowl is placed
before the counter is closed, food is heated before it is served. Nothing in
the graph consults the live scene.

Whether a node can *physically* start right now is a different question and
is answered only by its contract (Layer 3): `reachable(024_bowl)` and
`gripper_empty()` must be present in the current environment snapshot before
`pick_bowl_specialized` may be admitted. Contracts never encode task order,
and edges never restate physical preconditions. Keeping the two apart
is what lets one contract be reused by many skill nodes in different places
of the graph.

Two skill nodes that reference the same contract with the same arguments are
redundant; use one node. Separate Layer-2 candidates are appropriate when the
contracts they reference differ in target scope or failure/recovery behavior.
In the SetTable starter, the fixed-object contract and the `all`-object
contract have different target scopes and form an explicit primary/fallback
candidate pair.

`graph.ready_nodes(completed)` checks graph relations only. It intentionally
does not inspect facts, contracts, checkpoints, or environments. Use
`SkillRuntime.ready_nodes(...)` for complete Layer-3/4 admission.

### Cross-subgraph edges are sub-goal-level by default

A `CrossSubgraphEdge` endpoint is either a node id or `None`, meaning *any
achiever of that sub-goal*:

```python
from mshab.skills import CrossSubgraphEdge, SkillRelation

CrossSubgraphEdge("bowl_retrieved", "bowl_placed", None,
                  "navigate_bowl_to_destination", SkillRelation.ENABLES)
```

This is what makes alternatives usable. Pinning the relation to
`pick_bowl_specialized` would silently make that one candidate mandatory: a
rollout that recovered through `pick_bowl_generic` could never enable the next
sub-goal. `graph.prerequisite_groups(node_id)` therefore returns
*disjunctive* groups -- at least one member of each group must be completed --
and `ready_nodes` schedules against those groups rather than a flat set.

### Candidate graph versus one-node decisions

`SkillGraph` stores every candidate, including specialized and
generic fallback nodes. Consequently, its `execution_order()` is a stable
topological order over **all candidates**; it is not an execution plan and must
not be sent directly to Layer 3.

Something has to choose **one next `SkillNode` per call**. At this stage that
chooser is rule-based: `SkillPlanner`. A trained decision model is a later
option and must satisfy the same interface; either way the sequence is
produced incrementally, never emitted all at once:

```text
input:  candidate SkillGraph
        + completed/failed skill-node history
        + current environment summary

output: one selected SkillNode id
```

Each decision must:

1. respect sub-goal dependency order;
2. choose an instrumental node, one achiever, or a follow-up of that achiever
   from the active subgraph;
3. avoid executing unused `ALTERNATIVE_TO` candidates;
4. send only the selected node to Layer-3 grounding.

`SkillPlanner` in [`plan.py`](../plan.py) is the rule-based implementation of
that interface. It reads the candidate order out of the graph
rather than hard-coding it: within a sub-goal subgraph the achiever that starts
the `FALLBACK_TO` chain is the primary, and each `FALLBACK_TO` target is the next
candidate to try once its predecessor is reported failed. The same rule
selects instrumental prerequisites: for every prerequisite chain the planner
wants the member that already completed, else the first member that has not
failed. It also wants the achiever's *follow-ups*: the non-achiever nodes the
achiever enables inside its own subgraph, directly or through other
follow-ups, resolved along their own fallback chains the same way. A
follow-up is work that belongs to the sub-goal but comes once its predicate
holds, for example navigating back to a drawer and closing it after the
object was placed; the planner runs it after the achiever and before the
next sub-goal. A node that merely shares a prerequisite with the achiever,
such as the setup of an alternative candidate, is not a follow-up.
`planner.remaining(subgoal_id, completed, failed)` lists what the planner
still wants from one sub-goal. Before returning a
node it also checks `graph.ready_nodes(completed)`: Layer 1 fixes only the
coarse sub-goal order, and the finer Layer-2 relations are what the planner
obeys.

```python
from mshab.skills import SkillPlanner, build_set_table_graph

subgoals, graph = build_set_table_graph()
planner = SkillPlanner(subgoals, graph)

planner.decide(completed=(), failed=())        # one SkillNode per call
planner.plan()                                 # the 16-step nominal path
planner.plan(failed=("pick_bowl_specialized",))  # recovers via pick_bowl_generic
```

`SkillPlanner` is not a policy in the Layer-4 sense. It is the current Layer-2
decision method, the executable specification any later trained decision model
must satisfy, and the source of the catalog's `execution_plans` section.

### Failure handling has two levels

1. **A skill node fails: fall back inside its subgraph.** The failed node's
   `FALLBACK_TO` successor is the next candidate, for achievers and
   instrumental nodes alike; the sub-goal, its subgraph, and the rest of the
   plan stay as they are. Two rules make the substitution sound: a fallback
   inherits the prerequisites of the nodes it stands in for, and any member of
   a chain satisfies the causal edges of every member, so a node wired to the
   primary is enabled by the fallback as well. `plan(failed=...)` above is
   this level.
2. **A sub-goal fails: replan Layer 1.** When the failed node has no fallback
   left, `decide()` raises `NoViableCandidate`: the sub-goal as a whole has
   failed. There is no fallback between sub-goals and no substitute sub-goal in
   the graph. Recovery decomposes the goal again from the current environment
   state and decides the new sub-goal sequence afresh. That replan is the job
   of the goal -> sub-goal proposer behind the `GraphProposer` boundary:
   `TaskController` in `mshab/experiments/planning/` catches
   `NoViableCandidate`, asks the proposer again with the failure, the history,
   and the current facts, and swaps in the new layers. The packaged SetTable
   runner has no such loop, so there `NoViableCandidate` still ends the run.

