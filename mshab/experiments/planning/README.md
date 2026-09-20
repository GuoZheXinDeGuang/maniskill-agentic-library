# Proposer boundary

The interface between the controller and whatever generates the upper two
layers: the scripted pseudo proposer today, a real model later. A *proposer*
proposes Layers 1 and 2; the *planner*, `SkillPlanner` in `mshab.skills`,
decides which node runs next on a graph that already exists. Stage 3 of
[`../docs/higher-layers-plan.md`](../docs/higher-layers-plan.md). Everything
here is simulator-independent and standard library only.

## Two calls

```text
call 1  decompose:      goal + scene + contract inventory  ->  ordered sub-goals
call 2  plan_subgraph:  one sub-goal + the same scene       ->  one SkillSubgraph   (once per sub-goal, independently)
assemble_patch:         sub-goals + subgraphs               ->  SkillGraphPatch
```

Granularity is the experimental knob and is controlled in call 1
(`free`, `coarse`, `fine`). Call 2 is the same for every granularity, so a
difference in outcome is attributable to the decomposition.

## Documents (`documents.py`)

| Document | Fields | Consumed by |
| --- | --- | --- |
| `DecompositionRequest` | `task`, `goal`, plus the `PlanningContext`: `contracts`, `entities`, `facts`, `history`, `failure`, `granularity`, `attempt` | the proposer |
| `DecompositionResponse` | ordered `subgoals` `[{id, predicate}]`, `rationale` | `SubGoalGraph.from_sequence` |
| `SubgraphRequest` | `task`, `goal`, `subgoal`, `contracts`, `entities`, `facts`, `neighbours {previous, next}` | the proposer |
| `SubgraphResponse` | `subgraph {subgoal_id, nodes, edges}`, `rationale` | `SkillSubgraph.from_dict` |

`contracts` carries the library's `Contract.as_dict()` records verbatim; the
boundary defines no second contract serialization.
`PlanningContext.initial(library, task, entities=..., facts=..., granularity=...)`
builds the context of a first plan; `context.replan(failure, history, facts)`
the context of the next attempt after a sub-goal failed. Every document has a
strict `from_dict` through `mshab.skills.schema`: unknown keys, bad
identifiers, non-scalar arguments, and a contract outside the task namespace
are rejected at the boundary. A proposer never emits the derived
`achiever_nodes` and `execution_order` views; `SkillSubgraph.from_dict`
recomputes them.

## Proposer (`proposer.py`)

`GraphProposer` is a `SkillGraphBuilder` with two abstract methods, `decompose()`
and `plan_subgraph()`. `propose()` runs both calls and hands the answers to
`assemble_patch()`, so `proposer.build(goal, task, context, library)` works
like any other builder. The assembler adds Layer 1 as the ordered sequence
with consecutive dependencies and one sub-goal-level `ENABLES` cross edge
from every sub-goal to each root node of the next subgraph; it never invents
or reorders sub-goals.

`ScriptedProposer` is the pseudo proposer. Answers are keyed by request
fingerprint: `(task, goal, granularity, attempt, failed sub-goal)` for a
decomposition and `(task, sub-goal id, predicate)` for a subgraph. An unknown
fingerprint raises `UnscriptedRequest` instead of answering with a default.
`gold_proposer(names, library)` in the granularity experiment cuts the tables
out of the committed gold graphs and `scenario_proposer()` adds a scenario's
replan answers; `as_dict()`/`from_dict()` and `save()`/`load()` move tables
through JSON. This package depends on `mshab.skills` only.

## Validator (`validator.py`)

`ProposalValidator(library, task).plan(proposer, goal, context)` runs call 1, then
call 2 for every sub-goal, then assembles, applies the patch to fresh graphs
with the library, validates, and runs `SkillPlanner.plan()`. It returns a
`ValidatedProposal` holding both layers, the patch, the plan, and every request
and response verbatim (`as_dict()` is the trace record). Otherwise it raises
`ProposalRejected` with `Rejection(stage, subgoal_id, message)` entries:

| Stage | What failed |
| --- | --- |
| `decomposition` | the proposer errored, returned the wrong type, or the sequence is not a Layer-1 chain |
| `subgraph` | per sub-goal: wrong owner, no achiever, a node that does not ground (unknown contract, bad arguments), a node id another sub-goal already uses; all sub-goals are checked before the round stops |
| `assembly` | subgraphs do not match the sub-goal sequence |
| `graph` | `SkillGraphPatch.apply` or `SkillGraph.validate` refused the whole |
| `plan` | `SkillPlanner` cannot choose, for example two achievers with no `FALLBACK_TO` order |

Nothing live is touched on the way: a rejected round leaves whatever graphs
the controller holds byte identical.

## Symbolic environment (`symbolic.py`)

`SymbolicEnvironmentAdapter` is an `EnvironmentAdapter` whose whole state is
a set of predicate facts. `step({"add": [...], "remove": [...]})` applies a
fact change and then two world rules the contracts cannot express, because a
contract only retracts predicates it names: `holding(x)` retracts
`gripper_empty()` and every `at(x,...)`; a step that adds `reachable(y)`
retracts every other `reachable(...)`. `SymbolicPolicyExecutor` executes a
grounding by adding its effects and retracting its deletes, unless a
`ScriptedFailure` for that grounding says otherwise for that attempt: then the
failure's own `add`/`remove` happen and the execution reports its
`failure_mode`. Failures are keyed by contract type and grounded target, not
by node id, so a scenario also applies to graphs whose node ids a proposer
chose. `bind_symbolic_policy(library)` registers one `SymbolicPolicy`, bound
after every existing binding, so it is the ready default when no checkpoint
is downloaded. Admission, invariant monitoring, and effect verification are
not re-implemented here; they run in `SkillRuntime`.

## Controller (`controller.py`)

`TaskController(library, task, attempts_per_node=2, max_replans=2).run(goal,
proposer, environment, executor, granularity=..., goal_facts=...)` is the
`execute -> observe -> re-decide` loop the skill-library guide asks for:

```text
proposal = ProposalValidator.plan(proposer, goal, PlanningContext.initial(...))
loop:
    absorb facts: the next sub-goal in line whose predicate already holds is
                  achieved, and the nodes of its subgraph are skipped
    node = SkillPlanner.decide(completed, failed)         # one node, or None: done
    result = SkillRuntime.execute_node(graph, node.id, executor)
    success            -> completed
    failure            -> retry while the node's attempts remain, else failed
    NoViableCandidate  -> Failure(sub-goal, node, mode, missing effects / preconditions)
                          ProposalValidator.plan(proposer, goal, context.replan(...))
                          swap both layers; achievement is derived from the facts again
```

Achievement follows the guide: a sub-goal counts as achieved when its
predicate holds at the moment it becomes next in line, and that stays
recorded even if a later step retracts the predicate. Judging only the
sub-goal next in line matters: a transient predicate that happens to hold
early, such as `reachable(x)` before the robot moves on, must not mark a
later sub-goal done. Each decision has one of four outcomes: `success`,
`failed`, `admission_failed` (a precondition or invariant was missing when
`SkillRuntime` admitted the node), or `skipped`. A run ends with status
`success`, `goal_not_reached`, `proposal_rejected`, or `replans_exhausted`.
Success is judged on `goal_facts` in the final facts, independently of how
the proposer decomposed the goal.

`RunResult` (`as_dict()`, `save(path)`) is the trace: every proposal with its
requests and responses verbatim or its rejections, every decision with the
facts it added and removed, every replan with the failure that caused it, the
initial and final facts, and `metrics`:

| Metric | Meaning |
| --- | --- |
| `subgoals`, `subgraphs`, `nodes`, `mean_nodes_per_subgraph` | structure of the first accepted proposal |
| `node_executions`, `successful_executions`, `failed_executions`, `admission_failures`, `retries` | what ran; retries are executions with attempt > 1 |
| `skipped_nodes` | nodes of sub-goals that were already achieved when they came up |
| `redundant_executions` | successful executions whose effects all held before they ran |
| `achieved_subgoals` | distinct sub-goal predicates achieved over the whole run, replans included |
| `goal_facts_achieved` / `goal_facts` | the success criterion, fact by fact |
| `replans`, `replanning_span` | accepted replans, and how many of their sub-goals did not already hold at the moment the replan was made |
| `recovery_success` | success after at least one failure or replan |

The scenarios that exercise the loop, and the helper that runs one, live in
[`../granularity/higher_layers/scenarios.py`](../granularity/higher_layers/scenarios.py).

```python
from pathlib import Path

from mshab.experiments.granularity import build_granularity_library
from mshab.experiments.granularity.higher_layers import GOLD_GRAPHS, gold_proposer
from mshab.experiments.planning import ProposalValidator, PlanningContext

library = build_granularity_library(Path("../mshab-assets/data/mshab_checkpoints"))
proposer = gold_proposer(GOLD_GRAPHS, library)
validator = ProposalValidator(library, "granularity")
context = PlanningContext.initial(library, "granularity", granularity="fine")
validated = validator.plan(
    proposer, "Tidy the house: move every object to its target receptacle.", context
)
validated.plan.order            # the 20 nominal node ids
validated.as_dict()             # requests, responses, and the plan
```
