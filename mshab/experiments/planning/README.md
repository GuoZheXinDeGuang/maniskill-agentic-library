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
`ScriptedProposer.from_gold_graphs(names, library)` cuts the tables out of the
committed gold graphs; `as_dict()`/`from_dict()` and `save()`/`load()` move
them through JSON, which is how stage-4 scenarios will add replan answers.

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

```python
from pathlib import Path

from mshab.experiments.granularity import build_granularity_library
from mshab.experiments.granularity.higher_layers import GOLD_GRAPHS
from mshab.experiments.planning import ProposalValidator, PlanningContext, ScriptedProposer

library = build_granularity_library(Path("../mshab-assets/data/mshab_checkpoints"))
proposer = ScriptedProposer.from_gold_graphs(GOLD_GRAPHS, library)
validator = ProposalValidator(library, "granularity")
context = PlanningContext.initial(library, "granularity", granularity="fine")
validated = validator.plan(
    proposer, "Tidy the house: move every object to its target receptacle.", context
)
validated.plan.order            # the 20 nominal node ids
validated.as_dict()             # requests, responses, and the plan
```
