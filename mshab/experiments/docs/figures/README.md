# Figures of the higher-layers experiment

Pictures of what [`higher-layers-plan.md`](../higher-layers-plan.md) describes:
the pipeline from the request to the trace, the hand-authored gold graphs the
model is measured against, and, once a run has been extracted, what DeepSeek
actually proposed (the sub-goal chains of Layer 1 and the skill-node graphs
of Layer 2, including the replans inside scenario runs).

Every figure is deterministic SVG written by
`python -m mshab.experiments.docs.figures.render`. The DeepSeek figures are
drawn from compact extracts in [`extracts/`](extracts/), not from the
gitignored run directories, so they regenerate on a clean clone;
`tests/granularity/test_experiment_figures.py` checks that they do.

The experiment's task moved from TidyHouse to SetTable on 2026-09-21. The
TidyHouse extracts and figures of the 2026-09-20 runs were removed with it
(commit `d7fa8de` holds them); no SetTable run of the model has been
extracted yet, so `extracts/` is empty and the two figures below are the
only ones.

## The pipeline

![pipeline](pipeline.svg)

Blue band: the two proposer calls, the assembler, and the validator with its
retry rounds; the objects are those of [`../../planning/`](../../planning/README.md).
Green band: the controller loop, which is the same on the symbolic environment
(stages 4 and 5) and on MS-HAB (stage 6, [`../../rollout/`](../../rollout/README.md)).
Grey band: what stays fixed, the contract library of
[`../../granularity/`](../../granularity/README.md) and the gold graphs.

## The gold graphs

The two references of the granularity axis, rendered one row per sub-goal by
the granularity experiment itself (`python -m mshab.experiments.granularity.render`):

| Graph | Sub-goals | Nodes | Rendering |
| --- | --- | --- | --- |
| `set_table_coarse` | 2 | 16 | [`set_table_coarse.svg`](../../granularity/graphs/set_table_coarse.svg) |
| `set_table_fine` | 16 | 16 | [`set_table_fine.svg`](../../granularity/graphs/set_table_fine.svg) |

![set_table_coarse](../../granularity/graphs/set_table_coarse.svg)

The same renderer draws the DeepSeek graphs, so a model's graph and its
reference read alike: orange border = the achiever of the row's sub-goal,
purple arrows = relations inside a subgraph, orange arrows = the
cross-subgraph `ENABLES` edges the assembler adds.

## Layer 1: the sub-goal chains side by side

![subgoal_chains](subgoal_chains.svg)

One column per decomposition, top to bottom in execution order, one row per
skill node: a coarse sub-goal spans the eight rows of its segment, a fine
sub-goal one. The two gold columns come first; extracted DeepSeek proposals
(static ones, then the initial proposal of each scenario run) follow. A
dashed red box is a predicate that is in neither gold graph.

## Layer 2: the DeepSeek node graphs

A *static* proposal is the answer to the goal under the nominal facts, before
anything runs (`evaluate.py --no-scenarios`); it is drawn as
`deepseek_<granularity>_static.svg` with the nodes that have no counterpart
in the gold subgraph of the same predicate flagged. A scenario run is drawn
as `deepseek_<scenario>_<granularity>_replans.svg`: one block per proposal of
the run, the initial plan, then every Layer-1 replan with the failure that
triggered it, the model's rationale, and what the controller did with each
skill node (green success, red failed, yellow admission refused, grey
skipped).

## Data and regeneration

```bash
# redraw every figure from extracts/
python -m mshab.experiments.docs.figures.render

# refresh extracts/ from evaluation runs first (a run directory holds summary.json,
# exchanges.json, and the traces), then redraw
python -m mshab.experiments.docs.figures.render --extract mshab_exps/planning/<run> ...
```

An extract keeps the accepted responses, the request's entities, facts,
history, and failure, the controller's decisions and replans, and the
evaluation numbers; it drops the contract inventory every request repeats.
Extracts are named by granularity (`static_<granularity>.json`) and by
scenario (`scenario_<name>_<granularity>.json`), so a later run at the same
setting replaces the earlier extract. The figure files follow the same names.
