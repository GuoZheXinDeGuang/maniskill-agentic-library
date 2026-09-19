# Graph-granularity experiment

This package holds the two lower layers shared by the coarse- and
fine-grained graph experiments, as one `ContractLibrary` from
`mshab.skills`, and the hand-authored *gold graphs* built over them. The
plan that the package follows is
[`../docs/higher-layers-plan.md`](../docs/higher-layers-plan.md).

![Contract and policy layers](artifacts/contract_policy_layers.svg)

## Code structure

```text
lower_layers/
├── manifest.py   the 53 downloaded MS-HAB RL checkpoints as PolicySpec rows
└── library.py    build_granularity_library(): 5 generic contracts, 53 policies, 53 bindings

higher_layers/
├── builders.py   TidyHouseGraphBuilder(granularity), SetTableGenericGraphBuilder
├── gold.py       registry, validation, JSON document, loader
└── render.py     SVG rendering and the graphs/ artifacts

graphs/           committed gold graphs: <name>.json and <name>.svg
```

Everything else is the ordinary skill-library object model: `Contract` and
`CheckpointPolicy` from `mshab.skills.model`, bindings and policy selection
from `mshab.skills.library.ContractLibrary`. The package adds no storage or
relation classes of its own.

## Layer 3: five generic contracts

`library.py` registers one contract per `ContractType`, all with task
`granularity` and target `all`, so their ids are
`mshab.granularity.<type>.all`:

| Contract | Inputs | Preconditions | Effects | Deletes |
| --- | --- | --- | --- | --- |
| `NavigateContract` | `target` | `present(target)` | `reachable(target)` | — |
| `PickContract` | `object` | `reachable(object)`, `gripper_empty()` | `holding(object)` | `gripper_empty()` |
| `PlaceContract` | `object`, `destination` | `holding(object)`, `reachable(destination)` | `at(object,destination)`, `gripper_empty()` | `holding(object)` |
| `OpenContract` | `articulation` | `reachable(articulation)`, `closed(articulation)` | `open(articulation)` | `closed(articulation)` |
| `CloseContract` | `articulation` | `reachable(articulation)`, `open(articulation)` | `closed(articulation)` | `open(articulation)` |

A skill node grounds one of them with a concrete object, destination, or
articulation; the graphs that do so are the experiment's own test graphs, so
the ids keep the `granularity` task segment.

## Layer 4: the 53-policy manifest

`manifest.py` enumerates every `<family>/<task>/<type>/<target>` leaf of the
official RL download as a `PolicySpec` row and validates that the row names a
checkpoint that exists: navigation has only `all`, open and close exist only
for SetTable's `fridge` and `kitchen_counter`, and pick and place cover each
task's own object categories plus `all`.

- 3 Navigate policies.
- 23 Pick policies.
- 23 Place policies.
- 2 Open policies.
- 2 Close policies.

`PolicySpec.policy(checkpoint_root)` builds the `CheckpointPolicy`; its
`target` is the leaf's target segment. Readiness is read from the filesystem
at run time and never recorded.

## Bindings

Each policy is bound to the one contract of its type, in sorted policy-id
order:

```text
mshab.granularity.pick.all      <- rl.prepare_groceries.pick.002_master_chef_can, ..., rl.tidy_house.pick.all
mshab.granularity.navigate.all  <- rl.prepare_groceries.navigate.all, rl.set_table.navigate.all, rl.tidy_house.navigate.all
```

53 policies, 53 bindings. Binding order alone would let the apple checkpoint
execute a bowl pick, so which policy runs a grounding is decided with the
grounded arguments:

```python
library.select_policy("mshab.granularity.pick.all", arguments={"object": "024_bowl"})
# a ready checkpoint with target 024_bowl first, then one with target all;
# rl.set_table.pick.013_apple is never a candidate
```

`SkillRuntime` passes the grounded arguments through automatically. Fallback
and alternative relations belong to Layer 2; there are none in Layer 4 or in
the bindings.

## Gold graphs

A gold graph is a hand-authored, validated Layer-1/2 skill graph that exists
only for this experiment. It is the answer the scripted proposer returns and
the reference a real model's output is compared against; it is not a library
for later work to build on.

Every skill node is one contract call, which is one MS-HAB atomic subtask.
That is the only unit a policy can execute and the only unit the environment
verifies, so node granularity is fixed; what the experiment varies is how
many sub-goals own those nodes. With one generic contract per type there is
exactly one candidate node per role, so these graphs have no `FALLBACK_TO`
chains (the packaged SetTable graph keeps covering Layer-2 fallback).

| Graph | Sub-goals | Nodes | What it is |
| --- | --- | --- | --- |
| `tidy_house_coarse` | 5 | 20 | One `at(object,destination)` sub-goal per transfer; its subgraph is navigate -> pick -> navigate -> place |
| `tidy_house_fine` | 20 | 20 | `reachable -> holding -> reachable(destination) -> at` per transfer; every subgraph is one node |
| `set_table_generic` | 8 | 16 | The packaged SetTable order with one node per role on the generic contracts |

The two TidyHouse variants hold the same node ids, contracts, arguments, and
nominal execution order; only sub-goal ownership differs.
`TidyHouseGraphBuilder` takes `context={"transfers": ((object, receptacle),
...)}` for other episodes. PrepareGroceries is not built yet.

![tidy_house_coarse](graphs/tidy_house_coarse.svg)

Each committed `graphs/<name>.json` carries the `SkillGraphPatch` that builds
the graph, plus metadata, a structural summary, and the nominal plan.
`load_gold_graph(path, library)` rebuilds both layers from the patch alone,
revalidates them, grounds every node, plans, and rejects a document whose
summary or plan is stale.

```python
from pathlib import Path

from mshab.experiments.granularity import build_gold_graph, build_granularity_library, load_gold_graph
from mshab.experiments.granularity.higher_layers import gold_graph_path

library = build_granularity_library(Path("../mshab-assets/data/mshab_checkpoints"))
gold = build_gold_graph("tidy_house_fine", library)
gold.plan.order                       # 20 node ids
subgoals, graph = load_gold_graph(gold_graph_path("tidy_house_coarse"), library)
```

## Build and inspect

From the repository root:

```bash
MS_ASSET_DIR=../mshab-assets \
PYTHONPATH=. \
python -m mshab.experiments.granularity.render
```

This generates:

- `artifacts/library.json`: the five contracts with their policies in
  preference order, the 53 manifest rows with their contract, and a summary.
- `artifacts/contract_policy_layers.svg`: the diagram above.
- `graphs/<name>.json` and `graphs/<name>.svg` for every gold graph.

All of them exclude absolute checkpoint paths and local readiness state,
making them stable across machines.

## Python usage

```python
from pathlib import Path

from mshab.experiments.granularity import build_granularity_library, contract_id

library = build_granularity_library(Path("../mshab-assets/data/mshab_checkpoints"))

pick = library.get(contract_id("pick"))
grounded = pick.bind({"object": "024_bowl"})
library.applicable_policies(pick.id, grounded.arguments)   # bowl checkpoints, then all
library.select_policy(pick.id, arguments=grounded.arguments)  # first ready of those
```

The proposer boundary whose scripted implementation answers with these gold
graphs lives in [`../planning/`](../planning/README.md).
