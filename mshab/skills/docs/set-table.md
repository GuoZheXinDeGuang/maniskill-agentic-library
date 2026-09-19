# The packaged SetTable graph

The complete manual SetTable graph, how to run a graph-selected sequence in MS-
HAB and record video, and the smaller apple starter stack.

Part of the [MS-HAB skill library guide](../README.md).

## Complete generated SetTable graph

The first complete manual SetTable graph follows the official 16-step task
order: bowl from `kitchen_counter`, then apple from `fridge`. It includes 8
sub-goals, **8 sub-goal-owned skill subgraphs**, 20 candidate skill nodes, 24
internal edges, 7 cross-subgraph edges, 20 grounded skills, 11 canonical
contracts, 11 RL checkpoint policies, and 15 contract-policy bindings. The
cross-subgraph edges leave their source endpoint open (any achiever of that
sub-goal), so the flattened `graph.edges` view expands them into 11 concrete
edges, for 35 in total.

The generated machine-readable graph is
[`catalogs/set_table.json`](../catalogs/set_table.json).

<p align="center">
  <img src="../../../docs/static/images/set_table_skill_graph.svg"
       alt="Complete SetTable four-layer skill graph" width="100%" />
</p>

Rebuild both artifacts. This works on a clean clone: checkpoint availability
is runtime inventory state and is deliberately excluded from the canonical
catalog.

```bash
python scripts/generate_set_table_skill_graph.py
```

Build it directly in Python:

```python
from pathlib import Path

from mshab.skills import build_set_table_stack

set_table = build_set_table_stack(
    Path("/root/.maniskill/data/mshab_checkpoints")
)
```

### Execute a graph-selected SetTable sequence and record video

The skill-node sequence comes from the catalog's `execution_plans`, which
`SkillPlanner` generated. The bridge script then copies only scene-specific
grounding (object instance ids, articulations, poses, and build/init configs)
from one official MS-HAB `PlanData`; it does not use the official sequence as
the planner.

The commands below run inside the Docker container described in the
[repository README](../../../README.md#setup-and-installation); it presets
`MS_ASSET_DIR`, `MSHAB_EXPS_DIR`, and the checkpoint mount.

Inspect the 16 graph decisions without loading CUDA or starting the simulator:

```bash
docker compose run --rm -e DRY_RUN=True mshab \
  ./scripts/evaluate_set_table_graph_plan.sh nominal
```

Run that graph-selected plan with the downloaded per-object policies and save
one annotated video:

```bash
docker compose run --rm mshab \
  ./scripts/evaluate_set_table_graph_plan.sh nominal
```

Videos and tensorboard logs go to `MSHAB_EXPS_DIR`; on the host they appear
under:

```text
./mshab_exps/set_table-graph-plan/nominal_plan0_seed0/eval_videos/*.mp4
```

Run the graph's all-primary-failed fallback plan with the generic pick/place
policies:

```bash
docker compose run --rm mshab \
  ./scripts/evaluate_set_table_graph_plan.sh recovery_all_primaries_failed
```

This first-stage runner executes a path already selected from the graph. It
does not yet observe a policy failure mid-episode and call the planner again;
that online `execute -> observe -> re-decide` loop is the next environment
integration milestone. A 16-step episode can therefore end on the first subtask
whose policy fails; `subtask_fail_counts.json` in the run directory records
which index that was.

Because a single rollout can die on one unlucky spawn, `ATTEMPTS` (or an
explicit `SEEDS` list) runs several independent rollouts concurrently, each on
its own seed, and `MAX_PARALLEL` bounds how many simulator processes are alive
at once -- every attempt is a full GPU process, so that is the knob that bounds
VRAM:

```bash
docker compose run --rm -e ATTEMPTS=6 -e MAX_PARALLEL=2 \
  mshab ./scripts/evaluate_set_table_graph_plan.sh nominal
```

Each attempt gets its own run directory (`<RUN_NAME>_seed<N>`), video, and
`console.log`. The runner then prints one summary naming, per attempt, whether
it succeeded and which contract and target it died on:

```text
nominal_plan0_seed0   FAILED  failed at: mshab.set_table.place.024_bowl (024_bowl) x1
nominal_plan0_seed1   SUCCESS
```

It exits non-zero only if some attempt crashed outright or produced no
result; a rollout that merely fails its task still exits 0. With `ATTEMPTS=1`
(the default) the run directory name and the streamed simulator output are
unchanged from before; the exit status is now the runner's own -- 1 on a
crash, instead of the simulator's exit code.

Useful overrides are ordinary environment variables, passed with `-e`:

```bash
docker compose run --rm \
  -e PLAN_INDEX=3 -e SEED=7 -e INFO_ON_VIDEO=True -e RUN_NAME=my_settable_run \
  mshab ./scripts/evaluate_set_table_graph_plan.sh nominal
```

The runner needs a GPU, the ReplicaCAD SetTable assets, and the SetTable RL
checkpoints (see [Download checkpoints](running.md#download-checkpoints)).
It prints the exact output video directory when evaluation finishes.

## Smaller apple starter stack

The repository also includes a smaller manual stack around the downloaded
SetTable apple policies.

```python
from pathlib import Path

from mshab.skills import build_set_table_starter

stack = build_set_table_starter(
    Path("/root/.maniskill/data/mshab_checkpoints")
)

# Layer 1: sub-goal order
print(stack.subgoal_graph.execution_order())

# Layer 2 candidate partial order (not an execution plan)
print(stack.skill_graph.execution_order())

# Layer 3: one skill node grounded to its contract
print(stack.grounded_skills["pick_object_specialized"].preconditions)

# Layer 4: the policies bound to one contract, in preference order
print([p.id for p in stack.library.policies_for("mshab.set_table.pick.013_apple")])
# ['rl.set_table.pick.013_apple', 'rl.set_table.pick.all']
```

The starter Layer-1 sub-goals are:

```text
open(fridge)
    -> holding(013_apple)
    -> at(013_apple,dining_table)
    -> closed(fridge)
```

The main Layer-2 skill-node path is:

```text
navigate_to_source
    -> open_source
    -> navigate_to_object
    -> pick_object_specialized
    -> navigate_to_destination
    -> place_object_specialized
    -> navigate_back_to_source
    -> close_source
```

The nodes that reference the generic `pick.all` and `place.all` contracts are
represented as `IS_A`/`ALTERNATIVE_TO`/`FALLBACK_TO` candidates, not as nested
composite skills.

Layer 1 and Layer 2 can also be built without checkpoints or an environment:

```python
from mshab.skills import build_set_table_apple_graph

subgoals, graph = build_set_table_apple_graph(
    object="013_apple",
    source="fridge",
    destination="dining_table",
)
```

