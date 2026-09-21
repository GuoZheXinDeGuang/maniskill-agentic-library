# Data pipeline: how a SetTable run is produced

What object holds the data at each step of an actual SetTable run, what it
hands to the next one, and what is dropped at every boundary.

Part of the [MS-HAB skill library guide](../README.md).

Running SetTable today is a three-stage pipeline, and only the last stage
touches a simulator. The graph is used as a **plan compiler**: Layers 1-4 are
evaluated ahead of time, frozen into JSON, and MS-HAB's own evaluator replays
the result. See [What is not on this path yet](#what-is-not-on-this-path-yet)
for the runtime objects this leaves unused.

```text
A. authoring            stdlib only, no simulator
   SetTableGraphBuilder ─patch─> SubGoalGraph + SkillGraph
   build_set_table_library ────> ContractLibrary (contracts, policies, bindings)
   SkillGrounder ──────────────> GroundedSkill per node
   SkillPlanner ───────────────> SkillPlan (nominal, recovery)
   LibraryCatalog.as_dict ─────> mshab/skills/catalogs/set_table.json
                                          │
B. grounding            stdlib only, no simulator
   official .../set_table/sequential/train/all.json
                                          │
   build_set_table_graph_plan.py ─────────┴─> <MS_ASSET_DIR>/.../custom/
                                              graph_<RUN_NAME>.json
                                          │
C. execution            GPU, official MS-HAB stack
   evaluate_set_table_graph_plan.sh ──────┴─> python -m mshab.evaluate
   plan_data_from_file ─> PlanData/TaskPlan/Subtask ─> SequentialTask-v0
   <ckpt>/rl/set_table/<type>/<target>/policy.pt ───> per-step actions
                                              └─> eval_videos/, output.txt,
                                                  subtask_fail_counts.json
```

## Stage A: authoring the catalog

Entry point: `scripts/generate_set_table_skill_graph.py`, which calls
`build_set_table_stack(checkpoint_root)`
([tasks/set_table/stack.py](../tasks/set_table/stack.py)).

| Step | Producer | Object handed on | What it carries |
| --- | --- | --- | --- |
| 1 | `SetTableGraphBuilder.propose()` | `SkillGraphPatch` | Sub-goals, sub-goal dependencies, four subgraphs per object, internal edges, cross-subgraph edges. Pure data, no simulator types |
| 2 | `SkillGraphBuilder.build()` | `SubGoalGraph`, `SkillGraph` | The patch applied atomically to two fresh aggregates on staging copies |
| 3 | `build_set_table_library()` | `ContractLibrary` | 11 `Contract` objects from the hard-coded manifest, 11 `CheckpointPolicy`, 11 own bindings plus 4 generic ones |
| 4 | `SkillGrounder.ground(node)` | `GroundedSkill` per node | `contract.bind(node.arguments)`: predicate templates formatted into concrete predicate strings |
| 5 | `SkillPlanner.plan()` | `SkillPlan` | `selections` (one achiever per sub-goal) and `order` (16 node ids) |
| 6 | `LibraryCatalog.as_dict()` | `set_table.json` | All four layers plus `execution_plans` and a `summary` |

Three details of that chain are easy to miss:

- A `SkillNode` carries only `id`, `contract_id`, `arguments`, and `achieves`.
  The checkpoint root reaches the library but never the graph.
- Grounding fills the target parameter itself when a node leaves it out: a
  node on a specialized contract may pass `{}`, and `Contract.bind()` injects
  the contract's own target, which is why `close_apple_source` grounds to
  `closed(fridge)` with empty arguments.
- The planner runs **here**, not at execution time. `plan()` is the decision
  loop rolled forward with an empty environment, and both the nominal path and
  the all-primaries-failed recovery path are stored in the catalog.

What the JSON boundary drops: every Python object, the policies'
`ready/missing/partial` status, and absolute checkpoint paths (only the path
relative to `checkpoint_root` is kept). That is what makes the catalog
reproducible on a clean clone.

## Stage B: grounding the plan against one official episode

`scripts/build_set_table_graph_plan.py` reads two JSON files and writes one.
It never imports Torch or ManiSkill.

| Output field | Comes from |
| --- | --- |
| `dataset` | official `all.json` |
| `plans[0].subtasks` | official plan, copied verbatim (`obj_id`, `articulation_type`, goal poses) |
| `plans[0].build_config_name`, `init_config_name` | official plan |
| `selection.execution_plan` | CLI argument (`nominal` / `recovery_all_primaries_failed`) |
| `selection.skill_decisions[i].node_id`, `.contract_id` | catalog plan order and `skill_graph.nodes` |
| `selection.skill_decisions[i].target` | `node.arguments[<target parameter>]`, else the contract record's `target` |
| `selection.recommended_policy_type` | `rl_all_obj` if any selected pick/place contract has target `all`, else `rl_per_obj` |
| `selection.estimated_max_episode_steps` | `SUBTASK_HORIZONS`, a constant in this script |

The two inputs meet exactly once, positionally. For each index the script
checks that the node's contract type equals the source subtask's `type`, that
a pick/place subtask's `obj_id` starts with the node's target, and that an
open/close subtask's `articulation_type` equals it; a length mismatch aborts.

Note what this does *not* do: the subtasks are copied in their original order,
so the graph cannot reorder or drop a step. Its plan is **checked against**
the official sequence, not substituted for it. The nominal and recovery plans
have identical step types at identical positions and differ only in four node
identities (`*_specialized` vs `*_generic`), which reach the simulator as a
single flag, `recommended_policy_type`.

So the division of labour today is: **the official plan supplies the sequence
and the scene grounding; the graph supplies node identity and one consistency
check.** Nothing else from Layers 3 and 4 travels further — MS-HAB ignores the
whole `selection` block, which exists so the shell runner and a human can map
subtask indices back to graph nodes.

## Stage C: execution

`scripts/evaluate_set_table_graph_plan.sh` reads two values back out of the
file it just produced — `selection.recommended_policy_type` and
`selection.estimated_max_episode_steps` — and passes them to
`python -m mshab.evaluate` as `policy_type=` and `eval_env.max_episode_steps=`.
From there the data is entirely in official MS-HAB objects:

| Step | Object | Note |
| --- | --- | --- |
| `make_env()` | `EnvConfig` | `task_plan_fp` points at `graph_<RUN_NAME>.json` |
| `plan_data_from_file()` | `PlanData` → `TaskPlan` → `PickSubtask`/`PlaceSubtask`/`NavigateSubtask`/`OpenSubtask`/`CloseSubtask` | `dataset` becomes `scene_builder_cls` |
| `gym.make("SequentialTask-v0")` | `SequentialTaskEnv` | One sequential env for all 16 subtasks; the per-contract `env_id`s in the catalog (`PickSubtaskTrain-v0`, ...) are **not** used here |
| policy loading | `get_policy_act_fn` per `<family>/<task>/<type>/<target>` | Driven by `POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS` and the directory layout; the whole family is loaded before the rollout |
| `act(obs)` | action tensor | `uenv.subtask_pointer` gives the current subtask type; for per-object types the SAPIEN object name is matched to a target name, and that target's policy fills its slice of the batch |

Outputs land in `$MSHAB_EXPS_DIR/set_table-graph-plan/<run_name>/`:
`eval_videos/*.mp4`, `output.txt` (the results dict the runner's summary greps
`success_once` out of), `subtask_fail_counts.json`, tensorboard logs, plus the
runner's own `console.log` and `exit_status`.

The return path to the graph is one integer: a key `i` in
`subtask_fail_counts.json` is the index into `selection.skill_decisions`, which
is how the attempt summary can print the contract and target a rollout died on.

## What is not on this path yet

| Object | Intended role | Why it is idle today |
| --- | --- | --- |
| `SkillRuntime` | Admit a node, monitor invariants, dispatch a policy | Nothing calls it in a rollout; MS-HAB's own subtask success checks stand in |
| `GroundedSkill` predicates | Preconditions, effects, verification against live facts | Generated in stage A and serialized; never evaluated against a snapshot |
| `ContractLibrary` bindings, `select_policy()` | Choose which policy executes a contract | `mshab.evaluate` selects by directory convention plus `policy_type` |
| `PolicyExecutor` | Load a checkpoint and step the environment | `mshab.evaluate` has its own `act()` loop |
| `EnvironmentAdapter`, `EnvironmentSnapshot` | Entity resolution and fact extraction | No SetTable entity/fact extractor is wired up |
| `SkillPlanner` at runtime | Re-decide after each node | Runs only in stage A; the sequence is frozen into the catalog |

That is the SetTable path. The online `execute -> observe -> re-decide` loop
exists: `TaskController` in `mshab/experiments/planning/` runs it on a
symbolic environment, and `mshab/experiments/rollout/` (stage 6 of the
[higher-layers plan](../../experiments/docs/higher-layers-plan.md)) runs it
on MS-HAB for TidyHouse, where every object in the table above is on the
path: the adapter extracts facts from the environment's own checkers, the
runtime admits and verifies each node, the library selects the checkpoint,
the executor steps it, and the planner decides after every node. Bringing
the packaged SetTable graph onto that path needs a SetTable episode (its
articulations included) in place of the TidyHouse one.

## One fact, two sources

Two values are currently defined in both the contract layer and the scripts,
and the scripts win:

- **Episode horizon.** `NavigateContract.max_episode_steps` is 1000 in the
  catalog, while `SUBTASK_HORIZONS["navigate"]` in the bridge script is 500.
  The 16-step nominal plan is therefore run with a 5600-step budget; the
  contracts imply 9600.
- **Policy choice.** The catalog records 15 contract-policy bindings in
  preference order, but execution picks policies from
  `POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS` and the single `policy_type` flag.

Both disappear once stage C goes through `SkillRuntime`, which reads the
horizon off the contract and the policy off the library; the TidyHouse
rollout does exactly that.
