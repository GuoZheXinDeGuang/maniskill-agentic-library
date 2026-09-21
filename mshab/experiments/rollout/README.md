# MS-HAB rollout (stage 6)

The controller loop of [`../planning/`](../planning/README.md) on the real
simulator: one official SetTable episode, the RL checkpoints as policies,
and a proposer (the rule-based pseudo model, or DeepSeek) planning both upper
layers and replanning when a sub-goal fails. Stage 6 of
[`../docs/higher-layers-plan.md`](../docs/higher-layers-plan.md).

`TaskController`, `ProposalValidator`, `SkillRuntime`, and the granularity
library are reused unchanged. What this package adds is the two objects the
guide listed as simulator follow-ups, an environment they can drive, and a
runner:

```text
episode.py       one official plan as the symbolic scene            stdlib only
proposer.py      SetTableRuleProposer, the pseudo model              stdlib only
run.py           python -m mshab.experiments.rollout                 stdlib until the simulator starts
environment.py   SkillRollout-v0 behind MSHabEnvironmentAdapter      torch, ManiSkill
executor.py      CheckpointPolicyExecutor: SAC/PPO checkpoints       torch, ManiSkill
mshab/envs/skill_rollout.py   SkillRollout-v0 itself                 torch, ManiSkill
```

Until 2026-09-21 this package rolled out TidyHouse episodes; the first GPU
runs of that version are recorded in the README of commit `d7fa8de`. The
SetTable version below has not been run on the GPU yet: the CPU tests cover
the episode, the facts, the node -> subtask map, the rule proposer, and the
dry run, and the simulator path changed only where SetTable needs it (the
articulation state as facts).

## Run it

Inside the Docker container (the GPU, the ReplicaCAD assets, and the
checkpoint mount are described in the
[repository README](../../../README.md#setup-and-installation)):

```bash
# the rule proposer on plan 0 of the train split, coarse sub-goals, with video
docker compose run --rm mshab python -m mshab.experiments.rollout --granularity coarse

# fine sub-goals, another episode, no video
docker compose run --rm mshab python -m mshab.experiments.rollout --granularity fine --plan-index 42 --no-video

# the real model choosing its own decomposition (DEEPSEEK_API_KEY from the host or .env)
docker compose run --rm mshab python -m mshab.experiments.rollout --proposer deepseek --granularity free

# no simulator: the episode, the validated proposal, and every node's plan subtask
python -m mshab.experiments.rollout --dry-run --granularity coarse
```

Options: `--proposer rule|deepseek`, `--granularity free|coarse|fine` (the
rule proposer answers `coarse` and `fine`), `--split`, `--plan-index` (default
0; every official SetTable plan moves one bowl and one apple), `--seed`,
`--attempts-per-node` (2), `--max-replans` (2), `--retries` (the validator's
budget for the model, 2), `--give-up-after` (the rule proposer gives an
object up after its segment caused this many replans, 2), `--no-video`,
`--output`, and the DeepSeek transport options of the stage-5 sweep.
`--task-plan`, `--rearrange-root`, and `--checkpoint-root` point at other
files than the official ones under `MS_ASSET_DIR`.

A run writes to `$MSHAB_EXPS_DIR/rollout/<proposer>-<granularity>-plan<N>-<timestamp>/`
(`./mshab_exps/rollout/` on the host):

| File | Contents |
| --- | --- |
| `episode.json` | segments, entities, goal, goal facts, the plan subtask of every role |
| `set_table_plan.json` | the one official plan the environment loaded |
| `trace.json` | the controller's `RunResult`: proposals, decisions, replans, facts, metrics |
| `executions.json` | every policy run: grounding, policy, plan subtask, simulator steps, outcome |
| `summary.json` | status, success, metrics, simulator steps, elapsed time |
| `exchanges.json` | every model call, when DeepSeek was the proposer |
| `videos/*.mp4` | the whole run from the robot's camera, with the checkers on the frame |
| `proposal.json` | `--dry-run` only: the validated proposal and the node -> subtask map |

## The episode (`episode.py`)

An official sequential plan is 16 atomic subtasks: per object a *segment*
`navigate, open, navigate, pick, navigate, place, navigate, close`, with the
object instance id (`024_bowl-0`), the articulation the object is in
(`articulation_type` `kitchen_counter` or `fridge`, `articulation_id`
`kitchen_counter-0`), and the place goal (a position and a rectangle on the
table). `SetTableEpisode.from_plan` reads one plan and derives:

- **Entities.** Objects are named by category, so `pick(024_bowl)` grounds to
  the `024_bowl` checkpoint; a second instance of the same category is
  `024_bowl_2` and falls back to the `all` checkpoint. Storages are named by
  articulation type (`kitchen_counter`, `fridge`; a second fridge would be
  `fridge_2`), which is also the target of their open and close checkpoints.
  Receptacles take their scene names from the episode config's
  `goal_receptacles` (`frl_apartment_table_02`, ...), which
  `read_goal_receptacles` checks against the plan's objects before trusting
  the order; without the config they are `receptacle_1`, `receptacle_2`, ...
  by distinct goal rectangle. Segments are labelled by the object with its
  category prefix dropped (`bowl`, `apple`), so an official episode gets the
  gold graphs' own sub-goal and node ids.
- **The goal text** names the results only: *"Set the table: put 024_bowl and
  013_apple on frl_apartment_table_02, then close the storage they came
  from."* It does not say which storage holds which object; a proposer that
  opens the wrong one has its pick time out and replans.
- **Goal facts**: one `at(object,receptacle)` per segment, one
  `closed(storage)` per storage.
- **Grounded node -> plan subtask** (`subtask_for`). The MS-HAB policies
  observe the object, articulation, and goal of *the subtask the environment
  is pointed at*, so a node is executed by pointing the environment at its
  subtask first. `pick(object)` and `place(object,receptacle)` are the
  segment's own (a place to a receptacle the plan does not deliver the object
  to is `UnsupportedGrounding`, which the executor reports as the failure
  mode `no_matching_subtask` and the controller replans around);
  `open(storage)` and `close(storage)` the segment's that still needs the
  storage; `navigate(object)` the navigation before its pick;
  `navigate(storage)` the navigation before the open while the storage is
  closed and the one before the close once it stands open;
  `navigate(receptacle)` the navigation before the place of the segment whose
  object is held, else the first not yet delivered.
- **Facts from measurements** (`facts`). The environment measures every
  object, goal, and articulation each step (`SceneMeasurements`):
  `holding(x)` is the grasp check, `at(x,r)` the object inside its goal and
  not grasped (MS-HAB's own place success), `reachable(x)` MS-HAB's
  navigation success for that target without the arm terms (an object in the
  gripper counts as reachable, as it stays reachable after `pick` on the
  symbolic environment), `closed(storage)` the close checker's joint term and
  `open(storage)` the open checker's, with a storage that is measured and not
  closed counting as open (a half-open drawer can be closed but not opened),
  `gripper_empty()` when nothing is grasped, `collision_safe()` the pointed
  subtask's cumulative-force limit, and `present(...)` for every entity.

## The environment (`mshab/envs/skill_rollout.py`, `environment.py`)

`SequentialTask-v0` is a state machine over its plan: it advances the subtask
pointer when the current subtask's checkers pass, ends the episode on the
horizon or the force limit, and reports whole-task success. Those decisions
belong to the skill runtime here, so `SkillRollout-v0` keeps the plan, the
merged actors and articulations, the observations, and the per-subtask
checkers, and changes three things: the pointer moves only through
`point_at(index)`; `evaluate()` reports the pointed subtask's checkers as
`subtask_success` and never advances or ends anything; `scene_measurements()`
runs the grasp, place, navigation, and articulation-joint checks for every
subtask of the plan, not only the pointed one.

`make_rollout_env` builds the official `make_env` wrapper chain (depth
observations, three stacked frames, the episode recorder, the Fetch action
wrapper, the vector wrapper) around it with one environment, the plan's
navigation arm checks off as in the official runners, and a time limit that
covers every attempt of every node of every replan, because the vector
wrapper resets the scene when the limit is reached. `RolloutEnvironmentAdapter`
is `MSHabEnvironmentAdapter` with the episode's entities, a fact extractor that
turns `scene_measurements()` into `SetTableEpisode.facts`, and the fixed seed
the controller's argument-free `reset()` needs.

## The executor (`executor.py`)

`CheckpointPolicyExecutor` is the `PolicyExecutor` for `CheckpointPolicy`
objects. On first use it loads the checkpoint the way `mshab.evaluate` does
(SAC for pick, place, open, and close, PPO for navigation; `config.yml` is
read without the command line so the runner's own arguments are not parsed
as overrides). An execution is:

```text
index = episode.subtask_for(contract type, grounded arguments, current facts)
environment.point_at(index)
one step without moving              # the observation now describes this subtask
repeat up to contract.max_episode_steps times:
    action = policy(observation); step; monitor(snapshot)
    stop when MS-HAB's checker for the subtask passes
```

Two checks decide success: MS-HAB's own subtask checker ends the policy run
(the arm at rest, the robot still, the object grasped or placed, the handle
joint past its threshold), and `SkillRuntime` then verifies the contract's
effects on the facts, exactly as on the symbolic environment. The horizon is
the contract's (navigation 1000, manipulation 200 steps). When the runtime's
invariant monitor stops a skill for breaking the contact-force limit, the
executor clears the force count and refreshes the facts before re-raising, so
the retry is admitted with a clean slate as an MS-HAB subtask would start.
`executions` records every run with its plan subtask and simulator steps,
which the controller's trace does not carry.

Policy choice is the library's: `build_granularity_library(root,
task_families=("set_table",))` binds the 11 SetTable checkpoints, and
`select_policy(..., arguments=...)` prefers the object's own checkpoint over
`all`. Without the family filter the PrepareGroceries checkpoint of the same
object would come first in binding order.

## The rule proposer (`proposer.py`)

The scripted proposer answers from tables cut out of the gold graphs and has
no answer for a replan it was not given, and a real episode's names are in no
table. `SetTableRuleProposer` plays the pseudo model with rules over the
request instead: `decompose` returns one segment per object not yet on the
table, a held object first, at the requested granularity through
`SetTableGraphBuilder`, with the steps the facts leave open (`remaining_steps`:
open only while the storage is closed, pick only while the object is not
held, close whenever the storage was or will be open); a segment whose
sub-goal failed is kept, so the next graph retries it, until it has caused
`give_up_after` replans, after which only the closing of its storage remains.
`plan_subgraph` returns the gold subgraph of that sub-goal for the same
steps. Unlike a real model the rules know the true scene, the storage each
object is in included. On the gold graphs' own segments the rules reproduce
them exactly (a test checks this), so a nominal rollout with them executes
the gold plan on MS-HAB.

## Tests

`tests/rollout/test_rollout.py` runs on CPU with the standard library: the
episode from a plan document and an episode config (names, a shared storage,
duplicate categories, the node -> subtask map, facts from measurements, the
file checks), the rule proposer against the gold graphs and through the
validator (replans, giving up, a held object), the builder's steps, and the
runner's `--dry-run`. The simulator, the environment subclass, and the
executor are exercised only by the GPU run.
