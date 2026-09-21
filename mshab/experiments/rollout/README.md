# MS-HAB rollout (stage 6)

The controller loop of [`../planning/`](../planning/README.md) on the real
simulator: one official TidyHouse episode, the RL checkpoints as policies,
and a proposer (the rule-based pseudo model, or DeepSeek) planning both upper
layers and replanning when a sub-goal fails. Stage 6 of
[`../docs/higher-layers-plan.md`](../docs/higher-layers-plan.md).

`TaskController`, `ProposalValidator`, `SkillRuntime`, and the granularity
library are reused unchanged. What this package adds is the two objects the
guide listed as simulator follow-ups, an environment they can drive, and a
runner:

```text
episode.py       one official plan as the symbolic scene            stdlib only
proposer.py      TidyHouseRuleProposer, the pseudo model             stdlib only
run.py           python -m mshab.experiments.rollout                 stdlib until the simulator starts
environment.py   SkillRollout-v0 behind MSHabEnvironmentAdapter      torch, ManiSkill
executor.py      CheckpointPolicyExecutor: SAC/PPO checkpoints       torch, ManiSkill
mshab/envs/skill_rollout.py   SkillRollout-v0 itself                 torch, ManiSkill
```

## Run it

Inside the Docker container (the GPU, the ReplicaCAD assets, and the
checkpoint mount are described in the
[repository README](../../../README.md#setup-and-installation)):

```bash
# the rule proposer on plan 6 of the train split, coarse sub-goals, with video
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
6, the first train plan whose five objects are five distinct categories, so
every object name is also its specialised checkpoint's target), `--seed`,
`--attempts-per-node` (2), `--max-replans` (2), `--retries` (the validator's
budget for the model, 2), `--give-up-after` (the rule proposer drops a
transfer after this many replans, 2), `--no-video`, `--output`, and the
DeepSeek transport options of the stage-5 sweep. `--task-plan`,
`--rearrange-root`, and `--checkpoint-root` point at other files than the
official ones under `MS_ASSET_DIR`.

A run writes to `$MSHAB_EXPS_DIR/rollout/<proposer>-<granularity>-plan<N>-<timestamp>/`
(`./mshab_exps/rollout/` on the host):

| File | Contents |
| --- | --- |
| `episode.json` | transfers, entities, goal, goal facts, the plan subtask of every role |
| `tidy_house_plan.json` | the one official plan the environment loaded |
| `trace.json` | the controller's `RunResult`: proposals, decisions, replans, facts, metrics |
| `executions.json` | every policy run: grounding, policy, plan subtask, simulator steps, outcome |
| `summary.json` | status, success, metrics, simulator steps, elapsed time |
| `exchanges.json` | every model call, when DeepSeek was the proposer |
| `videos/*.mp4` | the whole run from the robot's camera, with the checkers on the frame |
| `proposal.json` | `--dry-run` only: the validated proposal and the node -> subtask map |

## The episode (`episode.py`)

An official sequential plan is 20 atomic subtasks, `navigate, pick, navigate,
place` per transfer, with object instance ids (`007_tuna_fish_can-0`) and
place goals (a position and a rectangle on a receptacle). `TidyHouseEpisode.
from_plan` reads one plan and derives:

- **Entities.** Objects are named by category, so `pick(024_bowl)` grounds to
  the `024_bowl` checkpoint; a second instance of the same category is
  `024_bowl_2` and falls back to the `all` checkpoint. Receptacles take their
  scene names from the episode config's `goal_receptacles`
  (`frl_apartment_table_01`, `kitchen_counter`, ...), which
  `read_goal_receptacles` checks against the plan's objects before trusting
  the order; without the config they are `receptacle_1`, `receptacle_2`, ...
  by distinct goal rectangle. Two transfers to the same receptacle share one
  entity.
- **The goal text** pairs every object with its receptacle: *"Tidy the house:
  move 007_tuna_fish_can to frl_apartment_table_01, ..."*. The stage-5 smoke
  test showed that a goal without the pairing leaves a model guessing
  destinations.
- **Goal facts**, one `at(object,receptacle)` per transfer.
- **Grounded node -> plan subtask** (`subtask_for`). The MS-HAB policies
  observe the object and goal of *the subtask the environment is pointed at*,
  so a node is executed by pointing the environment at its subtask first.
  `navigate(object)` is that transfer's first navigation, `navigate(receptacle)`
  the navigation before the place of the transfer whose object is held (else
  the first undelivered one), `pick(object)` and `place(object,receptacle)`
  the transfer's own; a place to a receptacle the plan does not deliver the
  object to, or an `open`/`close`, is `UnsupportedGrounding`, which the
  executor reports as the failure mode `no_matching_subtask` and the
  controller replans around.
- **Facts from measurements** (`facts`). The environment measures every
  object and goal each step (`SceneMeasurements`): `holding(x)` is the grasp
  check, `at(x,r)` the object inside its goal and not grasped (MS-HAB's own
  place success), `reachable(x)` MS-HAB's navigation success for that target
  without the arm terms (an object in the gripper counts as reachable, as it
  stays reachable after `pick` on the symbolic environment; otherwise a fine
  replan asks the navigation policy to reach the object it is carrying),
  `gripper_empty()` when nothing is grasped,
  `collision_safe()` the pointed subtask's cumulative-force limit, and
  `present(...)` for every entity.

## The environment (`mshab/envs/skill_rollout.py`, `environment.py`)

`SequentialTask-v0` is a state machine over its plan: it advances the subtask
pointer when the current subtask's checkers pass, ends the episode on the
horizon or the force limit, and reports whole-task success. Those decisions
belong to the skill runtime here, so `SkillRollout-v0` keeps the plan, the
merged actors, the observations, and the per-subtask checkers, and changes
three things: the pointer moves only through `point_at(index)`; `evaluate()`
reports the pointed subtask's checkers as `subtask_success` and never
advances or ends anything; `scene_measurements()` runs the grasp, place, and
navigation checks for every subtask of the plan, not only the pointed one.

`make_rollout_env` builds the official `make_env` wrapper chain (depth
observations, three stacked frames, the episode recorder, the Fetch action
wrapper, the vector wrapper) around it with one environment, the plan's
navigation arm checks off as in the official runners, and a time limit that
covers every attempt of every node of every replan, because the vector
wrapper resets the scene when the limit is reached. `RolloutEnvironmentAdapter`
is `MSHabEnvironmentAdapter` with the episode's entities, a fact extractor that
turns `scene_measurements()` into `TidyHouseEpisode.facts`, and the fixed seed
the controller's argument-free `reset()` needs.

## The executor (`executor.py`)

`CheckpointPolicyExecutor` is the `PolicyExecutor` for `CheckpointPolicy`
objects. On first use it loads the checkpoint the way `mshab.evaluate` does
(SAC for pick and place, PPO for navigation; `config.yml` is read without the
command line so the runner's own arguments are not parsed as overrides). An
execution is:

```text
index = episode.subtask_for(contract type, grounded arguments, current facts)
environment.point_at(index)
one step without moving              # the observation now describes this subtask
repeat up to contract.max_episode_steps times:
    action = policy(observation); step; monitor(snapshot)
    stop when MS-HAB's checker for the subtask passes
```

Two checks decide success: MS-HAB's own subtask checker ends the policy run
(the arm at rest, the robot still, the object grasped or placed), and
`SkillRuntime` then verifies the contract's effects on the facts, exactly as
on the symbolic environment. The horizon is the contract's (navigation 1000,
manipulation 200 steps), which the data-pipeline note said would replace the
scripts' constants once execution went through the runtime. When the runtime's
invariant monitor stops a skill for breaking the contact-force limit, the
executor clears the force count and refreshes the facts before re-raising, so
the retry is admitted with a clean slate as an MS-HAB subtask would start.
`executions` records every run with its plan subtask and simulator steps,
which the controller's trace does not carry.

Policy choice is the library's: `build_granularity_library(root,
task_families=("tidy_house",))` binds the 21 TidyHouse checkpoints, and
`select_policy(..., arguments=...)` prefers the object's own checkpoint over
`all`. Without the family filter the PrepareGroceries checkpoint of the same
object would come first in binding order.

## The rule proposer (`proposer.py`)

The scripted proposer answers from tables cut out of the gold graphs and has
no answer for a replan it was not given, and a real episode's objects and
receptacles are not in any table. `TidyHouseRuleProposer` plays the pseudo
model with rules over the request instead: `decompose` returns the transfers
whose object is not yet at its destination, a held object first, at the
requested granularity through `TidyHouseGraphBuilder` with the transfers'
original numbers (`object_4_delivered` still names the fourth transfer after
the second was given up); a transfer whose sub-goal failed is kept, so the
next graph retries it, until it has caused `give_up_after` replans.
`plan_subgraph` returns the gold subgraph of that sub-goal, shortened to
`navigate -> place` for a coarse transfer whose object is already held. On
the gold graphs' own transfers it reproduces them exactly (a test checks
this), so a nominal rollout with it executes the gold plan on MS-HAB.

## First runs (2026-09-20)

Rule proposer, RTX 4090, seed 0, attempts 2, replans 2, `give_up_after` 2.
Every run wrote its trace, executions, and video under
`mshab_exps/rollout/`. Simulator time is 200 to 300 steps per second; a run
is one to two minutes including scene loading and checkpoint loading.

| Run | Status | Goal facts | Executions (failed) | Skipped | Replans | Sim steps |
| --- | --- | --- | --- | --- | --- | --- |
| plan 6, coarse | `replans_exhausted` | 0 / 5 | 11 (6) | 0 | 2 | 834 |
| plan 6, fine | `replans_exhausted` | 0 / 5 | 10 (6) | 5 | 2 | 901 |
| plan 10, coarse | `replans_exhausted` | 3 / 5 | 20 (7) | 0 | 2 | 1245 |
| plan 21, coarse | `replans_exhausted` | 2 / 5 | 22 (7) | 0 | 2 | 1532 |
| plan 42, coarse | `proposal_rejected` | 4 / 5 | 28 (6) | 0 | 1 | 1576 |

What the runs showed:

- **The pipeline reproduces the official evaluator.** On plan 6 the first
  three nodes took 84, 36, and 90 steps with the official checkpoints, and
  the `007_tuna_fish_can` place then broke the 7500 cumulative-force limit
  on every attempt. `mshab.evaluate` (`rl_per_obj`, same seed, the run's
  `tidy_house_plan.json` as `task_plan_fp`) fails the same episode at the
  same subtask index 3 after 259 steps. The failure is the policy's.
- **Whole transfers succeed and the recoveries of the stage-4 scenarios
  happen for real.** Plans 10, 21, and 42 delivered two to four objects. On
  plan 21 the gelatin box was dropped by a failed place, the retry was not
  admitted (`holding` missing), the replan navigated back, picked it up
  again, and placed it: the `object_dropped` scenario on the simulator. A
  pick that broke the force limit was retried and succeeded three times.
- **Granularity is visible.** After the first replan of plan 6 the fine
  decomposition absorbed `reachable`, `holding`, and `reachable(table)` and
  skipped their nodes; the coarse one re-ran the navigation (1 step, already
  there). Before a held object counted as reachable, the fine replan sent the
  navigation policy after the can in the robot's own gripper for 644 steps.
- **The naming rules hold.** Plan 42 has two sugar boxes; `004_sugar_box_2`
  was navigated to, picked, and placed with the `all` checkpoints, the other
  one with its own.
- **Where the rules end.** With `give_up_after` 2 and `max_replans` 2 an
  episode has little slack: plan 42's last object was dropped on a place
  timeout, re-picked, and then reported not held one step later; the second
  admission failure counted as the transfer's second replan, the rules had
  nothing left to propose, and the run ended `proposal_rejected` with four
  objects delivered. An admission retry reads the same snapshot as the first
  attempt, so it fails the same way; whether a refresh step belongs between
  attempts is a controller question, not an adapter one.

## Tests

`tests/rollout/test_rollout.py` runs on CPU with the standard library: the episode
from a plan document and an episode config (names, shared receptacles,
duplicate categories, the node -> subtask map, facts from measurements, the
file checks), the rule proposer against the gold graphs and through the
validator (replans, giving up, a held object), the builder's transfer
numbers, and the runner's `--dry-run`. The simulator, the environment
subclass, and the executor are exercised only by the GPU run.
