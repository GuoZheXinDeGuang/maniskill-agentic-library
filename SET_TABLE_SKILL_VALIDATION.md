# SetTable single-skill validation

Environment fixed for the first milestone:

- task: `set_table`
- robot: Fetch (MS-HAB default)
- policy: official per-object RL checkpoint
- simulator: ManiSkill 3.0.0b18 / SAPIEN 3.0.0b1
- Python: 3.11

Activate the isolated environment and run one skill at a time:

```bash
source ../mshab-venv/bin/activate
./scripts/evaluate_set_table_skill.sh navigate
./scripts/evaluate_set_table_skill.sh open fridge
./scripts/evaluate_set_table_skill.sh pick 013_apple
./scripts/evaluate_set_table_skill.sh place 013_apple
./scripts/evaluate_set_table_skill.sh close fridge
```

The runner performs a CUDA preflight check because the official evaluation
configuration uses ManiSkill's GPU simulation backend. Verify `nvidia-smi`
works in the same shell before starting a rollout.

Use `MAX_TRAJECTORIES` and `NUM_ENVS` to scale beyond the default smoke test:

```bash
MAX_TRAJECTORIES=100 NUM_ENVS=21 ./scripts/evaluate_set_table_skill.sh pick 013_apple
```

Record a rollout video (keep `NUM_ENVS=1` for a readable smoke test):

```bash
RECORD_VIDEO=True MAX_TRAJECTORIES=1 NUM_ENVS=1 \
  ./scripts/evaluate_set_table_skill.sh pick 013_apple
```

Set `INFO_ON_VIDEO=True` only when the green per-step diagnostic overlay is
needed:

```bash
RECORD_VIDEO=True INFO_ON_VIDEO=True MAX_TRAJECTORIES=1 NUM_ENVS=1 \
  ./scripts/evaluate_set_table_skill.sh pick 013_apple
```

Videos and evaluation logs are written below
`../../mshab_exps/set_table_single_skill/<skill>/<object>/`.

Record one row per run:

| skill | object | episodes | success rate | mean steps | failure states | observation shape | action shape | checkpoint/config | termination condition |
|---|---|---:|---:|---:|---|---|---|---|---|
| navigate | all | | | | | | | `rl/set_table/navigate/all` | `success` |
| open | fridge | | | | | | | `rl/set_table/open/fridge` | `articulation_open` |
| pick | 013_apple | | | | | | | `rl/set_table/pick/013_apple` | `is_grasped` |
| place | 013_apple | | | | | | | `rl/set_table/place/013_apple` | `obj_at_goal` |
| close | fridge | | | | | | | `rl/set_table/close/fridge` | `articulation_closed` |

Do not mark a skill validated from video alone. A validated row needs a completed
evaluation run and the environment's programmatic success signal.

## First composed chain

The apple starts inside a closed fridge, so the minimal physically valid chain
includes Open and a second navigation:

```text
Navigate(fridge) -> Open(fridge) -> Navigate(apple)
-> Pick(apple) -> Navigate(table) -> Place(apple)
```

Run one continuous rollout with automatic checkpoint switching:

```bash
RECORD_VIDEO=True INFO_ON_VIDEO=True MAX_TRAJECTORIES=1 NUM_ENVS=1 \
  ./scripts/evaluate_set_table_apple_chain.sh
```

For a clean video, use `INFO_ON_VIDEO=False INVISIBLE_GOALS=True`.

The general runner defaults to permissive composition mode
(`CONTINUOUS_TASK=True`) so later skill handoffs remain observable after a
subtask timeout. For strict evaluation, set `CONTINUOUS_TASK=False`; completion
in permissive mode is not a valid strict task success if an earlier `fail`
signal occurred.

## General skill-chain runner

```bash
./scripts/evaluate_skill_chain.sh TASK CHAIN_NAME SELECTION
```

`SELECTION` addresses subtasks in an official sequential TaskPlan, either as a
half-open range or as increasing comma-separated indices. Examples:

```bash
# Apple segment from SetTable plan 0
EXPECTED_TYPES=navigate,open,navigate,pick,navigate,place \
  ./scripts/evaluate_skill_chain.sh set_table apple_to_table 8:14

# Select explicit nodes instead of a contiguous range
./scripts/evaluate_skill_chain.sh set_table selected_nodes 8,9,10,11,12,13

# Use another official episode/scene
PLAN_INDEX=12 ./scripts/evaluate_skill_chain.sh set_table apple_scene_12 8:14
```

The generated plan prints every selected skill and grounded target before
execution. A syntactically valid slice is not necessarily physically valid:
the user/planner must preserve prerequisites such as Open before picking an
object from a closed fridge.
