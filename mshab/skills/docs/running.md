# Installation, checkpoints, and tests

Where the Docker setup lives, how to download the policy checkpoints, and how
to run the CPU-only test suite.

Part of the [MS-HAB skill library guide](../README.md).

## Installation

Docker setup lives in the
[repository README](../../../README.md#setup-and-installation). Inside the
container `MS_ASSET_DIR` is `/root/.maniskill`, and `mshab.evaluate` reads
policies from `$MS_ASSET_DIR/data/mshab_checkpoints`.

## Download checkpoints

The policies are neither in the image nor in the assets volume. They go to a
host directory that `docker-compose.yml` bind-mounts read-only at
`$MS_ASSET_DIR/data/mshab_checkpoints`. Its default is
`/data/mshab/mshab_checkpoints`; override it with `MSHAB_CKPT_DIR`.

```bash
mkdir -p /data/mshab/mshab_checkpoints
```

Download everything -- 16GiB total (`rl` 2.9GB, `bc` 2.3GB, `dp` 12GB). The
HuggingFace repository is public, so no login is needed. `--user` keeps the
files owned by you rather than root:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HOME=/tmp/hf \
  -v /data/mshab/mshab_checkpoints:/out \
  --entrypoint hf mshab:latest \
  download arth-shukla/mshab_checkpoints --local-dir /out
```

Add `--include "rl/set_table/**"` to that command for only the 11 SetTable RL
policies used by the graph runner (~630MB).

`mshab.evaluate` loads *every* policy registered for the task and policy family
before the rollout starts, so a partial download within a family is not enough.
`task=set_table` with `policy_type=rl_*` needs all 11 policies listed under
[Layer 4](layers-3-4.md#layer-4-policies). Expected layout:

```text
$MSHAB_CKPT_DIR/
└── <family>/<task>/<contract-type>/<target>/
    ├── config.yml
    └── policy.pt
```

If the checkpoints live elsewhere, export the override before any
`docker compose` command:

```bash
export MSHAB_CKPT_DIR=/my/path/mshab_checkpoints
```

## Tests

All tests live under `tests/`, one folder per package they test (see
[`tests/README.md`](../../../tests/README.md)). Run the complete CPU-only suite:

```bash
docker compose run --rm mshab python -m unittest discover -s tests -p 'test_*.py' -v
```

Run only the manual-graph -> repeated skill-node-decision test:

```bash
docker compose run --rm mshab python -m unittest tests.skills.test_set_table_graph_decisions -v
```

`tests/skills/test_skill_library_checkpoints.py` validates the 11 downloaded SetTable
policies when the checkpoint directory exists and skips cleanly otherwise.
Tests inspect checkpoint files but do not load policy tensors or create a GPU
simulator. Use the [SetTable evaluation runner](set-table.md#execute-a-graph-selected-settable-sequence-and-record-video)
for full policy rollouts.

The suite has no third-party dependencies, so dropping the `docker compose
run --rm mshab` prefix also works with any local Python 3.9+. The real
model behind the proposer boundary (`mshab/experiments/planning/deepseek.py`)
is the one place that needs a package: the `openai` SDK from the `planning`
extra (`pip install -e ".[planning]"`; the Docker image installs it). Its
tests use a fake transport, so they run without the package, without a key,
and without network access. Running the model itself needs
`DEEPSEEK_API_KEY`, which `docker-compose.yml` passes through from the host
environment or a `.env` file next to it:

```bash
docker compose run --rm mshab python -m mshab.experiments.granularity.evaluate --samples 3
```

Two tests in `test_set_table_graph_decisions.py` reach the official SetTable
task plan through a hard-coded `<repo>/../mshab-assets/...` path instead of
`MS_ASSET_DIR`, so they skip inside the container even though the plan is
present in the assets volume.

## Run the controller on MS-HAB

The stage-6 rollout (`mshab/experiments/rollout/`) executes a proposer's plan
for one official TidyHouse episode with the RL checkpoints, replanning
through the proposer when a sub-goal fails. It needs the GPU, the ReplicaCAD
assets, and the TidyHouse checkpoints (`rl/tidy_house/**`, 21 policies):

```bash
docker compose run --rm mshab python -m mshab.experiments.rollout --granularity coarse
docker compose run --rm mshab python -m mshab.experiments.rollout --proposer deepseek --granularity free
```

Runs land under `./mshab_exps/rollout/` with the trace, the executions, the
plan the environment loaded, and a video. `--dry-run` validates the proposal
and maps its nodes to plan subtasks without a simulator, so it also works
outside the container. See the [rollout README](../../experiments/rollout/README.md).

