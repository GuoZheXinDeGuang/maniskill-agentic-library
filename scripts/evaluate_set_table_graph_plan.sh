#!/usr/bin/env bash
set -euo pipefail

# Execute a SkillCatalog plan in MS-HAB and record one video per attempt.
# Usage: ./scripts/evaluate_set_table_graph_plan.sh [nominal|recovery_all_primaries_failed]
#
# A single rollout is flaky: one unlucky spawn can fail a subtask (typically the
# `place` of the bowl) even when the plan itself is sound.  Set ATTEMPTS (or
# SEEDS) to run several independent rollouts concurrently and get a per-attempt
# summary of which subtask each one died on:
#
#   ATTEMPTS=6 ./scripts/evaluate_set_table_graph_plan.sh
#   SEEDS="0 7 11" MAX_PARALLEL=3 ./scripts/evaluate_set_table_graph_plan.sh
#
# Every attempt is a full GPU simulator process, so MAX_PARALLEL is what bounds
# VRAM, not ATTEMPTS.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSHAB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$MSHAB_ROOT/../.." && pwd)"

EXECUTION_PLAN="${1:-nominal}"
SPLIT="${SPLIT:-train}"
PLAN_INDEX="${PLAN_INDEX:-0}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-1}"
NUM_ENVS="${NUM_ENVS:-1}"
SEED="${SEED:-0}"
ATTEMPTS="${ATTEMPTS:-1}"
SEEDS="${SEEDS:-}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
INFO_ON_VIDEO="${INFO_ON_VIDEO:-True}"
INVISIBLE_GOALS="${INVISIBLE_GOALS:-False}"
DRY_RUN="${DRY_RUN:-False}"

# ---------------------------------------------------------------------------
# Resolve the seeds to attempt.
# ---------------------------------------------------------------------------
if [[ -n "$SEEDS" ]]; then
    read -r -a SEED_LIST <<<"$SEEDS"
else
    if [[ ! "$ATTEMPTS" =~ ^[0-9]+$ ]] || (( ATTEMPTS < 1 )); then
        echo "ATTEMPTS must be a positive integer, got: $ATTEMPTS" >&2
        exit 2
    fi
    SEED_LIST=()
    for (( offset = 0; offset < ATTEMPTS; offset++ )); do
        SEED_LIST+=( $(( SEED + offset )) )
    done
fi
ATTEMPT_COUNT=${#SEED_LIST[@]}
if (( ATTEMPT_COUNT == 0 )); then
    echo "no seeds to run" >&2
    exit 2
fi
for attempt_seed in "${SEED_LIST[@]}"; do
    if [[ ! "$attempt_seed" =~ ^[0-9]+$ ]]; then
        echo "seeds must be non-negative integers, got: $attempt_seed" >&2
        exit 2
    fi
done

if [[ ! "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || (( MAX_PARALLEL < 1 )); then
    echo "MAX_PARALLEL must be a positive integer, got: $MAX_PARALLEL" >&2
    exit 2
fi
(( MAX_PARALLEL > ATTEMPT_COUNT )) && MAX_PARALLEL=$ATTEMPT_COUNT

# A single attempt keeps the historical run name so existing output directories
# stay put; a sweep names the shared prefix and suffixes each attempt's seed.
if (( ATTEMPT_COUNT > 1 )); then
    RUN_NAME="${RUN_NAME:-${EXECUTION_PLAN}_plan${PLAN_INDEX}}"
else
    RUN_NAME="${RUN_NAME:-${EXECUTION_PLAN}_plan${PLAN_INDEX}_seed${SEED_LIST[0]}}"
fi

if [[ ! "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "RUN_NAME may contain only letters, numbers, underscores, and hyphens" >&2
    exit 2
fi

attempt_run_name() {
    if (( ATTEMPT_COUNT > 1 )); then
        echo "${RUN_NAME}_seed${1}"
    else
        echo "$RUN_NAME"
    fi
}

# Docker sets MS_ASSET_DIR=/root/.maniskill; the fallback is ManiSkill's own default.
export MS_ASSET_DIR="${MS_ASSET_DIR:-$HOME/.maniskill}"
# Evaluation outputs (videos, tensorboard). Docker points this inside the
# bind-mounted repo so results land in ./mshab_exps on the host.
MSHAB_EXPS_DIR="${MSHAB_EXPS_DIR:-$WORKSPACE_ROOT/mshab_exps}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mshab-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

CATALOG="$MSHAB_ROOT/mshab/skills/catalogs/set_table.json"
SOURCE_PLAN="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/set_table/sequential/$SPLIT/all.json"
CHAIN_PLAN="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/set_table/custom/graph_${RUN_NAME}.json"
CKPT_DIR="$MS_ASSET_DIR/data/mshab_checkpoints/rl/set_table"

for required in "$CATALOG" "$SOURCE_PLAN"; do
    if [[ ! -f "$required" ]]; then
        echo "missing required file: $required" >&2
        exit 1
    fi
done

# The grounded plan is seed-independent, so build it once and share it across
# attempts (read-only during evaluation).
python "$SCRIPT_DIR/build_set_table_graph_plan.py" \
    "$SOURCE_PLAN" \
    "$CHAIN_PLAN" \
    --catalog "$CATALOG" \
    --execution-plan "$EXECUTION_PLAN" \
    --source-plan-index "$PLAN_INDEX"

if [[ "$DRY_RUN" == "True" ]]; then
    echo "dry run complete; grounded graph plan: $CHAIN_PLAN"
    exit 0
fi

if ! python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA is not visible to PyTorch; the official MS-HAB GPU backend cannot run." >&2
    exit 1
fi
if [[ ! -d "$CKPT_DIR" ]]; then
    echo "missing SetTable checkpoints: $CKPT_DIR" >&2
    exit 1
fi

POLICY_TYPE="${POLICY_TYPE:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selection"]["recommended_policy_type"])' "$CHAIN_PLAN")}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selection"]["estimated_max_episode_steps"])' "$CHAIN_PLAN")}"

cd "$MSHAB_ROOT"

# ---------------------------------------------------------------------------
# Run the attempts, at most MAX_PARALLEL simulator processes at a time.
# ---------------------------------------------------------------------------
run_attempt() {
    local attempt_seed="$1"
    local run_name="$2"
    local exp_dir="$MSHAB_EXPS_DIR/set_table-graph-plan/$run_name"
    local status=0

    mkdir -p "$exp_dir"
    # Drop results from a previous run of the same name so the summary below
    # never reports stale numbers for an attempt that crashed early.
    rm -f "$exp_dir/exit_status" "$exp_dir/output.txt" "$exp_dir/subtask_fail_counts.json"

    local -a eval_cmd=(
        python -m mshab.evaluate configs/evaluate.yml
        seed="$attempt_seed"
        task=set_table
        policy_type="$POLICY_TYPE"
        save_trajectory=False
        max_trajectories="$MAX_TRAJECTORIES"
        eval_env.env_id=SequentialTask-v0
        eval_env.task_plan_fp="$CHAIN_PLAN"
        eval_env.spawn_data_fp=null
        eval_env.num_envs="$NUM_ENVS"
        eval_env.max_episode_steps="$MAX_EPISODE_STEPS"
        eval_env.continuous_task=False
        eval_env.frame_stack=3
        eval_env.stack=null
        eval_env.record_video=True
        eval_env.info_on_video="$INFO_ON_VIDEO"
        eval_env.save_video_freq=1
        eval_env.extra_stat_keys='<list>success, fail, subtask, subtask_type, subtasks_steps_left, robot_force, robot_cumulative_force</list>'
        eval_env.env_kwargs.invisible_goals_in_human_render="$INVISIBLE_GOALS"
        eval_env.env_kwargs.task_cfgs.navigate.ignore_arm_checkers=True
        logger.workspace="$MSHAB_EXPS_DIR"
        logger.exp_name="set_table-graph-plan/$run_name"
        logger.clear_out=False
        logger.tensorboard=True
        logger.wandb=False
    )

    export SAPIEN_NO_DISPLAY=1
    set +e
    if (( ATTEMPT_COUNT == 1 )); then
        # A lone attempt keeps the simulator's progress bar on the terminal.
        "${eval_cmd[@]}" 2>&1 | tee "$exp_dir/console.log"
        status=${PIPESTATUS[0]}
    else
        "${eval_cmd[@]}" >"$exp_dir/console.log" 2>&1
        status=$?
    fi
    set -e

    echo "$status" >"$exp_dir/exit_status"
    return 0
}

echo "running $ATTEMPT_COUNT attempt(s) (seeds: ${SEED_LIST[*]}) with up to $MAX_PARALLEL in parallel"

ATTEMPT_NAMES=()
for attempt_seed in "${SEED_LIST[@]}"; do
    run_name="$(attempt_run_name "$attempt_seed")"
    ATTEMPT_NAMES+=( "$run_name" )
    while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
        wait -n || true
    done
    echo "  -> launching seed=$attempt_seed run_name=$run_name"
    run_attempt "$attempt_seed" "$run_name" &
done
wait

# ---------------------------------------------------------------------------
# Summarise: success, and which subtask each failed attempt died on.
# ---------------------------------------------------------------------------
python - "$CHAIN_PLAN" "$MSHAB_EXPS_DIR/set_table-graph-plan" "${ATTEMPT_NAMES[@]}" <<'PY'
import json
import re
import sys
from pathlib import Path

chain_plan = Path(sys.argv[1])
root = Path(sys.argv[2])
run_names = sys.argv[3:]

decisions = json.loads(chain_plan.read_text())["selection"]["skill_decisions"]


def label(index):
    try:
        decision = decisions[int(index)]
    except (IndexError, ValueError):
        return "subtask {}".format(index)
    return "{} ({})".format(decision["contract_id"], decision["target"])


def success_of(exp_dir):
    output = exp_dir / "output.txt"
    if not output.is_file():
        return None
    match = re.search(r"'success_once':\s*tensor\(([0-9.eE+-]+)", output.read_text())
    if match is None:
        return None
    return float(match.group(1))


print("")
print("attempt summary")
print("-" * 78)
crashed = 0
succeeded = 0
for run_name in run_names:
    exp_dir = root / run_name
    status_file = exp_dir / "exit_status"
    status = status_file.read_text().strip() if status_file.is_file() else "?"
    success = success_of(exp_dir)

    if status != "0":
        verdict = "CRASHED (exit {})".format(status)
        crashed += 1
    elif success is None:
        verdict = "NO RESULT"
        crashed += 1
    elif success > 0:
        verdict = "SUCCESS"
        succeeded += 1
    else:
        verdict = "FAILED"

    detail = ""
    fail_counts = exp_dir / "subtask_fail_counts.json"
    if verdict == "FAILED" and fail_counts.is_file():
        counts = json.loads(fail_counts.read_text())
        detail = "  failed at: " + ", ".join(
            "{} x{}".format(label(k), v) for k, v in sorted(counts.items(), key=lambda kv: int(kv[0]))
        )
    print("{:<48} {}{}".format(run_name, verdict, detail))
    print("{:<48}   log: {}".format("", exp_dir / "console.log"))
    if (exp_dir / "eval_videos").is_dir():
        print("{:<48}   video: {}".format("", exp_dir / "eval_videos"))

print("-" * 78)
print(
    "{}/{} succeeded, {} crashed".format(succeeded, len(run_names), crashed)
)
sys.exit(1 if crashed else 0)
PY
