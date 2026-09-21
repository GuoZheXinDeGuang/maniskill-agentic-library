"""Roll a proposer's plan out on MS-HAB: the stage-6 command line.

    python -m mshab.experiments.rollout --granularity coarse                 # rule proposer, plan 0
    python -m mshab.experiments.rollout --granularity fine --plan-index 42
    python -m mshab.experiments.rollout --proposer deepseek --granularity free
    python -m mshab.experiments.rollout --dry-run                            # no simulator

One run is one official SetTable episode: the plan is read, the episode's
entities and goal are derived, the proposer is asked through the validator,
and ``TaskController`` executes the graph on ``SkillRollout-v0`` with the RL
checkpoints, replanning through the same proposer when a sub-goal fails.
The run directory holds ``episode.json``, the one-plan ``set_table_plan.json``
the environment loaded, ``trace.json`` (the controller's ``RunResult``),
``executions.json`` (every policy run with its subtask and simulator
steps), ``summary.json``, the model's ``exchanges.json`` when one was used,
and ``videos/`` unless recording is off.  ``--dry-run`` stops after the
validated proposal and writes ``proposal.json`` with every node's plan
subtask instead; it needs neither a GPU nor torch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from mshab.experiments.granularity.higher_layers.builders import GOLD_GRANULARITIES
from mshab.experiments.granularity.lower_layers.library import (
    EXPERIMENT_TASK,
    build_granularity_library,
)
from mshab.experiments.granularity.paths import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_REARRANGE_ROOT,
    DEFAULT_ROLLOUT_DIR,
)
from mshab.experiments.planning.controller import (
    DEFAULT_ATTEMPTS_PER_NODE,
    DEFAULT_MAX_REPLANS,
    Outcome,
    RunResult,
    TaskController,
)
from mshab.experiments.planning.deepseek import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS,
    DeepSeekProposer,
)
from mshab.experiments.planning.documents import GRANULARITIES, PlanningContext
from mshab.experiments.planning.proposer import GraphProposer, ProposerUnavailable
from mshab.experiments.planning.validator import ProposalRejected, ProposalValidator
from mshab.experiments.rollout.episode import (
    SEGMENT_SUBTASK_TYPES,
    SetTableEpisode,
    UnsupportedGrounding,
    load_episode,
    sequential_plan_path,
)
from mshab.experiments.rollout.proposer import (
    DEFAULT_GIVE_UP_AFTER,
    SetTableRuleProposer,
)
from mshab.skills.library import ContractLibrary


ROLLOUT_SCHEMA_VERSION = "mshab.rollout-summary.v1"
TASK_FAMILY = "set_table"
#: Every official SetTable plan moves one bowl and one apple, so any plan
#: names the specialised checkpoints; the first one is the default.
DEFAULT_PLAN_INDEX = 0
#: The validator's retry budget for the real model, as in the stage-5 sweep.
DEFAULT_MODEL_RETRIES = 2
PROPOSERS = ("rule", "deepseek")
PLAN_FILE_NAME = "set_table_plan.json"


def nominal_facts(episode: SetTableEpisode) -> tuple:
    """The facts the episode starts from when nothing has happened yet.

    Every storage is closed; where the objects are is not a fact.
    """

    facts = {"present({})".format(entity.name) for entity in episode.entities()}
    facts |= {"closed({})".format(source) for source in episode.sources}
    facts |= {"gripper_empty()", "collision_safe()"}
    return tuple(sorted(facts))


def step_budget(
    library: ContractLibrary,
    episode: SetTableEpisode,
    attempts_per_node: int,
    max_replans: int,
) -> int:
    """Simulator steps the whole run can take at most: the time limit to set.

    Every node runs for its contract's horizon plus the refresh step and, on a
    broken invariant, the step that clears the contact force; every node may
    be attempted ``attempts_per_node`` times, and every replan may run the
    whole plan again.  The vector wrapper resets the environment at the time
    limit, so the budget must never be reached.
    """

    horizons = {
        contract.contract_type_name: contract.max_episode_steps + 2
        for contract in library.find(task=EXPERIMENT_TASK)
    }
    per_pass = sum(horizons[kind] for kind in SEGMENT_SUBTASK_TYPES) * len(episode.segments)
    return per_pass * attempts_per_node * (1 + max_replans) + 1


def node_subtasks(
    proposal_plan_order: Sequence[str],
    skill_graph,
    library: ContractLibrary,
    episode: SetTableEpisode,
    facts: Sequence[str],
) -> List[Dict[str, Any]]:
    """Which plan subtask every node of a validated proposal stands for.

    The facts are rolled forward with every node's effects and deletes, as the
    symbolic environment would, so a navigation to a storage resolves to the
    navigation before its open while it is closed and to the one before its
    close once it stands open.
    """

    rows = []
    facts = frozenset(facts)
    for node_id in proposal_plan_order:
        node = skill_graph.nodes[node_id]
        grounded = library.get(node.contract_id).bind(node.arguments)
        row: Dict[str, Any] = {
            "node_id": node_id,
            "grounding": grounded.id,
            "subtask_index": None,
            "subtask": None,
        }
        try:
            index = episode.subtask_for(
                grounded.contract.contract_type_name, grounded.arguments, facts
            )
        except UnsupportedGrounding as exc:
            row["error"] = str(exc)
        else:
            row["subtask_index"] = index
            row["subtask"] = episode.subtask_label(index)
        rows.append(row)
        facts = grounded.apply_to(facts)
    return rows


# -- reporting -------------------------------------------------------------------------


def episode_lines(episode: SetTableEpisode) -> List[str]:
    lines = [
        "episode: plan {} of {} ({}, {})".format(
            episode.plan_index, episode.dataset, episode.build_config_name, episode.init_config_name
        ),
        "goal: {}".format(episode.goal),
        "{:>2}  {:<8} {:<12} {:<18} {:<20} {:<16} {:<20} {}".format(
            "#", "label", "object", "storage", "destination", "instance",
            "storage instance", "receptacle instance",
        ),
    ]
    for item in episode.segments:
        lines.append(
            "{:>2}  {:<8} {:<12} {:<18} {:<20} {:<16} {:<20} {}".format(
                item.index, item.label, item.object, item.source, item.destination,
                item.object_instance, item.source_instance,
                item.receptacle_instance or "(unnamed: receptacle by goal rectangle)",
            )
        )
    return lines


def result_lines(result: RunResult, executions: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "",
        "status: {}   success: {}".format(result.status.value, result.success),
        "{:>4}  {:<28} {:<26} {:<16} {:<8} {:>5}  {}".format(
            "step", "node", "sub-goal", "outcome", "attempt", "steps", "detail"
        ),
    ]
    pending = list(executions)
    for decision in result.decisions:
        steps = ""
        detail = decision.failure_mode or ""
        if decision.outcome in (Outcome.SUCCESS, Outcome.FAILED) and pending:
            record = pending.pop(0)
            steps = str(record["steps"])
            if record.get("subtask") is not None:
                detail = "{} {}".format(record["subtask"], detail).strip()
            if record.get("reason"):
                detail = "{}: {}".format(detail, record["reason"])
        elif decision.outcome is Outcome.ADMISSION_FAILED:
            detail = "missing {}".format(list(decision.missing_preconditions))
        elif decision.outcome is Outcome.SKIPPED:
            detail = decision.note
        lines.append(
            "{:>4}  {:<28} {:<26} {:<16} {:<8} {:>5}  {}".format(
                decision.step, decision.node_id, decision.subgoal_id, decision.outcome.value,
                decision.attempt, steps, detail,
            )
        )
    for replan in result.replans:
        lines.append(
            "replan after step {}: {} failed ({}) -> {}".format(
                replan.after_step, replan.failure.subgoal_id, replan.failure.failure_mode,
                "accepted, {} sub-goals, span {}".format(replan.subgoals, replan.span)
                if replan.accepted
                else "rejected: " + "; ".join(item.message for item in replan.rejections),
            )
        )
    metrics = result.metrics
    lines.append(
        "metrics: " + ", ".join(
            "{}={}".format(key, metrics[key])
            for key in (
                "subgoals", "nodes", "node_executions", "failed_executions", "retries",
                "skipped_nodes", "replans", "replanning_span", "goal_facts_achieved", "goal_facts",
            )
        )
    )
    lines.append("simulator steps: {}".format(sum(record["steps"] for record in executions)))
    return lines


# -- command line -----------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll a proposer's SetTable plan out on MS-HAB (stage 6)."
    )
    parser.add_argument("--proposer", choices=PROPOSERS, default="rule",
                        help="the rule-based pseudo model, or the real model")
    parser.add_argument("--granularity", choices=GRANULARITIES, default="coarse",
                        help="the rule proposer answers coarse and fine only")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--plan-index", type=int, default=DEFAULT_PLAN_INDEX,
                        help="which official plan; default {}".format(DEFAULT_PLAN_INDEX))
    parser.add_argument("--task-plan", type=Path, default=None,
                        help="an all.json to read instead of the official split's")
    parser.add_argument("--rearrange-root", type=Path, default=DEFAULT_REARRANGE_ROOT,
                        help="the rearrange dataset: plans and episode configs")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts-per-node", type=int, default=DEFAULT_ATTEMPTS_PER_NODE)
    parser.add_argument("--max-replans", type=int, default=DEFAULT_MAX_REPLANS)
    parser.add_argument("--retries", type=int, default=DEFAULT_MODEL_RETRIES,
                        help="how often the validator asks a refused call again")
    parser.add_argument("--give-up-after", type=int, default=DEFAULT_GIVE_UP_AFTER,
                        help="rule proposer: replans a segment may cause before its object is given up")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-info-on-video", action="store_true")
    parser.add_argument("--output", type=Path, default=None,
                        help="run directory; default {}/<proposer>-<granularity>-plan<N>-<timestamp>".format(DEFAULT_ROLLOUT_DIR))
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the initial proposal and map its nodes to plan subtasks; no simulator")
    parser.add_argument("--model", default=DEEPSEEK_DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEEPSEEK_BASE_URL)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--no-json-mode", action="store_true")
    return parser.parse_args(argv)


def make_proposer(args: argparse.Namespace, episode: SetTableEpisode) -> GraphProposer:
    if args.proposer == "rule":
        if args.granularity not in GOLD_GRANULARITIES:
            sys.exit(
                "the rule proposer decomposes at {}; use --proposer deepseek for {!r}".format(
                    "/".join(GOLD_GRANULARITIES), args.granularity
                )
            )
        return SetTableRuleProposer(episode, give_up_after=args.give_up_after)
    try:
        return DeepSeekProposer(
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            json_mode=not args.no_json_mode,
        )
    except ProposerUnavailable as exc:
        sys.exit(str(exc))


def _write(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main(argv: Optional[Sequence[str]] = None, log: Callable[[str], None] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    log = log or (lambda line: print(line, flush=True))
    rearrange_root = Path(args.rearrange_root)
    plan_path = (
        Path(args.task_plan)
        if args.task_plan is not None
        else sequential_plan_path(rearrange_root, args.split)
    )
    if not plan_path.is_file():
        sys.exit("task plan not found: {}".format(plan_path))
    episode, plan = load_episode(plan_path, args.plan_index, rearrange_root)
    library = build_granularity_library(Path(args.checkpoint_root), task_families=(TASK_FAMILY,))
    proposer = make_proposer(args, episode)
    description = proposer.describe() if hasattr(proposer, "describe") else {"proposer": type(proposer).__name__}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    label = args.model if args.proposer == "deepseek" else args.proposer
    output = (
        Path(args.output)
        if args.output is not None
        else DEFAULT_ROLLOUT_DIR / "{}-{}-plan{}-{}".format(label, args.granularity, args.plan_index, stamp)
    )
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "episode.json", episode.as_dict())
    # ``mshab.evaluate`` insists on the task name in the plan path, so the
    # same file can be handed to the official evaluator for a comparison run.
    plan_file = output / PLAN_FILE_NAME
    plan_file.write_text(json.dumps(plan) + "\n")
    for line in episode_lines(episode):
        log(line)
    validator = ProposalValidator(library, EXPERIMENT_TASK, retries=args.retries)
    summary: Dict[str, Any] = {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "proposer": description,
        "granularity": args.granularity,
        "plan_index": args.plan_index,
        "plan_path": str(plan_path),
        "seed": args.seed,
        "attempts_per_node": args.attempts_per_node,
        "max_replans": args.max_replans,
        "retries": args.retries,
        "output": str(output),
    }

    if args.dry_run:
        context = PlanningContext.initial(
            library,
            EXPERIMENT_TASK,
            entities=episode.entities(),
            facts=nominal_facts(episode),
            granularity=args.granularity,
        )
        try:
            proposal = validator.plan(proposer, episode.goal, context)
        except ProposalRejected as exc:
            summary.update(dry_run=True, accepted=False, rejections=exc.as_dict())
            _write(output / "proposal.json", {"accepted": False, "rejections": exc.as_dict(),
                                              "rounds": [item.as_dict() for item in exc.rounds]})
            log("proposal rejected: {}".format(exc))
        else:
            rows = node_subtasks(
                proposal.plan.order, proposal.skill_graph, library, episode, context.facts
            )
            record = proposal.as_dict()
            record.update(accepted=True, node_subtasks=rows)
            _write(output / "proposal.json", record)
            log("\nvalidated proposal: {} sub-goals, {} nodes; node -> plan subtask".format(
                len(proposal.subgoal_graph.subgoals), len(proposal.skill_graph.nodes)))
            for row in rows:
                log("  {:<28} {:<48} -> {} {}".format(
                    row["node_id"], row["grounding"],
                    "?" if row["subtask_index"] is None else row["subtask_index"],
                    row["subtask"] or row.get("error", ""),
                ))
            summary.update(
                dry_run=True, accepted=True,
                subgoals=len(proposal.subgoal_graph.subgoals),
                nodes=len(proposal.skill_graph.nodes),
                unmapped_nodes=sum(row["subtask_index"] is None for row in rows),
            )
        _write(output / "summary.json", summary)
        log("wrote {}".format(output))
        return summary

    # Only now does torch enter the process.
    from mshab.experiments.rollout.environment import (
        RolloutEnvironmentAdapter,
        make_rollout_env,
    )
    from mshab.experiments.rollout.executor import CheckpointPolicyExecutor

    budget = step_budget(library, episode, args.attempts_per_node, args.max_replans)
    log("\nstarting {} with a time limit of {} steps".format("SkillRollout-v0", budget))
    env = make_rollout_env(
        plan_file,
        max_episode_steps=budget,
        video_dir=None if args.no_video else output / "videos",
        info_on_video=not args.no_info_on_video,
    )
    adapter = RolloutEnvironmentAdapter(env, episode, library, EXPERIMENT_TASK, seed=args.seed)
    executor = CheckpointPolicyExecutor(episode, log=log)
    controller = TaskController(
        library,
        EXPERIMENT_TASK,
        attempts_per_node=args.attempts_per_node,
        max_replans=args.max_replans,
        validator=validator,
    )
    started = time.monotonic()
    try:
        result = controller.run(
            episode.goal,
            proposer,
            adapter,
            executor,
            granularity=args.granularity,
            goal_facts=episode.goal_facts,
        )
    finally:
        # Also writes the video.
        adapter.close()
    elapsed = time.monotonic() - started
    result.save(output / "trace.json")
    _write(output / "executions.json", executor.executions)
    if isinstance(proposer, DeepSeekProposer):
        proposer.save_exchanges(output / "exchanges.json")
    summary.update(
        dry_run=False,
        status=result.status.value,
        success=result.success,
        metrics=dict(result.metrics),
        simulator_steps=executor.simulator_steps,
        elapsed=round(elapsed, 1),
        initial_facts=list(result.initial_facts),
        final_facts=list(result.final_facts),
    )
    _write(output / "summary.json", summary)
    for line in result_lines(result, executor.executions):
        log(line)
    log("elapsed: {:.1f}s".format(elapsed))
    log("wrote {}".format(output))
    return summary


if __name__ == "__main__":
    main()
