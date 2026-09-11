#!/usr/bin/env python3
"""Standalone smoke test for the MS-HAB skill library.

This test is intentionally CPU-only: it validates checkpoint discovery,
contracts, backend selection, grounding, composition, and JSON serialization.
It does not load policy weights or start a ManiSkill environment.

Run from the ManiSkill-HAB repository root:

    python scripts/test_skill_library.py --expected-count 11
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make the smoke test runnable from a fresh clone before ``pip install -e .``.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mshab.skills import (
    ArtifactStatus,
    FunctionalGoal,
    FunctionalGoalGraph,
    SkillCompositionGraph,
    SkillLibrary,
    SkillNode,
    SkillRelation,
    SkillType,
)


DEFAULT_ASSET_ROOT = Path(
    os.environ.get("MS_ASSET_DIR", str(REPO_ROOT.parent / "mshab-assets"))
).expanduser()
DEFAULT_CHECKPOINT_ROOT = DEFAULT_ASSET_ROOT / "data" / "mshab_checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="root with family/task/type/target/{config.yml,policy.pt}",
    )
    parser.add_argument(
        "--task",
        default="set_table",
        help="task whose discovered skills should be tested",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="optional exact number of skills expected for --task",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="optional path at which to write the discovered JSON index",
    )
    return parser.parse_args()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def exercise_apple_graph(library: SkillLibrary, task: str) -> None:
    """Exercise grounding and graph relations for the SetTable apple skills."""

    picks = library.find(
        task=task,
        skill_type=SkillType.PICK,
        target="013_apple",
    )
    navigators = library.find(
        task=task,
        skill_type=SkillType.NAVIGATE,
        target="all",
    )
    generic_picks = library.find(
        task=task,
        skill_type=SkillType.PICK,
        target="all",
    )
    if not picks or not navigators or not generic_picks:
        print("[SKIP] apple composition: required pick/navigation skills not found")
        return

    apple_pick = picks[0]
    generic_pick = generic_picks[0]
    navigate = navigators[0]
    pick_backend = apple_pick.backend()
    navigate_backend = navigate.backend()

    pick_call = apple_pick.bind({}, backend_key=pick_backend.key)
    generic_pick_call = generic_pick.bind(
        {"object": "013_apple"}, backend_key=generic_pick.backend().key
    )
    navigate_call = navigate.bind(
        {"goal": "013_apple"}, backend_key=navigate_backend.key
    )

    check(
        pick_call.arguments == {"object": "013_apple"},
        "specialized PickSkill did not ground its object",
    )
    check(
        pick_call.contract.preconditions
        == ("reachable(013_apple)", "gripper_empty()"),
        "PickSkill preconditions were grounded incorrectly",
    )
    check(
        pick_call.contract.can_start(
            {"reachable(013_apple)", "gripper_empty()"}
        ),
        "PickSkill should be startable from the supplied facts",
    )
    check(
        pick_call.contract.achieved({"holding(013_apple)"}),
        "PickSkill effects should be achieved by the supplied facts",
    )
    check(
        pick_call.contract.verified({"holding(013_apple)"}),
        "PickSkill verification predicate should hold",
    )

    goals = FunctionalGoalGraph("Retrieve the apple")
    goals.add_goal(FunctionalGoal("apple_reachable", "reachable(013_apple)"))
    goals.add_goal(FunctionalGoal("apple_retrieved", "holding(013_apple)"))
    goals.add_dependency("apple_reachable", "apple_retrieved")

    graph = SkillCompositionGraph(task=task, goal_graph=goals)
    graph.add_node(
        SkillNode("navigate_to_apple", navigate_call, achieves=("apple_reachable",))
    )
    graph.add_node(
        SkillNode("pick_apple_specialized", pick_call, achieves=("apple_retrieved",))
    )
    graph.add_node(
        SkillNode("pick_apple_generic", generic_pick_call, achieves=("apple_retrieved",))
    )
    graph.relate(
        "navigate_to_apple", "pick_apple_specialized", SkillRelation.ENABLES
    )
    graph.relate("pick_apple_generic", "navigate_to_apple", SkillRelation.REQUIRES)
    graph.relate(
        "pick_apple_specialized", "pick_apple_generic", SkillRelation.IS_A
    )
    graph.relate(
        "pick_apple_specialized",
        "pick_apple_generic",
        SkillRelation.ALTERNATIVE_TO,
    )
    graph.relate(
        "pick_apple_specialized",
        "pick_apple_generic",
        SkillRelation.FALLBACK_TO,
    )

    order = graph.execution_order()
    check(
        order.index("navigate_to_apple") < order.index("pick_apple_specialized"),
        "ENABLES relation did not constrain execution order",
    )
    check(
        [item.id for item in graph.alternatives("pick_apple_specialized")]
        == ["pick_apple_generic"],
        "ALTERNATIVE_TO relation was not queryable",
    )
    check(
        [item.id for item in graph.fallbacks("pick_apple_specialized")]
        == ["pick_apple_generic"],
        "FALLBACK_TO relation was not queryable",
    )
    check(
        [item.id for item in graph.candidates_for_goal("apple_retrieved")]
        == ["pick_apple_generic", "pick_apple_specialized"],
        "ACHIEVED_BY goal-to-candidate mapping was not queryable",
    )
    check(not graph.uncovered_goals(), "every smoke-test goal should have a candidate")
    check(
        {edge.relation for edge in graph.edges} == set(SkillRelation),
        "smoke graph did not exercise the complete skill relation vocabulary",
    )
    check(json.dumps(graph.as_dict()), "composition graph was not serializable")
    print("[PASS] grounding, goal graph, and skill-skill relations")


def main() -> int:
    args = parse_args()
    root = args.checkpoint_root.expanduser().resolve()
    print("checkpoint root:", root)

    try:
        library = SkillLibrary.from_checkpoint_root(root)
        task_skills = library.find(task=args.task)
        check(task_skills, "no skills discovered for task {!r}".format(args.task))
        if args.expected_count is not None:
            check(
                len(task_skills) == args.expected_count,
                "expected {} {} skills, found {}".format(
                    args.expected_count, args.task, len(task_skills)
                ),
            )

        print("\ndiscovered {} skill(s) for {}:".format(len(task_skills), args.task))
        for skill in task_skills:
            backend_summary = ", ".join(
                "{}={}".format(key, backend.status.value)
                for key, backend in sorted(skill.backends.items())
            )
            print("  {:55s} {}".format(skill.id, backend_summary))
            for backend in skill.backends.values():
                check(
                    backend.status == ArtifactStatus.READY,
                    "{} backend {} is {}".format(
                        skill.id, backend.key, backend.status.value
                    ),
                )

        exercise_apple_graph(library, args.task)

        payload = library.to_dict()
        json.dumps(payload)
        check(
            payload["count"] == len(library.find()),
            "serialized index count does not match registry",
        )
        print("[PASS] JSON serialization")

        if args.export:
            library.save_index(args.export)
            print("[PASS] wrote index:", args.export.resolve())
    except (AssertionError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print("\n[FAIL] {}".format(exc), file=sys.stderr)
        return 1

    print("\nSkill library smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
