"""CPU-only integration tests for the downloaded SetTable checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import SkipTest, TestCase

from mshab.skills import (
    ArtifactStatus,
    SkillGrounder,
    SkillLibrary,
    SkillType,
    build_set_table_stack,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = Path(
    os.environ.get("MS_ASSET_DIR", str(REPOSITORY_ROOT.parent / "mshab-assets"))
).expanduser()
CHECKPOINT_ROOT = DEFAULT_ASSET_ROOT / "data" / "mshab_checkpoints"


class DownloadedCheckpointTests(TestCase):
    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT_ROOT.exists():
            raise SkipTest(
                "downloaded checkpoints are unavailable at {}".format(
                    CHECKPOINT_ROOT
                )
            )
        cls.library = SkillLibrary.from_checkpoint_root(CHECKPOINT_ROOT)

    def test_all_eleven_set_table_skills_are_discovered_and_ready(self):
        skills = self.library.find(task="set_table")

        self.assertEqual(len(skills), 11)
        for skill in skills:
            self.assertTrue(skill.ready, skill.id)
            for backend in skill.backends.values():
                self.assertEqual(backend.status, ArtifactStatus.READY)
        self.assertTrue(json.dumps(self.library.to_dict()))

    def test_downloaded_apple_pick_grounds_to_expected_contract(self):
        specialized = self.library.find(
            task="set_table",
            skill_type=SkillType.PICK,
            target="013_apple",
        )[0]
        invocation = specialized.bind({}, backend_key=specialized.backend().key)

        self.assertEqual(invocation.arguments, {"object": "013_apple"})
        self.assertEqual(
            invocation.contract.preconditions,
            ("reachable(013_apple)", "gripper_empty()"),
        )
        self.assertEqual(
            invocation.contract.effects,
            ("holding(013_apple)",),
        )

    def test_complete_manual_graph_grounds_all_twenty_nodes(self):
        stack = build_set_table_stack(CHECKPOINT_ROOT)
        contracts = SkillGrounder(self.library).contracts(stack.skill_graph)

        self.assertEqual(len(stack.goal_graph.goals), 8)
        self.assertEqual(len(stack.skill_graph.subgraphs), 8)
        self.assertEqual(len(stack.skill_graph.nodes), 20)
        self.assertEqual(len(contracts), 20)
        self.assertEqual(
            contracts["place_apple_specialized"].effects,
            ("at(013_apple,dining_table)", "gripper_empty()"),
        )
