from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mshab.skills import (
    ArtifactStatus,
    CheckpointBackend,
    FunctionalGoal,
    FunctionalGoalGraph,
    PickSkill,
    SkillCompositionGraph,
    SkillLibrary,
    SkillNode,
    SkillRelation,
    SkillType,
)


class SkillModelTests(TestCase):
    def test_specialized_atomic_skill_binds_target_and_contract(self):
        skill = PickSkill(task="set_table", target="013_apple")
        invocation = skill.bind({})

        self.assertEqual(invocation.arguments, {"object": "013_apple"})
        self.assertEqual(
            invocation.contract.preconditions,
            ("reachable(013_apple)", "gripper_empty()"),
        )
        self.assertEqual(invocation.contract.effects, ("holding(013_apple)",))

        with self.assertRaisesRegex(ValueError, "specialized"):
            skill.bind({"object": "024_bowl"})

    def test_checkpoint_backend_reports_missing_partial_and_ready(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = CheckpointBackend(
                key="rl",
                family="rl",
                checkpoint_path=root / "policy.pt",
                config_path=root / "config.yml",
            )
            self.assertEqual(backend.status, ArtifactStatus.MISSING)
            (root / "config.yml").write_text("name: ppo\n")
            self.assertEqual(backend.status, ArtifactStatus.PARTIAL)
            (root / "policy.pt").write_bytes(b"weights")
            self.assertEqual(backend.status, ArtifactStatus.READY)

    def test_discovery_groups_policy_families_as_alternative_backends(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in ("rl", "bc"):
                leaf = root / family / "set_table" / "pick" / "013_apple"
                leaf.mkdir(parents=True)
                (leaf / "config.yml").write_text("name: {}\n".format(family))
                (leaf / "policy.pt").write_bytes(b"weights")

            library = SkillLibrary.from_checkpoint_root(root)
            skills = library.find(task="set_table", skill_type=SkillType.PICK)

            self.assertEqual(len(skills), 1)
            self.assertEqual(set(skills[0].backends), {"rl", "bc"})
            self.assertTrue(skills[0].ready)

    def test_composition_is_expressed_only_with_graph_relations(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for skill_type, target in (
                ("navigate", "all"),
                ("pick", "013_apple"),
                ("pick", "all"),
            ):
                leaf = root / "rl" / "set_table" / skill_type / target
                leaf.mkdir(parents=True)
                (leaf / "config.yml").write_text("name: ppo\n")
                (leaf / "policy.pt").write_bytes(b"weights")

            library = SkillLibrary.from_checkpoint_root(root)
            navigate = library.get("mshab.set_table.navigate.all")
            pick = library.get("mshab.set_table.pick.013_apple")
            generic_pick = library.get("mshab.set_table.pick.all")

            goals = FunctionalGoalGraph("Retrieve the apple")
            goals.add_goal(FunctionalGoal("reachable", "reachable(013_apple)"))
            goals.add_goal(FunctionalGoal("retrieved", "holding(013_apple)"))
            goals.add_dependency("reachable", "retrieved")

            graph = SkillCompositionGraph("set_table", goal_graph=goals)
            graph.add_node(
                SkillNode(
                    "navigate",
                    navigate.bind({"goal": "013_apple"}, backend_key="rl"),
                    achieves=("reachable",),
                )
            )
            graph.add_node(
                SkillNode(
                    "pick",
                    pick.bind({}, backend_key="rl"),
                    achieves=("retrieved",),
                )
            )
            graph.add_node(
                SkillNode(
                    "generic_pick",
                    generic_pick.bind({"object": "013_apple"}, backend_key="rl"),
                    achieves=("retrieved",),
                )
            )
            graph.relate("navigate", "pick", SkillRelation.ENABLES)
            graph.relate("generic_pick", "navigate", SkillRelation.REQUIRES)
            graph.relate("pick", "generic_pick", SkillRelation.IS_A)
            graph.relate("pick", "generic_pick", SkillRelation.ALTERNATIVE_TO)
            graph.relate("pick", "generic_pick", SkillRelation.FALLBACK_TO)

            order = graph.execution_order()
            self.assertLess(order.index("navigate"), order.index("pick"))
            self.assertLess(order.index("navigate"), order.index("generic_pick"))
            self.assertEqual(
                tuple(item.id for item in graph.prerequisites("pick")),
                ("navigate",),
            )
            self.assertEqual(
                tuple(item.id for item in graph.alternatives("pick")),
                ("generic_pick",),
            )
            self.assertEqual(
                tuple(item.id for item in graph.fallbacks("pick")),
                ("generic_pick",),
            )
            self.assertEqual(
                tuple(item.id for item in graph.candidates_for_goal("retrieved")),
                ("generic_pick", "pick"),
            )
            self.assertEqual(graph.uncovered_goals(), ())
            self.assertEqual({edge.relation for edge in graph.edges}, set(SkillRelation))
            self.assertEqual(
                tuple(item.id for item in graph.ready_nodes({"present(013_apple)"})),
                ("navigate",),
            )

            with self.assertRaisesRegex(ValueError, "causal cycle"):
                graph.relate("pick", "navigate", SkillRelation.ENABLES)
            self.assertEqual(len(graph.edges), 5)  # failed relation was rolled back

    def test_invocation_arguments_are_read_only(self):
        call = PickSkill(task="set_table", target="013_apple").bind({})
        with self.assertRaises(TypeError):
            call.arguments["object"] = "024_bowl"
