import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mshab.experiments.granularity import build_granularity_library, contract_id
from mshab.experiments.granularity.higher_layers import (
    SCENARIOS,
    build_gold_graph,
    gold_graphs_for,
    gold_proposer,
    run_scenario,
)
from mshab.experiments.planning import (
    EntityDescription,
    ScriptedFailure,
    ScriptedProposer,
    SymbolicEnvironmentAdapter,
    SymbolicPolicy,
    SymbolicPolicyExecutor,
    bind_symbolic_policy,
)
from mshab.skills.model import ArtifactStatus


TASK = "granularity"


class _Base(TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.library = build_granularity_library(Path(self.temporary_directory.name))
        bind_symbolic_policy(self.library, TASK)

    def run_scenario(self, name, granularity, **options):
        return run_scenario(SCENARIOS[name], granularity, self.library, **options)


class SymbolicEnvironmentTests(_Base):
    def test_symbolic_policy_is_the_ready_default_without_checkpoints(self):
        pick = contract_id("pick")
        policy = self.library.select_policy(pick, arguments={"object": "024_bowl"})
        self.assertEqual(policy.id, SymbolicPolicy.ID)
        self.assertEqual(policy.status, ArtifactStatus.READY)
        self.assertEqual(policy.target, "all")
        # It is appended after the 23 checkpoint bindings, never before them.
        self.assertEqual(self.library.policies_for(pick)[-1].id, SymbolicPolicy.ID)
        self.assertEqual(len(self.library.policies_for(pick)), 24)

    def test_facts_follow_contract_effects_and_world_rules(self):
        environment = SymbolicEnvironmentAdapter(
            self.library,
            TASK,
            ("present(024_bowl)", "present(dining_table)", "gripper_empty()", "collision_safe()"),
            (EntityDescription("024_bowl", "object"), EntityDescription("dining_table", "receptacle")),
        )
        self.assertTrue(environment.supports_contract_env("PickSubtaskTrain-v0"))
        self.assertEqual(sorted(environment.description.entities), ["024_bowl", "dining_table"])
        snapshot = environment.reset()
        self.assertEqual(snapshot.step_index, 0)
        executor = SymbolicPolicyExecutor()
        library = self.library

        def run(contract_type, arguments):
            grounded = library.get(contract_id(contract_type)).bind(arguments)
            return executor.execute(grounded, library.policy(SymbolicPolicy.ID), environment, lambda s: None)

        self.assertTrue(run("navigate", {"target": "024_bowl"}).success)
        self.assertIn("reachable(024_bowl)", environment.facts)
        self.assertTrue(run("pick", {"object": "024_bowl"}).success)
        self.assertIn("holding(024_bowl)", environment.facts)
        self.assertNotIn("gripper_empty()", environment.facts)
        # Rule 2: arriving somewhere else retracts the earlier reachable fact.
        run("navigate", {"target": "dining_table"})
        self.assertNotIn("reachable(024_bowl)", environment.facts)
        self.assertIn("reachable(dining_table)", environment.facts)
        run("place", {"object": "024_bowl", "destination": "dining_table"})
        self.assertIn("at(024_bowl,dining_table)", environment.facts)
        self.assertIn("gripper_empty()", environment.facts)
        # Rule 1: picking the object up again retracts where it was.
        run("navigate", {"target": "024_bowl"})
        run("pick", {"object": "024_bowl"})
        self.assertNotIn("at(024_bowl,dining_table)", environment.facts)
        self.assertEqual(environment.snapshot().step_index, 6)
        with self.assertRaisesRegex(TypeError, "symbolic action"):
            environment.step(["reachable(x)"])

    def test_scripted_failures_match_a_grounding_on_the_listed_attempts(self):
        failure = ScriptedFailure(
            "place", "024_bowl", attempts=(1,), failure_mode="object_dropped",
            add=("gripper_empty()",), remove=("holding(024_bowl)",),
        )
        self.assertEqual(ScriptedFailure.from_dict(failure.as_dict()), failure)
        self.assertEqual(ScriptedFailure.from_dict({"contract_type": "pick", "target": "x", "attempts": None}).attempts, None)
        with self.assertRaisesRegex(ValueError, "positive integers"):
            ScriptedFailure("pick", "x", attempts=(0,))
        with self.assertRaisesRegex(ValueError, "same grounding"):
            SymbolicPolicyExecutor((failure, ScriptedFailure("place", "024_bowl")))

        environment = SymbolicEnvironmentAdapter(
            self.library, TASK,
            ("present(024_bowl)", "present(dining_table)", "holding(024_bowl)", "reachable(dining_table)", "collision_safe()"),
        )
        environment.reset()
        executor = SymbolicPolicyExecutor((failure,))
        grounded = self.library.get(contract_id("place")).bind({"object": "024_bowl", "destination": "dining_table"})
        policy = self.library.policy(SymbolicPolicy.ID)
        first = executor.execute(grounded, policy, environment, lambda s: None)
        self.assertFalse(first.success)
        self.assertEqual(first.failure_mode, "object_dropped")
        self.assertNotIn("holding(024_bowl)", environment.facts)
        self.assertNotIn("at(024_bowl,dining_table)", environment.facts)
        # The second attempt is not scripted to fail; the symbolic executor
        # applies the effects regardless of preconditions, which is what
        # SkillRuntime's admission check exists for.
        second = executor.execute(grounded, policy, environment, lambda s: None)
        self.assertTrue(second.success)
        self.assertEqual(executor.attempts[("place", "024_bowl")], 2)


class ScenarioTests(_Base):
    def test_nominal_runs_the_gold_plan_at_both_granularities(self):
        for granularity in ("coarse", "fine"):
            result = self.run_scenario("nominal", granularity)
            gold = build_gold_graph("tidy_house_" + granularity, self.library)
            self.assertEqual(result.status, "success")
            self.assertTrue(result.success)
            self.assertEqual([d.node_id for d in result.decisions], list(gold.plan.order))
            self.assertTrue(all(d.outcome == "success" for d in result.decisions))
            self.assertTrue(all(d.policy_id == SymbolicPolicy.ID for d in result.decisions))
            metrics = result.metrics
            self.assertEqual(metrics["node_executions"], 20)
            self.assertEqual(metrics["retries"], 0)
            self.assertEqual(metrics["replans"], 0)
            self.assertEqual(metrics["skipped_nodes"], 0)
            self.assertEqual(metrics["redundant_executions"], 0)
            self.assertEqual(metrics["goal_facts_achieved"], 5)
            self.assertEqual(metrics["achieved_subgoals"], len(gold.subgoal_graph.subgoals))
            self.assertEqual(metrics["subgoals"], len(gold.subgoal_graph.subgoals))
            self.assertFalse(metrics["recovery_success"])
            self.assertEqual(metrics["proposal_retries"], 0)
            self.assertEqual(metrics["proposer_calls"], 1 + len(gold.subgoal_graph.subgoals))
            self.assertEqual(len(result.proposals), 1)
            self.assertTrue(result.proposals[0]["accepted"])
            self.assertEqual(len(result.proposals[0]["rounds"]), 1)

    def test_a_single_failure_is_retried_without_a_replan(self):
        for granularity in ("coarse", "fine"):
            result = self.run_scenario("pick_fails_once", granularity)
            self.assertTrue(result.success)
            metrics = result.metrics
            self.assertEqual(metrics["node_executions"], 21)
            self.assertEqual(metrics["failed_executions"], 1)
            self.assertEqual(metrics["retries"], 1)
            self.assertEqual(metrics["replans"], 0)
            self.assertTrue(metrics["recovery_success"])
            attempts = [d for d in result.decisions if d.node_id == "pick_object_2"]
            self.assertEqual([(d.attempt, d.outcome) for d in attempts], [(1, "failed"), (2, "success")])
            self.assertEqual(attempts[0].failure_mode, "scripted_failure")
            self.assertEqual(attempts[0].missing_effects, ("holding(003_cracker_box)",))

    def test_an_exhausted_node_fails_its_sub_goal_and_triggers_a_replan(self):
        expected = {
            "coarse": {"failed_subgoal": "object_2_delivered", "span": 3, "achieved": 4},
            "fine": {"failed_subgoal": "object_2_holding", "span": 12, "achieved": 17},
        }
        for granularity, want in expected.items():
            result = self.run_scenario("pick_exhausted", granularity)
            self.assertEqual(result.status, "goal_not_reached")
            self.assertFalse(result.success)
            metrics = result.metrics
            self.assertEqual(metrics["node_executions"], 19)
            self.assertEqual(metrics["failed_executions"], 2)
            self.assertEqual(metrics["replans"], 1)
            self.assertEqual(metrics["replanning_span"], want["span"])
            self.assertEqual(metrics["goal_facts_achieved"], 4)
            self.assertEqual(metrics["achieved_subgoals"], want["achieved"])
            self.assertFalse(metrics["recovery_success"])
            replan = result.replans[0]
            self.assertTrue(replan.accepted)
            self.assertEqual(replan.failure.subgoal_id, want["failed_subgoal"])
            self.assertEqual(replan.failure.node_id, "pick_object_2")
            self.assertEqual(replan.failure.failure_mode, "scripted_failure")
            self.assertEqual(len(result.proposals), 2)
            self.assertEqual(result.proposals[1]["attempt"], 1)
            request = result.proposals[1]["decomposition"]["request"]
            self.assertEqual(request["failure"]["subgoal_id"], want["failed_subgoal"])
            self.assertIn("at(002_master_chef_can,kitchen_counter)", request["facts"])
            self.assertEqual(request["history"]["failed_nodes"], ["pick_object_2"])

    def test_a_dropped_object_is_planned_again_after_admission_fails(self):
        expected = {
            "coarse": {"failed_subgoal": "object_2_delivered", "span": 4},
            # 15, not 16: the robot is already at the dining table when the replan
            # is made, so destination_2_reachable holds at that moment.
            "fine": {"failed_subgoal": "object_2_placed", "span": 15},
        }
        for granularity, want in expected.items():
            result = self.run_scenario("object_dropped", granularity)
            self.assertTrue(result.success, granularity)
            metrics = result.metrics
            self.assertEqual(metrics["node_executions"], 25)
            self.assertEqual(metrics["failed_executions"], 2)
            self.assertEqual(metrics["admission_failures"], 1)
            self.assertEqual(metrics["replans"], 1)
            self.assertEqual(metrics["replanning_span"], want["span"])
            self.assertTrue(metrics["recovery_success"])
            attempts = [d for d in result.decisions if d.node_id == "place_object_2"]
            self.assertEqual(
                [(d.attempt, d.outcome) for d in attempts],
                [(1, "failed"), (2, "admission_failed"), (1, "success")],
            )
            self.assertEqual(attempts[0].failure_mode, "object_dropped")
            self.assertIn("holding(003_cracker_box)", attempts[0].facts_removed)
            self.assertEqual(attempts[1].missing_preconditions, ("holding(003_cracker_box)",))
            replan = result.replans[0]
            self.assertEqual(replan.failure.subgoal_id, want["failed_subgoal"])
            self.assertEqual(replan.failure.failure_mode, "precondition_violated")
            self.assertEqual(replan.failure.missing_preconditions, ("holding(003_cracker_box)",))

    def test_an_already_delivered_object_is_skipped_only_at_coarse_granularity(self):
        coarse = self.run_scenario("object_already_delivered", "coarse")
        fine = self.run_scenario("object_already_delivered", "fine")
        self.assertTrue(coarse.success and fine.success)
        self.assertEqual(coarse.metrics["skipped_nodes"], 4)
        self.assertEqual(coarse.metrics["node_executions"], 16)
        self.assertEqual(coarse.metrics["replans"], 0)
        skipped = [d for d in coarse.decisions if d.outcome == "skipped"]
        self.assertEqual(
            [d.node_id for d in skipped],
            ["navigate_to_object_1", "pick_object_1", "navigate_to_destination_1", "place_object_1"],
        )
        self.assertTrue(all(d.subgoal_id == "object_1_delivered" for d in skipped))
        # Fine sub-goals judge readiness one predicate at a time, so the first
        # object is picked up and put back: four executions the coarse
        # decomposition did not need.
        self.assertEqual(fine.metrics["skipped_nodes"], 0)
        self.assertEqual(fine.metrics["node_executions"], 20)
        self.assertEqual(fine.metrics["replans"], 0)
        self.assertEqual(fine.metrics["failed_executions"], 0)

    def test_set_table_generic_runs_with_articulations(self):
        scenario = SCENARIOS["set_table_nominal"]
        self.assertEqual(scenario.goal, build_gold_graph("set_table_generic", self.library).spec.goal)
        self.assertEqual(
            scenario.initial_facts,
            (
                "closed(fridge)", "closed(kitchen_counter)", "collision_safe()", "gripper_empty()",
                "present(013_apple)", "present(024_bowl)", "present(dining_table)",
                "present(fridge)", "present(kitchen_counter)",
            ),
        )
        self.assertEqual(
            scenario.goal_facts,
            ("at(024_bowl,dining_table)", "at(013_apple,dining_table)", "closed(kitchen_counter)", "closed(fridge)"),
        )
        self.assertEqual(
            [(item.name, item.kind) for item in scenario.entities],
            [
                ("024_bowl", "object"), ("013_apple", "object"), ("kitchen_counter", "articulation"),
                ("fridge", "articulation"), ("dining_table", "receptacle"),
            ],
        )
        # The scripted proposer is found through the scenario's goal: the one
        # gold graph authored for SetTable, keyed as "free".
        result = self.run_scenario("set_table_nominal", "free")
        self.assertTrue(result.success)
        self.assertEqual(result.metrics["node_executions"], 16)
        self.assertEqual(result.metrics["redundant_executions"], 0)
        self.assertIn("open(kitchen_counter)", result.achieved_predicates)
        self.assertNotIn("open(kitchen_counter)", result.final_facts)
        with self.assertRaisesRegex(ValueError, "no gold graph answers"):
            self.run_scenario("set_table_nominal", "coarse")
        self.assertEqual(gold_graphs_for(scenario.goal), ("set_table_generic",))

    def test_rejected_proposals_and_exhausted_replans_end_the_run(self):
        result = run_scenario(SCENARIOS["nominal"], "coarse", self.library, proposer=ScriptedProposer())
        self.assertEqual(result.status, "proposal_rejected")
        self.assertFalse(result.success)
        self.assertFalse(result.proposals[0]["accepted"])
        self.assertEqual(result.proposals[0]["rejections"][0]["stage"], "decomposition")
        self.assertEqual(len(result.proposals[0]["rounds"]), 1)
        self.assertEqual(result.metrics["node_executions"], 0)
        self.assertEqual(result.metrics["proposer_calls"], 1)

        result = self.run_scenario("pick_exhausted", "coarse", max_replans=0)
        self.assertEqual(result.status, "replans_exhausted")
        self.assertEqual(result.metrics["node_executions"], 7)
        self.assertEqual(result.metrics["replans"], 0)

        # A replan the scripted proposer has no answer for is a rejection too.
        result = run_scenario(
            SCENARIOS["pick_exhausted"], "coarse", self.library,
            proposer=gold_proposer(["tidy_house_coarse"], self.library),
        )
        self.assertEqual(result.status, "proposal_rejected")
        self.assertEqual(len(result.replans), 1)
        self.assertFalse(result.replans[0].accepted)
        self.assertEqual(result.replans[0].rejections[0].stage, "decomposition")

    def test_trace_is_a_json_document(self):
        result = self.run_scenario("object_dropped", "fine")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            result.save(path)
            document = json.loads(path.read_text())
        self.assertEqual(document["status"], "success")
        self.assertEqual(
            sorted(document),
            sorted(
                ["task", "goal", "granularity", "status", "success", "metrics", "proposals",
                 "decisions", "replans", "initial_facts", "final_facts", "goal_facts", "achieved_predicates"]
            ),
        )
        self.assertEqual(len(document["decisions"]), 25)
        self.assertEqual(document["replans"][0]["failure"]["missing_preconditions"], ["holding(003_cracker_box)"])
        self.assertEqual(SCENARIOS["object_dropped"].as_dict()["failures"][0]["failure_mode"], "object_dropped")
