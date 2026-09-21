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
        with self.assertRaisesRegex(TypeError, "ScriptedFailure instances"):
            SymbolicPolicyExecutor((failure, {"contract_type": "place"}))

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

    def test_a_failure_is_suspended_while_its_unless_facts_hold(self):
        # An object inside a closed drawer cannot be picked: the failure applies
        # on every attempt made while the drawer is closed and on none made
        # while it is open.  A second, one-off failure of the same grounding is
        # consulted only when the standing one does not apply.
        physics = ScriptedFailure(
            "pick", "024_bowl", attempts=None, failure_mode="execution_timeout",
            unless=("open(kitchen_counter)",),
        )
        flaky = ScriptedFailure("pick", "024_bowl", attempts=(3,))
        self.assertEqual(ScriptedFailure.from_dict(physics.as_dict()), physics)
        self.assertEqual(physics.as_dict()["unless"], ["open(kitchen_counter)"])
        self.assertTrue(physics.applies(1, ["closed(kitchen_counter)"]))
        self.assertFalse(physics.applies(1, ["open(kitchen_counter)"]))
        self.assertFalse(flaky.applies(1, []))
        with self.assertRaisesRegex(ValueError, "unless facts"):
            ScriptedFailure("pick", "x", unless=("",))
        environment = SymbolicEnvironmentAdapter(
            self.library, TASK,
            ("present(024_bowl)", "present(kitchen_counter)", "reachable(024_bowl)",
             "closed(kitchen_counter)", "gripper_empty()", "collision_safe()"),
        )
        environment.reset()
        executor = SymbolicPolicyExecutor((physics, flaky))
        grounded = self.library.get(contract_id("pick")).bind({"object": "024_bowl"})
        policy = self.library.policy(SymbolicPolicy.ID)
        first = executor.execute(grounded, policy, environment, lambda s: None)
        self.assertFalse(first.success)
        self.assertEqual(first.failure_mode, "execution_timeout")
        self.assertNotIn("holding(024_bowl)", environment.facts)
        environment.step({"add": ["open(kitchen_counter)"], "remove": ["closed(kitchen_counter)"]})
        second = executor.execute(grounded, policy, environment, lambda s: None)
        self.assertTrue(second.success)
        self.assertIn("holding(024_bowl)", environment.facts)
        environment.step({"add": ["gripper_empty()"], "remove": ["holding(024_bowl)"]})
        third = executor.execute(grounded, policy, environment, lambda s: None)
        self.assertEqual((third.success, third.failure_mode), (False, "scripted_failure"))


class ScenarioTests(_Base):
    BOWL_NODES = (
        "navigate_to_bowl_source", "open_bowl_source", "navigate_to_bowl", "pick_bowl",
        "navigate_bowl_to_destination", "place_bowl", "navigate_back_to_bowl_source", "close_bowl_source",
    )

    def test_set_table_scenarios_describe_the_scene(self):
        scenario = SCENARIOS["nominal"]
        self.assertEqual(scenario.goal, build_gold_graph("set_table_coarse", self.library).spec.goal)
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
        # The storage physics: the proposer is never told where the objects are.
        self.assertEqual(
            [(item.target, item.unless, item.failure_mode) for item in scenario.failures],
            [
                ("024_bowl", ("open(kitchen_counter)",), "execution_timeout"),
                ("013_apple", ("open(fridge)",), "execution_timeout"),
            ],
        )
        self.assertNotIn("at(024_bowl,kitchen_counter)", scenario.initial_facts)
        self.assertIn("open(fridge)", SCENARIOS["storage_already_open"].initial_facts)
        self.assertEqual(
            [item.unless for item in SCENARIOS["object_elsewhere"].failures],
            [("open(fridge)",), ("open(kitchen_counter)",)],
        )
        self.assertEqual(gold_graphs_for(scenario.goal), ("set_table_coarse", "set_table_fine"))
        document = SCENARIOS["object_elsewhere"].as_dict()
        json.dumps(document)
        self.assertEqual(document["replans"][0]["context"]["steps"], {"apple": ["pick", "place", "close"]})

    def test_nominal_runs_the_gold_plan_at_both_granularities(self):
        for granularity in ("coarse", "fine"):
            result = self.run_scenario("nominal", granularity)
            gold = build_gold_graph("set_table_" + granularity, self.library)
            self.assertEqual(result.status, "success")
            self.assertTrue(result.success)
            self.assertEqual([d.node_id for d in result.decisions], list(gold.plan.order))
            self.assertTrue(all(d.outcome == "success" for d in result.decisions))
            self.assertTrue(all(d.policy_id == SymbolicPolicy.ID for d in result.decisions))
            metrics = result.metrics
            self.assertEqual(metrics["node_executions"], 16)
            self.assertEqual(metrics["retries"], 0)
            self.assertEqual(metrics["replans"], 0)
            self.assertEqual(metrics["skipped_nodes"], 0)
            self.assertEqual(metrics["redundant_executions"], 0)
            self.assertEqual(metrics["goal_facts_achieved"], 4)
            # Distinct achieved predicates: the fine graph reaches the same
            # storage and the table twice, so 13 of its 16 sub-goals count.
            predicates = {item.predicate for item in gold.subgoal_graph.subgoals.values()}
            self.assertEqual(metrics["achieved_subgoals"], len(predicates))
            self.assertEqual(metrics["subgoals"], len(gold.subgoal_graph.subgoals))
            self.assertFalse(metrics["recovery_success"])
            self.assertEqual(metrics["proposal_retries"], 0)
            self.assertEqual(metrics["proposer_calls"], 1 + len(gold.subgoal_graph.subgoals))
            self.assertEqual(len(result.proposals), 1)
            self.assertTrue(result.proposals[0]["accepted"])
            self.assertEqual(len(result.proposals[0]["rounds"]), 1)
        # Coarse: the drawer is closed again inside the bowl's sub-goal, before
        # the apple starts, and the sub-goal counts as achieved only then.
        coarse = self.run_scenario("nominal", "coarse")
        self.assertEqual(tuple(d.node_id for d in coarse.decisions[:8]), self.BOWL_NODES)
        self.assertEqual(coarse.achieved_predicates, ("at(024_bowl,dining_table)", "at(013_apple,dining_table)"))
        self.assertIn("closed(kitchen_counter)", coarse.final_facts)

    def test_a_single_failure_is_retried_without_a_replan(self):
        for granularity in ("coarse", "fine"):
            result = self.run_scenario("pick_fails_once", granularity)
            self.assertTrue(result.success)
            metrics = result.metrics
            self.assertEqual(metrics["node_executions"], 17)
            self.assertEqual(metrics["failed_executions"], 1)
            self.assertEqual(metrics["retries"], 1)
            self.assertEqual(metrics["replans"], 0)
            self.assertTrue(metrics["recovery_success"])
            attempts = [d for d in result.decisions if d.node_id == "pick_bowl"]
            self.assertEqual([(d.attempt, d.outcome) for d in attempts], [(1, "failed"), (2, "success")])
            self.assertEqual(attempts[0].failure_mode, "scripted_failure")
            self.assertEqual(attempts[0].missing_effects, ("holding(024_bowl)",))

    def test_an_exhausted_node_fails_its_sub_goal_and_triggers_a_replan(self):
        expected = {
            "coarse": {"failed_subgoal": "bowl_delivered", "span": 2, "achieved": 2,
                       "replan": ["bowl_storage_closed", "apple_delivered"]},
            "fine": {"failed_subgoal": "bowl_holding", "span": 9, "achieved": 11,
                     "replan": ["bowl_source_reachable_again", "bowl_source_closed"]},
        }
        for granularity, want in expected.items():
            result = self.run_scenario("pick_exhausted", granularity)
            self.assertEqual(result.status, "goal_not_reached")
            self.assertFalse(result.success)
            metrics = result.metrics
            self.assertEqual(metrics["node_executions"], 15)
            self.assertEqual(metrics["failed_executions"], 2)
            self.assertEqual(metrics["replans"], 1)
            self.assertEqual(metrics["replanning_span"], want["span"])
            self.assertEqual(metrics["goal_facts_achieved"], 3)
            self.assertEqual(metrics["achieved_subgoals"], want["achieved"])
            self.assertFalse(metrics["recovery_success"])
            replan = result.replans[0]
            self.assertTrue(replan.accepted)
            self.assertEqual(replan.failure.subgoal_id, want["failed_subgoal"])
            self.assertEqual(replan.failure.node_id, "pick_bowl")
            self.assertEqual(replan.failure.failure_mode, "scripted_failure")
            self.assertEqual(len(result.proposals), 2)
            self.assertEqual(result.proposals[1]["attempt"], 1)
            request = result.proposals[1]["decomposition"]["request"]
            self.assertEqual(request["failure"]["subgoal_id"], want["failed_subgoal"])
            self.assertIn("open(kitchen_counter)", request["facts"])
            self.assertEqual(request["history"]["failed_nodes"], ["pick_bowl"])
            replanned = [item["id"] for item in result.proposals[1]["decomposition"]["response"]["subgoals"]]
            self.assertEqual(replanned[: len(want["replan"])], want["replan"])
            # The given-up bowl's drawer is closed again before the apple.
            self.assertEqual(
                [d.node_id for d in result.decisions[5:7]],
                ["navigate_back_to_bowl_source", "close_bowl_source"],
            )
            self.assertIn("closed(kitchen_counter)", result.final_facts)

    def test_a_dropped_object_is_planned_again_after_admission_fails(self):
        expected = {
            "coarse": {"failed_subgoal": "bowl_delivered", "span": 2},
            # 11, not 14: the robot stands at the table when the replan is made,
            # and the drawer already stands open.
            "fine": {"failed_subgoal": "bowl_placed", "span": 11},
        }
        for granularity, want in expected.items():
            result = self.run_scenario("object_dropped", granularity)
            self.assertTrue(result.success, granularity)
            metrics = result.metrics
            self.assertEqual(metrics["node_executions"], 21)
            self.assertEqual(metrics["failed_executions"], 2)
            self.assertEqual(metrics["admission_failures"], 1)
            self.assertEqual(metrics["replans"], 1)
            self.assertEqual(metrics["replanning_span"], want["span"])
            self.assertTrue(metrics["recovery_success"])
            attempts = [d for d in result.decisions if d.node_id == "place_bowl"]
            self.assertEqual(
                [(d.attempt, d.outcome) for d in attempts],
                [(1, "failed"), (2, "admission_failed"), (1, "success")],
            )
            self.assertEqual(attempts[0].failure_mode, "object_dropped")
            self.assertIn("holding(024_bowl)", attempts[0].facts_removed)
            self.assertEqual(attempts[1].missing_preconditions, ("holding(024_bowl)",))
            replan = result.replans[0]
            self.assertEqual(replan.failure.subgoal_id, want["failed_subgoal"])
            self.assertEqual(replan.failure.failure_mode, "precondition_violated")
            self.assertEqual(replan.failure.missing_preconditions, ("holding(024_bowl)",))
            # The drawer that already stands open is not opened again.
            self.assertEqual([d.node_id for d in result.decisions].count("open_bowl_source"), 1)
            self.assertEqual([d.node_id for d in result.decisions][7:9], ["navigate_to_bowl", "pick_bowl"])

    def test_an_already_delivered_object_is_skipped_only_at_coarse_granularity(self):
        coarse = self.run_scenario("object_already_delivered", "coarse")
        fine = self.run_scenario("object_already_delivered", "fine")
        self.assertTrue(coarse.success and fine.success)
        self.assertEqual(coarse.metrics["skipped_nodes"], 8)
        self.assertEqual(coarse.metrics["node_executions"], 8)
        self.assertEqual(coarse.metrics["replans"], 0)
        skipped = [d for d in coarse.decisions if d.outcome == "skipped"]
        self.assertEqual(tuple(d.node_id for d in skipped), self.BOWL_NODES)
        self.assertTrue(all(d.subgoal_id == "bowl_delivered" for d in skipped))
        # Fine sub-goals judge readiness one predicate at a time, so the drawer
        # is opened, the bowl picked up off the table and put back, and the
        # drawer closed again: eight executions the coarse decomposition did
        # not need.
        self.assertEqual(fine.metrics["skipped_nodes"], 0)
        self.assertEqual(fine.metrics["node_executions"], 16)
        self.assertEqual(fine.metrics["replans"], 0)
        self.assertEqual(fine.metrics["failed_executions"], 0)

    def test_an_open_storage_is_absorbed_at_fine_and_replanned_at_coarse_granularity(self):
        coarse = self.run_scenario("storage_already_open", "coarse")
        fine = self.run_scenario("storage_already_open", "fine")
        self.assertTrue(coarse.success and fine.success)
        # Coarse: the open node of the apple's sub-goal cannot be admitted
        # (the fridge is not closed), the sub-goal fails, and the replan
        # fetches the apple without opening anything.
        self.assertEqual(coarse.metrics["admission_failures"], 2)
        self.assertEqual(coarse.metrics["replans"], 1)
        self.assertEqual(coarse.metrics["replanning_span"], 1)
        self.assertEqual(coarse.metrics["node_executions"], 17)
        self.assertEqual(coarse.metrics["skipped_nodes"], 0)
        replan = coarse.replans[0]
        self.assertEqual((replan.failure.subgoal_id, replan.failure.node_id), ("apple_delivered", "open_apple_source"))
        self.assertEqual(replan.failure.missing_preconditions, ("closed(fridge)",))
        self.assertEqual(
            [item["id"] for item in coarse.proposals[1]["decomposition"]["response"]["subgoals"]],
            ["apple_delivered"],
        )
        # Fine: the open(fridge) sub-goal already holds when its turn comes.
        self.assertEqual(fine.metrics["replans"], 0)
        self.assertEqual(fine.metrics["skipped_nodes"], 1)
        self.assertEqual(fine.metrics["node_executions"], 15)
        skipped = [d for d in fine.decisions if d.outcome == "skipped"]
        self.assertEqual([(d.node_id, d.subgoal_id) for d in skipped], [("open_apple_source", "apple_source_open")])

    def test_an_object_in_the_other_storage_is_found_after_a_replan(self):
        expected = {"coarse": ("bowl_delivered", 2), "fine": ("bowl_holding", 12)}
        for granularity, (failed_subgoal, span) in expected.items():
            result = self.run_scenario("object_elsewhere", granularity)
            self.assertTrue(result.success, granularity)
            metrics = result.metrics
            self.assertEqual(metrics["node_executions"], 19)
            self.assertEqual(metrics["failed_executions"], 2)
            self.assertEqual(metrics["replans"], 1)
            self.assertEqual(metrics["replanning_span"], span)
            self.assertEqual(metrics["goal_facts_achieved"], 4)
            attempts = [d for d in result.decisions if d.node_id == "pick_bowl"]
            # Two picks behind the closed fridge time out; the third, once the
            # fridge is open, succeeds.
            self.assertEqual(
                [(d.attempt, d.outcome, d.failure_mode) for d in attempts],
                [(1, "failed", "execution_timeout"), (2, "failed", "execution_timeout"), (1, "success", None)],
            )
            replan = result.replans[0]
            self.assertEqual(replan.failure.subgoal_id, failed_subgoal)
            self.assertEqual(replan.failure.missing_effects, ("holding(024_bowl)",))
            opened = [d for d in result.decisions if d.node_id == "open_bowl_source" and d.outcome == "success"]
            self.assertEqual(len(opened), 2)
            # The drawer opened first is closed at the end, after the apple was taken out of it.
            self.assertEqual(result.decisions[-1].node_id, "close_apple_source")
            self.assertTrue({"closed(fridge)", "closed(kitchen_counter)"} <= set(result.final_facts))

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
        self.assertEqual(result.metrics["node_executions"], 5)
        self.assertEqual(result.metrics["replans"], 0)

        # A replan the scripted proposer has no answer for is a rejection too.
        result = run_scenario(
            SCENARIOS["pick_exhausted"], "coarse", self.library,
            proposer=gold_proposer(["set_table_coarse"], self.library),
        )
        self.assertEqual(result.status, "proposal_rejected")
        self.assertEqual(len(result.replans), 1)
        self.assertFalse(result.replans[0].accepted)
        self.assertEqual(result.replans[0].rejections[0].stage, "decomposition")
        with self.assertRaisesRegex(ValueError, "no gold graph answers"):
            self.run_scenario("nominal", "free")

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
        self.assertEqual(len(document["decisions"]), 21)
        self.assertEqual(document["replans"][0]["failure"]["missing_preconditions"], ["holding(024_bowl)"])
        self.assertEqual(SCENARIOS["object_dropped"].as_dict()["failures"][-1]["failure_mode"], "object_dropped")
