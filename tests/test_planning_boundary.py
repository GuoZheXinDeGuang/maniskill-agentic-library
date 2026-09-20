import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mshab.experiments.granularity import build_granularity_library, contract_id
from mshab.experiments.granularity.higher_layers import GOLD_GRAPHS, build_gold_graph, gold_proposer
from mshab.experiments.planning import (
    DecompositionRequest,
    DecompositionResponse,
    EntityDescription,
    Failure,
    History,
    Neighbours,
    ProposalRejected,
    ProposalValidator,
    PlanningContext,
    GraphProposer,
    ScriptedProposer,
    SubgraphRequest,
    SubgraphResponse,
    UnscriptedRequest,
    assemble_patch,
    root_nodes,
)
from mshab.skills import SkillGraph, SkillNode, SkillRelation, SkillSubgraph, SubGoal, SubGoalGraph
from mshab.skills.schema import SchemaError


TASK = "granularity"
ENTITIES = (
    EntityDescription("024_bowl", "object"),
    EntityDescription("dining_table", "location"),
)
FACTS = ("present(024_bowl)", "gripper_empty()", "present(024_bowl)")


class _Base(TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.library = build_granularity_library(Path(self.temporary_directory.name))
        self.validator = ProposalValidator(self.library, TASK)

    def context(self, granularity="free"):
        return PlanningContext.initial(
            self.library, TASK, entities=ENTITIES, facts=FACTS, granularity=granularity
        )

    def gold_context(self, gold):
        return self.context(gold.spec.granularity or "free")


class DocumentTests(_Base):
    def test_context_describes_the_library_and_normalises_facts(self):
        context = self.context("coarse")
        self.assertEqual(
            [item["id"] for item in context.contracts],
            sorted(self.library.get(contract_id(t)).id for t in ("navigate", "pick", "place", "open", "close")),
        )
        # Contract records are the library's own as_dict(), not a second schema.
        pick = next(item for item in context.contracts if item["contract_type"] == "pick")
        self.assertEqual(pick, self.library.get(contract_id("pick")).as_dict())
        self.assertEqual(pick["target_parameter"], "object")
        self.assertEqual(
            [(p["name"], p["type"], p["required"]) for p in pick["parameters"]],
            [("object", "entity", True)],
        )
        self.assertEqual(pick["preconditions"], ["reachable({object})", "gripper_empty()"])
        with self.assertRaisesRegex(SchemaError, "contracts\\[0\\].id"):
            PlanningContext(({"contract_type": "pick"},))
        self.assertEqual(context.facts, ("gripper_empty()", "present(024_bowl)"))
        self.assertEqual(context.attempt, 0)
        self.assertIsNone(context.failure)
        with self.assertRaisesRegex(ValueError, "granularity must be one of"):
            self.context("medium")
        with self.assertRaisesRegex(ValueError, "attempt must be"):
            PlanningContext(context.contracts, attempt=-1)

        replanned = context.replan(
            Failure("object_2_delivered", "pick_object_2", "execution_timeout", ("holding(003_cracker_box)",)),
            History(("object_1_delivered",), ("navigate_to_object_1",), ("pick_object_2",)),
            ("gripper_empty()",),
        )
        self.assertEqual(replanned.attempt, 1)
        self.assertEqual(replanned.failure.subgoal_id, "object_2_delivered")
        self.assertEqual(replanned.granularity, "coarse")
        self.assertEqual(replanned.contracts, context.contracts)

    def test_all_four_documents_round_trip_strictly(self):
        context = self.context("fine").replan(
            Failure("object_2_delivered", None, None, ()),
            History(),
            FACTS,
        )
        request = DecompositionRequest(TASK, "Tidy up", context)
        self.assertEqual(request.fingerprint(), (TASK, "Tidy up", "fine", 1, "object_2_delivered"))
        restored = DecompositionRequest.from_dict(json.loads(json.dumps(request.as_dict())))
        self.assertEqual(restored.as_dict(), request.as_dict())
        self.assertEqual(restored.fingerprint(), request.fingerprint())

        response = DecompositionResponse(
            (SubGoal("a", "holding(024_bowl)"), SubGoal("b", "at(024_bowl,dining_table)")),
            rationale="pick then place",
        )
        self.assertEqual(DecompositionResponse.from_dict(response.as_dict()).as_dict(), response.as_dict())
        with self.assertRaisesRegex(ValueError, "cannot repeat"):
            DecompositionResponse((SubGoal("a", "x()"), SubGoal("a", "y()")))
        with self.assertRaisesRegex(ValueError, "at least one"):
            DecompositionResponse(())

        subgraph_request = SubgraphRequest(
            TASK, "Tidy up", SubGoal("b", "at(024_bowl,dining_table)"),
            context.contracts, context.entities, FACTS, Neighbours("holding(024_bowl)", None),
        )
        self.assertEqual(subgraph_request.fingerprint(), (TASK, "b", "at(024_bowl,dining_table)"))
        restored = SubgraphRequest.from_dict(json.loads(json.dumps(subgraph_request.as_dict())))
        self.assertEqual(restored.as_dict(), subgraph_request.as_dict())
        self.assertEqual(restored.neighbours, Neighbours("holding(024_bowl)", None))

        subgraph = SkillSubgraph("b", TASK)
        subgraph.add_node(SkillNode("nav", contract_id("navigate"), {"target": "dining_table"}))
        subgraph.add_node(
            SkillNode("place", contract_id("place"), {"object": "024_bowl", "destination": "dining_table"}, ("b",))
        )
        subgraph.relate("nav", "place", SkillRelation.ENABLES)
        subgraph_response = SubgraphResponse(subgraph, "navigate then place")
        document = subgraph_response.as_dict()
        self.assertEqual(sorted(document["subgraph"]), ["edges", "nodes", "subgoal_id"])
        restored = SubgraphResponse.from_dict(json.loads(json.dumps(document)), task=TASK)
        self.assertEqual(restored.as_dict(), document)
        self.assertEqual(restored.subgraph.execution_order(), ("nav", "place"))

        # Strictness: unknown keys, a foreign task namespace, a bad granularity.
        with self.assertRaises(SchemaError):
            DecompositionResponse.from_dict({**response.as_dict(), "policy": "rl"})
        with self.assertRaisesRegex(ValueError, "outside task namespace"):
            SubgraphResponse.from_dict(document, task="set_table")
        with self.assertRaisesRegex(SchemaError, "granularity"):
            DecompositionRequest.from_dict({**request.as_dict(), "granularity": "medium"})
        with self.assertRaises(SchemaError):
            SubgraphRequest.from_dict({**subgraph_request.as_dict(), "attempt": 0})


class ScriptedProposerTests(_Base):
    def setUp(self):
        super().setUp()
        self.proposer = gold_proposer(GOLD_GRAPHS, self.library)

    def test_reproduces_every_gold_graph_through_the_validator(self):
        for name in GOLD_GRAPHS:
            gold = build_gold_graph(name, self.library)
            validated = self.validator.plan(self.proposer, gold.spec.goal, self.gold_context(gold))
            self.assertEqual(validated.subgoal_graph.as_dict(), gold.subgoal_graph.as_dict())
            self.assertEqual(validated.skill_graph.as_dict(), gold.skill_graph.as_dict())
            self.assertEqual(validated.plan.order, gold.plan.order)
            self.assertEqual(validated.patch.as_dict(), gold.patch.as_dict())
            order = gold.subgoal_graph.execution_order()
            self.assertEqual(len(validated.subgraph_requests), len(order))
            first, second = validated.subgraph_requests[:2]
            self.assertIsNone(first.neighbours.previous)
            self.assertEqual(first.neighbours.next, gold.subgoal_graph.subgoals[order[1]].predicate)
            self.assertEqual(second.neighbours.previous, gold.subgoal_graph.subgoals[order[0]].predicate)
            trace = validated.as_dict()
            self.assertEqual(trace["nominal_plan"]["order"], list(gold.plan.order))
            self.assertEqual(len(trace["subgraphs"]), len(order))
            json.dumps(trace)

    def test_coarse_and_fine_are_different_scripted_answers_to_the_same_goal(self):
        coarse = build_gold_graph("tidy_house_coarse", self.library)
        fine = self.validator.plan(self.proposer, coarse.spec.goal, self.context("fine"))
        self.assertEqual(len(fine.subgoal_graph.subgoals), 20)
        self.assertEqual(
            len(self.validator.plan(self.proposer, coarse.spec.goal, self.context("coarse")).subgoal_graph.subgoals),
            5,
        )
        with self.assertRaises(ProposalRejected) as raised:
            self.validator.plan(self.proposer, coarse.spec.goal, self.context("free"))
        self.assertEqual(raised.exception.rejections[0].stage, "decomposition")
        self.assertIn("no scripted decomposition", raised.exception.rejections[0].message)

    def test_unscripted_requests_raise_instead_of_defaulting(self):
        with self.assertRaisesRegex(UnscriptedRequest, "no scripted decomposition"):
            self.proposer.decompose(DecompositionRequest(TASK, "Make coffee", self.context()))
        request = SubgraphRequest(TASK, "g", SubGoal("object_9_delivered", "at(x,y)"), ())
        with self.assertRaisesRegex(UnscriptedRequest, "no scripted subgraph"):
            self.proposer.plan_subgraph(request)

    def test_proposer_is_a_skill_graph_builder(self):
        gold = build_gold_graph("set_table_generic", self.library)
        subgoals, graph = self.proposer.build(
            gold.spec.goal, TASK, context=self.gold_context(gold).as_dict(), library=self.library
        )
        self.assertEqual(subgoals.as_dict(), gold.subgoal_graph.as_dict())
        self.assertEqual(graph.as_dict(), gold.skill_graph.as_dict())

    def test_script_round_trips_through_json(self):
        document = json.loads(json.dumps(self.proposer.as_dict()))
        self.assertEqual(len(document["decompositions"]), 3)
        self.assertEqual(len(document["subgraphs"]), 5 + 20 + 8)
        restored = ScriptedProposer.from_dict(document)
        self.assertEqual(restored.as_dict(), document)
        gold = build_gold_graph("tidy_house_fine", self.library)
        validated = self.validator.plan(restored, gold.spec.goal, self.gold_context(gold))
        self.assertEqual(validated.skill_graph.as_dict(), gold.skill_graph.as_dict())
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "script.json"
            self.proposer.save(path)
            self.assertEqual(ScriptedProposer.load(path).as_dict(), document)


class PlanValidatorTests(_Base):
    def setUp(self):
        super().setUp()
        self.proposer = gold_proposer(["tidy_house_coarse"], self.library)
        self.gold = build_gold_graph("tidy_house_coarse", self.library)
        self.goal = self.gold.spec.goal

    def _override(self, subgoal_id, predicate, subgraph):
        self.proposer.script_subgraph((TASK, subgoal_id, predicate), SubgraphResponse(subgraph))

    def test_malformed_subgraphs_name_their_sub_goal_and_leave_graphs_untouched(self):
        accepted = self.validator.plan(self.proposer, self.goal, self.context("coarse"))
        before = (accepted.subgoal_graph.as_dict(), accepted.skill_graph.as_dict())

        unknown_contract = SkillSubgraph("object_2_delivered", TASK)
        unknown_contract.add_node(
            SkillNode("fly_2", "mshab.granularity.fly.all", {"target": "x"}, ("object_2_delivered",))
        )
        self._override("object_2_delivered", "at(003_cracker_box,dining_table)", unknown_contract)
        wrong_argument = SkillSubgraph("object_4_delivered", TASK)
        wrong_argument.add_node(
            SkillNode("place_object_4", contract_id("place"), {"item": "005_tomato_soup_can"}, ("object_4_delivered",))
        )
        self._override("object_4_delivered", "at(005_tomato_soup_can,tv_stand)", wrong_argument)

        with self.assertRaises(ProposalRejected) as raised:
            self.validator.plan(self.proposer, self.goal, self.context("coarse"))
        rejections = raised.exception.rejections
        self.assertEqual([item.stage for item in rejections], ["subgraph", "subgraph"])
        self.assertEqual([item.subgoal_id for item in rejections], ["object_2_delivered", "object_4_delivered"])
        self.assertIn("unknown contract 'mshab.granularity.fly.all'", rejections[0].message)
        self.assertIn("invalid contract arguments", rejections[1].message)
        self.assertEqual(raised.exception.as_dict()[0]["subgoal_id"], "object_2_delivered")
        self.assertEqual((accepted.subgoal_graph.as_dict(), accepted.skill_graph.as_dict()), before)

    def test_structural_mistakes_are_attributed(self):
        # A subgraph handed back for a different sub-goal.  The scripted proposer
        # refuses to be scripted that way, so a small proposer of its own
        # produces the answer, which also exercises the GraphProposer interface.
        other = SkillSubgraph("somewhere_else", TASK)
        other.add_node(SkillNode("nav_x", contract_id("navigate"), {"target": "x"}, ("somewhere_else",)))
        scripted = self.proposer

        class WrongOwner(GraphProposer):
            def decompose(self, request):
                return scripted.decompose(request)

            def plan_subgraph(self, request):
                if request.subgoal.id == "object_3_delivered":
                    return SubgraphResponse(other)
                return scripted.plan_subgraph(request)

        with self.assertRaises(ProposalRejected) as raised:
            self.validator.plan(WrongOwner(), self.goal, self.context("coarse"))
        self.assertEqual(raised.exception.rejections[0].subgoal_id, "object_3_delivered")
        self.assertIn("is for sub-goal 'somewhere_else'", raised.exception.rejections[0].message)
        with self.assertRaisesRegex(ValueError, "is owned by 'somewhere_else'"):
            self._override("object_3_delivered", "at(004_sugar_box,coffee_table)", other)

        # A node id that another sub-goal already uses.
        reused = SkillSubgraph("object_3_delivered", TASK)
        reused.add_node(SkillNode("pick_object_1", contract_id("pick"), {"object": "004_sugar_box"}, ("object_3_delivered",)))
        self._override("object_3_delivered", "at(004_sugar_box,coffee_table)", reused)
        with self.assertRaises(ProposalRejected) as raised:
            self.validator.plan(self.proposer, self.goal, self.context("coarse"))
        self.assertEqual(raised.exception.rejections[0].subgoal_id, "object_3_delivered")
        self.assertIn("already used by sub-goal 'object_1_delivered'", raised.exception.rejections[0].message)

        # No achiever at all.
        empty = SkillSubgraph("object_3_delivered", TASK)
        empty.add_node(SkillNode("nav_3", contract_id("navigate"), {"target": "004_sugar_box"}))
        self._override("object_3_delivered", "at(004_sugar_box,coffee_table)", empty)
        with self.assertRaises(ProposalRejected) as raised:
            self.validator.plan(self.proposer, self.goal, self.context("coarse"))
        self.assertIn("has no achiever", raised.exception.rejections[0].message)

    def test_plan_stage_rejects_ambiguous_achievers(self):
        # Two achievers without a FALLBACK_TO order pass subgraph validation but
        # cannot be planned: the one-node decision has no primary.
        ambiguous = SkillSubgraph("object_3_delivered", TASK)
        ambiguous.add_node(
            SkillNode("place_a", contract_id("place"), {"object": "004_sugar_box", "destination": "coffee_table"}, ("object_3_delivered",))
        )
        ambiguous.add_node(
            SkillNode("place_b", contract_id("place"), {"object": "004_sugar_box", "destination": "tv_stand"}, ("object_3_delivered",))
        )
        self._override("object_3_delivered", "at(004_sugar_box,coffee_table)", ambiguous)
        with self.assertRaises(ProposalRejected) as raised:
            self.validator.plan(self.proposer, self.goal, self.context("coarse"))
        self.assertEqual(raised.exception.rejections[0].stage, "plan")
        self.assertIn("no FALLBACK_TO order", raised.exception.rejections[0].message)

    def test_assembler_enables_every_root_of_the_next_subgraph(self):
        first = SkillSubgraph("bowl_held", TASK)
        first.add_node(SkillNode("nav_bowl", contract_id("navigate"), {"target": "024_bowl"}))
        first.add_node(SkillNode("pick_bowl", contract_id("pick"), {"object": "024_bowl"}, ("bowl_held",)))
        first.relate("nav_bowl", "pick_bowl", SkillRelation.ENABLES)
        second = SkillSubgraph("bowl_placed", TASK)
        second.add_node(SkillNode("nav_table", contract_id("navigate"), {"target": "dining_table"}))
        second.add_node(SkillNode("nav_chair", contract_id("navigate"), {"target": "chair"}))
        second.add_node(
            SkillNode("place_bowl", contract_id("place"), {"object": "024_bowl", "destination": "dining_table"}, ("bowl_placed",))
        )
        second.relate("nav_table", "place_bowl", SkillRelation.ENABLES)
        second.relate("nav_chair", "place_bowl", SkillRelation.ENABLES)
        self.assertEqual(root_nodes(second), ("nav_chair", "nav_table"))

        decomposition = DecompositionResponse(
            (SubGoal("bowl_held", "holding(024_bowl)"), SubGoal("bowl_placed", "at(024_bowl,dining_table)"))
        )
        patch = assemble_patch(TASK, decomposition.subgoals, [first, second])
        self.assertEqual([d.as_dict() for d in patch.subgoal_dependencies], [{"source": "bowl_held", "target": "bowl_placed"}])
        self.assertEqual(
            sorted((edge.source_node, edge.target_node) for edge in patch.cross_edges),
            [(None, "nav_chair"), (None, "nav_table")],
        )
        subgoals = SubGoalGraph("Place the bowl")
        graph = SkillGraph(TASK, subgoal_graph=subgoals)
        patch.apply(subgoals, graph, library=self.library)
        graph.validate()
        self.assertEqual(graph.execution_order()[0], "nav_bowl")
        with self.assertRaisesRegex(ValueError, "do not match the decomposition"):
            assemble_patch(TASK, decomposition.subgoals, [second, first])
