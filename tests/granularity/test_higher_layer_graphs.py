import json
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mshab.experiments.granularity import build_granularity_library, split_contract_id
from mshab.experiments.granularity.higher_layers import (
    GOLD_GRAPHS,
    SetTableGraphBuilder,
    build_gold_graph,
    closing_subgoal_id,
    coarse_subgoal_id,
    fine_subgoal_ids,
    gold_graph_document,
    gold_graph_path,
    gold_graph_svg,
    load_gold_graph,
    segment_node_ids,
    segment_subgoal,
)
from mshab.skills import LibraryCatalog, SkillPlanner, SkillRelation
from mshab.skills.schema import SchemaError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPOSITORY_ROOT / "mshab" / "skills" / "catalogs" / "set_table.json"

# name -> (sub-goals, nodes)
EXPECTED = {
    "set_table_coarse": (2, 16),
    "set_table_fine": (16, 16),
}
BOWL_NODES = (
    "navigate_to_bowl_source",
    "open_bowl_source",
    "navigate_to_bowl",
    "pick_bowl",
    "navigate_bowl_to_destination",
    "place_bowl",
    "navigate_back_to_bowl_source",
    "close_bowl_source",
)
_TARGET_ARGUMENT = {
    "navigate": "target",
    "pick": "object",
    "place": "object",
    "open": "articulation",
    "close": "articulation",
}


def _type_and_target(node):
    _, contract_type, target = split_contract_id(node.contract_id)
    if target == "all":
        target = node.arguments[_TARGET_ARGUMENT[contract_type]]
    return contract_type, target


class GoldGraphTests(TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.library = build_granularity_library(Path(self.temporary_directory.name))

    def test_every_gold_graph_is_valid_grounded_and_plannable(self):
        self.assertEqual(list(GOLD_GRAPHS), list(EXPECTED))
        for name, (subgoals, nodes) in EXPECTED.items():
            gold = build_gold_graph(name, self.library)
            graph = gold.skill_graph
            self.assertEqual(len(gold.subgoal_graph.subgoals), subgoals)
            self.assertEqual(len(graph.subgraphs), subgoals)
            self.assertEqual(len(graph.nodes), nodes)
            self.assertEqual(len(graph.cross_edges), subgoals - 1)
            self.assertEqual(len(gold.plan.order), nodes)
            self.assertEqual(set(gold.plan.order), set(graph.nodes))
            self.assertEqual(set(gold.plan.selections), set(graph.subgraphs))
            for subgraph in graph.subgraphs.values():
                # One generic contract per type: one candidate per role.
                self.assertEqual(len(subgraph.achievers), 1)
                self.assertEqual(
                    {edge.relation for edge in subgraph.edges} - {SkillRelation.ENABLES},
                    set(),
                )
            for node in graph.nodes.values():
                self.assertTrue(node.contract_id.startswith("mshab.granularity."))
        with self.assertRaisesRegex(KeyError, "unknown gold graph"):
            build_gold_graph("tidy_house_coarse")

    def test_coarse_and_fine_differ_only_in_sub_goal_ownership(self):
        coarse = build_gold_graph("set_table_coarse", self.library)
        fine = build_gold_graph("set_table_fine", self.library)

        def roles(gold):
            return {
                node.id: (node.contract_id, dict(node.arguments))
                for node in gold.skill_graph.nodes.values()
            }

        self.assertEqual(roles(coarse), roles(fine))
        self.assertEqual(coarse.plan.order, fine.plan.order)
        self.assertEqual(coarse.plan.order[:8], BOWL_NODES)
        self.assertEqual(coarse.plan.order[8], "navigate_to_apple_source")
        self.assertEqual(
            coarse.subgoal_graph.execution_order(), ("bowl_delivered", "apple_delivered")
        )
        self.assertEqual(fine.subgoal_graph.execution_order()[:8], fine_subgoal_ids("bowl"))
        self.assertEqual(
            fine_subgoal_ids("bowl"),
            (
                "bowl_source_reachable",
                "bowl_source_open",
                "bowl_reachable",
                "bowl_holding",
                "bowl_destination_reachable",
                "bowl_placed",
                "bowl_source_reachable_again",
                "bowl_source_closed",
            ),
        )
        self.assertEqual(
            coarse.subgoal_graph.subgoals["bowl_delivered"].predicate,
            "at(024_bowl,dining_table)",
        )
        self.assertEqual(
            fine.subgoal_graph.subgoals["bowl_placed"].predicate, "at(024_bowl,dining_table)"
        )
        self.assertEqual(
            fine.subgoal_graph.subgoals["bowl_source_closed"].predicate, "closed(kitchen_counter)"
        )
        self.assertEqual(
            fine.subgoal_graph.subgoals["apple_source_open"].predicate, "open(fridge)"
        )
        self.assertEqual(
            gold_graph_document(coarse)["summary"],
            {
                "subgoals": 2,
                "subgraphs": 2,
                "nodes": 16,
                "internal_edges": 14,
                "cross_edges": 1,
                "mean_nodes_per_subgraph": 8.0,
            },
        )
        self.assertEqual(
            gold_graph_document(fine)["summary"],
            {
                "subgoals": 16,
                "subgraphs": 16,
                "nodes": 16,
                "internal_edges": 0,
                "cross_edges": 15,
                "mean_nodes_per_subgraph": 1.0,
            },
        )

    def test_coarse_follow_ups_run_after_the_achiever(self):
        # The place node achieves at(bowl, table); closing the drawer again is
        # the achiever's follow-up inside the same sub-goal, and the planner
        # runs it before the apple's segment starts.
        coarse = build_gold_graph("set_table_coarse", self.library)
        subgraph = coarse.skill_graph.subgraph_for_subgoal("bowl_delivered")
        self.assertEqual([node.id for node in subgraph.achievers], ["place_bowl"])
        planner = SkillPlanner(coarse.subgoal_graph, coarse.skill_graph)
        self.assertEqual(planner.remaining("bowl_delivered"), BOWL_NODES)
        self.assertEqual(
            planner.remaining("bowl_delivered", completed=BOWL_NODES[:6]),
            ("navigate_back_to_bowl_source", "close_bowl_source"),
        )
        self.assertEqual(planner.remaining("bowl_delivered", completed=BOWL_NODES), ())
        self.assertEqual(planner.decide(completed=BOWL_NODES[:6]).id, "navigate_back_to_bowl_source")
        self.assertEqual(planner.decide(completed=BOWL_NODES).id, "navigate_to_apple_source")

    def test_builders_take_context_and_reject_other_tasks(self):
        builder = SetTableGraphBuilder("coarse")
        subgoals, graph = builder.build(
            "Put the bowl on the table",
            "granularity",
            context={"segments": (("bowl", "024_bowl", "kitchen_counter"),), "destination": "table"},
            library=self.library,
        )
        self.assertEqual(subgoals.execution_order(), ("bowl_delivered",))
        self.assertEqual(len(graph.nodes), 8)
        self.assertEqual(graph.nodes["place_bowl"].arguments["destination"], "table")

        # Steps: what a segment still needs after a failure or a replan.
        patch = builder.propose(
            "goal", "granularity",
            {"steps": {"bowl": ("close",), "apple": ("pick", "place", "close")}},
        )
        self.assertEqual(
            [(item.id, item.predicate) for item in patch.subgoals],
            [
                ("bowl_storage_closed", "closed(kitchen_counter)"),
                ("apple_delivered", "at(013_apple,dining_table)"),
            ],
        )
        self.assertEqual(
            sorted(patch.skill_subgraphs[0].nodes),
            ["close_bowl_source", "navigate_back_to_bowl_source"],
        )
        self.assertEqual(
            patch.skill_subgraphs[1].execution_order(),
            (
                "navigate_to_apple",
                "pick_apple",
                "navigate_apple_to_destination",
                "place_apple",
                "navigate_back_to_apple_source",
                "close_apple_source",
            ),
        )
        fine = SetTableGraphBuilder("fine").propose(
            "goal", "granularity", {"steps": {"bowl": ("place", "close")}}
        )
        self.assertEqual(
            [item.id for item in fine.subgoals][:4],
            ["bowl_destination_reachable", "bowl_placed", "bowl_source_reachable_again", "bowl_source_closed"],
        )
        self.assertEqual(len(fine.subgoals), 12)

        with self.assertRaisesRegex(ValueError, "not task 'set_table'"):
            builder.build("goal", "set_table")
        with self.assertRaisesRegex(ValueError, "granularity must be one of"):
            SetTableGraphBuilder("medium")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            builder.build("goal", "granularity", context={"segments": ()})
        with self.assertRaisesRegex(ValueError, "labels must be distinct"):
            builder.propose(
                "goal", "granularity",
                {"segments": (("bowl", "024_bowl", "fridge"), ("bowl", "013_apple", "fridge"))},
            )
        for steps, message in (
            ({"bowl": ("open", "place")}, "opens its storage without picking"),
            ({"bowl": ("pick",)}, "picks its object without placing"),
            ({"cup": ("close",)}, "steps name segment"),
            ({"bowl": ()}, "at least one step"),
            ({"bowl": ("wash",)}, "unknown steps"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                builder.propose("goal", "granularity", {"steps": steps})

    def test_set_table_graphs_follow_the_packaged_plan(self):
        catalog = LibraryCatalog.from_dict(json.loads(CATALOG_PATH.read_text()))
        packaged = [
            _type_and_target(catalog.skill_graph.nodes[node_id])
            for node_id in catalog.execution_plans["nominal"].order
        ]
        for name in GOLD_GRAPHS:
            gold = build_gold_graph(name, self.library)
            ours = [
                _type_and_target(gold.skill_graph.nodes[node_id])
                for node_id in gold.plan.order
            ]
            self.assertEqual(ours, packaged, name)
            # 20 packaged candidates collapse to 16 nodes, one per role.
            self.assertEqual(len(gold.skill_graph.nodes), 16)
        self.assertEqual(len(catalog.skill_graph.nodes), 20)
        # The packaged graph cuts each object into four sub-goals; the coarse
        # gold graph owns the same nodes by one sub-goal per object.
        self.assertEqual(len(catalog.subgoal_graph.execution_order()), 8)

    def test_segment_helpers_name_and_parse_the_ids(self):
        labels = ("bowl", "apple")
        self.assertEqual(coarse_subgoal_id("bowl"), "bowl_delivered")
        self.assertEqual(closing_subgoal_id("apple"), "apple_storage_closed")
        self.assertEqual(len(fine_subgoal_ids("apple")), 8)
        self.assertEqual(segment_subgoal("bowl_holding", labels), ("bowl", "holding"))
        self.assertEqual(segment_subgoal("apple_delivered", labels), ("apple", "delivered"))
        self.assertEqual(segment_subgoal("bowl_storage_closed", labels), ("bowl", "storage_closed"))
        self.assertEqual(segment_subgoal("bowl_source_reachable_again", labels), ("bowl", "source_reachable_again"))
        self.assertIsNone(segment_subgoal("object_1_delivered", labels))
        self.assertIsNone(segment_subgoal("bowl_delivered", ("apple",)))
        self.assertEqual(segment_node_ids("bowl")["pick"], "pick_bowl")
        self.assertEqual(tuple(segment_node_ids("bowl")[role] for role in (
            "navigate_to_source", "open", "navigate_to_object", "pick",
            "navigate_to_destination", "place", "navigate_back_to_source", "close",
        )), BOWL_NODES)

    def test_documents_round_trip_and_match_the_committed_artifacts(self):
        for name in GOLD_GRAPHS:
            gold = build_gold_graph(name, self.library)
            document = gold_graph_document(gold)
            self.assertNotIn(self.temporary_directory.name, json.dumps(document))
            path = gold_graph_path(name)
            self.assertEqual(document, json.loads(path.read_text()))
            svg = gold_graph_svg(gold)
            ET.fromstring(svg)
            self.assertEqual(svg, path.with_suffix(".svg").read_text())

            subgoals, graph = load_gold_graph(path, self.library)
            self.assertEqual(subgoals.as_dict(), gold.subgoal_graph.as_dict())
            self.assertEqual(graph.as_dict(), gold.skill_graph.as_dict())

    def test_loading_rejects_a_stale_derived_view(self):
        document = json.loads(gold_graph_path("set_table_coarse").read_text())
        document["nominal_plan"]["order"] = list(reversed(document["nominal_plan"]["order"]))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "stale.json"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(SchemaError, "nominal_plan is stale"):
                load_gold_graph(path, self.library)
