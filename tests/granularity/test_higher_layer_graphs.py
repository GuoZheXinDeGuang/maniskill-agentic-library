import json
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mshab.experiments.granularity import build_granularity_library, split_contract_id
from mshab.experiments.granularity.higher_layers import (
    GOLD_GRAPHS,
    TidyHouseGraphBuilder,
    build_gold_graph,
    gold_graph_document,
    gold_graph_path,
    gold_graph_svg,
    load_gold_graph,
)
from mshab.skills import LibraryCatalog, SkillRelation
from mshab.skills.schema import SchemaError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPOSITORY_ROOT / "mshab" / "skills" / "catalogs" / "set_table.json"

# name -> (sub-goals, nodes)
EXPECTED = {
    "tidy_house_coarse": (5, 20),
    "tidy_house_fine": (20, 20),
    "set_table_generic": (8, 16),
}
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
            build_gold_graph("prepare_groceries")

    def test_coarse_and_fine_differ_only_in_sub_goal_ownership(self):
        coarse = build_gold_graph("tidy_house_coarse", self.library)
        fine = build_gold_graph("tidy_house_fine", self.library)

        def roles(gold):
            return {
                node.id: (node.contract_id, dict(node.arguments))
                for node in gold.skill_graph.nodes.values()
            }

        self.assertEqual(roles(coarse), roles(fine))
        self.assertEqual(coarse.plan.order, fine.plan.order)
        self.assertEqual(
            coarse.plan.order[:4],
            (
                "navigate_to_object_1",
                "pick_object_1",
                "navigate_to_destination_1",
                "place_object_1",
            ),
        )
        self.assertEqual(
            coarse.subgoal_graph.execution_order()[:2],
            ("object_1_delivered", "object_2_delivered"),
        )
        self.assertEqual(
            fine.subgoal_graph.execution_order()[:4],
            (
                "object_1_reachable",
                "object_1_holding",
                "destination_1_reachable",
                "object_1_placed",
            ),
        )
        self.assertEqual(
            coarse.subgoal_graph.subgoals["object_1_delivered"].predicate,
            "at(002_master_chef_can,kitchen_counter)",
        )
        self.assertEqual(
            fine.subgoal_graph.subgoals["object_1_placed"].predicate,
            "at(002_master_chef_can,kitchen_counter)",
        )
        self.assertEqual(
            gold_graph_document(coarse)["summary"],
            {
                "subgoals": 5,
                "subgraphs": 5,
                "nodes": 20,
                "internal_edges": 15,
                "cross_edges": 4,
                "mean_nodes_per_subgraph": 4.0,
            },
        )
        self.assertEqual(
            gold_graph_document(fine)["summary"],
            {
                "subgoals": 20,
                "subgraphs": 20,
                "nodes": 20,
                "internal_edges": 0,
                "cross_edges": 19,
                "mean_nodes_per_subgraph": 1.0,
            },
        )

    def test_builders_take_context_and_reject_other_tasks(self):
        builder = TidyHouseGraphBuilder("coarse")
        subgoals, graph = builder.build(
            "Put the bowl on the table",
            "granularity",
            context={"transfers": (("024_bowl", "dining_table"),)},
            library=self.library,
        )
        self.assertEqual(subgoals.execution_order(), ("object_1_delivered",))
        self.assertEqual(len(graph.nodes), 4)
        with self.assertRaisesRegex(ValueError, "not task 'set_table'"):
            builder.build("goal", "set_table")
        with self.assertRaisesRegex(ValueError, "granularity must be one of"):
            TidyHouseGraphBuilder("medium")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            builder.build("goal", "granularity", context={"transfers": ()})

    def test_set_table_generic_follows_the_packaged_plan(self):
        gold = build_gold_graph("set_table_generic", self.library)
        catalog = LibraryCatalog.from_dict(json.loads(CATALOG_PATH.read_text()))
        packaged = [
            _type_and_target(catalog.skill_graph.nodes[node_id])
            for node_id in catalog.execution_plans["nominal"].order
        ]
        ours = [
            _type_and_target(gold.skill_graph.nodes[node_id])
            for node_id in gold.plan.order
        ]
        self.assertEqual(ours, packaged)
        self.assertEqual(
            gold.subgoal_graph.execution_order(),
            catalog.subgoal_graph.execution_order(),
        )
        # 20 packaged candidates collapse to 16 nodes, one per role.
        self.assertEqual(len(catalog.skill_graph.nodes), 20)
        self.assertEqual(len(gold.skill_graph.nodes), 16)

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
        document = json.loads(gold_graph_path("tidy_house_coarse").read_text())
        document["nominal_plan"]["order"] = list(reversed(document["nominal_plan"]["order"]))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "stale.json"
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(SchemaError, "nominal_plan is stale"):
                load_gold_graph(path, self.library)
