import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mshab.experiments.granularity import build_granularity_library
from mshab.experiments.granularity.evaluate import (
    EVALUATION_GOALS,
    Evaluation,
    RejectionTally,
    decomposition_agreement,
    main,
    node_role,
    order_similarity,
    predicate_name,
    rejection_rule,
    scripted_factory,
    subgraph_agreement,
    validity_of,
)
from mshab.experiments.granularity.higher_layers import build_gold_graph
from mshab.experiments.planning import ScriptedProposer


class EvaluationTests(TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.library = build_granularity_library(Path(self.temporary_directory.name))

    def test_registry_covers_the_goal(self):
        self.assertEqual(list(EVALUATION_GOALS), ["set_table"])
        table = EVALUATION_GOALS["set_table"]
        self.assertEqual(table.references, {"coarse": "set_table_coarse", "fine": "set_table_fine"})
        self.assertEqual(table.scenarios[0], "nominal")
        self.assertEqual(len(table.scenarios), 7)
        self.assertEqual(table.reference_names("free"), ("set_table_coarse", "set_table_fine"))
        self.assertEqual(table.reference_names("fine"), ("set_table_fine",))
        self.assertEqual(table.nominal.goal, table.goal)

    def test_the_gold_proposer_agrees_with_its_own_gold_graph(self):
        with TemporaryDirectory() as tmp:
            evaluation = Evaluation(self.library, scripted_factory(self.library), retries=2, output=Path(tmp))
            summary = evaluation.run(["set_table"], ["coarse", "fine"], samples=1)
            for granularity, reference, span in (("coarse", "set_table_coarse", 2), ("fine", "set_table_fine", 9)):
                block = summary["goals"]["set_table"][granularity]
                self.assertEqual(block["validity"], {"accepted_first_try": 1, "accepted_after_retries": 0, "rejected": 0})
                self.assertEqual(block["rounds"], 1.0)
                self.assertEqual(block["calls"], 1 + block["decomposition"]["subgoals"]["mean"])
                self.assertEqual(block["decomposition"]["agreement"][reference]["exact_rate"], 1.0)
                self.assertEqual(block["decomposition"]["agreement"][reference]["order_similarity"], 1.0)
                self.assertEqual(block["subgraphs"][reference]["nodes_match_rate"], 1.0)
                self.assertEqual(block["subgraphs"][reference]["edges_match_rate"], 1.0)
                self.assertEqual(block["subgraphs"][reference]["graph_role_recall"], 1.0)
                self.assertEqual(sorted(block["scenarios"]), sorted(EVALUATION_GOALS["set_table"].scenarios))
                self.assertEqual(block["scenarios"]["nominal"]["success_rate"], 1.0)
                self.assertEqual(block["scenarios"]["nominal"]["statuses"], {"success": 1})
                self.assertEqual(block["scenarios"]["nominal"]["metrics"]["node_executions"], 16)
                self.assertEqual(block["scenarios"]["nominal"]["metrics"]["proposal_retries"], 0)
                self.assertEqual(block["scenarios"]["nominal"]["metrics"]["proposer_calls"], block["calls"])
                self.assertEqual(block["scenarios"]["pick_exhausted"]["success_rate"], 0.0)
                self.assertEqual(block["scenarios"]["pick_exhausted"]["metrics"]["replanning_span"], span)
                self.assertEqual(block["scenarios"]["object_dropped"]["metrics"]["recovery_success"], 1.0)
                self.assertEqual(block["scenarios"]["object_elsewhere"]["success_rate"], 1.0)
                self.assertEqual(block["errors"], 0)
            self.assertEqual(summary["goals"]["set_table"]["coarse"]["decomposition"]["vocabulary"], {"at": 2})
            self.assertEqual(
                summary["goals"]["set_table"]["fine"]["decomposition"]["vocabulary"],
                {"reachable": 8, "open": 2, "holding": 2, "at": 2, "closed": 2},
            )
            self.assertEqual(summary["rejections"], [])
            self.assertEqual(summary["samples"], 2)
            json.dumps(summary)
            root = Path(tmp) / "set_table"
            static = json.loads((root / "coarse" / "sample_00" / "static.json").read_text())
            self.assertTrue(static["accepted"])
            self.assertEqual(len(static["rounds"]), 1)
            self.assertEqual(static["evaluation"]["decomposition"]["subgoals"], 2)
            run = json.loads((root / "fine" / "sample_00" / "scenario_object_dropped.json").read_text())
            self.assertEqual(run["status"], "success")
            self.assertEqual(len(run["proposals"][0]["rounds"]), 1)

    def test_agreement_metrics_separate_the_two_granularities(self):
        coarse = build_gold_graph("set_table_coarse", self.library)
        fine = build_gold_graph("set_table_fine", self.library)
        subgoals = tuple(coarse.subgoal_graph.subgoals[i] for i in coarse.subgoal_graph.execution_order())
        agreement = decomposition_agreement(subgoals, fine.subgoal_graph)
        self.assertFalse(agreement["exact"])
        self.assertEqual(agreement["precision"], 1.0)
        # Recall is over the fine graph's 13 distinct predicates (it reaches
        # each storage and the table twice); order similarity over its 16 rows.
        self.assertAlmostEqual(agreement["recall"], 2 / 13)
        self.assertEqual(agreement["order_similarity"], 0.125)
        self.assertEqual((agreement["subgoals"], agreement["gold_subgoals"]), (2, 16))
        subgraphs = subgraph_agreement(coarse.subgoal_graph, coarse.skill_graph, fine)
        self.assertEqual(subgraphs["matched"], 2)
        self.assertEqual(subgraphs["nodes_match_rate"], 0.0)
        # A coarse segment has seven distinct roles (the storage is navigated
        # to twice); the fine gold subgraph of at(...) has one.
        self.assertAlmostEqual(subgraphs["node_jaccard"], 1 / 7)
        self.assertEqual(subgraphs["edge_jaccard"], 0.0)
        self.assertEqual(subgraphs["graph_role_precision"], 1.0)
        self.assertEqual(subgraphs["graph_role_recall"], 1.0)
        self.assertEqual(len(subgraphs["missing_predicates"]), 11)
        self.assertEqual(subgraphs["unmatched_predicates"], [])
        self.assertEqual(subgraphs["per_subgoal"][0]["gold_nodes"], 1)
        self.assertEqual(
            node_role(coarse.skill_graph.nodes["pick_bowl"]),
            ("pick", (("object", "024_bowl"),)),
        )
        self.assertEqual(order_similarity(["a", "b", "c"], ["a", "c"]), 2 / 3)
        self.assertEqual(order_similarity([], []), 1.0)
        self.assertEqual(predicate_name("at(x,y)"), "at")
        self.assertEqual(predicate_name("gripper_empty()"), "gripper_empty")

    def test_rejections_are_tallied_by_rule(self):
        self.assertEqual(
            rejection_rule("unknown contract 'mshab.granularity.fly.all'; available=['a', 'b']"),
            "unknown contract '...'; available=[...]",
        )
        tally = RejectionTally()
        tally.add("subgraph", "node id 'x' is already used by sub-goal 'y'")
        tally.add("subgraph", "node id 'z' is already used by sub-goal 'w'")
        tally.add("decomposition", "no scripted decomposition for ('granularity',)")
        rows = tally.as_list()
        self.assertEqual((rows[0]["stage"], rows[0]["count"]), ("subgraph", 2))
        self.assertEqual(rows[0]["rule"], "node id '...' is already used by sub-goal '...'")
        self.assertEqual(rows[0]["example"], "node id 'x' is already used by sub-goal 'y'")
        self.assertEqual(tally.total, 3)
        self.assertEqual(validity_of(1, True), "accepted_first_try")
        self.assertEqual(validity_of(2, True), "accepted_after_retries")
        self.assertEqual(validity_of(3, False), "rejected")

        # A proposer with no answers is refused at the decomposition every round.
        evaluation = Evaluation(
            self.library, lambda goal, granularity, scenario: ScriptedProposer(), retries=1
        )
        summary = evaluation.run(["set_table"], ["free"], samples=2, run_scenarios=False)
        block = summary["goals"]["set_table"]["free"]
        self.assertEqual(block["validity"], {"accepted_first_try": 0, "accepted_after_retries": 0, "rejected": 2})
        self.assertEqual(block["rounds"], 2.0)
        self.assertEqual(block["calls"], 2.0)
        self.assertEqual(block["scenarios"], {})
        self.assertEqual(summary["rejections"][0]["stage"], "decomposition")
        self.assertEqual(summary["rejections"][0]["count"], 4)
        self.assertIn("no scripted decomposition", summary["rejections"][0]["rule"])
        with self.assertRaisesRegex(KeyError, "unknown evaluation goal"):
            evaluation.run(["tidy_house"], ["free"], samples=1)
        with self.assertRaisesRegex(ValueError, "granularity must be one of"):
            evaluation.run(["set_table"], ["medium"], samples=1)

    def test_the_scripted_dry_run_writes_a_summary(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            with redirect_stdout(io.StringIO()) as printed:
                summary = main(
                    [
                        "--proposer", "scripted",
                        "--goals", "set_table",
                        "--granularities", "free", "coarse",
                        "--samples", "1",
                        "--scenarios", "nominal", "storage_already_open",
                        "--output", str(output),
                        "--checkpoint-root", tmp,
                    ]
                )
            self.assertEqual(list(summary["goals"]), ["set_table"])
            # The scripted proposer only answers granularities a gold graph was authored at.
            self.assertEqual(list(summary["goals"]["set_table"]), ["coarse"])
            table = summary["goals"]["set_table"]["coarse"]
            self.assertEqual(table["validity"]["accepted_first_try"], 1)
            self.assertEqual(table["decomposition"]["agreement"]["set_table_coarse"]["exact_rate"], 1.0)
            self.assertEqual(sorted(table["scenarios"]), ["nominal", "storage_already_open"])
            self.assertEqual(table["scenarios"]["nominal"]["success_rate"], 1.0)
            self.assertEqual(table["scenarios"]["nominal"]["metrics"]["node_executions"], 16)
            self.assertEqual(table["scenarios"]["storage_already_open"]["metrics"]["admission_failures"], 2)
            document = json.loads((output / "summary.json").read_text())
            self.assertEqual(document["proposer"]["proposer"], "ScriptedProposer")
            self.assertEqual(document["samples"], 1)
            self.assertFalse((output / "exchanges.json").exists())
            output_text = printed.getvalue()
            self.assertIn("[set_table coarse #0] static proposal", output_text)
            self.assertIn("set_table    coarse  n=1   first=1 retried=0 rejected=0", output_text)
            self.assertIn("vs set_table_coarse", output_text)
            self.assertIn("exact=1.00", output_text)
            self.assertNotIn("most frequent rejections", output_text)
            self.assertIn("wrote", output_text)
