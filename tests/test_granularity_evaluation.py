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

    def test_registry_covers_both_goals(self):
        tidy = EVALUATION_GOALS["tidy_house"]
        self.assertEqual(tidy.references, {"coarse": "tidy_house_coarse", "fine": "tidy_house_fine"})
        self.assertEqual(tidy.scenarios[0], "nominal")
        self.assertEqual(len(tidy.scenarios), 5)
        self.assertEqual(tidy.reference_names("free"), ("tidy_house_coarse", "tidy_house_fine"))
        self.assertEqual(tidy.reference_names("fine"), ("tidy_house_fine",))
        table = EVALUATION_GOALS["set_table"]
        self.assertEqual(table.references, {"free": "set_table_generic"})
        self.assertEqual(table.scenarios, ("set_table_nominal",))
        self.assertEqual(table.nominal.goal, table.goal)

    def test_the_gold_proposer_agrees_with_its_own_gold_graph(self):
        with TemporaryDirectory() as tmp:
            evaluation = Evaluation(self.library, scripted_factory(self.library), retries=2, output=Path(tmp))
            summary = evaluation.run(["tidy_house"], ["coarse", "fine"], samples=1)
            for granularity, reference, span in (("coarse", "tidy_house_coarse", 3), ("fine", "tidy_house_fine", 12)):
                block = summary["goals"]["tidy_house"][granularity]
                self.assertEqual(block["validity"], {"accepted_first_try": 1, "accepted_after_retries": 0, "rejected": 0})
                self.assertEqual(block["rounds"], 1.0)
                self.assertEqual(block["calls"], 1 + block["decomposition"]["subgoals"]["mean"])
                self.assertEqual(block["decomposition"]["agreement"][reference]["exact_rate"], 1.0)
                self.assertEqual(block["decomposition"]["agreement"][reference]["order_similarity"], 1.0)
                self.assertEqual(block["subgraphs"][reference]["nodes_match_rate"], 1.0)
                self.assertEqual(block["subgraphs"][reference]["edges_match_rate"], 1.0)
                self.assertEqual(block["subgraphs"][reference]["graph_role_recall"], 1.0)
                self.assertEqual(sorted(block["scenarios"]), sorted(EVALUATION_GOALS["tidy_house"].scenarios))
                self.assertEqual(block["scenarios"]["nominal"]["success_rate"], 1.0)
                self.assertEqual(block["scenarios"]["nominal"]["statuses"], {"success": 1})
                self.assertEqual(block["scenarios"]["nominal"]["metrics"]["node_executions"], 20)
                self.assertEqual(block["scenarios"]["nominal"]["metrics"]["proposal_retries"], 0)
                self.assertEqual(block["scenarios"]["nominal"]["metrics"]["proposer_calls"], block["calls"])
                self.assertEqual(block["scenarios"]["pick_exhausted"]["success_rate"], 0.0)
                self.assertEqual(block["scenarios"]["pick_exhausted"]["metrics"]["replanning_span"], span)
                self.assertEqual(block["scenarios"]["object_dropped"]["metrics"]["recovery_success"], 1.0)
                self.assertEqual(block["errors"], 0)
            self.assertEqual(summary["goals"]["tidy_house"]["coarse"]["decomposition"]["vocabulary"], {"at": 5})
            self.assertEqual(
                summary["goals"]["tidy_house"]["fine"]["decomposition"]["vocabulary"],
                {"reachable": 10, "holding": 5, "at": 5},
            )
            self.assertEqual(summary["rejections"], [])
            self.assertEqual(summary["samples"], 2)
            json.dumps(summary)
            root = Path(tmp) / "tidy_house"
            static = json.loads((root / "coarse" / "sample_00" / "static.json").read_text())
            self.assertTrue(static["accepted"])
            self.assertEqual(len(static["rounds"]), 1)
            self.assertEqual(static["evaluation"]["decomposition"]["subgoals"], 5)
            run = json.loads((root / "fine" / "sample_00" / "scenario_object_dropped.json").read_text())
            self.assertEqual(run["status"], "success")
            self.assertEqual(len(run["proposals"][0]["rounds"]), 1)

    def test_agreement_metrics_separate_the_two_granularities(self):
        coarse = build_gold_graph("tidy_house_coarse", self.library)
        fine = build_gold_graph("tidy_house_fine", self.library)
        subgoals = tuple(coarse.subgoal_graph.subgoals[i] for i in coarse.subgoal_graph.execution_order())
        agreement = decomposition_agreement(subgoals, fine.subgoal_graph)
        self.assertFalse(agreement["exact"])
        self.assertEqual(agreement["precision"], 1.0)
        self.assertEqual(agreement["recall"], 0.25)
        self.assertEqual(agreement["order_similarity"], 0.25)
        self.assertEqual((agreement["subgoals"], agreement["gold_subgoals"]), (5, 20))
        subgraphs = subgraph_agreement(coarse.subgoal_graph, coarse.skill_graph, fine)
        self.assertEqual(subgraphs["matched"], 5)
        self.assertEqual(subgraphs["nodes_match_rate"], 0.0)
        self.assertEqual(subgraphs["node_jaccard"], 0.25)
        self.assertEqual(subgraphs["edge_jaccard"], 0.0)
        self.assertEqual(subgraphs["graph_role_precision"], 1.0)
        self.assertEqual(subgraphs["graph_role_recall"], 1.0)
        self.assertEqual(len(subgraphs["missing_predicates"]), 15)
        self.assertEqual(subgraphs["unmatched_predicates"], [])
        self.assertEqual(subgraphs["per_subgoal"][0]["gold_nodes"], 1)
        self.assertEqual(
            node_role(coarse.skill_graph.nodes["pick_object_1"]),
            ("pick", (("object", "002_master_chef_can"),)),
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
            evaluation.run(["prepare_groceries"], ["free"], samples=1)
        with self.assertRaisesRegex(ValueError, "granularity must be one of"):
            evaluation.run(["set_table"], ["medium"], samples=1)

    def test_the_scripted_dry_run_writes_a_summary(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            with redirect_stdout(io.StringIO()) as printed:
                summary = main(
                    [
                        "--proposer", "scripted",
                        "--goals", "set_table", "tidy_house",
                        "--granularities", "free", "coarse",
                        "--samples", "1",
                        "--scenarios", "nominal", "set_table_nominal",
                        "--output", str(output),
                        "--checkpoint-root", tmp,
                    ]
                )
            self.assertEqual(sorted(summary["goals"]), ["set_table", "tidy_house"])
            # The scripted proposer only answers granularities a gold graph was authored at.
            self.assertEqual(list(summary["goals"]["tidy_house"]), ["coarse"])
            self.assertEqual(list(summary["goals"]["set_table"]), ["free"])
            table = summary["goals"]["set_table"]["free"]
            self.assertEqual(table["validity"]["accepted_first_try"], 1)
            self.assertEqual(table["decomposition"]["agreement"]["set_table_generic"]["exact_rate"], 1.0)
            self.assertEqual(table["scenarios"]["set_table_nominal"]["success_rate"], 1.0)
            self.assertEqual(table["scenarios"]["set_table_nominal"]["metrics"]["node_executions"], 16)
            self.assertEqual(list(summary["goals"]["tidy_house"]["coarse"]["scenarios"]), ["nominal"])
            document = json.loads((output / "summary.json").read_text())
            self.assertEqual(document["proposer"]["proposer"], "ScriptedProposer")
            self.assertEqual(document["samples"], 2)
            self.assertFalse((output / "exchanges.json").exists())
            output_text = printed.getvalue()
            self.assertIn("[set_table free #0] static proposal", output_text)
            self.assertIn("set_table    free    n=1   first=1 retried=0 rejected=0", output_text)
            self.assertIn("vs set_table_generic    exact=1.00", output_text)
            self.assertNotIn("most frequent rejections", output_text)
            self.assertIn("wrote", output_text)
