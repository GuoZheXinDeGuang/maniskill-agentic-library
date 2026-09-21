"""The committed figures of the higher-layers experiment regenerate identically
from the committed data, and the DeepSeek proposals in that data rebuild into
valid graphs."""

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

from mshab.experiments.docs.figures.render import (
    DATA_DIR,
    FIGURE_DIR,
    build_graphs,
    build_gold_graph,
    flag_extra_nodes,
    load_data,
    reference_for,
    write_figures,
)


class FigureTests(unittest.TestCase):
    def test_committed_figures_match_the_generator(self):
        with TemporaryDirectory() as tmp:
            written = write_figures(Path(tmp), DATA_DIR)
            self.assertTrue(written)
            for path in written:
                svg = path.read_text()
                ET.fromstring(svg)
                committed = FIGURE_DIR / path.name
                self.assertTrue(committed.exists(), "{} is not committed".format(path.name))
                self.assertEqual(svg, committed.read_text(), "{} is stale".format(path.name))

    def test_every_deepseek_proposal_rebuilds_into_valid_graphs(self):
        statics, scenarios = load_data(DATA_DIR)
        self.assertTrue(statics)
        self.assertTrue(scenarios)
        for record in statics:
            subgoals, graph = build_graphs(record["goal"], record["proposal"])
            self.assertEqual(
                len(subgoals.subgoals), len(record["proposal"]["decomposition"]["subgoals"])
            )
            reference = reference_for(record["granularity"])
            if reference is None:
                continue
            flagged = flag_extra_nodes(subgoals, graph, build_gold_graph(reference))
            achievers = {
                node.id
                for subgoal_id in subgoals.subgoals
                for node in graph.subgraph_for_subgoal(subgoal_id).achievers
            }
            # A flagged node is an extra step; the achiever of a sub-goal always
            # has its gold counterpart, otherwise the predicate itself is new.
            self.assertFalse(set(flagged) & achievers, record["granularity"])
        for record in scenarios:
            for proposal in record["proposals"]:
                if proposal.get("accepted"):
                    build_graphs(record["goal"], proposal)


if __name__ == "__main__":
    unittest.main()
