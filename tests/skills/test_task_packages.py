"""Task-package boundaries must preserve the stable public API."""

from unittest import TestCase

from mshab.skills import build_set_table_graph as public_build_set_table_graph
from mshab.skills.starter import (
    build_set_table_graph as compatibility_build_set_table_graph,
)
from mshab.skills.tasks.set_table import (
    build_set_table_graph as packaged_build_set_table_graph,
)


class TaskPackageTests(TestCase):
    def test_set_table_package_preserves_public_and_compatibility_imports(self):
        self.assertIs(public_build_set_table_graph, packaged_build_set_table_graph)
        self.assertIs(
            compatibility_build_set_table_graph, packaged_build_set_table_graph
        )

    def test_packaged_set_table_graph_keeps_reference_shape(self):
        subgoals, graph = packaged_build_set_table_graph()

        self.assertEqual(len(subgoals.subgoals), 8)
        self.assertEqual(len(graph.subgraphs), 8)
        self.assertEqual(len(graph.nodes), 20)
