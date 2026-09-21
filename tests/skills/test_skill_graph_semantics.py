"""Regression tests for the four Layer-1/2 semantics fixed in graph v1.

Each class pins one behaviour that the structural/count assertions elsewhere
could not have caught: the previous suite was fully green while the fallback
path was a dead end, a sealed subgraph was reopenable, and a rejected patch
left the live graph half-modified.
"""

from __future__ import annotations

import json
from unittest import TestCase

from mshab.skills import (
    SubGoal,
    SubGoalGraph,
    SubGoalDependency,
    SkillSubgraph,
    SkillSubgraphExtension,
    NoViableCandidate,
    SchemaError,
    SkillGraph,
    SkillEdge,
    SkillGraphPatch,
    ContractLibrary,
    SkillNode,
    SkillPlanner,
    SkillRelation,
    CrossSubgraphEdge,
    build_set_table_graph,
)


def _pick(node_id, target, goal, **arguments):
    return SkillNode(
        node_id, "mshab.set_table.pick." + target, arguments, achieves=(goal,)
    )


def _tiny_graph():
    """One goal, one achiever, plus a second goal downstream of it."""

    subgoals = SubGoalGraph("tiny")
    subgoals.add_subgoal(SubGoal("retrieved", "holding(024_bowl)"))
    subgoals.add_subgoal(SubGoal("placed", "at(024_bowl,table)"))
    subgoals.add_dependency("retrieved", "placed")
    graph = SkillGraph("set_table", subgoal_graph=subgoals)

    retrieved = SkillSubgraph("retrieved", "set_table")
    retrieved.add_node(_pick("pick_primary", "024_bowl", "retrieved"))
    graph.add_subgraph(retrieved)

    placed = SkillSubgraph("placed", "set_table")
    placed.add_node(
        SkillNode(
            "place_it",
            "mshab.set_table.place.024_bowl",
            {"destination": "table"},
            achieves=("placed",),
        )
    )
    graph.add_subgraph(placed)
    graph.relate_subgraphs("retrieved", "placed", None, "place_it")
    return subgoals, graph


class FallbackReachabilityTests(TestCase):
    """A goal-level ENABLES must be satisfiable by *any* achiever."""

    def test_recovering_through_the_fallback_unblocks_the_next_goal(self):
        _, graph = build_set_table_graph()
        completed = [
            "navigate_to_bowl_source",
            "open_bowl_source",
            "navigate_to_bowl",
            "pick_bowl_generic",
        ]

        ready = [node.id for node in graph.ready_nodes(completed)]

        # Before the fix this returned only pick_bowl_specialized -- the node
        # that had just failed -- so the rollout could never continue.
        self.assertIn("navigate_bowl_to_destination", ready)

    def test_downstream_prerequisite_is_a_disjunction_over_achievers(self):
        _, graph = build_set_table_graph()

        groups = graph.prerequisite_groups("navigate_bowl_to_destination")

        self.assertEqual(
            [sorted(group) for group in groups],
            [["pick_bowl_generic", "pick_bowl_specialized"]],
        )

    def test_every_cross_goal_relation_survives_a_full_generic_rollout(self):
        subgoals, graph = build_set_table_graph()
        planner = SkillPlanner(subgoals, graph)
        primaries = [
            planner.select_achiever(subgraph).id
            for subgraph in graph.subgraphs.values()
            if len(planner.fallback_chain(subgraph)) > 1
        ]

        recovery = planner.plan(failed=primaries)

        self.assertEqual(len(recovery.order), 16)
        self.assertEqual(recovery.order[-1], "close_apple_source")
        self.assertEqual(recovery.selections["bowl_retrieved"], "pick_bowl_generic")
        self.assertEqual(recovery.selections["apple_placed"], "place_apple_generic")


class ExecutionPlanTests(TestCase):
    """``execution_order()`` ranks candidates; ``SkillPlanner`` picks a path."""

    EXPECTED = (
        "navigate_to_bowl_source",
        "open_bowl_source",
        "navigate_to_bowl",
        "pick_bowl_specialized",
        "navigate_bowl_to_destination",
        "place_bowl_specialized",
        "navigate_back_to_bowl_source",
        "close_bowl_source",
    )

    def test_plan_selects_exactly_one_achiever_per_goal(self):
        subgoals, graph = build_set_table_graph()

        plan = SkillPlanner(subgoals, graph).plan()

        self.assertEqual(len(plan.order), 16)
        self.assertEqual(plan.order[:8], self.EXPECTED)
        self.assertEqual(len(plan.selections), 8)
        self.assertFalse([item for item in plan.order if item.endswith("_generic")])

    def test_candidate_partial_order_is_not_the_plan(self):
        subgoals, graph = build_set_table_graph()

        order = graph.execution_order()
        plan = SkillPlanner(subgoals, graph).plan()

        self.assertEqual(len(order), 20)
        self.assertNotEqual(tuple(order), plan.order)
        # Both picks appear, and the alphabetical tie-break even puts the
        # fallback first: following this order would execute the bowl twice.
        self.assertLess(
            order.index("pick_bowl_generic"), order.index("pick_bowl_specialized")
        )

    def test_exhausting_a_fallback_chain_raises_rather_than_looping(self):
        subgoals, graph = build_set_table_graph()
        planner = SkillPlanner(subgoals, graph)

        with self.assertRaisesRegex(NoViableCandidate, "every achiever"):
            planner.plan(failed=("pick_bowl_specialized", "pick_bowl_generic"))

    def test_goal_dependency_without_a_layer_two_relation_is_rejected(self):
        subgoals, graph = _tiny_graph()
        subgoals.add_subgoal(SubGoal("closed", "closed(fridge)"))
        subgoals.add_dependency("placed", "closed")
        orphan = SkillSubgraph("closed", "set_table")
        orphan.add_node(
            SkillNode(
                "close_it", "mshab.set_table.close.fridge", {}, achieves=("closed",)
            )
        )
        graph.add_subgraph(orphan)

        with self.assertRaisesRegex(ValueError, "without a Layer-2 ENABLES"):
            graph.validate()

    def test_planner_obeys_an_additional_layer_two_dependency(self):
        """Layer 2, not alphabetical Layer-1 order, controls readiness."""

        subgoals = SubGoalGraph("two otherwise independent outcomes")
        subgoals.add_subgoal(SubGoal("a", "a_done()"))
        subgoals.add_subgoal(SubGoal("b", "b_done()"))
        graph = SkillGraph("set_table", subgoal_graph=subgoals)

        for subgoal_id in ("a", "b"):
            subgraph = SkillSubgraph(subgoal_id, "set_table")
            subgraph.add_node(
                SkillNode(
                    subgoal_id,
                    "mshab.set_table.pick.all",
                    {"object": subgoal_id},
                    achieves=(subgoal_id,),
                )
            )
            graph.add_subgraph(subgraph)

        # Layer 1 deliberately leaves A/B unordered.  This Layer-2 relation is
        # therefore the only source of the required B -> A execution order.
        graph.relate("b", "a", SkillRelation.ENABLES)
        planner = SkillPlanner(subgoals, graph)

        self.assertEqual(planner.decide().id, "b")
        self.assertEqual(planner.decide(completed=("b",)).id, "a")
        self.assertEqual(planner.plan().order, ("b", "a"))


class SealingTests(TestCase):
    """A registered subgraph is frozen, not merely guarded on two methods."""

    def test_sealed_subgraph_rejects_every_attribute_assignment(self):
        _, graph = build_set_table_graph()
        subgraph = graph.subgraphs["bowl_retrieved"]

        for name, value in (("subgoal_id", "apple_placed"), ("_sealed", False)):
            with self.assertRaisesRegex(RuntimeError, "registered and sealed"):
                setattr(subgraph, name, value)

        self.assertEqual(subgraph.subgoal_id, "bowl_retrieved")
        self.assertTrue(subgraph.sealed)

    def test_sealed_subgraph_rejects_structural_mutation(self):
        _, graph = build_set_table_graph()
        subgraph = graph.subgraphs["bowl_retrieved"]

        with self.assertRaisesRegex(RuntimeError, "registered and sealed"):
            subgraph.add_node(_pick("smuggled", "all", "bowl_retrieved", object="x"))
        self.assertEqual(len(graph.nodes), 20)

    def test_unsealed_copy_is_independent_of_the_registered_original(self):
        _, graph = build_set_table_graph()

        clone = graph.subgraphs["bowl_retrieved"].unsealed_copy()
        clone.add_node(_pick("extra", "all", "bowl_retrieved", object="024_bowl"))

        self.assertIn("extra", clone.nodes)
        self.assertNotIn("extra", graph.nodes)


class SubgraphExtensionTests(TestCase):
    """Adding a candidate to a goal that is already registered."""

    def test_extension_adds_a_candidate_and_keeps_cross_goal_relations(self):
        subgoals, graph = _tiny_graph()
        patch = SkillGraphPatch().with_extension(
            "retrieved",
            nodes=(_pick("pick_backup", "all", "retrieved", object="024_bowl"),),
            edges=(
                SkillEdge("pick_primary", "pick_backup", SkillRelation.FALLBACK_TO),
            ),
        )

        patch.apply(subgoals, graph)

        self.assertEqual(len(graph.subgraphs["retrieved"].achievers), 2)
        self.assertEqual(graph.owner_of("pick_backup"), "retrieved")
        # The pre-existing goal-level ENABLES now covers the new candidate too.
        self.assertEqual(
            [sorted(group) for group in graph.prerequisite_groups("place_it")],
            [["pick_backup", "pick_primary"]],
        )
        planner = SkillPlanner(subgoals, graph)
        self.assertEqual(
            planner.plan(failed=("pick_primary",)).selections["retrieved"],
            "pick_backup",
        )

    def test_extension_rejects_a_node_id_owned_by_another_goal(self):
        subgoals, graph = _tiny_graph()
        patch = SkillGraphPatch().with_extension(
            "retrieved",
            nodes=(_pick("place_it", "all", "retrieved", object="024_bowl"),),
        )

        with self.assertRaisesRegex(ValueError, "duplicate skill nodes"):
            patch.apply(subgoals, graph)
        self.assertEqual(graph.owner_of("place_it"), "placed")

    def test_extension_rejects_a_node_claiming_another_goal(self):
        subgoals, graph = _tiny_graph()
        patch = SkillGraphPatch().with_extension(
            "retrieved",
            nodes=(_pick("pick_backup", "all", "placed", object="024_bowl"),),
        )

        with self.assertRaisesRegex(ValueError, "achieves"):
            patch.apply(subgoals, graph)


class PatchAtomicityTests(TestCase):
    """A rejected patch must leave Layer 1 and Layer 2 byte-identical."""

    def _snapshot(self, subgoals, graph):
        return json.dumps([subgoals.as_dict(), graph.as_dict()], sort_keys=True)

    def test_rejected_relation_leaves_both_graphs_untouched(self):
        subgoals, graph = _tiny_graph()
        before = self._snapshot(subgoals, graph)
        stranger = SkillSubgraph("inspected", "set_table")
        stranger.add_node(
            SkillNode(
                "inspect", "mshab.set_table.pick.all", {"object": "x"},
                achieves=("inspected",),
            )
        )
        patch = SkillGraphPatch(
            subgoals=(SubGoal("inspected", "inspected(x)"),),
            subgoal_dependencies=(SubGoalDependency("placed", "inspected"),),
            skill_subgraphs=(stranger,),
            cross_edges=(
                CrossSubgraphEdge("placed", "inspected", None, "does_not_exist"),
            ),
        )

        with self.assertRaises(ValueError):
            patch.apply(subgoals, graph)

        self.assertEqual(self._snapshot(subgoals, graph), before)
        self.assertNotIn("inspected", subgoals.subgoals)

    def test_rejected_extension_leaves_the_original_subgraph_installed(self):
        subgoals, graph = _tiny_graph()
        before = self._snapshot(subgoals, graph)
        patch = SkillGraphPatch().with_extension(
            "retrieved",
            nodes=(_pick("place_it", "all", "retrieved", object="x"),),
        )

        with self.assertRaises(ValueError):
            patch.apply(subgoals, graph)

        self.assertEqual(self._snapshot(subgoals, graph), before)

    def test_patch_rejects_foreign_objects_before_touching_the_graph(self):
        class NotASubgraph:
            subgoal_id = "spoofed"
            task = "set_table"
            nodes = {}
            edges = ()

        with self.assertRaisesRegex(TypeError, "SkillSubgraph"):
            SkillGraphPatch(skill_subgraphs=(NotASubgraph(),))

    def test_apply_rejects_a_skill_the_library_does_not_have(self):
        subgoals, graph = _tiny_graph()
        before = self._snapshot(subgoals, graph)
        invented = SkillSubgraph("inspected", "set_table")
        invented.add_node(
            SkillNode(
                "inspect", "mshab.set_table.teleport.moon", {},
                achieves=("inspected",),
            )
        )
        patch = SkillGraphPatch(
            subgoals=(SubGoal("inspected", "inspected(x)"),),
            skill_subgraphs=(invented,),
        )

        with self.assertRaisesRegex(KeyError, "unregistered contracts"):
            patch.apply(subgoals, graph, library=ContractLibrary())

        self.assertEqual(self._snapshot(subgoals, graph), before)


class PatchSchemaTests(TestCase):
    """Untrusted patch documents are parsed strictly or not at all."""

    VALID = {
        "subgoals": [{"id": "inspected", "predicate": "inspected(013_apple)"}],
        "skill_subgraphs": [
            {
                "subgoal_id": "inspected",
                "nodes": [
                    {
                        "id": "inspect_apple",
                        "contract_id": "mshab.set_table.pick.all",
                        "arguments": {"object": "013_apple"},
                        "achieves": ["inspected"],
                    }
                ],
                "edges": [],
            }
        ],
    }

    def test_patch_round_trips_through_dict(self):
        patch = SkillGraphPatch.from_dict(self.VALID, task="set_table")

        restored = SkillGraphPatch.from_dict(
            json.loads(json.dumps(patch.as_dict())), task="set_table"
        )

        self.assertEqual(restored.as_dict(), patch.as_dict())
        self.assertEqual(restored.subgoals[0].id, "inspected")
        self.assertEqual(
            dict(restored.skill_subgraphs[0].nodes["inspect_apple"].arguments),
            {"object": "013_apple"},
        )

    def test_set_table_catalog_subgraphs_round_trip(self):
        _, graph = build_set_table_graph()

        restored = SkillGraph.from_dict(
            json.loads(json.dumps(graph.as_dict()))
        )

        self.assertEqual(restored.as_dict(), graph.as_dict())

    def test_malformed_patches_are_rejected(self):
        cases = {
            "unknown key": {"subgoals": [], "wat": 1},
            "unknown goal key": {"subgoals": [{"id": "g", "predicate": "p()", "x": 1}]},
            "bad identifier": {"subgoals": [{"id": "../etc", "predicate": "p()"}]},
            "unknown relation": {
                "cross_edges": [
                    {"source_subgoal": "a", "target_subgoal": "b", "relation": "teleports"}
                ]
            },
            "nested argument": {
                "skill_subgraphs": [
                    {
                        "subgoal_id": "g",
                        "nodes": [
                            {
                                "id": "n",
                                "contract_id": "mshab.set_table.pick.all",
                                "arguments": {"waypoints": [1, 2, 3]},
                            }
                        ],
                    }
                ]
            },
            "subgoals not a list": {"subgoals": {"id": "g"}},
            "bad schema version": {"schema_version": "something.else"},
        }
        for label, payload in cases.items():
            with self.subTest(label):
                with self.assertRaises((SchemaError, ValueError)):
                    SkillGraphPatch.from_dict(payload, task="set_table")

    def test_nested_arguments_cannot_reach_a_skill_node(self):
        with self.assertRaisesRegex(TypeError, "must be a scalar"):
            SkillNode("n", "mshab.set_table.pick.all", {"waypoints": [1, 2, 3]})

    def test_skill_nodes_are_hashable(self):
        node = _pick("n", "all", "g", object="013_apple")

        self.assertEqual(len({node, _pick("n", "all", "g", object="013_apple")}), 1)


class FallbackShapeTests(TestCase):
    """A skill node falls back inside its own subgraph; a sub-goal is replanned."""

    def test_a_node_declares_at_most_one_fallback(self):
        subgoals, graph = _tiny_graph()
        SkillGraphPatch().with_extension(
            "retrieved",
            nodes=(_pick("pick_backup", "all", "retrieved", object="024_bowl"),),
            edges=(
                SkillEdge("pick_primary", "pick_backup", SkillRelation.FALLBACK_TO),
            ),
        ).apply(subgoals, graph)
        fork = SkillGraphPatch().with_extension(
            "retrieved",
            nodes=(_pick("pick_third", "024_bowl", "retrieved"),),
            edges=(
                SkillEdge("pick_primary", "pick_third", SkillRelation.FALLBACK_TO),
            ),
        )

        with self.assertRaisesRegex(ValueError, "more than one fallback"):
            fork.apply(subgoals, graph)
        self.assertEqual(
            [node.id for node in graph.subgraphs["retrieved"].achievers],
            ["pick_backup", "pick_primary"],
        )

    def test_fallback_chains_cannot_cycle(self):
        subgraph = SkillSubgraph("retrieved", "set_table")
        subgraph.add_node(_pick("pick_primary", "024_bowl", "retrieved"))
        subgraph.add_node(_pick("pick_backup", "all", "retrieved", object="024_bowl"))
        subgraph.relate("pick_primary", "pick_backup", SkillRelation.FALLBACK_TO)

        with self.assertRaisesRegex(ValueError, "FALLBACK_TO cycle"):
            subgraph.relate("pick_backup", "pick_primary", SkillRelation.FALLBACK_TO)
        self.assertEqual(len(subgraph.edges), 1)

    def test_fallback_never_crosses_subgraphs(self):
        subgoals, graph = _tiny_graph()

        with self.assertRaisesRegex(ValueError, "inside one sub-goal subgraph"):
            graph.relate_subgraphs(
                "retrieved", "placed", "pick_primary", "place_it",
                SkillRelation.FALLBACK_TO,
            )
        with self.assertRaisesRegex(ValueError, "inside one sub-goal subgraph"):
            CrossSubgraphEdge(
                "retrieved", "placed", "pick_primary", "place_it",
                SkillRelation.FALLBACK_TO,
            )

    def test_a_new_candidate_joins_the_end_of_the_chain(self):
        """The README ``with_extension`` example: chain, then sub-goal failure."""

        subgoals, graph = build_set_table_graph()
        SkillGraphPatch().with_extension(
            "bowl_retrieved",
            nodes=(
                SkillNode(
                    "pick_bowl_bc",
                    "mshab.set_table.pick.024_bowl",
                    {},
                    achieves=("bowl_retrieved",),
                ),
            ),
            edges=(
                SkillEdge("pick_bowl_generic", "pick_bowl_bc", SkillRelation.FALLBACK_TO),
            ),
        ).apply(subgoals, graph)
        planner = SkillPlanner(subgoals, graph)

        self.assertEqual(
            [
                node.id
                for node in planner.fallback_chain(graph.subgraphs["bowl_retrieved"])
            ],
            ["pick_bowl_specialized", "pick_bowl_generic", "pick_bowl_bc"],
        )
        self.assertEqual(
            planner.plan(
                failed=("pick_bowl_specialized", "pick_bowl_generic")
            ).selections["bowl_retrieved"],
            "pick_bowl_bc",
        )
        with self.assertRaises(NoViableCandidate):
            planner.plan(
                failed=("pick_bowl_specialized", "pick_bowl_generic", "pick_bowl_bc")
            )


class SubGoalSequenceTests(TestCase):
    """Layer 1 arrives as an ordered sequence; the graph stores it as a chain."""

    def test_from_sequence_keeps_the_generating_order(self):
        subgoals = SubGoalGraph.from_sequence(
            "Retrieve the apple",
            (
                SubGoal("source_open", "open(fridge)"),
                SubGoal("retrieved", "holding(013_apple)"),
                SubGoal("source_closed", "closed(fridge)"),
            ),
        )

        self.assertEqual(
            subgoals.execution_order(), ("source_open", "retrieved", "source_closed")
        )
        self.assertEqual(
            [(item.source, item.target) for item in subgoals.dependencies],
            [("source_open", "retrieved"), ("retrieved", "source_closed")],
        )
        self.assertEqual(
            [item.id for item in subgoals.ready_subgoals(())], ["source_open"]
        )

    def test_from_sequence_rejects_a_repeated_sub_goal(self):
        with self.assertRaisesRegex(ValueError, "duplicate sub-goal"):
            SubGoalGraph.from_sequence(
                "goal", (SubGoal("a", "p()"), SubGoal("a", "q()"))
            )

    def test_set_table_layer_one_is_a_sequence(self):
        subgoals, _ = build_set_table_graph()
        rebuilt = SubGoalGraph.from_sequence(
            subgoals.goal,
            tuple(subgoals.subgoals[item] for item in subgoals.execution_order()),
        )

        self.assertEqual(set(rebuilt.dependencies), set(subgoals.dependencies))
        self.assertEqual(rebuilt.execution_order(), subgoals.execution_order())


def _instrumental_fallback_graph():
    """Two sub-goals; the second has an instrumental node with its own fallback."""

    subgoals = SubGoalGraph.from_sequence(
        "fallback",
        (SubGoal("opened", "open(fridge)"), SubGoal("retrieved", "holding(013_apple)")),
    )
    graph = SkillGraph("set_table", subgoal_graph=subgoals)
    opened = SkillSubgraph("opened", "set_table")
    opened.add_node(
        SkillNode("open_fridge", "mshab.set_table.open.fridge", {}, achieves=("opened",))
    )
    graph.add_subgraph(opened)
    retrieved = SkillSubgraph("retrieved", "set_table")
    retrieved.add_node(
        SkillNode("navigate", "mshab.set_table.navigate.all", {"target": "013_apple"})
    )
    retrieved.add_node(
        SkillNode(
            "navigate_slow",
            "mshab.set_table.navigate.all",
            {"target": "013_apple", "speed": "slow"},
        )
    )
    retrieved.add_node(_pick("pick", "013_apple", "retrieved"))
    retrieved.relate("navigate", "pick", SkillRelation.ENABLES)
    retrieved.relate("navigate", "navigate_slow", SkillRelation.FALLBACK_TO)
    graph.add_subgraph(retrieved)
    graph.relate_subgraphs("opened", "retrieved", None, "navigate", SkillRelation.ENABLES)
    return subgoals, graph


class InstrumentalFallbackTests(TestCase):
    """Any node falls back inside its subgraph, not only achievers."""

    def test_nominal_plan_runs_the_primary(self):
        subgoals, graph = _instrumental_fallback_graph()

        self.assertEqual(
            SkillPlanner(subgoals, graph).plan().order,
            ("open_fridge", "navigate", "pick"),
        )

    def test_failed_instrumental_node_is_replaced_by_its_fallback(self):
        subgoals, graph = _instrumental_fallback_graph()

        plan = SkillPlanner(subgoals, graph).plan(failed=("navigate",))

        self.assertEqual(plan.order, ("open_fridge", "navigate_slow", "pick"))

    def test_fallback_inherits_prerequisites_and_satisfies_downstream_edges(self):
        subgoals, graph = _instrumental_fallback_graph()

        # navigate_slow waits for the fridge like its primary does.
        self.assertEqual([node.id for node in graph.ready_nodes(())], ["open_fridge"])
        self.assertEqual(
            [sorted(group) for group in graph.prerequisite_groups("navigate_slow")],
            [["open_fridge"]],
        )
        # pick is enabled by either navigation node.
        self.assertEqual(
            [sorted(group) for group in graph.prerequisite_groups("pick")],
            [["navigate", "navigate_slow"]],
        )
        self.assertIn(
            "pick",
            [node.id for node in graph.ready_nodes(("open_fridge", "navigate_slow"))],
        )

    def test_exhausted_instrumental_chain_fails_the_sub_goal(self):
        subgoals, graph = _instrumental_fallback_graph()

        with self.assertRaisesRegex(NoViableCandidate, "all failed"):
            SkillPlanner(subgoals, graph).plan(failed=("navigate", "navigate_slow"))

    def test_a_fallback_plays_the_same_role_as_its_primary(self):
        subgraph = SkillSubgraph("retrieved", "set_table")
        subgraph.add_node(
            SkillNode("navigate", "mshab.set_table.navigate.all", {"target": "013_apple"})
        )
        subgraph.add_node(_pick("pick", "013_apple", "retrieved"))

        with self.assertRaisesRegex(ValueError, "same role"):
            subgraph.relate("pick", "navigate", SkillRelation.FALLBACK_TO)

    def test_a_fallback_cannot_depend_on_its_primary(self):
        subgraph = SkillSubgraph("retrieved", "set_table")
        subgraph.add_node(_pick("pick", "013_apple", "retrieved"))
        subgraph.add_node(_pick("pick_again", "all", "retrieved", object="013_apple"))
        subgraph.relate("pick", "pick_again", SkillRelation.FALLBACK_TO)

        with self.assertRaisesRegex(ValueError, "one FALLBACK_TO chain"):
            subgraph.relate("pick", "pick_again", SkillRelation.ENABLES)
        self.assertEqual(len(subgraph.edges), 1)

    def test_two_primaries_cannot_share_one_fallback(self):
        subgraph = SkillSubgraph("retrieved", "set_table")
        for node_id in ("nav_a", "nav_b", "nav_c"):
            subgraph.add_node(
                SkillNode(node_id, "mshab.set_table.navigate.all", {"target": node_id})
            )
        subgraph.relate("nav_a", "nav_c", SkillRelation.FALLBACK_TO)

        with self.assertRaisesRegex(ValueError, "fallback of both"):
            subgraph.relate("nav_b", "nav_c", SkillRelation.FALLBACK_TO)


def _follow_up_graph():
    """One sub-goal whose achiever (place) enables two follow-ups (navigate back, close)."""

    subgoals = SubGoalGraph.from_sequence("set", [SubGoal("placed", "at(024_bowl,table)")])
    graph = SkillGraph("set_table", subgoal_graph=subgoals)
    placed = SkillSubgraph("placed", "set_table")
    placed.add_node(SkillNode("nav_table", "mshab.set_table.navigate.all", {"target": "table"}))
    placed.add_node(
        SkillNode("place", "mshab.set_table.place.024_bowl", {"destination": "table"}, achieves=("placed",))
    )
    placed.add_node(SkillNode("nav_back", "mshab.set_table.navigate.all", {"target": "fridge"}))
    placed.add_node(SkillNode("close", "mshab.set_table.close.fridge", {}))
    placed.add_node(SkillNode("close_slow", "mshab.set_table.close.fridge", {}))
    # A node that only shares a prerequisite with the achiever: the setup of
    # some other candidate, not a follow-up.
    placed.add_node(SkillNode("nav_photo", "mshab.set_table.navigate.all", {"target": "camera"}))
    placed.relate("nav_table", "place", SkillRelation.ENABLES)
    placed.relate("nav_table", "nav_photo", SkillRelation.ENABLES)
    placed.relate("place", "nav_back", SkillRelation.ENABLES)
    placed.relate("nav_back", "close", SkillRelation.ENABLES)
    placed.relate("close", "close_slow", SkillRelation.FALLBACK_TO)
    graph.add_subgraph(placed)
    return subgoals, graph


class FollowUpTests(TestCase):
    """Nodes the achiever enables inside its subgraph run after it, before the next sub-goal."""

    def test_follow_ups_are_planned_after_the_achiever(self):
        subgoals, graph = _follow_up_graph()
        planner = SkillPlanner(subgoals, graph)

        self.assertEqual(planner.plan().order, ("nav_table", "place", "nav_back", "close"))
        self.assertEqual(planner.plan().selections, {"placed": "place"})
        self.assertEqual(planner.remaining("placed"), ("nav_table", "place", "nav_back", "close"))
        self.assertEqual(planner.remaining("placed", completed=("nav_table", "place")), ("nav_back", "close"))
        self.assertEqual(planner.remaining("placed", completed=("nav_table", "place", "nav_back", "close")), ())
        self.assertEqual(planner.decide(completed=("nav_table", "place")).id, "nav_back")
        self.assertIsNone(planner.decide(completed=("nav_table", "place", "nav_back", "close")))

    def test_a_failed_follow_up_falls_back_and_an_exhausted_one_fails_the_sub_goal(self):
        subgoals, graph = _follow_up_graph()
        planner = SkillPlanner(subgoals, graph)

        self.assertEqual(planner.plan(failed=("close",)).order, ("nav_table", "place", "nav_back", "close_slow"))
        with self.assertRaisesRegex(NoViableCandidate, "all failed"):
            planner.plan(failed=("close", "close_slow"))
        with self.assertRaisesRegex(NoViableCandidate, "every achiever"):
            planner.remaining("placed", failed=("place",))
