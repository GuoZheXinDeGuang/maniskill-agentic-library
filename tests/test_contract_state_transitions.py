"""Contracts as state transitions: what a skill asserts *and* what it retracts.

Without delete effects the symbolic state is monotone, so a SetTable rollout
ends with ``open(kitchen_counter)`` and ``closed(kitchen_counter)`` both true
and the bowl still ``holding`` after it was placed.  These tests pin the
transition semantics that ``BoundContract.deletes``/``apply_to`` restore.
"""

from __future__ import annotations

from unittest import TestCase

from mshab.skills import (
    CloseSkill,
    OpenSkill,
    PickSkill,
    PlaceSkill,
    SkillContract,
    SkillParameter,
    ParameterType,
    SkillPlanner,
    build_set_table_graph,
)
from mshab.skills.library import SkillLibrary
from mshab.skills.model import NavigateSkill


def _set_table_library():
    library = SkillLibrary()
    library.register(NavigateSkill("set_table"))
    for target in ("024_bowl", "013_apple"):
        library.register(PickSkill("set_table", target))
        library.register(PlaceSkill("set_table", target))
    library.register(PickSkill("set_table", "all"))
    library.register(PlaceSkill("set_table", "all"))
    for target in ("kitchen_counter", "fridge"):
        library.register(OpenSkill("set_table", target))
        library.register(CloseSkill("set_table", target))
    return library


class DeleteEffectTests(TestCase):
    def test_open_and_close_retract_each_other(self):
        opened = OpenSkill("set_table", "fridge").bind({}).contract
        closed = CloseSkill("set_table", "fridge").bind({}).contract

        after_open = opened.apply_to({"closed(fridge)", "reachable(fridge)"})
        after_close = closed.apply_to(after_open)

        self.assertIn("open(fridge)", after_open)
        self.assertNotIn("closed(fridge)", after_open)
        self.assertIn("closed(fridge)", after_close)
        self.assertNotIn("open(fridge)", after_close)

    def test_place_releases_the_held_object(self):
        contract = (
            PlaceSkill("set_table", "024_bowl")
            .bind({"destination": "dining_table"})
            .contract
        )

        after = contract.apply_to({"holding(024_bowl)", "reachable(dining_table)"})

        self.assertIn("at(024_bowl,dining_table)", after)
        self.assertIn("gripper_empty()", after)
        self.assertNotIn("holding(024_bowl)", after)

    def test_pick_consumes_the_empty_gripper(self):
        contract = PickSkill("set_table", "024_bowl").bind({}).contract

        after = contract.apply_to({"gripper_empty()", "reachable(024_bowl)"})

        self.assertIn("holding(024_bowl)", after)
        self.assertNotIn("gripper_empty()", after)

    def test_a_predicate_cannot_be_both_asserted_and_retracted(self):
        with self.assertRaisesRegex(ValueError, "asserted and retracted"):
            SkillContract(
                parameters=(SkillParameter("object", ParameterType.ENTITY),),
                preconditions=(),
                effects=("holding({object})",),
                deletes=("holding({object})",),
            )

    def test_a_skill_cannot_retract_its_own_invariant(self):
        with self.assertRaisesRegex(ValueError, "retract its own invariants"):
            SkillContract(
                parameters=(SkillParameter("object", ParameterType.ENTITY),),
                preconditions=(),
                effects=("holding({object})",),
                invariants=("collision_safe()",),
                deletes=("collision_safe()",),
            )

    def test_grounding_cannot_create_an_assert_delete_conflict(self):
        contract = SkillContract(
            parameters=(SkillParameter("object", ParameterType.ENTITY),),
            preconditions=(),
            effects=("holding({object})",),
            deletes=("holding(013_apple)",),
        )

        with self.assertRaisesRegex(ValueError, "grounded predicates"):
            contract.bind({"object": "013_apple"})


class SetTableStateTransitionTests(TestCase):
    """The full 16-step plan must stay symbolically consistent throughout."""

    INITIAL = frozenset(
        {
            "present(kitchen_counter)",
            "present(024_bowl)",
            "present(dining_table)",
            "present(fridge)",
            "present(013_apple)",
            "closed(kitchen_counter)",
            "closed(fridge)",
            "gripper_empty()",
            "collision_safe()",
        }
    )

    def _rollout(self, failed=()):
        """Replay a plan symbolically, returning the final and per-goal states."""

        goals, graph = build_set_table_graph()
        library = _set_table_library()
        plan = SkillPlanner(goals, graph).plan(failed=failed)
        facts = set(self.INITIAL)
        at_goal_completion = {}
        for node_id in plan.order:
            node = graph.nodes[node_id]
            contract = library.get(node.skill_id).bind(node.arguments).contract
            missing = [item for item in contract.preconditions if item not in facts]
            self.assertEqual(
                missing, [], "{} cannot start: missing {}".format(node_id, missing)
            )
            self.assertTrue(
                set(contract.invariants).issubset(facts),
                "{} invariants do not hold".format(node_id),
            )
            facts = set(contract.apply_to(facts))
            for goal_id in node.achieves:
                at_goal_completion[goal_id] = frozenset(facts)
        return goals, facts, at_goal_completion

    def test_every_goal_predicate_holds_when_its_achiever_completes(self):
        # open(kitchen_counter) and holding(bowl) are transient: they are true
        # when their subgraph finishes, and correctly retracted later on.
        goals, _, at_goal_completion = self._rollout()

        self.assertEqual(set(at_goal_completion), set(goals.goals))
        for goal in goals.goals.values():
            self.assertIn(goal.predicate, at_goal_completion[goal.id], goal.id)
            self.assertTrue(
                goals.achieved(goal.id, at_goal_completion[goal.id]), goal.id
            )

    def test_nominal_plan_leaves_no_contradictory_state(self):
        _, facts, _ = self._rollout()

        for container in ("kitchen_counter", "fridge"):
            self.assertIn("closed({})".format(container), facts)
            self.assertNotIn("open({})".format(container), facts)
        for obj in ("024_bowl", "013_apple"):
            self.assertNotIn("holding({})".format(obj), facts)
        self.assertIn("gripper_empty()", facts)

    def test_fallback_rollout_is_symbolically_identical_to_the_nominal_one(self):
        _, nominal, _ = self._rollout()
        _, recovered, _ = self._rollout(
            failed=(
                "pick_bowl_specialized",
                "place_bowl_specialized",
                "pick_apple_specialized",
                "place_apple_specialized",
            )
        )

        self.assertEqual(nominal, recovered)
