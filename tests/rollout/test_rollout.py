import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mshab.experiments.granularity import build_granularity_library, contract_id
from mshab.experiments.granularity.higher_layers import (
    DEFAULT_SET_TABLE_SEGMENTS,
    SetTableGraphBuilder,
    build_gold_graph,
    fine_subgoal_ids,
)
from mshab.experiments.planning import (
    DecompositionRequest,
    Failure,
    History,
    Neighbours,
    PlanningContext,
    ProposalRejected,
    ProposalValidator,
    SubgraphRequest,
)
from mshab.experiments.rollout import (
    SceneMeasurements,
    SetTableEpisode,
    SetTableRuleProposer,
    UnsupportedGrounding,
    articulation_instance,
    habitat_instance,
    load_episode,
    object_category,
    read_goal_receptacles,
    remaining_steps,
    segment_label,
)
from mshab.experiments.rollout.run import main, node_subtasks, nominal_facts, step_budget
from mshab.skills.graph import SubGoal


TASK = "granularity"

# The official SetTable episode: the bowl out of the kitchen counter drawer,
# the apple out of the fridge, both onto one table.
OBJECTS = ("024_bowl-0", "013_apple-0")
ARTICULATIONS = (("kitchen_counter", "kitchen_counter-0"), ("fridge", "fridge-0"))
RECEPTACLES = ("frl_apartment_table_02_:0000", "frl_apartment_table_02_:0000")


def _corners(tag):
    offset = float(ord(tag))
    return [[offset, 0.0, 0.5], [offset + 1.0, 0.0, 0.5], [offset + 1.0, 1.0, 0.5], [offset, 1.0, 0.5]]


def make_plan(objects=OBJECTS, articulations=ARTICULATIONS, rectangles=("A", "A"),
              init_config_name="train/set_table/episode_7.json"):
    subtasks = []
    for number, (obj, (articulation_type, articulation_id), tag) in enumerate(
        zip(objects, articulations, rectangles)
    ):
        uid = "set_table-sequential-train-0-{}"
        config = {
            "articulation_type": articulation_type,
            "articulation_id": articulation_id,
            "articulation_handle_link_idx": 3,
            "articulation_handle_active_joint_idx": 1,
        }
        navigate = {"obj_id": None, "goal_pos": None, "articulation_config": None}
        subtasks += [
            {"type": "navigate", "uid": uid.format(8 * number), **navigate},
            {"type": "open", "uid": uid.format(8 * number + 1), "obj_id": obj,
             "articulation_relative_handle_pos": [0.1, 0.2, 0.3], **config},
            {"type": "navigate", "uid": uid.format(8 * number + 2), **navigate},
            {"type": "pick", "uid": uid.format(8 * number + 3), "obj_id": obj, "articulation_config": config},
            {"type": "navigate", "uid": uid.format(8 * number + 4), **navigate},
            {"type": "place", "uid": uid.format(8 * number + 5), "obj_id": obj,
             "goal_rectangle_corners": _corners(tag), "goal_pos": [float(ord(tag)) + 0.5, 0.5, 0.6],
             "validate_goal_rectangle_corners": True, "articulation_config": None},
            {"type": "navigate", "uid": uid.format(8 * number + 6), **navigate},
            {"type": "close", "uid": uid.format(8 * number + 7),
             "articulation_relative_handle_pos": [0.1, 0.2, 0.3], "remove_obj_id": None, **config},
        ]
    return {
        "subtasks": subtasks,
        "build_config_name": "v3_sc0_staging_00.scene_instance.json",
        "init_config_name": init_config_name,
    }


def make_episode_config(objects=OBJECTS, receptacles=RECEPTACLES):
    labels = {}
    for obj in objects:
        category, number = obj.rsplit("-", 1)
        labels["{}_:{:04d}".format(category, int(number))] = "any_targets|{}".format(len(labels))
    return {
        "episode_id": "7",
        "info": {"object_labels": labels},
        "target_receptacles": [["kitchen_counter_:0000", 3], ["fridge_:0000", 1]],
        "goal_receptacles": [[name, 0] for name in receptacles],
    }


def make_episode(**kwargs):
    return SetTableEpisode.from_plan(
        make_plan(), 0, dataset="ReplicaCADSetTableTrain", goal_receptacles=RECEPTACLES, **kwargs
    )


def gold_episode():
    """An episode whose segments are the gold graphs' default ones."""

    return SetTableEpisode.from_plan(
        make_plan(), 0, dataset="test", goal_receptacles=("dining_table_:0000",) * 2
    )


class EpisodeTests(TestCase):
    def test_names_entities_and_the_goal(self):
        episode = make_episode()
        self.assertEqual(object_category("024_bowl-0"), "024_bowl")
        self.assertEqual(articulation_instance("kitchen_counter-0"), ("kitchen_counter", 0))
        self.assertEqual(habitat_instance("frl_apartment_table_02_:0000"), ("frl_apartment_table_02", 0))
        self.assertEqual(segment_label("024_bowl"), "bowl")
        self.assertEqual(segment_label("024_bowl_2"), "bowl_2")
        self.assertEqual(segment_label("bowl"), "bowl")
        with self.assertRaisesRegex(ValueError, "object id"):
            object_category("024_bowl")
        with self.assertRaisesRegex(ValueError, "articulation id"):
            articulation_instance("fridge")
        self.assertEqual(episode.objects, ("024_bowl", "013_apple"))
        self.assertEqual(episode.labels, ("bowl", "apple"))
        self.assertEqual(episode.sources, ("kitchen_counter", "fridge"))
        self.assertEqual(episode.destinations, ("frl_apartment_table_02",))
        self.assertEqual([item.triple for item in episode.segments], list(DEFAULT_SET_TABLE_SEGMENTS))
        # The goal names the results, not which storage holds which object.
        self.assertEqual(
            episode.goal,
            "Set the table: put 024_bowl and 013_apple on frl_apartment_table_02, "
            "then close the storage they came from.",
        )
        self.assertEqual(
            episode.goal_facts,
            (
                "at(024_bowl,frl_apartment_table_02)",
                "at(013_apple,frl_apartment_table_02)",
                "closed(kitchen_counter)",
                "closed(fridge)",
            ),
        )
        self.assertEqual(
            [(item.name, item.kind) for item in episode.entities()],
            [
                ("024_bowl", "object"), ("013_apple", "object"),
                ("kitchen_counter", "articulation"), ("fridge", "articulation"),
                ("frl_apartment_table_02", "receptacle"),
            ],
        )
        environment_entities = {item.name: item for item in episode.environment_entities()}
        self.assertEqual(environment_entities["024_bowl"].environment_name, "024_bowl-0")
        self.assertEqual(environment_entities["024_bowl"].attributes["category"], "024_bowl")
        self.assertEqual(environment_entities["fridge"].environment_name, "fridge-0")
        self.assertEqual(environment_entities["fridge"].attributes["articulation_type"], "fridge")
        self.assertEqual(
            environment_entities["frl_apartment_table_02"].environment_name, "frl_apartment_table_02_:0000"
        )
        self.assertEqual(environment_entities["frl_apartment_table_02"].attributes["segments"], (1, 2))
        self.assertEqual(episode.subtask_count, 16)
        self.assertEqual(episode.segments[1].subtasks, (8, 9, 10, 11, 12, 13, 14, 15))
        self.assertEqual(episode.segments[1].subtask_of("close"), 15)
        document = json.loads(json.dumps(episode.as_dict()))
        self.assertEqual(document["segments"][1]["subtasks"]["place"], 13)
        self.assertEqual(document["goal"], episode.goal)

    def test_grounded_nodes_map_to_plan_subtasks(self):
        episode = make_episode()
        table = "frl_apartment_table_02"
        self.assertEqual(episode.subtask_for("navigate", {"target": "kitchen_counter"}), 0)
        self.assertEqual(episode.subtask_for("open", {"articulation": "kitchen_counter"}), 1)
        self.assertEqual(episode.subtask_for("navigate", {"target": "024_bowl"}), 2)
        self.assertEqual(episode.subtask_for("pick", {"object": "024_bowl"}), 3)
        self.assertEqual(episode.subtask_for("navigate", {"target": table}), 4)
        self.assertEqual(episode.subtask_for("place", {"object": "024_bowl", "destination": table}), 5)
        # A navigation to an open storage is the one before its close.
        self.assertEqual(episode.subtask_for("navigate", {"target": "kitchen_counter"}, ["open(kitchen_counter)"]), 6)
        self.assertEqual(episode.subtask_for("close", {"articulation": "kitchen_counter"}), 7)
        self.assertEqual(episode.subtask_for("navigate", {"target": "fridge"}), 8)
        self.assertEqual(episode.subtask_for("open", {"articulation": "fridge"}), 9)
        # The shared table resolves to the segment whose object is held, else the
        # first not yet delivered, else the first.
        self.assertEqual(episode.subtask_for("navigate", {"target": table}, ["holding(013_apple)"]), 12)
        self.assertEqual(episode.subtask_for("navigate", {"target": table}, ["at(024_bowl,{})".format(table)]), 12)
        self.assertEqual(
            episode.subtask_for(
                "navigate", {"target": table},
                ["at(024_bowl,{})".format(table), "at(013_apple,{})".format(table)],
            ),
            4,
        )
        with self.assertRaisesRegex(UnsupportedGrounding, "neither an object, a storage, nor a receptacle"):
            episode.subtask_for("navigate", {"target": "dining_table"})
        with self.assertRaisesRegex(UnsupportedGrounding, "not an object of this episode"):
            episode.subtask_for("pick", {"object": "003_cracker_box"})
        with self.assertRaisesRegex(UnsupportedGrounding, "delivers 024_bowl to frl_apartment_table_02, not kitchen_counter"):
            episode.subtask_for("place", {"object": "024_bowl", "destination": "kitchen_counter"})
        with self.assertRaisesRegex(UnsupportedGrounding, "not a storage of this episode"):
            episode.subtask_for("open", {"articulation": "sofa"})
        with self.assertRaisesRegex(UnsupportedGrounding, "no 'wash' subtask"):
            episode.subtask_for("wash", {"object": "024_bowl"})
        self.assertEqual(
            [episode.subtask_label(index) for index in range(8)],
            [
                "navigate(kitchen_counter)", "open(kitchen_counter)", "navigate(024_bowl)", "pick(024_bowl)",
                "navigate(frl_apartment_table_02)", "place(024_bowl,frl_apartment_table_02)",
                "navigate(kitchen_counter)", "close(kitchen_counter)",
            ],
        )
        with self.assertRaises(IndexError):
            episode.subtask_label(16)

    def test_duplicate_categories_shared_storages_and_unnamed_receptacles(self):
        # Two bowls, both in the one fridge: distinct object names and labels,
        # one storage entity, one closed(fridge) goal fact.
        plan = make_plan(("024_bowl-0", "024_bowl-1"), (("fridge", "fridge-0"), ("fridge", "fridge-0")), ("A", "B"))
        episode = SetTableEpisode.from_plan(plan, 0)
        self.assertEqual(episode.objects, ("024_bowl", "024_bowl_2"))
        self.assertEqual(episode.labels, ("bowl", "bowl_2"))
        self.assertEqual(episode.sources, ("fridge",))
        self.assertEqual(episode.destinations, ("receptacle_1", "receptacle_2"))
        self.assertIsNone(episode.segments[0].receptacle_instance)
        self.assertEqual(episode.goal_facts[-1], "closed(fridge)")
        self.assertEqual(
            episode.goal,
            "Set the table: put 024_bowl on receptacle_1 and 024_bowl_2 on receptacle_2, "
            "then close the storage they came from.",
        )
        # A shared storage resolves to the segment that still needs it.
        self.assertEqual(episode.subtask_for("open", {"articulation": "fridge"}), 1)
        self.assertEqual(episode.subtask_for("open", {"articulation": "fridge"}, ["at(024_bowl,receptacle_1)"]), 9)
        self.assertEqual(episode.subtask_for("close", {"articulation": "fridge"}, ["holding(024_bowl_2)"]), 15)
        # Two fridge instances are two storages, numbered by instance id.
        two = make_plan(("024_bowl-0", "013_apple-0"), (("fridge", "fridge-1"), ("fridge", "fridge-0")))
        self.assertEqual(SetTableEpisode.from_plan(two, 0).sources, ("fridge_2", "fridge"))
        # Objects too: the plan may visit the second bowl first.
        swapped = SetTableEpisode.from_plan(
            make_plan(("024_bowl-1", "024_bowl-0"), (("fridge", "fridge-0"), ("fridge", "fridge-0"))), 0
        )
        self.assertEqual(swapped.objects, ("024_bowl_2", "024_bowl"))
        with self.assertRaisesRegex(ValueError, "goal receptacles for"):
            SetTableEpisode.from_plan(plan, 0, goal_receptacles=RECEPTACLES[:1])

    def test_plan_shape_is_checked(self):
        plan = make_plan()
        plan["subtasks"][5]["obj_id"] = "013_apple-0"
        with self.assertRaisesRegex(ValueError, "picks '024_bowl-0' but places"):
            SetTableEpisode.from_plan(plan, 0)
        plan = make_plan()
        plan["subtasks"][7]["articulation_id"] = "fridge-0"
        with self.assertRaisesRegex(ValueError, "opens 'kitchen_counter-0' but closes"):
            SetTableEpisode.from_plan(plan, 0)
        plan = make_plan()
        plan["subtasks"].pop()
        with self.assertRaisesRegex(ValueError, "8 subtasks per segment"):
            SetTableEpisode.from_plan(plan, 0)
        plan = make_plan()
        plan["subtasks"][0]["type"] = "pick"
        with self.assertRaisesRegex(ValueError, "segment 1 is"):
            SetTableEpisode.from_plan(plan, 0)

    def test_facts_follow_the_measurements(self):
        episode = make_episode()
        bowl, apple = episode.segments
        present = {"present({})".format(item.name) for item in episode.entities()}
        # Nothing measured: nothing is held, at its goal, reached, open, or closed.
        self.assertEqual(
            episode.facts(SceneMeasurements({}, {}, {})), present | {"gripper_empty()", "collision_safe()"}
        )
        start = SceneMeasurements(
            grasped={bowl.pick: False, apple.pick: False},
            at_goal={bowl.place: False, apple.place: False},
            near={index: False for item in episode.segments for index in item.subtasks[::2]},
            opened={bowl.open: False, bowl.close: False, apple.open: False, apple.close: False},
            closed={bowl.open: True, bowl.close: True, apple.open: True, apple.close: True},
        )
        self.assertEqual(
            episode.facts(start),
            present | {"gripper_empty()", "collision_safe()", "closed(kitchen_counter)", "closed(fridge)"},
        )
        self.assertEqual(set(nominal_facts(episode)), episode.facts(start))
        midway = SceneMeasurements(
            grasped={bowl.pick: True},
            at_goal={bowl.place: True, apple.place: True},
            near={bowl.navigate_to_destination: True, apple.navigate_to_source: True},
            opened={bowl.open: True, apple.open: False},
            closed={bowl.close: False, apple.close: False, apple.open: False},
            collision_safe=False,
        )
        facts = episode.facts(midway)
        self.assertIn("holding(024_bowl)", facts)
        self.assertNotIn("gripper_empty()", facts)
        self.assertNotIn("collision_safe()", facts)
        # A grasped object is not at its goal, whatever the distance says.
        self.assertNotIn("at(024_bowl,frl_apartment_table_02)", facts)
        self.assertIn("at(013_apple,frl_apartment_table_02)", facts)
        self.assertIn("reachable(frl_apartment_table_02)", facts)
        self.assertIn("reachable(fridge)", facts)
        # An object in the gripper is reachable wherever the robot stands.
        self.assertIn("reachable(024_bowl)", facts)
        self.assertNotIn("reachable(013_apple)", facts)
        self.assertIn("open(kitchen_counter)", facts)
        self.assertNotIn("closed(kitchen_counter)", facts)
        # A measured storage that is not closed counts as open: a half-open
        # fridge can be closed but not opened.
        self.assertIn("open(fridge)", facts)
        self.assertNotIn("closed(fridge)", facts)
        self.assertEqual(SceneMeasurements({1: 1}, {}, {}).grasped[1], True)
        with self.assertRaisesRegex(TypeError, "subtask indices"):
            SceneMeasurements({}, {}, {}, opened={"1": True})
        json.dumps(midway.as_dict())

    def test_official_files_are_read_and_checked(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = root / "task_plans" / "set_table" / "sequential" / "train" / "all.json"
            plan_path.parent.mkdir(parents=True)
            plan_path.write_text(json.dumps({"dataset": "ReplicaCADSetTableTrain", "plans": [make_plan(), make_plan()]}))
            config_path = root / "train" / "set_table" / "episode_7.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(make_episode_config()))

            episode, document = load_episode(plan_path, 1, root)
            self.assertEqual(episode.plan_index, 1)
            self.assertEqual(episode.dataset, "ReplicaCADSetTableTrain")
            self.assertEqual(episode.destinations, ("frl_apartment_table_02",))
            self.assertEqual(document["dataset"], "ReplicaCADSetTableTrain")
            self.assertEqual(len(document["plans"]), 1)
            self.assertEqual(document["plans"][0]["subtasks"][3]["obj_id"], "024_bowl-0")

            # Without the config the receptacle is numbered by goal rectangle.
            unnamed, _ = load_episode(plan_path, 0, None)
            self.assertEqual(unnamed.destinations, ("receptacle_1",))
            self.assertIsNone(read_goal_receptacles(root, {"init_config_name": "train/set_table/episode_9.json"}))

            # A config whose objects disagree with the plan is refused.
            config_path.write_text(json.dumps(make_episode_config(objects=OBJECTS[::-1])))
            with self.assertRaisesRegex(ValueError, "lists objects"):
                load_episode(plan_path, 0, root)
            with self.assertRaises(IndexError):
                load_episode(plan_path, 2, None)


class RuleProposerTests(TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.library = build_granularity_library(Path(self.temporary_directory.name), task_families=("set_table",))
        self.validator = ProposalValidator(self.library, TASK)

    def context(self, episode, granularity, facts=None, failure=None, attempt=0, history=None):
        context = PlanningContext.initial(
            self.library, TASK, entities=episode.entities(),
            facts=nominal_facts(episode) if facts is None else facts, granularity=granularity,
        )
        if failure is None:
            return context
        return PlanningContext(
            contracts=context.contracts, entities=context.entities, facts=context.facts,
            history=history or History(), failure=failure, granularity=granularity, attempt=attempt,
        )

    def test_remaining_steps_follow_the_facts(self):
        bowl = gold_episode().segments[0]
        closed = {"closed(kitchen_counter)", "gripper_empty()"}
        self.assertEqual(remaining_steps(bowl, closed), ("open", "pick", "place", "close"))
        self.assertEqual(remaining_steps(bowl, {"open(kitchen_counter)"}), ("pick", "place", "close"))
        self.assertEqual(remaining_steps(bowl, {"open(kitchen_counter)", "holding(024_bowl)"}), ("place", "close"))
        self.assertEqual(remaining_steps(bowl, {"open(kitchen_counter)", "at(024_bowl,dining_table)"}), ("close",))
        self.assertEqual(remaining_steps(bowl, {"closed(kitchen_counter)", "at(024_bowl,dining_table)"}), ())
        # A given-up object is not fetched, but its open storage is still closed.
        self.assertEqual(remaining_steps(bowl, {"open(kitchen_counter)"}, given_up=True), ("close",))
        self.assertEqual(remaining_steps(bowl, closed, given_up=True), ())

    def test_reproduces_the_gold_graphs_on_the_official_episode(self):
        episode = gold_episode()
        self.assertEqual(tuple(item.triple for item in episode.segments), DEFAULT_SET_TABLE_SEGMENTS)
        for granularity in ("coarse", "fine"):
            proposer = SetTableRuleProposer(episode)
            proposal = self.validator.plan(proposer, episode.goal, self.context(episode, granularity))
            gold = build_gold_graph("set_table_" + granularity, self.library)
            self.assertEqual(proposal.patch.as_dict(), gold.patch.as_dict())
            self.assertEqual(proposal.plan.order, gold.plan.order)
            self.assertEqual(len(proposal.rounds), 1)
            # Every node stands for the official subtask at the same position
            # once the facts are rolled forward: the second navigation to a
            # storage is the one before its close only while it stands open.
            rows = node_subtasks(
                proposal.plan.order, proposal.skill_graph, self.library, episode, nominal_facts(episode)
            )
            self.assertEqual([row["subtask_index"] for row in rows], list(range(16)))
        self.assertEqual(SetTableRuleProposer(episode).describe()["proposer"], "SetTableRuleProposer")

    def test_answers_a_real_episode_at_both_granularities(self):
        episode = make_episode()
        for granularity, subgoals in (("coarse", 2), ("fine", 16)):
            proposal = self.validator.plan(
                SetTableRuleProposer(episode), episode.goal, self.context(episode, granularity)
            )
            self.assertEqual(len(proposal.subgoal_graph.subgoals), subgoals)
            self.assertEqual(len(proposal.skill_graph.nodes), 16)
            self.assertEqual(
                proposal.skill_graph.nodes["place_apple"].arguments["destination"], "frl_apartment_table_02"
            )

    def test_replans_keep_a_failed_segment_once_then_give_its_object_up(self):
        episode = gold_episode()
        proposer = SetTableRuleProposer(episode, give_up_after=2)
        # The drawer is open and the robot stands at the bowl when its pick fails.
        facts = sorted(
            set(nominal_facts(episode)) - {"closed(kitchen_counter)"}
            | {"open(kitchen_counter)", "reachable(024_bowl)"}
        )
        failure = Failure("bowl_delivered", "pick_bowl", "execution_timeout", ("holding(024_bowl)",))
        history = History((), ("navigate_to_bowl_source", "open_bowl_source", "navigate_to_bowl"), ("pick_bowl",))

        first = proposer.decompose(DecompositionRequest(TASK, episode.goal, self.context(episode, "coarse", facts, failure, 1, history)))
        self.assertEqual([item.id for item in first.subgoals], ["bowl_delivered", "apple_delivered"])
        self.assertIn("failed 1 time(s)", first.rationale)
        # The retried segment skips the opening the facts already show.
        request = SubgraphRequest(
            TASK, episode.goal, first.subgoals[0], PlanningContext.initial(self.library, TASK).contracts,
            entities=episode.entities(), facts=facts, neighbours=Neighbours(None, first.subgoals[1].predicate),
        )
        response = proposer.plan_subgraph(request)
        self.assertEqual(
            response.subgraph.execution_order(),
            ("navigate_to_bowl", "pick_bowl", "navigate_bowl_to_destination", "place_bowl",
             "navigate_back_to_bowl_source", "close_bowl_source"),
        )
        self.assertIn("['pick', 'place', 'close']", response.rationale)

        second = proposer.decompose(DecompositionRequest(TASK, episode.goal, self.context(episode, "coarse", facts, failure, 2, history)))
        self.assertEqual([item.id for item in second.subgoals], ["bowl_storage_closed", "apple_delivered"])
        self.assertEqual(second.subgoals[0].predicate, "closed(kitchen_counter)")
        self.assertIn("gave up the object(s) of segment(s) ['bowl']", second.rationale)
        self.assertEqual(proposer.failures["bowl"], 2)
        # The whole replanned graph validates and plans: the drawer is closed first.
        proposal = self.validator.plan(proposer, episode.goal, self.context(episode, "coarse", facts, failure, 3, history))
        self.assertEqual(proposal.plan.order[:2], ("navigate_back_to_bowl_source", "close_bowl_source"))
        self.assertEqual(len(proposal.skill_graph.nodes), 10)
        self.assertEqual(proposer.failures["bowl"], 3)
        # Fine ids name the segment too.
        fine = proposer.decompose(DecompositionRequest(TASK, episode.goal, self.context(episode, "fine", facts, Failure("bowl_holding", "pick_bowl", "execution_timeout"), 4, history)))
        self.assertEqual([item.id for item in fine.subgoals][:2], ["bowl_source_reachable_again", "bowl_source_closed"])
        self.assertEqual([item.id for item in fine.subgoals][2:], list(fine_subgoal_ids("apple")))
        self.assertEqual(proposer.failures["bowl"], 4)
        # A failure the ids do not name is not counted.
        proposer.decompose(DecompositionRequest(TASK, episode.goal, self.context(episode, "fine", facts, Failure("cup_placed", None, "no_viable_candidate"), 5, history)))
        self.assertEqual(sum(proposer.failures.values()), 4)

        # A segment whose object is in the gripper is never given up.
        held = sorted(set(facts) - {"gripper_empty()"} | {"holding(024_bowl)"})
        kept = proposer.decompose(DecompositionRequest(TASK, episode.goal, self.context(episode, "coarse", held)))
        self.assertEqual([item.id for item in kept.subgoals], ["bowl_delivered", "apple_delivered"])
        self.assertIn("kept although given up", kept.rationale)
        everything = sorted(set(nominal_facts(episode)) | set(episode.goal_facts))
        with self.assertRaisesRegex(ValueError, "nothing left to plan"):
            proposer.decompose(DecompositionRequest(TASK, episode.goal, self.context(episode, "coarse", everything)))
        with self.assertRaisesRegex(ValueError, "decomposes at"):
            proposer.decompose(DecompositionRequest(TASK, episode.goal, self.context(episode, "free")))
        with self.assertRaisesRegex(ValueError, "answers for task"):
            proposer.decompose(DecompositionRequest("set_table", episode.goal, self.context(episode, "coarse")))
        with self.assertRaisesRegex(ValueError, "at least 1"):
            SetTableRuleProposer(episode, give_up_after=0)

    def test_a_held_object_is_delivered_first_without_a_pick(self):
        episode = gold_episode()
        proposer = SetTableRuleProposer(episode)
        # The bowl is done and its drawer shut; the apple is in the gripper, the fridge open.
        holding = sorted(
            set(nominal_facts(episode)) - {"gripper_empty()", "closed(fridge)"}
            | {"holding(013_apple)", "open(fridge)", "at(024_bowl,dining_table)"}
        )
        decomposition = proposer.decompose(DecompositionRequest(TASK, episode.goal, self.context(episode, "coarse", holding)))
        self.assertEqual([item.id for item in decomposition.subgoals], ["apple_delivered"])
        request = SubgraphRequest(
            TASK, episode.goal, decomposition.subgoals[0], PlanningContext.initial(self.library, TASK).contracts,
            entities=episode.entities(), facts=holding, neighbours=Neighbours(None, None),
        )
        response = proposer.plan_subgraph(request)
        self.assertEqual(
            response.subgraph.execution_order(),
            ("navigate_apple_to_destination", "place_apple", "navigate_back_to_apple_source", "close_apple_source"),
        )
        self.assertEqual(response.subgraph.achievers[0].id, "place_apple")
        proposal = self.validator.plan(proposer, episode.goal, self.context(episode, "coarse", holding))
        self.assertEqual(len(proposal.skill_graph.nodes), 4)
        fine = self.validator.plan(proposer, episode.goal, self.context(episode, "fine", holding))
        self.assertEqual(
            fine.subgoal_graph.execution_order(),
            ("apple_destination_reachable", "apple_placed", "apple_source_reachable_again", "apple_source_closed"),
        )
        # Both objects held first: the held one comes before the untouched one.
        bowl_held = sorted(
            set(nominal_facts(episode)) - {"gripper_empty()", "closed(fridge)"} | {"holding(013_apple)", "open(fridge)"}
        )
        first = proposer.decompose(DecompositionRequest(TASK, episode.goal, self.context(episode, "coarse", bowl_held)))
        self.assertEqual([item.id for item in first.subgoals], ["apple_delivered", "bowl_delivered"])

    def test_subgraph_requests_are_checked(self):
        episode = make_episode()
        proposer = SetTableRuleProposer(episode)
        contracts = PlanningContext.initial(self.library, TASK).contracts

        def request(subgoal, facts=None):
            return SubgraphRequest(
                TASK, episode.goal, subgoal, contracts, entities=episode.entities(),
                facts=nominal_facts(episode) if facts is None else facts,
            )

        with self.assertRaisesRegex(ValueError, "not one the rule proposer decomposes into"):
            proposer.plan_subgraph(request(SubGoal("cup_placed", "at(024_bowl,frl_apartment_table_02)")))
        with self.assertRaisesRegex(ValueError, "should read"):
            proposer.plan_subgraph(request(SubGoal("bowl_delivered", "at(024_bowl,kitchen_counter)")))
        with self.assertRaisesRegex(ValueError, "leave nothing to do"):
            proposer.plan_subgraph(request(
                SubGoal("bowl_delivered", "at(024_bowl,frl_apartment_table_02)"),
                sorted(set(nominal_facts(episode)) | {"at(024_bowl,frl_apartment_table_02)"}),
            ))
        response = proposer.plan_subgraph(request(SubGoal("bowl_delivered", "at(024_bowl,frl_apartment_table_02)")))
        self.assertEqual(len(response.subgraph.nodes), 8)
        response = proposer.plan_subgraph(request(SubGoal("apple_source_open", "open(fridge)")))
        self.assertEqual(list(response.subgraph.nodes), ["open_apple_source"])
        # When everything is done the rules have nothing to answer, which the
        # validator records as a decomposition rejection.
        with self.assertRaises(ProposalRejected) as caught:
            self.validator.plan(
                proposer, episode.goal,
                self.context(episode, "coarse", sorted(set(nominal_facts(episode)) | set(episode.goal_facts))),
            )
        self.assertEqual(caught.exception.rejections[0].stage, "decomposition")
        self.assertIn("nothing left to plan", caught.exception.rejections[0].message)


class BuilderStepsTests(TestCase):
    def test_steps_select_the_segment_roles(self):
        builder = SetTableGraphBuilder("coarse")
        patch = builder.propose(
            "goal", TASK,
            {"segments": (("apple", "013_apple", "fridge"), ("bowl", "024_bowl", "kitchen_counter")),
             "steps": {"apple": ("place", "close")}},
        )
        self.assertEqual([item.id for item in patch.subgoals], ["apple_delivered", "bowl_delivered"])
        self.assertEqual(
            patch.skill_subgraphs[0].execution_order(),
            ("navigate_apple_to_destination", "place_apple", "navigate_back_to_apple_source", "close_apple_source"),
        )
        self.assertEqual(len(patch.skill_subgraphs[1].nodes), 8)
        with self.assertRaisesRegex(ValueError, "picks its object without placing"):
            builder.propose("goal", TASK, {"steps": {"bowl": ("open", "pick")}})


class RunTests(TestCase):
    def test_dry_run_writes_the_episode_and_the_mapped_proposal(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = root / "all.json"
            plan_path.write_text(json.dumps({"dataset": "ReplicaCADSetTableTrain", "plans": [make_plan()]}))
            config_path = root / "train" / "set_table" / "episode_7.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(make_episode_config()))
            output = root / "run"
            lines = []
            summary = main(
                [
                    "--dry-run", "--task-plan", str(plan_path), "--plan-index", "0",
                    "--rearrange-root", str(root), "--checkpoint-root", str(root / "ckpt"),
                    "--granularity", "fine", "--output", str(output),
                ],
                log=lines.append,
            )
            self.assertTrue(summary["dry_run"] and summary["accepted"])
            self.assertEqual((summary["subgoals"], summary["nodes"], summary["unmapped_nodes"]), (16, 16, 0))
            self.assertEqual(summary["proposer"]["proposer"], "SetTableRuleProposer")
            episode = json.loads((output / "episode.json").read_text())
            self.assertEqual(episode["entities"][2], {"name": "kitchen_counter", "kind": "articulation"})
            plan = json.loads((output / "set_table_plan.json").read_text())
            self.assertEqual(len(plan["plans"]), 1)
            proposal = json.loads((output / "proposal.json").read_text())
            self.assertEqual([row["subtask_index"] for row in proposal["node_subtasks"]], list(range(16)))
            self.assertEqual(proposal["node_subtasks"][5]["subtask"], "place(024_bowl,frl_apartment_table_02)")
            self.assertEqual(proposal["node_subtasks"][14]["subtask"], "navigate(fridge)")
            self.assertTrue(any("validated proposal: 16 sub-goals" in line for line in lines))
            self.assertTrue((output / "summary.json").is_file())

    def test_step_budget_covers_every_attempt_and_replan(self):
        with TemporaryDirectory() as tmp:
            library = build_granularity_library(Path(tmp), task_families=("set_table",))
        episode = make_episode()
        # (1002 + 202) x 4 steps per segment, two segments, two attempts, three passes.
        self.assertEqual(step_budget(library, episode, 2, 2), 4816 * 2 * 2 * 3 + 1)
        self.assertEqual(len(library.policies), 11)
        self.assertEqual(
            [policy.id for policy in library.applicable_policies(contract_id("pick"), {"object": "024_bowl"})],
            ["rl.set_table.pick.024_bowl", "rl.set_table.pick.all"],
        )
        self.assertEqual(
            [policy.id for policy in library.applicable_policies(contract_id("open"), {"articulation": "fridge"})],
            ["rl.set_table.open.fridge"],
        )
