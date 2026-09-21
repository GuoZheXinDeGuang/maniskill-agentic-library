import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from mshab.experiments.granularity import build_granularity_library, contract_id
from mshab.experiments.granularity.higher_layers import (
    DEFAULT_TIDY_HOUSE_TRANSFERS,
    TidyHouseGraphBuilder,
    build_gold_graph,
    coarse_subgoal_id,
    fine_subgoal_ids,
    transfer_index,
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
    TidyHouseEpisode,
    TidyHouseRuleProposer,
    UnsupportedGrounding,
    habitat_instance,
    load_episode,
    object_category,
    read_goal_receptacles,
)
from mshab.experiments.rollout.run import main, nominal_facts, step_budget
from mshab.skills.graph import SubGoal


TASK = "granularity"

# Plan 6 of the official train split, abridged: five distinct objects, the
# last two delivered to the same receptacle.
OBJECTS = (
    "007_tuna_fish_can-0",
    "004_sugar_box-0",
    "009_gelatin_box-0",
    "024_bowl-0",
    "002_master_chef_can-0",
)
RECEPTACLES = (
    "frl_apartment_table_01_:0000",
    "frl_apartment_chair_01_:0000",
    "kitchen_counter_:0000",
    "frl_apartment_tvstand_:0000",
    "frl_apartment_tvstand_:0000",
)
RECTANGLES = ("A", "B", "C", "D", "D")


def _corners(tag):
    offset = float(ord(tag))
    return [[offset, 0.0, 0.5], [offset + 1.0, 0.0, 0.5], [offset + 1.0, 1.0, 0.5], [offset, 1.0, 0.5]]


def make_plan(objects=OBJECTS, rectangles=RECTANGLES, init_config_name="train/tidy_house/episode_1002.json"):
    subtasks = []
    for number, (obj, tag) in enumerate(zip(objects, rectangles)):
        uid = "tidy_house-sequential-train-6-{}"
        subtasks.append({"type": "navigate", "uid": uid.format(4 * number), "obj_id": None, "goal_pos": None})
        subtasks.append({"type": "pick", "uid": uid.format(4 * number + 1), "obj_id": obj, "articulation_config": None})
        subtasks.append({"type": "navigate", "uid": uid.format(4 * number + 2), "obj_id": None, "goal_pos": None})
        subtasks.append({
            "type": "place", "uid": uid.format(4 * number + 3), "obj_id": obj,
            "goal_rectangle_corners": _corners(tag), "goal_pos": [float(ord(tag)) + 0.5, 0.5, 0.6],
            "validate_goal_rectangle_corners": True, "articulation_config": None,
        })
    return {
        "subtasks": subtasks,
        "build_config_name": "v3_sc2_staging_17.scene_instance.json",
        "init_config_name": init_config_name,
    }


def make_episode_config(objects=OBJECTS, receptacles=RECEPTACLES):
    labels = {}
    for obj in objects:
        category, number = obj.rsplit("-", 1)
        labels["{}_:{:04d}".format(category, int(number))] = "any_targets|{}".format(len(labels))
    return {
        "episode_id": "1002",
        "info": {"object_labels": labels},
        "target_receptacles": [[name, None] for name in receptacles],
        "goal_receptacles": [[name, None] for name in receptacles],
    }


def make_episode(**kwargs):
    return TidyHouseEpisode.from_plan(make_plan(), 6, dataset="ReplicaCADTidyHouseTrain", goal_receptacles=RECEPTACLES, **kwargs)


def gold_episode():
    """An episode whose transfers are the gold graphs' default ones."""

    objects = tuple("{}-0".format(obj) for obj, _ in DEFAULT_TIDY_HOUSE_TRANSFERS)
    receptacles = tuple("{}_:0000".format(dest) for _, dest in DEFAULT_TIDY_HOUSE_TRANSFERS)
    plan = make_plan(objects, tuple("ABCDE"))
    return TidyHouseEpisode.from_plan(plan, 0, dataset="test", goal_receptacles=receptacles)


class EpisodeTests(TestCase):
    def test_names_entities_and_the_goal(self):
        episode = make_episode()
        self.assertEqual(object_category("007_tuna_fish_can-0"), "007_tuna_fish_can")
        self.assertEqual(habitat_instance("frl_apartment_table_01_:0000"), ("frl_apartment_table_01", 0))
        with self.assertRaisesRegex(ValueError, "object id"):
            object_category("007_tuna_fish_can")
        self.assertEqual(
            episode.objects,
            ("007_tuna_fish_can", "004_sugar_box", "009_gelatin_box", "024_bowl", "002_master_chef_can"),
        )
        self.assertEqual(
            episode.destinations,
            ("frl_apartment_table_01", "frl_apartment_chair_01", "kitchen_counter", "frl_apartment_tvstand"),
        )
        self.assertEqual(episode.transfers[3].destination, episode.transfers[4].destination)
        self.assertEqual(
            episode.goal,
            "Tidy the house: move 007_tuna_fish_can to frl_apartment_table_01, "
            "004_sugar_box to frl_apartment_chair_01, 009_gelatin_box to kitchen_counter, "
            "024_bowl to frl_apartment_tvstand, and 002_master_chef_can to frl_apartment_tvstand.",
        )
        self.assertEqual(
            episode.goal_facts,
            (
                "at(007_tuna_fish_can,frl_apartment_table_01)",
                "at(004_sugar_box,frl_apartment_chair_01)",
                "at(009_gelatin_box,kitchen_counter)",
                "at(024_bowl,frl_apartment_tvstand)",
                "at(002_master_chef_can,frl_apartment_tvstand)",
            ),
        )
        self.assertEqual(
            [(item.name, item.kind) for item in episode.entities()],
            [(name, "object") for name in episode.objects]
            + [(name, "receptacle") for name in episode.destinations],
        )
        environment_entities = {item.name: item for item in episode.environment_entities()}
        self.assertEqual(environment_entities["024_bowl"].environment_name, "024_bowl-0")
        self.assertEqual(environment_entities["024_bowl"].attributes["category"], "024_bowl")
        self.assertEqual(
            environment_entities["frl_apartment_tvstand"].environment_name, "frl_apartment_tvstand_:0000"
        )
        self.assertEqual(environment_entities["frl_apartment_tvstand"].attributes["transfers"], (4, 5))
        self.assertEqual(episode.subtask_count, 20)
        self.assertEqual([item.subtasks for item in episode.transfers][1], (4, 5, 6, 7))
        document = json.loads(json.dumps(episode.as_dict()))
        self.assertEqual(document["transfers"][4]["subtasks"]["place"], 19)
        self.assertEqual(document["goal"], episode.goal)

    def test_grounded_nodes_map_to_plan_subtasks(self):
        episode = make_episode()
        self.assertEqual(episode.subtask_for("navigate", {"target": "004_sugar_box"}), 4)
        self.assertEqual(episode.subtask_for("pick", {"object": "004_sugar_box"}), 5)
        self.assertEqual(episode.subtask_for("navigate", {"target": "frl_apartment_chair_01"}), 6)
        self.assertEqual(
            episode.subtask_for("place", {"object": "004_sugar_box", "destination": "frl_apartment_chair_01"}), 7
        )
        # A shared receptacle resolves to the transfer whose object is held,
        # else to the first one not yet delivered, else to the first.
        shared = {"target": "frl_apartment_tvstand"}
        self.assertEqual(episode.subtask_for("navigate", shared), 14)
        self.assertEqual(episode.subtask_for("navigate", shared, ["holding(002_master_chef_can)"]), 18)
        self.assertEqual(episode.subtask_for("navigate", shared, ["at(024_bowl,frl_apartment_tvstand)"]), 18)
        self.assertEqual(
            episode.subtask_for(
                "navigate", shared,
                ["at(024_bowl,frl_apartment_tvstand)", "at(002_master_chef_can,frl_apartment_tvstand)"],
            ),
            14,
        )
        with self.assertRaisesRegex(UnsupportedGrounding, "neither an object nor a receptacle"):
            episode.subtask_for("navigate", {"target": "dining_table"})
        with self.assertRaisesRegex(UnsupportedGrounding, "not an object of this episode"):
            episode.subtask_for("pick", {"object": "013_apple"})
        with self.assertRaisesRegex(UnsupportedGrounding, "delivers 024_bowl to frl_apartment_tvstand, not kitchen_counter"):
            episode.subtask_for("place", {"object": "024_bowl", "destination": "kitchen_counter"})
        with self.assertRaisesRegex(UnsupportedGrounding, "no 'open' subtask"):
            episode.subtask_for("open", {"articulation": "fridge"})
        self.assertEqual(
            [episode.subtask_label(index) for index in range(4)],
            [
                "navigate(007_tuna_fish_can)", "pick(007_tuna_fish_can)",
                "navigate(frl_apartment_table_01)", "place(007_tuna_fish_can,frl_apartment_table_01)",
            ],
        )
        with self.assertRaises(IndexError):
            episode.subtask_label(20)

    def test_duplicate_categories_and_unnamed_receptacles(self):
        objects = ("003_cracker_box-0", "024_bowl-0", "003_cracker_box-1", "004_sugar_box-0", "005_tomato_soup_can-0")
        plan = make_plan(objects, ("A", "B", "A", "C", "B"))
        episode = TidyHouseEpisode.from_plan(plan, 0)
        self.assertEqual(
            episode.objects,
            ("003_cracker_box", "024_bowl", "003_cracker_box_2", "004_sugar_box", "005_tomato_soup_can"),
        )
        self.assertEqual(episode.destinations, ("receptacle_1", "receptacle_2", "receptacle_3"))
        self.assertEqual([item.destination for item in episode.transfers],
                         ["receptacle_1", "receptacle_2", "receptacle_1", "receptacle_3", "receptacle_2"])
        self.assertIsNone(episode.transfers[0].receptacle_instance)
        self.assertEqual(episode.environment_entities()[5].environment_name, "receptacle_1")
        # Two instances of the same receptacle model are two destinations,
        # numbered by instance id however the plan orders them.
        receptacles = (
            "frl_apartment_chair_01_:0001", "frl_apartment_chair_01_:0000", "frl_apartment_chair_01_:0001",
            "kitchen_counter_:0000", "frl_apartment_chair_01_:0000",
        )
        named = TidyHouseEpisode.from_plan(plan, 0, goal_receptacles=receptacles)
        self.assertEqual(
            named.destinations, ("frl_apartment_chair_01_2", "frl_apartment_chair_01", "kitchen_counter")
        )
        self.assertEqual(named.transfers[4].destination, "frl_apartment_chair_01")
        # Objects too: the plan may pick the second cracker box first.
        swapped = TidyHouseEpisode.from_plan(make_plan(("003_cracker_box-1",) + objects[1:2] + ("003_cracker_box-0",) + objects[3:], ("A", "B", "A", "C", "B")), 0)
        self.assertEqual(swapped.objects[0], "003_cracker_box_2")
        self.assertEqual(swapped.objects[2], "003_cracker_box")
        with self.assertRaisesRegex(ValueError, "goal receptacles for"):
            TidyHouseEpisode.from_plan(plan, 0, goal_receptacles=receptacles[:3])

    def test_plan_shape_is_checked(self):
        plan = make_plan()
        plan["subtasks"][3]["obj_id"] = "024_bowl-0"
        with self.assertRaisesRegex(ValueError, "picks '007_tuna_fish_can-0' but places"):
            TidyHouseEpisode.from_plan(plan, 0)
        plan = make_plan()
        plan["subtasks"].pop()
        with self.assertRaisesRegex(ValueError, "4 subtasks per transfer"):
            TidyHouseEpisode.from_plan(plan, 0)
        plan = make_plan()
        plan["subtasks"][0]["type"] = "pick"
        with self.assertRaisesRegex(ValueError, "transfer 1 is"):
            TidyHouseEpisode.from_plan(plan, 0)

    def test_facts_follow_the_measurements(self):
        episode = make_episode()
        tuna, sugar = episode.transfers[0], episode.transfers[1]
        present = {"present({})".format(item.name) for item in episode.entities()}
        nothing = SceneMeasurements(
            grasped={item.pick: False for item in episode.transfers},
            at_goal={item.place: False for item in episode.transfers},
            near={index: False for item in episode.transfers for index in (item.navigate_to_object, item.navigate_to_destination)},
        )
        self.assertEqual(episode.facts(nothing), present | {"gripper_empty()", "collision_safe()"})
        self.assertEqual(episode.facts(SceneMeasurements({}, {}, {})), present | {"gripper_empty()", "collision_safe()"})
        holding = SceneMeasurements(
            grasped={tuna.pick: True},
            at_goal={tuna.place: True, sugar.place: True},
            near={tuna.navigate_to_destination: True, sugar.navigate_to_object: True},
            collision_safe=False,
        )
        facts = episode.facts(holding)
        self.assertIn("holding(007_tuna_fish_can)", facts)
        self.assertNotIn("gripper_empty()", facts)
        self.assertNotIn("collision_safe()", facts)
        # A grasped object is not at its goal, whatever the distance says.
        self.assertNotIn("at(007_tuna_fish_can,frl_apartment_table_01)", facts)
        self.assertIn("at(004_sugar_box,frl_apartment_chair_01)", facts)
        self.assertIn("reachable(frl_apartment_table_01)", facts)
        self.assertIn("reachable(004_sugar_box)", facts)
        # An object in the gripper is reachable wherever the robot stands.
        self.assertIn("reachable(007_tuna_fish_can)", facts)
        self.assertNotIn("reachable(024_bowl)", facts)
        self.assertEqual(SceneMeasurements({1: 1}, {}, {}).grasped[1], True)
        with self.assertRaisesRegex(TypeError, "subtask indices"):
            SceneMeasurements({"1": True}, {}, {})
        json.dumps(holding.as_dict())

    def test_official_files_are_read_and_checked(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = root / "task_plans" / "tidy_house" / "sequential" / "train" / "all.json"
            plan_path.parent.mkdir(parents=True)
            plan_path.write_text(json.dumps({"dataset": "ReplicaCADTidyHouseTrain", "plans": [make_plan(), make_plan()]}))
            config_path = root / "train" / "tidy_house" / "episode_1002.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(make_episode_config()))

            episode, document = load_episode(plan_path, 1, root)
            self.assertEqual(episode.plan_index, 1)
            self.assertEqual(episode.dataset, "ReplicaCADTidyHouseTrain")
            self.assertEqual(episode.destinations[0], "frl_apartment_table_01")
            self.assertEqual(document["dataset"], "ReplicaCADTidyHouseTrain")
            self.assertEqual(len(document["plans"]), 1)
            self.assertEqual(document["plans"][0]["subtasks"][1]["obj_id"], "007_tuna_fish_can-0")

            # Without the config the receptacles are numbered by goal rectangle.
            unnamed, _ = load_episode(plan_path, 0, None)
            self.assertEqual(unnamed.destinations, ("receptacle_1", "receptacle_2", "receptacle_3", "receptacle_4"))
            self.assertEqual(unnamed.transfers[3].destination, unnamed.transfers[4].destination)
            self.assertIsNone(read_goal_receptacles(root, {"init_config_name": "train/tidy_house/episode_9.json"}))

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
        self.library = build_granularity_library(Path(self.temporary_directory.name), task_families=("tidy_house",))
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

    def test_reproduces_the_gold_graphs_on_their_own_transfers(self):
        episode = gold_episode()
        self.assertEqual(tuple(item.pair for item in episode.transfers), DEFAULT_TIDY_HOUSE_TRANSFERS)
        for granularity in ("coarse", "fine"):
            proposer = TidyHouseRuleProposer(episode)
            proposal = self.validator.plan(proposer, episode.goal, self.context(episode, granularity))
            gold = build_gold_graph("tidy_house_" + granularity, self.library)
            self.assertEqual(proposal.patch.as_dict(), gold.patch.as_dict())
            self.assertEqual(proposal.plan.order, gold.plan.order)
            self.assertEqual(len(proposal.rounds), 1)
            # Every node stands for the official subtask at the same position.
            order = [
                episode.subtask_for(
                    self.library.get(node.contract_id).contract_type_name, node.arguments
                )
                for node in (proposal.skill_graph.nodes[node_id] for node_id in proposal.plan.order)
            ]
            self.assertEqual(order, list(range(20)))
        self.assertEqual(TidyHouseRuleProposer(episode).describe()["proposer"], "TidyHouseRuleProposer")

    def test_answers_the_real_episode_at_both_granularities(self):
        episode = make_episode()
        for granularity, subgoals in (("coarse", 5), ("fine", 20)):
            proposal = self.validator.plan(
                TidyHouseRuleProposer(episode), episode.goal, self.context(episode, granularity)
            )
            self.assertEqual(len(proposal.subgoal_graph.subgoals), subgoals)
            self.assertEqual(len(proposal.skill_graph.nodes), 20)
            self.assertIn("frl_apartment_tvstand", proposal.skill_graph.nodes["place_object_5"].arguments["destination"])

    def test_replans_keep_a_failed_transfer_once_then_drop_it(self):
        episode = make_episode()
        proposer = TidyHouseRuleProposer(episode, give_up_after=2)
        delivered = ["at(007_tuna_fish_can,frl_apartment_table_01)"]
        facts = list(nominal_facts(episode)) + delivered
        failure = Failure("object_2_delivered", "pick_object_2", "execution_timeout", ("holding(004_sugar_box)",))
        history = History(("object_1_delivered",), ("navigate_to_object_1", "pick_object_1"), ("pick_object_2",))

        first = proposer.decompose(DecompositionRequest(TASK, episode.goal, self.context(episode, "coarse", facts, failure, 1, history)))
        self.assertEqual(
            [item.id for item in first.subgoals],
            ["object_2_delivered", "object_3_delivered", "object_4_delivered", "object_5_delivered"],
        )
        self.assertIn("failed 1 time(s)", first.rationale)
        second = proposer.decompose(DecompositionRequest(TASK, episode.goal, self.context(episode, "coarse", facts, failure, 2, history)))
        self.assertEqual(
            [item.id for item in second.subgoals],
            ["object_3_delivered", "object_4_delivered", "object_5_delivered"],
        )
        self.assertIn("gave up transfer(s) [2]", second.rationale)
        self.assertEqual(proposer.failures[2], 2)
        # The sub-goal ids keep the transfers' original numbers, and the
        # whole replanned graph still validates and plans.
        proposal = self.validator.plan(proposer, episode.goal, self.context(episode, "coarse", facts, failure, 3, history))
        self.assertEqual(proposal.plan.order[0], "navigate_to_object_3")
        self.assertEqual(len(proposal.skill_graph.nodes), 12)
        self.assertEqual(proposer.failures[2], 3)
        # Fine ids name the transfer too.
        fine = proposer.decompose(DecompositionRequest(TASK, episode.goal, self.context(episode, "fine", facts, Failure("object_3_holding", "pick_object_3", "execution_timeout"), 4, history)))
        self.assertEqual([item.id for item in fine.subgoals][:4], list(fine_subgoal_ids(3)))
        self.assertEqual(proposer.failures[3], 1)
        # A failure the ids do not name is not counted.
        proposer.decompose(DecompositionRequest(TASK, episode.goal, self.context(episode, "fine", facts, Failure("bowl_placed", None, "no_viable_candidate"), 5, history)))
        self.assertEqual(sum(proposer.failures.values()), 4)

        # A transfer whose object is in the gripper is never given up: with
        # transfer 2 at three failures, holding its object keeps it, first.
        held = facts + ["holding(004_sugar_box)"]
        kept = proposer.decompose(DecompositionRequest(TASK, episode.goal, self.context(episode, "coarse", held)))
        self.assertEqual([item.id for item in kept.subgoals][:2], ["object_2_delivered", "object_3_delivered"])
        self.assertIn("kept although given up", kept.rationale)
        everything = facts + [item.delivered for item in episode.transfers]
        with self.assertRaisesRegex(ValueError, "nothing left to plan"):
            proposer.decompose(DecompositionRequest(TASK, episode.goal, self.context(episode, "coarse", everything)))
        with self.assertRaisesRegex(ValueError, "decomposes at"):
            proposer.decompose(DecompositionRequest(TASK, episode.goal, self.context(episode, "free")))
        with self.assertRaisesRegex(ValueError, "answers for task"):
            proposer.decompose(DecompositionRequest("set_table", episode.goal, self.context(episode, "coarse")))
        with self.assertRaisesRegex(ValueError, "at least 1"):
            TidyHouseRuleProposer(episode, give_up_after=0)

    def test_a_held_object_is_delivered_first_without_a_pick(self):
        episode = make_episode()
        proposer = TidyHouseRuleProposer(episode)
        holding = sorted(set(nominal_facts(episode)) - {"gripper_empty()"}) + ["holding(009_gelatin_box)", "at(007_tuna_fish_can,frl_apartment_table_01)"]
        decomposition = proposer.decompose(DecompositionRequest(TASK, episode.goal, self.context(episode, "coarse", holding)))
        self.assertEqual(
            [item.id for item in decomposition.subgoals],
            ["object_3_delivered", "object_2_delivered", "object_4_delivered", "object_5_delivered"],
        )
        request = SubgraphRequest(
            TASK, episode.goal, decomposition.subgoals[0], PlanningContext.initial(self.library, TASK).contracts,
            entities=episode.entities(), facts=holding, neighbours=Neighbours(None, decomposition.subgoals[1].predicate),
        )
        response = proposer.plan_subgraph(request)
        self.assertEqual(response.subgraph.execution_order(), ("navigate_to_destination_3", "place_object_3"))
        self.assertIn("already held", response.rationale)
        self.assertEqual(response.subgraph.achievers[0].id, "place_object_3")
        # The whole proposal validates and plans from that state.
        proposal = self.validator.plan(proposer, episode.goal, self.context(episode, "coarse", holding))
        self.assertEqual(proposal.plan.order[:2], ("navigate_to_destination_3", "place_object_3"))
        self.assertEqual(len(proposal.skill_graph.nodes), 14)
        # Fine sub-goals are not shortened: the controller absorbs the ones
        # whose predicates already hold.
        fine = self.validator.plan(proposer, episode.goal, self.context(episode, "fine", holding))
        self.assertEqual(fine.plan.order[:4], ("navigate_to_object_3", "pick_object_3", "navigate_to_destination_3", "place_object_3"))

    def test_subgraph_requests_are_checked(self):
        episode = make_episode()
        proposer = TidyHouseRuleProposer(episode)
        contracts = PlanningContext.initial(self.library, TASK).contracts

        def request(subgoal):
            return SubgraphRequest(TASK, episode.goal, subgoal, contracts, entities=episode.entities(), facts=nominal_facts(episode))

        with self.assertRaisesRegex(ValueError, "not one the rule proposer decomposes into"):
            proposer.plan_subgraph(request(SubGoal("bowl_placed", "at(024_bowl,frl_apartment_tvstand)")))
        with self.assertRaisesRegex(ValueError, "not one the rule proposer decomposes into"):
            proposer.plan_subgraph(request(SubGoal(coarse_subgoal_id(9), "at(x,y)")))
        with self.assertRaisesRegex(ValueError, "should read"):
            proposer.plan_subgraph(request(SubGoal(coarse_subgoal_id(4), "at(024_bowl,kitchen_counter)")))
        response = proposer.plan_subgraph(request(SubGoal(coarse_subgoal_id(4), "at(024_bowl,frl_apartment_tvstand)")))
        self.assertEqual(len(response.subgraph.nodes), 4)
        # When every transfer is delivered the rules have nothing to answer,
        # which the validator records as a decomposition rejection.
        with self.assertRaises(ProposalRejected) as caught:
            self.validator.plan(
                proposer, episode.goal,
                self.context(episode, "coarse", list(nominal_facts(episode)) + list(episode.goal_facts)),
            )
        self.assertEqual(caught.exception.rejections[0].stage, "decomposition")
        self.assertIn("nothing left to plan", caught.exception.rejections[0].message)


class BuilderIndicesTests(TestCase):
    def test_transfer_indices_number_the_ids(self):
        builder = TidyHouseGraphBuilder("coarse")
        patch = builder.propose(
            "goal", TASK,
            {"transfers": (("024_bowl", "dining_table"), ("013_apple", "dining_table")), "transfer_indices": (4, 2)},
        )
        self.assertEqual([item.id for item in patch.subgoals], ["object_4_delivered", "object_2_delivered"])
        self.assertEqual(sorted(patch.skill_subgraphs[1].nodes), ["navigate_to_destination_2", "navigate_to_object_2", "pick_object_2", "place_object_2"])
        self.assertEqual(transfer_index("object_4_delivered"), 4)
        self.assertEqual(transfer_index("destination_2_reachable"), 2)
        self.assertIsNone(transfer_index("object_delivered"))
        for bad in ((1,), (0, 1), (1, 1), (True, 2)):
            with self.assertRaises(ValueError):
                builder.propose("goal", TASK, {"transfers": (("a", "b"), ("c", "d")), "transfer_indices": bad})


class RunTests(TestCase):
    def test_dry_run_writes_the_episode_and_the_mapped_proposal(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_path = root / "all.json"
            plan_path.write_text(json.dumps({"dataset": "ReplicaCADTidyHouseTrain", "plans": [make_plan()]}))
            config_path = root / "train" / "tidy_house" / "episode_1002.json"
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
            self.assertEqual((summary["subgoals"], summary["nodes"], summary["unmapped_nodes"]), (20, 20, 0))
            self.assertEqual(summary["proposer"]["proposer"], "TidyHouseRuleProposer")
            episode = json.loads((output / "episode.json").read_text())
            self.assertEqual(episode["entities"][5], {"name": "frl_apartment_table_01", "kind": "receptacle"})
            plan = json.loads((output / "tidy_house_plan.json").read_text())
            self.assertEqual(len(plan["plans"]), 1)
            proposal = json.loads((output / "proposal.json").read_text())
            self.assertEqual([row["subtask_index"] for row in proposal["node_subtasks"]], list(range(20)))
            self.assertEqual(proposal["node_subtasks"][3]["subtask"], "place(007_tuna_fish_can,frl_apartment_table_01)")
            self.assertTrue(any("validated proposal: 20 sub-goals" in line for line in lines))
            self.assertTrue((output / "summary.json").is_file())

    def test_step_budget_covers_every_attempt_and_replan(self):
        with TemporaryDirectory() as tmp:
            library = build_granularity_library(Path(tmp), task_families=("tidy_house",))
        episode = make_episode()
        # (1002 + 202 + 1002 + 202) steps per transfer, five transfers, two
        # attempts, three passes.
        self.assertEqual(step_budget(library, episode, 2, 2), 2408 * 5 * 2 * 3 + 1)
        self.assertEqual(len(library.policies), 21)
        self.assertEqual(
            [policy.id for policy in library.applicable_policies(contract_id("pick"), {"object": "024_bowl"})],
            ["rl.tidy_house.pick.024_bowl", "rl.tidy_house.pick.all"],
        )
