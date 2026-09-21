"""One official TidyHouse episode as the symbolic scene the controller plans over.

MS-HAB fixes a TidyHouse episode as a sequential task plan: five object
transfers, each ``navigate -> pick -> navigate -> place``, 20 atomic subtasks.
This module reads that plan as plain JSON, without a simulator, and derives
what the upper layers and the environment adapter need from it:

- the symbolic entities: one name per object and one per goal receptacle,
  with the plan's instance ids behind them;
- the goal text, which pairs every object with its receptacle, and the goal
  facts the run is judged on;
- which subtask of the plan a grounded skill node stands for.  The MS-HAB
  policies observe the object and the goal of *the subtask the environment
  is pointed at*, so executing ``pick(024_bowl)`` means pointing the
  environment at the bowl's pick subtask first;
- the facts of the scene from per-subtask measurements the environment
  reports every step.

Standard library only: the tests build episodes from a plan document and
never start a simulator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from mshab.experiments.planning.documents import EntityDescription
from mshab.skills.environment import EnvironmentEntity


#: The atomic subtasks of one transfer, in the official order.
TRANSFER_SUBTASK_TYPES = ("navigate", "pick", "navigate", "place")
OBJECT_KIND = "object"
RECEPTACLE_KIND = "receptacle"
#: Relative to the rearrange dataset root: the official sequential plans.
SEQUENTIAL_PLAN_PATH = "task_plans/tidy_house/sequential/{split}/all.json"
EPISODE_SCHEMA_VERSION = "mshab.rollout-episode.v1"


class UnsupportedGrounding(ValueError):
    """A grounded skill node names nothing this episode's plan can execute."""


def object_category(instance_id: str) -> str:
    """``007_tuna_fish_can-0`` -> ``007_tuna_fish_can``, the checkpoint target."""

    category, separator, number = instance_id.rpartition("-")
    if not separator or not category or not number.isdigit():
        raise ValueError(
            "{!r} is not a <category>-<number> object id".format(instance_id)
        )
    return category


def habitat_instance(name: str) -> Tuple[str, int]:
    """``frl_apartment_table_01_:0000`` -> ``("frl_apartment_table_01", 0)``.

    The rearrange episode configs name every object and receptacle instance
    this way; the plan generator turns the objects into ``<category>-<n>``.
    """

    base, separator, number = name.rpartition("_:")
    if not separator or not base or not number.isdigit():
        raise ValueError("{!r} is not a <name>_:<number> instance id".format(name))
    return base, int(number)


def _symbolic_names(instances: Sequence[Tuple[str, str]]) -> Dict[str, str]:
    """One symbolic name per distinct instance key.

    Distinct instances that share a base name get ``base``, ``base_2``,
    ``base_3`` in the order of their instance ids (``-0`` before ``-1``,
    ``_:0000`` before ``_:0001``), whatever order the plan visits them in.
    The first keeps the bare name, so an object whose category appears once
    in the episode still matches the checkpoint trained for it.
    """

    distinct: Dict[str, str] = {}
    for key, base in instances:
        distinct.setdefault(key, base)
    by_base: Dict[str, List[str]] = {}
    for key, base in distinct.items():
        by_base.setdefault(base, []).append(key)
    names: Dict[str, str] = {}
    for base, keys in by_base.items():
        for ordinal, key in enumerate(sorted(keys), start=1):
            names[key] = base if ordinal == 1 else "{}_{}".format(base, ordinal)
    return names


@dataclass(frozen=True)
class Transfer:
    """One object transfer of the plan and the four subtasks that implement it."""

    index: int
    object: str
    destination: str
    object_instance: str
    receptacle_instance: Optional[str]
    navigate_to_object: int
    pick: int
    navigate_to_destination: int
    place: int
    goal_position: Optional[Tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        if self.index < 1:
            raise ValueError("transfer index is 1-based")
        if not self.object or not self.destination or not self.object_instance:
            raise ValueError("a transfer names an object, a destination, and an instance")
        if self.goal_position is not None:
            object.__setattr__(
                self, "goal_position", tuple(float(value) for value in self.goal_position)
            )

    @property
    def pair(self) -> Tuple[str, str]:
        return (self.object, self.destination)

    @property
    def subtasks(self) -> Tuple[int, int, int, int]:
        return (self.navigate_to_object, self.pick, self.navigate_to_destination, self.place)

    @property
    def held(self) -> str:
        return "holding({})".format(self.object)

    @property
    def delivered(self) -> str:
        return "at({},{})".format(self.object, self.destination)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "object": self.object,
            "destination": self.destination,
            "object_instance": self.object_instance,
            "receptacle_instance": self.receptacle_instance,
            "subtasks": {
                "navigate_to_object": self.navigate_to_object,
                "pick": self.pick,
                "navigate_to_destination": self.navigate_to_destination,
                "place": self.place,
            },
            "goal_position": None if self.goal_position is None else list(self.goal_position),
        }


def _frozen_bools(payload: Mapping[Any, Any], name: str) -> Mapping[int, bool]:
    result = {}
    for key, value in dict(payload).items():
        if isinstance(key, bool) or not isinstance(key, int):
            raise TypeError("{} keys are subtask indices, got {!r}".format(name, key))
        result[key] = bool(value)
    return MappingProxyType(result)


@dataclass(frozen=True)
class SceneMeasurements:
    """What the environment measured about every object and goal of the plan.

    Keyed by subtask index: ``grasped[i]`` for the pick subtask of an object,
    ``at_goal[i]`` for its place subtask, ``near[i]`` for a navigate subtask
    (the base is close to and facing that subtask's goal).  The environment
    checks all of them every step, not only the subtask it is pointed at;
    turning them into predicates is :meth:`TidyHouseEpisode.facts`.
    """

    grasped: Mapping[int, bool]
    at_goal: Mapping[int, bool]
    near: Mapping[int, bool]
    collision_safe: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "grasped", _frozen_bools(self.grasped, "grasped"))
        object.__setattr__(self, "at_goal", _frozen_bools(self.at_goal, "at_goal"))
        object.__setattr__(self, "near", _frozen_bools(self.near, "near"))
        object.__setattr__(self, "collision_safe", bool(self.collision_safe))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "grasped": {str(key): value for key, value in self.grasped.items()},
            "at_goal": {str(key): value for key, value in self.at_goal.items()},
            "near": {str(key): value for key, value in self.near.items()},
            "collision_safe": self.collision_safe,
        }


@dataclass(frozen=True)
class TidyHouseEpisode:
    """One official episode: its transfers, entities, goal, and subtask mapping."""

    plan_index: int
    dataset: str
    build_config_name: str
    init_config_name: str
    transfers: Tuple[Transfer, ...]

    def __post_init__(self) -> None:
        transfers = tuple(self.transfers)
        if not transfers:
            raise ValueError("an episode has at least one transfer")
        if [item.index for item in transfers] != list(range(1, len(transfers) + 1)):
            raise ValueError("transfers are numbered 1, 2, ... in plan order")
        objects = [item.object for item in transfers]
        if len(set(objects)) != len(objects):
            raise ValueError("every transfer moves a distinct symbolic object")
        for item in transfers:
            if item.object in self.destinations_of(transfers):
                raise ValueError("{!r} is both an object and a receptacle".format(item.object))
        object.__setattr__(self, "transfers", transfers)

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_plan(
        cls,
        plan: Mapping[str, Any],
        plan_index: int,
        *,
        dataset: str = "",
        goal_receptacles: Optional[Sequence[str]] = None,
    ) -> "TidyHouseEpisode":
        """Read one plan of the official ``all.json`` document.

        ``goal_receptacles`` are the episode config's receptacle instance ids
        in transfer order (see :func:`read_goal_receptacles`); they give the
        destinations their scene names.  Without them destinations are
        ``receptacle_1``, ``receptacle_2``, ... by distinct goal rectangle.
        """

        subtasks = list(plan["subtasks"])
        types = [str(item.get("type")) for item in subtasks]
        width = len(TRANSFER_SUBTASK_TYPES)
        if not subtasks or len(subtasks) % width:
            raise ValueError(
                "a TidyHouse plan has {} subtasks per transfer, got {}".format(
                    width, len(subtasks)
                )
            )
        groups = [subtasks[start : start + width] for start in range(0, len(subtasks), width)]
        for number, group in enumerate(groups, start=1):
            group_types = tuple(str(item.get("type")) for item in group)
            if group_types != TRANSFER_SUBTASK_TYPES:
                raise ValueError(
                    "transfer {} is {}, expected {}".format(
                        number, group_types, TRANSFER_SUBTASK_TYPES
                    )
                )
            if group[1].get("obj_id") != group[3].get("obj_id"):
                raise ValueError(
                    "transfer {} picks {!r} but places {!r}".format(
                        number, group[1].get("obj_id"), group[3].get("obj_id")
                    )
                )
        instances = [str(group[1]["obj_id"]) for group in groups]
        object_names = _symbolic_names(
            [(instance, object_category(instance)) for instance in instances]
        )
        if goal_receptacles is not None:
            receptacles = [str(item) for item in goal_receptacles]
            if len(receptacles) != len(groups):
                raise ValueError(
                    "{} goal receptacles for {} transfers".format(len(receptacles), len(groups))
                )
            destination_names = _symbolic_names(
                [(item, habitat_instance(item)[0]) for item in receptacles]
            )
            destinations = [destination_names[item] for item in receptacles]
        else:
            receptacles = [None] * len(groups)
            keys = [_goal_key(group[3]) for group in groups]
            distinct = list(dict.fromkeys(keys))
            destinations = [
                "receptacle_{}".format(distinct.index(key) + 1) for key in keys
            ]
        transfers = []
        for number, (group, instance, receptacle, destination) in enumerate(
            zip(groups, instances, receptacles, destinations), start=1
        ):
            base = (number - 1) * width
            goal_position = group[3].get("goal_pos")
            transfers.append(
                Transfer(
                    index=number,
                    object=object_names[instance],
                    destination=destination,
                    object_instance=instance,
                    receptacle_instance=receptacle,
                    navigate_to_object=base,
                    pick=base + 1,
                    navigate_to_destination=base + 2,
                    place=base + 3,
                    goal_position=None if goal_position is None else tuple(goal_position),
                )
            )
        return cls(
            plan_index=int(plan_index),
            dataset=str(dataset),
            build_config_name=str(plan.get("build_config_name", "")),
            init_config_name=str(plan.get("init_config_name", "")),
            transfers=tuple(transfers),
        )

    # -- the symbolic scene -----------------------------------------------------

    @staticmethod
    def destinations_of(transfers: Iterable[Transfer]) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(item.destination for item in transfers))

    @property
    def objects(self) -> Tuple[str, ...]:
        return tuple(item.object for item in self.transfers)

    @property
    def destinations(self) -> Tuple[str, ...]:
        """Distinct receptacles, in first-appearance order."""

        return self.destinations_of(self.transfers)

    @property
    def subtask_count(self) -> int:
        return len(self.transfers) * len(TRANSFER_SUBTASK_TYPES)

    @property
    def goal(self) -> str:
        """The goal text, pairing every object with its receptacle."""

        parts = ["{} to {}".format(item.object, item.destination) for item in self.transfers]
        if len(parts) == 1:
            listing = parts[0]
        else:
            listing = ", ".join(parts[:-1]) + ", and " + parts[-1]
        return "Tidy the house: move {}.".format(listing)

    @property
    def goal_facts(self) -> Tuple[str, ...]:
        return tuple(item.delivered for item in self.transfers)

    def entities(self) -> Tuple[EntityDescription, ...]:
        """What a proposer is told: objects first, then receptacles."""

        return tuple(EntityDescription(name, OBJECT_KIND) for name in self.objects) + tuple(
            EntityDescription(name, RECEPTACLE_KIND) for name in self.destinations
        )

    def environment_entities(self) -> Tuple[EnvironmentEntity, ...]:
        """The adapter's entity map: symbolic name -> the plan's instance."""

        entities = []
        for item in self.transfers:
            entities.append(
                EnvironmentEntity(
                    item.object,
                    OBJECT_KIND,
                    item.object_instance,
                    {"category": object_category(item.object_instance), "transfer": item.index},
                )
            )
        for destination in self.destinations:
            owners = [item for item in self.transfers if item.destination == destination]
            entities.append(
                EnvironmentEntity(
                    destination,
                    RECEPTACLE_KIND,
                    owners[0].receptacle_instance or destination,
                    {"transfers": tuple(item.index for item in owners)},
                )
            )
        return tuple(entities)

    def transfer(self, object_name: str) -> Transfer:
        for item in self.transfers:
            if item.object == object_name:
                return item
        raise UnsupportedGrounding(
            "{!r} is not an object of this episode; objects: {}".format(
                object_name, list(self.objects)
            )
        )

    # -- grounding -> subtask -----------------------------------------------------

    def subtask_for(
        self,
        contract_type: str,
        arguments: Mapping[str, Any],
        facts: Iterable[str] = (),
    ) -> int:
        """The plan subtask a grounded node stands for.

        A navigate node targets an object (its transfer's first navigate) or
        a receptacle; a receptacle shared by several transfers resolves to the
        transfer whose object is held, else the first not yet delivered.  A
        place node must name the destination the plan delivers the object to:
        a proposer that picked another receptacle gets
        :class:`UnsupportedGrounding`, which the executor reports as a failed
        execution and the controller replans around.
        """

        facts = set(facts)
        if contract_type == "navigate":
            target = str(arguments.get("target"))
            for item in self.transfers:
                if item.object == target:
                    return item.navigate_to_object
            candidates = [item for item in self.transfers if item.destination == target]
            if not candidates:
                raise UnsupportedGrounding(
                    "{!r} is neither an object nor a receptacle of this episode; "
                    "entities: {}".format(target, [entity.name for entity in self.entities()])
                )
            for item in candidates:
                if item.held in facts:
                    return item.navigate_to_destination
            for item in candidates:
                if item.delivered not in facts:
                    return item.navigate_to_destination
            return candidates[0].navigate_to_destination
        if contract_type == "pick":
            return self.transfer(str(arguments.get("object"))).pick
        if contract_type == "place":
            item = self.transfer(str(arguments.get("object")))
            destination = str(arguments.get("destination"))
            if item.destination != destination:
                raise UnsupportedGrounding(
                    "this episode delivers {} to {}, not {}".format(
                        item.object, item.destination, destination
                    )
                )
            return item.place
        raise UnsupportedGrounding(
            "TidyHouse has no {!r} subtask; its plan is {}".format(
                contract_type, "/".join(TRANSFER_SUBTASK_TYPES)
            )
        )

    def subtask_label(self, index: int) -> str:
        """``pick(007_tuna_fish_can)`` and the like, for logs and summaries."""

        width = len(TRANSFER_SUBTASK_TYPES)
        if not 0 <= index < self.subtask_count:
            raise IndexError("subtask {} is outside the plan".format(index))
        item = self.transfers[index // width]
        role = TRANSFER_SUBTASK_TYPES[index % width]
        if index == item.navigate_to_object:
            return "navigate({})".format(item.object)
        if index == item.navigate_to_destination:
            return "navigate({})".format(item.destination)
        if role == "pick":
            return "pick({})".format(item.object)
        return "place({},{})".format(item.object, item.destination)

    # -- measurements -> facts ------------------------------------------------------

    def facts(self, measurements: SceneMeasurements) -> FrozenSet[str]:
        """The scene as predicates the contracts of the granularity library use.

        ``holding`` follows the grasp check; ``at`` needs the object inside
        its goal and not grasped, as MS-HAB's own place success does;
        ``reachable`` is MS-HAB's navigation success for that target without
        the arm checks, and an object in the gripper is reachable wherever
        the robot stands (the symbolic environment keeps ``reachable(x)``
        after ``pick(x)`` too; asking the navigation policy to reach an
        object it is carrying is a 600-step detour); ``gripper_empty`` holds
        when no object is grasped; ``collision_safe`` is the pointed
        subtask's cumulative-force limit.
        """

        facts = {"present({})".format(entity.name) for entity in self.entities()}
        if measurements.collision_safe:
            facts.add("collision_safe()")
        holding = False
        for item in self.transfers:
            grasped = measurements.grasped.get(item.pick, False)
            if grasped:
                facts.add(item.held)
                holding = True
            elif measurements.at_goal.get(item.place, False):
                facts.add(item.delivered)
            if grasped or measurements.near.get(item.navigate_to_object, False):
                facts.add("reachable({})".format(item.object))
            if measurements.near.get(item.navigate_to_destination, False):
                facts.add("reachable({})".format(item.destination))
        if not holding:
            facts.add("gripper_empty()")
        return frozenset(facts)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "plan_index": self.plan_index,
            "dataset": self.dataset,
            "build_config_name": self.build_config_name,
            "init_config_name": self.init_config_name,
            "goal": self.goal,
            "goal_facts": list(self.goal_facts),
            "entities": [entity.as_dict() for entity in self.entities()],
            "transfers": [item.as_dict() for item in self.transfers],
        }


def _goal_key(place: Mapping[str, Any]) -> Tuple[float, ...]:
    """What identifies a destination when the episode config is not at hand."""

    corners = place.get("goal_rectangle_corners")
    if corners:
        return tuple(round(float(value), 2) for corner in corners for value in corner)
    position = place.get("goal_pos")
    if position:
        return tuple(round(float(value), 2) for value in position)
    raise ValueError("a place subtask needs goal_rectangle_corners or goal_pos")


# -- the official files -----------------------------------------------------------


def sequential_plan_path(rearrange_root: Path, split: str = "train") -> Path:
    if split not in ("train", "val"):
        raise ValueError("split must be train or val, got {!r}".format(split))
    return Path(rearrange_root) / SEQUENTIAL_PLAN_PATH.format(split=split)


def read_plan(path: Path, plan_index: int) -> Tuple[str, Dict[str, Any]]:
    """``(dataset, plan)`` of one plan in an official ``all.json`` document."""

    document = json.loads(Path(path).read_text())
    plans = document.get("plans")
    if not isinstance(plans, list) or not plans:
        raise ValueError("{} contains no plans".format(path))
    if not 0 <= plan_index < len(plans):
        raise IndexError(
            "plan index {} is outside {} plans of {}".format(plan_index, len(plans), path)
        )
    return str(document.get("dataset", "")), dict(plans[plan_index])


def plan_document(dataset: str, plan: Mapping[str, Any]) -> Dict[str, Any]:
    """A one-plan ``PlanData`` document the environment factory can load."""

    return {"dataset": dataset, "plans": [dict(plan)]}


def read_goal_receptacles(
    rearrange_root: Optional[Path], plan: Mapping[str, Any]
) -> Optional[Tuple[str, ...]]:
    """The goal receptacle instance of every transfer, from the episode config.

    The plan generator walked the config's objects, target receptacles, and
    goal receptacles in step, so the k-th transfer's receptacle is the k-th
    entry.  The objects are checked against the plan before that order is
    trusted.  ``None`` when the config is not available.
    """

    if rearrange_root is None:
        return None
    name = plan.get("init_config_name")
    if not name:
        return None
    path = Path(rearrange_root) / str(name)
    if not path.is_file():
        return None
    episode = json.loads(path.read_text())
    labels = list(episode.get("info", {}).get("object_labels", {}))
    picks = [
        str(item["obj_id"]) for item in plan["subtasks"] if item.get("type") == "pick"
    ]
    derived = ["{}-{}".format(*habitat_instance(label)) for label in labels]
    if derived != picks:
        raise ValueError(
            "episode config {} lists objects {}, the plan picks {}".format(path, derived, picks)
        )
    receptacles = episode.get("goal_receptacles")
    if not isinstance(receptacles, list) or len(receptacles) != len(picks):
        raise ValueError("episode config {} has no goal receptacle per object".format(path))
    return tuple(str(item[0]) for item in receptacles)


def load_episode(
    plan_path: Path,
    plan_index: int,
    rearrange_root: Optional[Path] = None,
) -> Tuple[TidyHouseEpisode, Dict[str, Any]]:
    """One episode and its one-plan document, from the official files."""

    dataset, plan = read_plan(plan_path, plan_index)
    receptacles = read_goal_receptacles(rearrange_root, plan)
    episode = TidyHouseEpisode.from_plan(
        plan, plan_index, dataset=dataset, goal_receptacles=receptacles
    )
    return episode, plan_document(dataset, plan)
