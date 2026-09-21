"""One official SetTable episode as the symbolic scene the controller plans over.

MS-HAB fixes a SetTable episode as a sequential task plan: two segments, the
bowl out of the kitchen counter drawer and the apple out of the fridge, each
``navigate -> open -> navigate -> pick -> navigate -> place -> navigate ->
close``, 16 atomic subtasks.  This module reads that plan as plain JSON,
without a simulator, and derives what the upper layers and the environment
adapter need from it:

- the symbolic entities: one name per object, one per storage (the
  articulation an object is taken out of), one per goal receptacle, with the
  plan's instance ids behind them;
- the goal text and the goal facts the run is judged on.  The goal does not
  say which storage holds which object: a proposer that opens the wrong one
  has to notice and plan around it;
- which subtask of the plan a grounded skill node stands for.  The MS-HAB
  policies observe the object, articulation, and goal of *the subtask the
  environment is pointed at*, so executing ``pick(024_bowl)`` means pointing
  the environment at the bowl's pick subtask first;
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


#: The atomic subtasks of one segment, in the official order.
SEGMENT_SUBTASK_TYPES = (
    "navigate", "open", "navigate", "pick", "navigate", "place", "navigate", "close",
)
#: The segment role of each subtask, in the same order: the builder's names.
SEGMENT_SUBTASK_ROLES = (
    "navigate_to_source",
    "open",
    "navigate_to_object",
    "pick",
    "navigate_to_destination",
    "place",
    "navigate_back_to_source",
    "close",
)
OBJECT_KIND = "object"
ARTICULATION_KIND = "articulation"
RECEPTACLE_KIND = "receptacle"
#: Relative to the rearrange dataset root: the official sequential plans.
SEQUENTIAL_PLAN_PATH = "task_plans/set_table/sequential/{split}/all.json"
EPISODE_SCHEMA_VERSION = "mshab.rollout-episode.v2"


class UnsupportedGrounding(ValueError):
    """A grounded skill node names nothing this episode's plan can execute."""


def object_category(instance_id: str) -> str:
    """``024_bowl-0`` -> ``024_bowl``, the checkpoint target."""

    category, separator, number = instance_id.rpartition("-")
    if not separator or not category or not number.isdigit():
        raise ValueError(
            "{!r} is not a <category>-<number> object id".format(instance_id)
        )
    return category


def articulation_instance(instance_id: str) -> Tuple[str, int]:
    """``kitchen_counter-0`` -> ``("kitchen_counter", 0)``: the plan's articulation ids."""

    base, separator, number = instance_id.rpartition("-")
    if not separator or not base or not number.isdigit():
        raise ValueError(
            "{!r} is not a <articulation>-<number> articulation id".format(instance_id)
        )
    return base, int(number)


def habitat_instance(name: str) -> Tuple[str, int]:
    """``frl_apartment_table_01_:0000`` -> ``("frl_apartment_table_01", 0)``.

    The rearrange episode configs name every object and receptacle instance
    this way; the plan generator turns the objects into ``<category>-<n>``.
    """

    base, separator, number = name.rpartition("_:")
    if not separator or not base or not number.isdigit():
        raise ValueError("{!r} is not a <name>_:<number> instance id".format(name))
    return base, int(number)


def segment_label(object_name: str) -> str:
    """``024_bowl`` -> ``bowl``: the segment label the gold builders use.

    The YCB category prefix is dropped so that an episode of the official
    objects gets the gold graphs' own labels; a second instance keeps its
    suffix (``024_bowl_2`` -> ``bowl_2``).
    """

    head, separator, rest = object_name.partition("_")
    if separator and head.isdigit() and rest:
        return rest
    return object_name


def _symbolic_names(instances: Sequence[Tuple[str, str]]) -> Dict[str, str]:
    """One symbolic name per distinct instance key.

    Distinct instances that share a base name get ``base``, ``base_2``,
    ``base_3`` in the order of their instance ids (``-0`` before ``-1``,
    ``_:0000`` before ``_:0001``), whatever order the plan visits them in.
    The first keeps the bare name, so an object whose category appears once
    in the episode still matches the checkpoint trained for it, and the
    articulation keeps the name its open and close checkpoints carry.
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
class Segment:
    """One object of the plan, its storage, and the eight subtasks that move it."""

    index: int
    label: str
    object: str
    source: str
    destination: str
    object_instance: str
    source_instance: str
    receptacle_instance: Optional[str]
    navigate_to_source: int
    open: int
    navigate_to_object: int
    pick: int
    navigate_to_destination: int
    place: int
    navigate_back_to_source: int
    close: int
    goal_position: Optional[Tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        if self.index < 1:
            raise ValueError("segment index is 1-based")
        for name in ("label", "object", "source", "destination", "object_instance", "source_instance"):
            if not getattr(self, name):
                raise ValueError("a segment needs a non-empty {}".format(name))
        if self.goal_position is not None:
            object.__setattr__(
                self, "goal_position", tuple(float(value) for value in self.goal_position)
            )

    @property
    def triple(self) -> Tuple[str, str, str]:
        """``(label, object, source)``: the builder's segment description."""

        return (self.label, self.object, self.source)

    @property
    def subtasks(self) -> Tuple[int, ...]:
        return (
            self.navigate_to_source,
            self.open,
            self.navigate_to_object,
            self.pick,
            self.navigate_to_destination,
            self.place,
            self.navigate_back_to_source,
            self.close,
        )

    def subtask_of(self, role: str) -> int:
        """The plan subtask of one segment role (``pick``, ``navigate_to_source``, ...)."""

        if role not in SEGMENT_SUBTASK_ROLES:
            raise KeyError("unknown segment role {!r}".format(role))
        return getattr(self, role)

    @property
    def held(self) -> str:
        return "holding({})".format(self.object)

    @property
    def delivered(self) -> str:
        return "at({},{})".format(self.object, self.destination)

    @property
    def source_open(self) -> str:
        return "open({})".format(self.source)

    @property
    def source_closed(self) -> str:
        return "closed({})".format(self.source)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "object": self.object,
            "source": self.source,
            "destination": self.destination,
            "object_instance": self.object_instance,
            "source_instance": self.source_instance,
            "receptacle_instance": self.receptacle_instance,
            "subtasks": {role: self.subtask_of(role) for role in SEGMENT_SUBTASK_ROLES},
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
    """What the environment measured about every object, goal, and articulation of the plan.

    Keyed by subtask index: ``grasped[i]`` for the pick subtask of an object,
    ``at_goal[i]`` for its place subtask, ``near[i]`` for a navigate subtask
    (the base is close to and facing that subtask's goal), ``opened[i]`` and
    ``closed[i]`` for an open or close subtask (its articulation's handle
    joint beyond the open threshold, or within the closed one).  The
    environment checks all of them every step, not only the subtask it is
    pointed at; turning them into predicates is :meth:`SetTableEpisode.facts`.
    """

    grasped: Mapping[int, bool]
    at_goal: Mapping[int, bool]
    near: Mapping[int, bool]
    opened: Mapping[int, bool] = MappingProxyType({})
    closed: Mapping[int, bool] = MappingProxyType({})
    collision_safe: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "grasped", _frozen_bools(self.grasped, "grasped"))
        object.__setattr__(self, "at_goal", _frozen_bools(self.at_goal, "at_goal"))
        object.__setattr__(self, "near", _frozen_bools(self.near, "near"))
        object.__setattr__(self, "opened", _frozen_bools(self.opened, "opened"))
        object.__setattr__(self, "closed", _frozen_bools(self.closed, "closed"))
        object.__setattr__(self, "collision_safe", bool(self.collision_safe))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "grasped": {str(key): value for key, value in self.grasped.items()},
            "at_goal": {str(key): value for key, value in self.at_goal.items()},
            "near": {str(key): value for key, value in self.near.items()},
            "opened": {str(key): value for key, value in self.opened.items()},
            "closed": {str(key): value for key, value in self.closed.items()},
            "collision_safe": self.collision_safe,
        }


@dataclass(frozen=True)
class SetTableEpisode:
    """One official episode: its segments, entities, goal, and subtask mapping."""

    plan_index: int
    dataset: str
    build_config_name: str
    init_config_name: str
    segments: Tuple[Segment, ...]

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        if not segments:
            raise ValueError("an episode has at least one segment")
        if [item.index for item in segments] != list(range(1, len(segments) + 1)):
            raise ValueError("segments are numbered 1, 2, ... in plan order")
        for name in ("label", "object"):
            values = [getattr(item, name) for item in segments]
            if len(set(values)) != len(values):
                raise ValueError("every segment has a distinct {}".format(name))
        others = set(self.sources_of(segments)) | set(self.destinations_of(segments))
        for item in segments:
            if item.object in others:
                raise ValueError(
                    "{!r} is both an object and a storage or receptacle".format(item.object)
                )
        object.__setattr__(self, "segments", segments)

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_plan(
        cls,
        plan: Mapping[str, Any],
        plan_index: int,
        *,
        dataset: str = "",
        goal_receptacles: Optional[Sequence[str]] = None,
    ) -> "SetTableEpisode":
        """Read one plan of the official ``all.json`` document.

        ``goal_receptacles`` are the episode config's receptacle instance ids
        in segment order (see :func:`read_goal_receptacles`); they give the
        destinations their scene names.  Without them destinations are
        ``receptacle_1``, ``receptacle_2``, ... by distinct goal rectangle.
        """

        subtasks = list(plan["subtasks"])
        width = len(SEGMENT_SUBTASK_TYPES)
        if not subtasks or len(subtasks) % width:
            raise ValueError(
                "a SetTable plan has {} subtasks per segment, got {}".format(
                    width, len(subtasks)
                )
            )
        groups = [subtasks[start : start + width] for start in range(0, len(subtasks), width)]
        for number, group in enumerate(groups, start=1):
            group_types = tuple(str(item.get("type")) for item in group)
            if group_types != SEGMENT_SUBTASK_TYPES:
                raise ValueError(
                    "segment {} is {}, expected {}".format(
                        number, group_types, SEGMENT_SUBTASK_TYPES
                    )
                )
            if group[3].get("obj_id") != group[5].get("obj_id"):
                raise ValueError(
                    "segment {} picks {!r} but places {!r}".format(
                        number, group[3].get("obj_id"), group[5].get("obj_id")
                    )
                )
            if group[1].get("articulation_id") != group[7].get("articulation_id"):
                raise ValueError(
                    "segment {} opens {!r} but closes {!r}".format(
                        number, group[1].get("articulation_id"), group[7].get("articulation_id")
                    )
                )
            if not group[1].get("articulation_type"):
                raise ValueError("segment {} names no articulation_type".format(number))
        instances = [str(group[3]["obj_id"]) for group in groups]
        object_names = _symbolic_names(
            [(instance, object_category(instance)) for instance in instances]
        )
        articulations = [str(group[1]["articulation_id"]) for group in groups]
        source_names = _symbolic_names(
            [
                (articulation, str(group[1]["articulation_type"]))
                for articulation, group in zip(articulations, groups)
            ]
        )
        if goal_receptacles is not None:
            receptacles = [str(item) for item in goal_receptacles]
            if len(receptacles) != len(groups):
                raise ValueError(
                    "{} goal receptacles for {} segments".format(len(receptacles), len(groups))
                )
            destination_names = _symbolic_names(
                [(item, habitat_instance(item)[0]) for item in receptacles]
            )
            destinations = [destination_names[item] for item in receptacles]
        else:
            receptacles = [None] * len(groups)
            keys = [_goal_key(group[5]) for group in groups]
            distinct = list(dict.fromkeys(keys))
            destinations = [
                "receptacle_{}".format(distinct.index(key) + 1) for key in keys
            ]
        labels = [segment_label(object_names[instance]) for instance in instances]
        if len(set(labels)) != len(labels):
            labels = [object_names[instance] for instance in instances]
        segments = []
        for number, (group, instance, articulation, receptacle, destination, label) in enumerate(
            zip(groups, instances, articulations, receptacles, destinations, labels), start=1
        ):
            base = (number - 1) * width
            goal_position = group[5].get("goal_pos")
            segments.append(
                Segment(
                    index=number,
                    label=label,
                    object=object_names[instance],
                    source=source_names[articulation],
                    destination=destination,
                    object_instance=instance,
                    source_instance=articulation,
                    receptacle_instance=receptacle,
                    navigate_to_source=base,
                    open=base + 1,
                    navigate_to_object=base + 2,
                    pick=base + 3,
                    navigate_to_destination=base + 4,
                    place=base + 5,
                    navigate_back_to_source=base + 6,
                    close=base + 7,
                    goal_position=None if goal_position is None else tuple(goal_position),
                )
            )
        return cls(
            plan_index=int(plan_index),
            dataset=str(dataset),
            build_config_name=str(plan.get("build_config_name", "")),
            init_config_name=str(plan.get("init_config_name", "")),
            segments=tuple(segments),
        )

    # -- the symbolic scene -----------------------------------------------------

    @staticmethod
    def sources_of(segments: Iterable[Segment]) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(item.source for item in segments))

    @staticmethod
    def destinations_of(segments: Iterable[Segment]) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(item.destination for item in segments))

    @property
    def objects(self) -> Tuple[str, ...]:
        return tuple(item.object for item in self.segments)

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(item.label for item in self.segments)

    @property
    def sources(self) -> Tuple[str, ...]:
        """Distinct storages, in first-appearance order."""

        return self.sources_of(self.segments)

    @property
    def destinations(self) -> Tuple[str, ...]:
        """Distinct receptacles, in first-appearance order."""

        return self.destinations_of(self.segments)

    @property
    def subtask_count(self) -> int:
        return len(self.segments) * len(SEGMENT_SUBTASK_TYPES)

    @property
    def goal(self) -> str:
        """The goal text: the objects, where they go, and that their storage ends closed.

        It does not pair an object with the storage it is in.
        """

        if len(self.destinations) == 1:
            listing = "{} on {}".format(_listing(self.objects), self.destinations[0])
        else:
            listing = _listing(
                ["{} on {}".format(item.object, item.destination) for item in self.segments]
            )
        return "Set the table: put {}, then close the storage {} came from.".format(
            listing, "it" if len(self.objects) == 1 else "they"
        )

    @property
    def goal_facts(self) -> Tuple[str, ...]:
        return tuple(item.delivered for item in self.segments) + tuple(
            "closed({})".format(source) for source in self.sources
        )

    def entities(self) -> Tuple[EntityDescription, ...]:
        """What a proposer is told: objects, then storages, then receptacles."""

        return (
            tuple(EntityDescription(name, OBJECT_KIND) for name in self.objects)
            + tuple(EntityDescription(name, ARTICULATION_KIND) for name in self.sources)
            + tuple(EntityDescription(name, RECEPTACLE_KIND) for name in self.destinations)
        )

    def environment_entities(self) -> Tuple[EnvironmentEntity, ...]:
        """The adapter's entity map: symbolic name -> the plan's instance."""

        entities = []
        for item in self.segments:
            entities.append(
                EnvironmentEntity(
                    item.object,
                    OBJECT_KIND,
                    item.object_instance,
                    {"category": object_category(item.object_instance), "segment": item.index},
                )
            )
        for source in self.sources:
            owners = [item for item in self.segments if item.source == source]
            entities.append(
                EnvironmentEntity(
                    source,
                    ARTICULATION_KIND,
                    owners[0].source_instance,
                    {
                        "articulation_type": articulation_instance(owners[0].source_instance)[0],
                        "segments": tuple(item.index for item in owners),
                    },
                )
            )
        for destination in self.destinations:
            owners = [item for item in self.segments if item.destination == destination]
            entities.append(
                EnvironmentEntity(
                    destination,
                    RECEPTACLE_KIND,
                    owners[0].receptacle_instance or destination,
                    {"segments": tuple(item.index for item in owners)},
                )
            )
        return tuple(entities)

    def segment(self, object_name: str) -> Segment:
        for item in self.segments:
            if item.object == object_name:
                return item
        raise UnsupportedGrounding(
            "{!r} is not an object of this episode; objects: {}".format(
                object_name, list(self.objects)
            )
        )

    def segment_by_label(self, label: str) -> Segment:
        for item in self.segments:
            if item.label == label:
                return item
        raise KeyError("{!r} is not a segment label; labels: {}".format(label, list(self.labels)))

    # -- grounding -> subtask -----------------------------------------------------

    def subtask_for(
        self,
        contract_type: str,
        arguments: Mapping[str, Any],
        facts: Iterable[str] = (),
    ) -> int:
        """The plan subtask a grounded node stands for.

        ``pick(object)`` and ``place(object, destination)`` are the object's
        own subtasks; a place to a receptacle the plan does not deliver the
        object to is :class:`UnsupportedGrounding`, which the executor reports
        as a failed execution and the controller replans around.  A storage
        shared by several segments resolves to the segment whose object still
        needs it: ``open(storage)`` and ``navigate(storage)`` while it is
        closed go to the segment not yet delivered, ``close(storage)`` and
        ``navigate(storage)`` while it is open to the navigation before its
        close.  ``navigate(receptacle)`` resolves to the segment whose object
        is held, else the first not yet delivered.
        """

        facts = set(facts)
        if contract_type == "navigate":
            target = str(arguments.get("target"))
            for item in self.segments:
                if item.object == target:
                    return item.navigate_to_object
            by_source = [item for item in self.segments if item.source == target]
            if by_source:
                item = self._active_segment(by_source, facts)
                if item.source_open in facts:
                    return item.navigate_back_to_source
                return item.navigate_to_source
            by_destination = [item for item in self.segments if item.destination == target]
            if by_destination:
                return self._active_segment(by_destination, facts).navigate_to_destination
            raise UnsupportedGrounding(
                "{!r} is neither an object, a storage, nor a receptacle of this episode; "
                "entities: {}".format(target, [entity.name for entity in self.entities()])
            )
        if contract_type == "pick":
            return self.segment(str(arguments.get("object"))).pick
        if contract_type == "place":
            item = self.segment(str(arguments.get("object")))
            destination = str(arguments.get("destination"))
            if item.destination != destination:
                raise UnsupportedGrounding(
                    "this episode delivers {} to {}, not {}".format(
                        item.object, item.destination, destination
                    )
                )
            return item.place
        if contract_type in ("open", "close"):
            articulation = str(arguments.get("articulation"))
            candidates = [item for item in self.segments if item.source == articulation]
            if not candidates:
                raise UnsupportedGrounding(
                    "{!r} is not a storage of this episode; storages: {}".format(
                        articulation, list(self.sources)
                    )
                )
            item = self._active_segment(candidates, facts)
            return item.open if contract_type == "open" else item.close
        raise UnsupportedGrounding(
            "SetTable has no {!r} subtask; its plan is {}".format(
                contract_type, "/".join(SEGMENT_SUBTASK_TYPES)
            )
        )

    @staticmethod
    def _active_segment(candidates: Sequence[Segment], facts: Iterable[str]) -> Segment:
        """Of several segments sharing an entity: the held one, else the first undelivered, else the first."""

        facts = set(facts)
        for item in candidates:
            if item.held in facts:
                return item
        for item in candidates:
            if item.delivered not in facts:
                return item
        return candidates[0]

    def subtask_label(self, index: int) -> str:
        """``pick(024_bowl)`` and the like, for logs and summaries."""

        width = len(SEGMENT_SUBTASK_TYPES)
        if not 0 <= index < self.subtask_count:
            raise IndexError("subtask {} is outside the plan".format(index))
        item = self.segments[index // width]
        role = SEGMENT_SUBTASK_ROLES[index % width]
        if role in ("navigate_to_source", "navigate_back_to_source"):
            return "navigate({})".format(item.source)
        if role == "navigate_to_object":
            return "navigate({})".format(item.object)
        if role == "navigate_to_destination":
            return "navigate({})".format(item.destination)
        if role == "open":
            return "open({})".format(item.source)
        if role == "pick":
            return "pick({})".format(item.object)
        if role == "place":
            return "place({},{})".format(item.object, item.destination)
        return "close({})".format(item.source)

    # -- measurements -> facts ------------------------------------------------------

    def facts(self, measurements: SceneMeasurements) -> FrozenSet[str]:
        """The scene as predicates the contracts of the granularity library use.

        ``holding`` follows the grasp check; ``at`` needs the object inside
        its goal and not grasped, as MS-HAB's own place success does;
        ``reachable`` is MS-HAB's navigation success for that target without
        the arm checks, and an object in the gripper is reachable wherever
        the robot stands (the symbolic environment keeps ``reachable(x)``
        after ``pick(x)`` too); ``closed(storage)`` is the close checker's
        joint term and ``open(storage)`` the open checker's, with a storage
        that is measured and not closed counting as open, so a half-open
        drawer can be closed but not opened; ``gripper_empty`` holds when no
        object is grasped; ``collision_safe`` is the pointed subtask's
        cumulative-force limit.
        """

        facts = {"present({})".format(entity.name) for entity in self.entities()}
        if measurements.collision_safe:
            facts.add("collision_safe()")
        holding = False
        for item in self.segments:
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
            if measurements.near.get(item.navigate_to_source, False) or measurements.near.get(
                item.navigate_back_to_source, False
            ):
                facts.add("reachable({})".format(item.source))
        for source in self.sources:
            indices = [
                index
                for item in self.segments
                if item.source == source
                for index in (item.open, item.close)
            ]
            measured = any(
                index in measurements.opened or index in measurements.closed for index in indices
            )
            is_closed = any(measurements.closed.get(index, False) for index in indices)
            is_open = any(measurements.opened.get(index, False) for index in indices)
            if is_closed:
                facts.add("closed({})".format(source))
            if is_open or (measured and not is_closed):
                facts.add("open({})".format(source))
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
            "segments": [item.as_dict() for item in self.segments],
        }


def _listing(parts: Sequence[str]) -> str:
    parts = list(parts)
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return "{} and {}".format(*parts)
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


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
    """The goal receptacle instance of every segment, from the episode config.

    The plan generator walked the config's objects, target receptacles, and
    goal receptacles in step, so the k-th segment's receptacle is the k-th
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
) -> Tuple[SetTableEpisode, Dict[str, Any]]:
    """One episode and its one-plan document, from the official files."""

    dataset, plan = read_plan(plan_path, plan_index)
    receptacles = read_goal_receptacles(rearrange_root, plan)
    episode = SetTableEpisode.from_plan(
        plan, plan_index, dataset=dataset, goal_receptacles=receptacles
    )
    return episode, plan_document(dataset, plan)
