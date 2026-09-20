"""The proposer boundary: what a proposer is asked and what it must answer.

Four JSON documents cross the boundary between the controller and any
proposer, scripted or a real model:

```text
DecompositionRequest  goal + scene + contract inventory  ->  DecompositionResponse  ordered sub-goals
SubgraphRequest       one sub-goal + the same scene       ->  SubgraphResponse       one SkillSubgraph
```

Requests are built by the controller from library and environment objects.
Responses are parsed strictly through :mod:`mshab.skills.schema`, so a model
can put nothing but sub-goals, skill nodes, and edges into a graph.  When the
validator refuses an answer and asks again, the same request travels back
with its ``rejections`` filled in; that list is the third kind of document a
proposer reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from mshab.skills import schema
from mshab.skills.environment import EnvironmentDescription
from mshab.skills.graph import SkillSubgraph, SubGoal
from mshab.skills.library import ContractLibrary


SCHEMA_VERSION = "mshab.planning.v1"
# Tuples, not ``str`` enums like ``Outcome``/``Status`` in controller.py. Both
# are closed vocabularies checked at construction, so a misspelling raises
# where it is written; neither is ever compared against a bare literal to pick
# a branch, which is the case an enum protects and the reason the controller's
# two vocabularies are enums. They are also fed straight to argparse
# ``choices`` and interpolated into error messages, both of which want the
# plain values.
GRANULARITIES = ("free", "coarse", "fine")
STAGES = ("decomposition", "subgraph", "assembly", "graph", "plan")


def _require_identifiers(
    payload: Mapping[str, Any], key: str, *, where: str
) -> Tuple[str, ...]:
    values = schema.require_str_tuple(payload, key, where=where)
    for value in values:
        schema.require_identifier({key: value}, key, where=where)
    return values


def _require_granularity(value: Any) -> str:
    if value not in GRANULARITIES:
        raise ValueError(
            "granularity must be one of {}, got {!r}".format(GRANULARITIES, value)
        )
    return value


def _require_images(images: Iterable[Any]) -> Tuple[str, ...]:
    images = tuple(images)
    if any(not isinstance(item, str) or not item for item in images):
        raise ValueError("images must be non-empty strings (paths, URLs, or data URIs)")
    return images


def _require_rejections(items: Iterable[Any]) -> Tuple["Rejection", ...]:
    rejections = tuple(items)
    if any(not isinstance(item, Rejection) for item in rejections):
        raise TypeError("request rejections must be Rejection instances")
    return rejections


# -- what the validator tells the proposer about its previous answer ---------------


@dataclass(frozen=True)
class Rejection:
    """One reason an answer was refused, attributed to a stage and a sub-goal.

    Rejections are produced by the validator and travel back to the proposer
    on the retry of the same request (``DecompositionRequest.rejections``,
    ``SubgraphRequest.rejections``), so they are boundary documents too.
    """

    stage: str
    subgoal_id: Optional[str]
    message: str

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError("unknown rejection stage {!r}".format(self.stage))

    def as_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "subgoal_id": self.subgoal_id, "message": self.message}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Rejection":
        where = "rejection"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(payload, where=where, required=("stage", "subgoal_id", "message"))
        stage = schema.require_str(payload, "stage", where=where)
        if stage not in STAGES:
            raise schema.SchemaError(
                "{}.stage must be one of {}, got {!r}".format(where, STAGES, stage)
            )
        return cls(
            stage=stage,
            subgoal_id=schema.optional_identifier(payload, "subgoal_id", where=where),
            message=schema.optional_str(payload, "message", where=where),
        )


# -- what the proposer is told about the world -----------------------------------


def _require_contract_records(
    items: Iterable[Any], where: str
) -> Tuple[Dict[str, Any], ...]:
    """Contract records are ``Contract.as_dict()`` verbatim; only their shape is checked.

    The boundary defines no second contract serialization: the library's own
    record is what a proposer reads, and it is copied, not re-described.
    """

    records = []
    for index, item in enumerate(items):
        record_where = "{}.contracts[{}]".format(where, index)
        record = schema.require_mapping(item, where=record_where)
        schema.require_contract_id(record, "id", where=record_where)
        records.append(dict(record))
    return tuple(records)


@dataclass(frozen=True)
class EntityDescription:
    """A symbolic scene entity the proposer may name in arguments."""

    name: str
    kind: str

    def as_dict(self) -> Dict[str, str]:
        return {"name": self.name, "kind": self.kind}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EntityDescription":
        where = "entity"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(payload, where=where, required=("name", "kind"))
        return cls(
            schema.require_str(payload, "name", where=where),
            schema.require_str(payload, "kind", where=where),
        )


@dataclass(frozen=True)
class History:
    """What has already happened in this run."""

    achieved_subgoals: Tuple[str, ...] = ()
    completed_nodes: Tuple[str, ...] = ()
    failed_nodes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("achieved_subgoals", "completed_nodes", "failed_nodes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "achieved_subgoals": list(self.achieved_subgoals),
            "completed_nodes": list(self.completed_nodes),
            "failed_nodes": list(self.failed_nodes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "History":
        where = "history"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            required=("achieved_subgoals", "completed_nodes", "failed_nodes"),
        )
        return cls(
            _require_identifiers(payload, "achieved_subgoals", where=where),
            _require_identifiers(payload, "completed_nodes", where=where),
            _require_identifiers(payload, "failed_nodes", where=where),
        )


@dataclass(frozen=True)
class Failure:
    """Why a replan is requested: the sub-goal that failed and the evidence."""

    subgoal_id: str
    node_id: Optional[str] = None
    failure_mode: Optional[str] = None
    missing_effects: Tuple[str, ...] = ()
    missing_preconditions: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.subgoal_id:
            raise ValueError("failure must name the failed sub-goal")
        object.__setattr__(self, "missing_effects", tuple(self.missing_effects))
        object.__setattr__(
            self, "missing_preconditions", tuple(self.missing_preconditions)
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "subgoal_id": self.subgoal_id,
            "node_id": self.node_id,
            "failure_mode": self.failure_mode,
            "missing_effects": list(self.missing_effects),
            "missing_preconditions": list(self.missing_preconditions),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Failure":
        where = "failure"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            required=("subgoal_id",),
            optional=("node_id", "failure_mode", "missing_effects", "missing_preconditions"),
        )
        return cls(
            subgoal_id=schema.require_identifier(payload, "subgoal_id", where=where),
            node_id=schema.optional_identifier(payload, "node_id", where=where),
            failure_mode=schema.optional_str(payload, "failure_mode", where=where) or None,
            missing_effects=schema.require_str_tuple(
                payload, "missing_effects", where=where
            ),
            missing_preconditions=schema.require_str_tuple(
                payload, "missing_preconditions", where=where
            ),
        )


@dataclass(frozen=True)
class PlanningContext:
    """Everything a proposer is told besides the goal: inventory, scene, history."""

    contracts: Tuple[Mapping[str, Any], ...]
    entities: Tuple[EntityDescription, ...] = ()
    facts: Tuple[str, ...] = ()
    history: History = field(default_factory=History)
    failure: Optional[Failure] = None
    granularity: str = "free"
    attempt: int = 0
    #: Image references of the current scene, for a vision-capable proposer.
    #: Reserved: the symbolic environment never fills it, and the text-only
    #: DeepSeek proposer refuses a request that carries one.
    images: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "contracts", _require_contract_records(self.contracts, "planning_context")
        )
        object.__setattr__(self, "entities", tuple(self.entities))
        facts = tuple(sorted(set(self.facts)))
        if any(not isinstance(fact, str) or not fact for fact in facts):
            raise ValueError("facts must be non-empty strings")
        object.__setattr__(self, "facts", facts)
        object.__setattr__(self, "images", _require_images(self.images))
        _require_granularity(self.granularity)
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 0:
            raise ValueError("attempt must be a non-negative integer")

    @classmethod
    def initial(
        cls,
        library: ContractLibrary,
        task: str,
        *,
        entities: Iterable[EntityDescription] = (),
        facts: Iterable[str] = (),
        granularity: str = "free",
    ) -> "PlanningContext":
        """The context of a first plan: the library's contracts for ``task``."""

        return cls(
            contracts=contract_records(library, task),
            entities=tuple(entities),
            facts=tuple(facts),
            granularity=granularity,
        )

    def replan(
        self,
        failure: Failure,
        history: History,
        facts: Iterable[str],
        images: Iterable[str] = (),
    ) -> "PlanningContext":
        """The context of the next attempt after a sub-goal failed."""

        return PlanningContext(
            contracts=self.contracts,
            entities=self.entities,
            facts=tuple(facts),
            history=history,
            failure=failure,
            granularity=self.granularity,
            attempt=self.attempt + 1,
            images=tuple(images),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "contracts": [dict(record) for record in self.contracts],
            "entities": [item.as_dict() for item in self.entities],
            "facts": list(self.facts),
            "history": self.history.as_dict(),
            "failure": None if self.failure is None else self.failure.as_dict(),
            "granularity": self.granularity,
            "attempt": self.attempt,
            "images": list(self.images),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanningContext":
        where = "planning_context"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload, where=where, required=_CONTEXT_KEYS, optional=_OPTIONAL_CONTEXT_KEYS
        )
        return cls._from_fields(payload, where)

    @classmethod
    def _from_fields(cls, payload: Mapping[str, Any], where: str) -> "PlanningContext":
        failure = payload.get("failure")
        granularity = payload.get("granularity")
        try:
            _require_granularity(granularity)
        except ValueError as exc:
            raise schema.SchemaError("{}: {}".format(where, exc)) from exc
        return cls(
            contracts=_require_contract_records(
                schema.require_sequence(payload, "contracts", where=where), where
            ),
            entities=tuple(
                EntityDescription.from_dict(item)
                for item in schema.require_sequence(payload, "entities", where=where)
            ),
            facts=schema.require_str_tuple(payload, "facts", where=where),
            history=History.from_dict(
                schema.require_mapping(payload.get("history"), where="{}.history".format(where))
            ),
            failure=None if failure is None else Failure.from_dict(failure),
            granularity=granularity,
            attempt=schema.require_non_negative_int(payload, "attempt", where=where),
            images=schema.require_str_tuple(payload, "images", where=where),
        )


_CONTEXT_KEYS = (
    "contracts",
    "entities",
    "facts",
    "history",
    "failure",
    "granularity",
    "attempt",
)
# ``images`` is reserved and may be absent from older documents.
_OPTIONAL_CONTEXT_KEYS = ("images",)


def contract_records(
    library: ContractLibrary, task: Optional[str] = None
) -> Tuple[Dict[str, Any], ...]:
    """The library's ``Contract.as_dict()`` records for ``task``, sorted by id."""

    return tuple(contract.as_dict() for contract in library.find(task=task))


def entity_descriptions(
    description: EnvironmentDescription,
) -> Tuple[EntityDescription, ...]:
    return tuple(
        EntityDescription(entity.name, entity.kind)
        for entity in sorted(description.entities.values(), key=lambda item: item.name)
    )


# -- call 1: goal -> ordered sub-goals ----------------------------------------------


@dataclass(frozen=True)
class DecompositionRequest:
    """Call 1.  ``rejections`` is non-empty only on a retry of the same request:
    the reasons the previous answer was refused, for the proposer to fix."""

    task: str
    goal: str
    context: PlanningContext
    rejections: Tuple[Rejection, ...] = ()

    def __post_init__(self) -> None:
        if not self.task or not self.goal:
            raise ValueError("decomposition request needs a task and a goal")
        object.__setattr__(self, "rejections", _require_rejections(self.rejections))

    @property
    def images(self) -> Tuple[str, ...]:
        return self.context.images

    def fingerprint(self) -> Tuple[str, str, str, int, Optional[str]]:
        """What a scripted answer is keyed on; a retry keeps its fingerprint."""

        failure = self.context.failure
        return (
            self.task,
            self.goal,
            self.context.granularity,
            self.context.attempt,
            None if failure is None else failure.subgoal_id,
        )

    def with_rejections(self, rejections: Sequence[Rejection]) -> "DecompositionRequest":
        """The same request, asked again after these rejections."""

        return replace(self, rejections=tuple(rejections))

    def as_dict(self) -> Dict[str, Any]:
        document = {"schema_version": SCHEMA_VERSION, "task": self.task, "goal": self.goal}
        document.update(self.context.as_dict())
        document["rejections"] = [item.as_dict() for item in self.rejections]
        return document

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecompositionRequest":
        where = "decomposition_request"
        payload = schema.require_mapping(payload, where=where)
        schema.require_schema_version(payload, where=where, expected=SCHEMA_VERSION)
        schema.require_keys(
            payload,
            where=where,
            required=("task", "goal") + _CONTEXT_KEYS,
            optional=("schema_version", "rejections") + _OPTIONAL_CONTEXT_KEYS,
        )
        return cls(
            task=schema.require_identifier(payload, "task", where=where),
            goal=schema.require_str(payload, "goal", where=where),
            context=PlanningContext._from_fields(payload, where),
            rejections=tuple(
                Rejection.from_dict(item)
                for item in schema.require_sequence(payload, "rejections", where=where)
            ),
        )


@dataclass(frozen=True)
class DecompositionResponse:
    """An ordered sub-goal sequence; ``SubGoalGraph.from_sequence`` consumes it."""

    subgoals: Tuple[SubGoal, ...]
    rationale: str = ""

    def __post_init__(self) -> None:
        subgoals = tuple(self.subgoals)
        if not subgoals:
            raise ValueError("a decomposition must contain at least one sub-goal")
        ids = [subgoal.id for subgoal in subgoals]
        if len(ids) != len(set(ids)):
            raise ValueError("a decomposition cannot repeat a sub-goal id")
        object.__setattr__(self, "subgoals", subgoals)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "subgoals": [subgoal.as_dict() for subgoal in self.subgoals],
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecompositionResponse":
        where = "decomposition_response"
        payload = schema.require_mapping(payload, where=where)
        schema.require_schema_version(payload, where=where, expected=SCHEMA_VERSION)
        schema.require_keys(
            payload,
            where=where,
            required=("subgoals",),
            optional=("rationale", "schema_version"),
        )
        return cls(
            subgoals=tuple(
                SubGoal.from_dict(item)
                for item in schema.require_sequence(payload, "subgoals", where=where)
            ),
            rationale=schema.optional_str(payload, "rationale", where=where),
        )


# -- call 2: one sub-goal -> one skill subgraph -------------------------------------


@dataclass(frozen=True)
class Neighbours:
    """The predicates before and after the sub-goal being planned; context only."""

    previous: Optional[str] = None
    next: Optional[str] = None

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {"previous": self.previous, "next": self.next}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Neighbours":
        where = "neighbours"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(payload, where=where, required=("previous", "next"))
        return cls(
            previous=schema.optional_str(payload, "previous", where=where) or None,
            next=schema.optional_str(payload, "next", where=where) or None,
        )


@dataclass(frozen=True)
class SubgraphRequest:
    """Call 2, asked once per sub-goal.  ``images`` mirrors the context's
    reserved field; ``rejections`` is non-empty only on a retry of the same
    request."""

    task: str
    goal: str
    subgoal: SubGoal
    contracts: Tuple[Mapping[str, Any], ...]
    entities: Tuple[EntityDescription, ...] = ()
    facts: Tuple[str, ...] = ()
    neighbours: Neighbours = field(default_factory=Neighbours)
    images: Tuple[str, ...] = ()
    rejections: Tuple[Rejection, ...] = ()

    def __post_init__(self) -> None:
        if not self.task or not self.goal:
            raise ValueError("subgraph request needs a task and a goal")
        object.__setattr__(
            self, "contracts", _require_contract_records(self.contracts, "subgraph_request")
        )
        object.__setattr__(self, "entities", tuple(self.entities))
        object.__setattr__(self, "facts", tuple(sorted(set(self.facts))))
        object.__setattr__(self, "images", _require_images(self.images))
        object.__setattr__(self, "rejections", _require_rejections(self.rejections))

    def fingerprint(self) -> Tuple[str, str, str]:
        """What a scripted answer is keyed on: the sub-goal's id and predicate."""

        return (self.task, self.subgoal.id, self.subgoal.predicate)

    def with_rejections(self, rejections: Sequence[Rejection]) -> "SubgraphRequest":
        """The same request, asked again after these rejections."""

        return replace(self, rejections=tuple(rejections))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "task": self.task,
            "goal": self.goal,
            "subgoal": self.subgoal.as_dict(),
            "contracts": [dict(record) for record in self.contracts],
            "entities": [item.as_dict() for item in self.entities],
            "facts": list(self.facts),
            "neighbours": self.neighbours.as_dict(),
            "images": list(self.images),
            "rejections": [item.as_dict() for item in self.rejections],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SubgraphRequest":
        where = "subgraph_request"
        payload = schema.require_mapping(payload, where=where)
        schema.require_schema_version(payload, where=where, expected=SCHEMA_VERSION)
        schema.require_keys(
            payload,
            where=where,
            required=("task", "goal", "subgoal", "contracts", "entities", "facts", "neighbours"),
            optional=("schema_version", "images", "rejections"),
        )
        return cls(
            task=schema.require_identifier(payload, "task", where=where),
            goal=schema.require_str(payload, "goal", where=where),
            subgoal=SubGoal.from_dict(payload["subgoal"]),
            contracts=_require_contract_records(
                schema.require_sequence(payload, "contracts", where=where), where
            ),
            entities=tuple(
                EntityDescription.from_dict(item)
                for item in schema.require_sequence(payload, "entities", where=where)
            ),
            facts=schema.require_str_tuple(payload, "facts", where=where),
            neighbours=Neighbours.from_dict(payload["neighbours"]),
            images=schema.require_str_tuple(payload, "images", where=where),
            rejections=tuple(
                Rejection.from_dict(item)
                for item in schema.require_sequence(payload, "rejections", where=where)
            ),
        )


def lean_subgraph_dict(subgraph: SkillSubgraph) -> Dict[str, Any]:
    """The part of ``SkillSubgraph.as_dict()`` a proposer has to decide.

    ``achiever_nodes`` and ``execution_order`` are derived views;
    ``SkillSubgraph.from_dict`` recomputes them, so a proposer never emits them.
    """

    return {
        "subgoal_id": subgraph.subgoal_id,
        "nodes": [
            node.as_dict()
            for node in sorted(subgraph.nodes.values(), key=lambda item: item.id)
        ],
        "edges": [edge.as_dict() for edge in subgraph.edges],
    }


@dataclass(frozen=True)
class SubgraphResponse:
    """One skill subgraph for one sub-goal: nodes and their internal relations."""

    subgraph: SkillSubgraph
    rationale: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "subgraph": lean_subgraph_dict(self.subgraph),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, task: str) -> "SubgraphResponse":
        """Parse strictly; ``task`` comes from the request, never the payload."""

        where = "subgraph_response"
        payload = schema.require_mapping(payload, where=where)
        schema.require_schema_version(payload, where=where, expected=SCHEMA_VERSION)
        schema.require_keys(
            payload,
            where=where,
            required=("subgraph",),
            optional=("rationale", "schema_version"),
        )
        return cls(
            subgraph=SkillSubgraph.from_dict(payload["subgraph"], task=task),
            rationale=schema.optional_str(payload, "rationale", where=where),
        )
