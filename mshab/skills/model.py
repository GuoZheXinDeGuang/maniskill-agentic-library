"""Object model for executable MS-HAB skills.

The classes in this module describe *what* a skill promises separately from
*how* it is executed.  This matters for a skill library: ``pick(apple)`` is one
semantic capability even when RL, BC, DP, or a remote VLA can all execute it.

This module deliberately has no torch or ManiSkill imports.  Listing and
planning with a skill library must not allocate a GPU or load checkpoints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


class SkillType(str, Enum):
    """The primitive skill vocabulary currently implemented by MS-HAB."""

    NAVIGATE = "navigate"
    PICK = "pick"
    PLACE = "place"
    OPEN = "open"
    CLOSE = "close"


class ParameterType(str, Enum):
    """Planner-facing parameter types used by skill contracts."""

    ENTITY = "entity"
    LOCATION = "location"
    ARTICULATION = "articulation"
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"


class ExecutorType(str, Enum):
    """Execution engines represented in the architecture diagram."""

    POLICY = "policy"
    VLA = "vla"
    NAVIGATION = "navigation"
    CONTROLLER = "controller"
    SCRIPT = "script"
    STATE_MACHINE = "state_machine"


class ArtifactStatus(str, Enum):
    """Local availability of an execution backend's artifacts."""

    MISSING = "missing"
    PARTIAL = "partial"
    READY = "ready"


@dataclass(frozen=True)
class SkillParameter:
    name: str
    type: ParameterType
    description: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", ParameterType(self.type))
        if not self.name or not self.name.isidentifier():
            raise ValueError("skill parameter name must be a non-empty identifier")


@dataclass(frozen=True)
class BoundContract:
    """A contract whose predicate templates have concrete arguments."""

    preconditions: Tuple[str, ...]
    effects: Tuple[str, ...]
    invariants: Tuple[str, ...]
    verification: Tuple[str, ...]
    failure_modes: Tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "preconditions",
            "effects",
            "invariants",
            "verification",
            "failure_modes",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def can_start(self, facts: Iterable[str]) -> bool:
        return set(self.preconditions).issubset(set(facts))

    def achieved(self, facts: Iterable[str]) -> bool:
        return set(self.effects).issubset(set(facts))

    def verified(self, facts: Iterable[str]) -> bool:
        """Whether the explicit success predicates hold after execution."""

        return set(self.verification).issubset(set(facts))


@dataclass(frozen=True)
class SkillContract:
    """Executable pre/post-condition contract for one semantic skill.

    Predicates use ``str.format`` placeholders matching parameter names, for
    example ``holding({object})``.  They stay lightweight strings here; a
    simulator adapter is responsible for evaluating them against live state.
    """

    parameters: Tuple[SkillParameter, ...]
    preconditions: Tuple[str, ...]
    effects: Tuple[str, ...]
    invariants: Tuple[str, ...] = ()
    verification: Tuple[str, ...] = ()
    failure_modes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "parameters",
            "preconditions",
            "effects",
            "invariants",
            "verification",
            "failure_modes",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("skill contract has duplicate parameter names")
        if not self.effects:
            raise ValueError("skill contract must declare at least one effect")
        if not self.verification:
            object.__setattr__(self, "verification", tuple(self.effects))
        for group_name, predicates in (
            ("preconditions", self.preconditions),
            ("effects", self.effects),
            ("invariants", self.invariants),
            ("verification", self.verification),
        ):
            if any(not predicate.strip() for predicate in predicates):
                raise ValueError("{} cannot contain empty predicates".format(group_name))

    def bind(self, arguments: Mapping[str, Any]) -> BoundContract:
        values = dict(arguments)
        known = {parameter.name: parameter for parameter in self.parameters}
        missing = sorted(
            parameter.name
            for parameter in self.parameters
            if parameter.required and parameter.name not in values
        )
        unknown = sorted(set(values) - set(known))
        if missing or unknown:
            raise ValueError(
                "invalid skill arguments: missing={}, unknown={}".format(missing, unknown)
            )
        for name, value in values.items():
            _validate_parameter(known[name], value)

        def ground(predicates: Sequence[str]) -> Tuple[str, ...]:
            try:
                return tuple(predicate.format(**values) for predicate in predicates)
            except KeyError as exc:
                raise ValueError(
                    "predicate template references undeclared/unbound parameter {!r}".format(
                        exc.args[0]
                    )
                ) from exc

        return BoundContract(
            preconditions=ground(self.preconditions),
            effects=ground(self.effects),
            invariants=ground(self.invariants),
            verification=ground(self.verification),
            failure_modes=self.failure_modes,
        )


def _validate_parameter(parameter: SkillParameter, value: Any) -> None:
    expected = {
        ParameterType.ENTITY: str,
        ParameterType.LOCATION: str,
        ParameterType.ARTICULATION: str,
        ParameterType.STRING: str,
        ParameterType.INTEGER: int,
        ParameterType.FLOAT: (int, float),
        ParameterType.BOOLEAN: bool,
    }[parameter.type]
    if isinstance(value, bool) and parameter.type in (
        ParameterType.INTEGER,
        ParameterType.FLOAT,
    ):
        valid = False
    else:
        valid = isinstance(value, expected)
    if not valid:
        raise TypeError(
            "argument {!r} must be {}, got {}".format(
                parameter.name, parameter.type.value, type(value).__name__
            )
        )


class ExecutionBackend(ABC):
    """One interchangeable implementation of an atomic skill."""

    def __init__(self, key: str, executor_type: ExecutorType) -> None:
        if not key:
            raise ValueError("backend key cannot be empty")
        self.key = key
        self.executor_type = executor_type

    @property
    @abstractmethod
    def status(self) -> ArtifactStatus:
        """Whether everything needed to execute this backend is available."""

    @abstractmethod
    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable library record."""


class CheckpointBackend(ExecutionBackend):
    """A local learned policy represented by ``config.yml`` + ``policy.pt``."""

    def __init__(
        self,
        key: str,
        family: str,
        checkpoint_path: Path,
        config_path: Path,
        policy_type: Optional[str] = None,
        checkpoint_sha256: Optional[str] = None,
    ) -> None:
        super().__init__(key=key, executor_type=ExecutorType.POLICY)
        self.family = family
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.policy_type = policy_type or family
        self.checkpoint_sha256 = checkpoint_sha256

    @property
    def status(self) -> ArtifactStatus:
        present = (self.checkpoint_path.is_file(), self.config_path.is_file())
        if all(present):
            return ArtifactStatus.READY
        if any(present):
            return ArtifactStatus.PARTIAL
        return ArtifactStatus.MISSING

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "executor_type": self.executor_type.value,
            "family": self.family,
            "policy_type": self.policy_type,
            "checkpoint": str(self.checkpoint_path),
            "config": str(self.config_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class SkillInvocation:
    """One grounded call to a skill, suitable for a graph node."""

    skill: "Skill" = field(repr=False)
    arguments: Mapping[str, Any]
    contract: BoundContract
    backend_key: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))

    @property
    def id(self) -> str:
        args = ",".join(
            "{}={}".format(key, self.arguments[key]) for key in sorted(self.arguments)
        )
        return "{}({})".format(self.skill.id, args)


class Skill(ABC):
    """Semantic interface shared by every library-managed executable skill."""

    def __init__(
        self,
        name: str,
        task: str,
        contract: SkillContract,
        description: str = "",
    ) -> None:
        if not name or not task:
            raise ValueError("skill name and task must be non-empty")
        self.name = name
        self.task = task
        self.contract = contract
        self.description = description

    @property
    def id(self) -> str:
        return "mshab.{}.{}".format(self.task, self.name)

    @property
    @abstractmethod
    def ready(self) -> bool:
        """Whether at least one complete execution path is available."""

    def bind(
        self, arguments: Mapping[str, Any], backend_key: Optional[str] = None
    ) -> SkillInvocation:
        return SkillInvocation(
            skill=self,
            arguments=dict(arguments),
            contract=self.contract.bind(arguments),
            backend_key=backend_key,
        )

    @abstractmethod
    def as_dict(self) -> Dict[str, Any]:
        pass


class AtomicSkill(Skill):
    """A smallest dispatchable skill with one or more executor backends."""

    def __init__(
        self,
        skill_type: SkillType,
        task: str,
        target: str,
        contract: SkillContract,
        env_id: str,
        max_episode_steps: int,
        description: str = "",
    ) -> None:
        if not target:
            raise ValueError("atomic skill target must be non-empty")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        super().__init__(
            name="{}.{}".format(skill_type.value, target),
            task=task,
            contract=contract,
            description=description,
        )
        self.skill_type = skill_type
        self.target = target
        self.env_id = env_id
        self.max_episode_steps = max_episode_steps
        self._backends: Dict[str, ExecutionBackend] = {}

    @property
    def backends(self) -> Mapping[str, ExecutionBackend]:
        return dict(self._backends)

    def add_backend(self, backend: ExecutionBackend) -> None:
        if backend.key in self._backends:
            raise ValueError(
                "skill {} already has backend {!r}".format(self.id, backend.key)
            )
        self._backends[backend.key] = backend

    def backend(self, key: Optional[str] = None) -> ExecutionBackend:
        if key is not None:
            try:
                return self._backends[key]
            except KeyError as exc:
                raise KeyError(
                    "skill {} has no backend {!r}; available={}".format(
                        self.id, key, sorted(self._backends)
                    )
                ) from exc
        ready = [
            backend
            for backend in self._backends.values()
            if backend.status == ArtifactStatus.READY
        ]
        if not ready:
            raise RuntimeError("skill {} has no ready backend".format(self.id))
        return ready[0]

    @property
    def ready(self) -> bool:
        return any(
            backend.status == ArtifactStatus.READY
            for backend in self._backends.values()
        )

    def bind(
        self, arguments: Mapping[str, Any], backend_key: Optional[str] = None
    ) -> SkillInvocation:
        values = dict(arguments)
        target_parameter = _TARGET_PARAMETER[self.skill_type]
        if self.target != "all":
            supplied = values.get(target_parameter)
            if supplied is not None and supplied != self.target:
                raise ValueError(
                    "skill {} is specialized for {!r}, not {!r}".format(
                        self.id, self.target, supplied
                    )
                )
            values[target_parameter] = self.target
        if backend_key is not None:
            self.backend(backend_key)
        return super().bind(values, backend_key=backend_key)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": "atomic",
            "task": self.task,
            "skill_type": self.skill_type.value,
            "target": self.target,
            "description": self.description,
            "contract": _contract_dict(self.contract),
            "execution": {
                "env_id": self.env_id,
                "max_episode_steps": self.max_episode_steps,
                "backends": {
                    key: backend.as_dict()
                    for key, backend in sorted(self._backends.items())
                },
            },
            "ready": self.ready,
        }


_TARGET_PARAMETER = {
    SkillType.NAVIGATE: "goal",
    SkillType.PICK: "object",
    SkillType.PLACE: "object",
    SkillType.OPEN: "articulation",
    SkillType.CLOSE: "articulation",
}


class NavigateSkill(AtomicSkill):
    def __init__(self, task: str, target: str = "all") -> None:
        super().__init__(
            SkillType.NAVIGATE,
            task,
            target,
            SkillContract(
                parameters=(SkillParameter("goal", ParameterType.ENTITY),),
                preconditions=("present({goal})",),
                effects=("reachable({goal})",),
                verification=("reachable({goal})",),
                failure_modes=("navigation_timeout", "collision_limit"),
            ),
            env_id="NavigateSubtaskTrain-v0",
            max_episode_steps=1000,
            description="Move to a pose from which the grounded goal is reachable.",
        )


class PickSkill(AtomicSkill):
    def __init__(self, task: str, target: str) -> None:
        super().__init__(
            SkillType.PICK,
            task,
            target,
            SkillContract(
                parameters=(SkillParameter("object", ParameterType.ENTITY),),
                preconditions=("reachable({object})", "gripper_empty()"),
                effects=("holding({object})",),
                invariants=("collision_safe()",),
                verification=("holding({object})",),
                failure_modes=("grasp_failed", "object_dropped", "force_limit"),
            ),
            env_id="PickSubtaskTrain-v0",
            max_episode_steps=200,
            description="Grasp and hold one grounded object.",
        )


class PlaceSkill(AtomicSkill):
    def __init__(self, task: str, target: str) -> None:
        super().__init__(
            SkillType.PLACE,
            task,
            target,
            SkillContract(
                parameters=(
                    SkillParameter("object", ParameterType.ENTITY),
                    SkillParameter("destination", ParameterType.LOCATION),
                ),
                preconditions=("holding({object})", "reachable({destination})"),
                effects=("at({object},{destination})", "gripper_empty()"),
                invariants=("collision_safe()",),
                verification=("at({object},{destination})",),
                failure_modes=("placement_failed", "object_dropped", "force_limit"),
            ),
            env_id="PlaceSubtaskTrain-v0",
            max_episode_steps=200,
            description="Place a held object at a grounded destination.",
        )


class OpenSkill(AtomicSkill):
    def __init__(self, task: str, target: str) -> None:
        super().__init__(
            SkillType.OPEN,
            task,
            target,
            SkillContract(
                parameters=(
                    SkillParameter("articulation", ParameterType.ARTICULATION),
                ),
                preconditions=("reachable({articulation})", "closed({articulation})"),
                effects=("open({articulation})",),
                invariants=("collision_safe()",),
                verification=("open({articulation})",),
                failure_modes=("handle_not_grasped", "joint_blocked", "force_limit"),
            ),
            env_id="OpenSubtaskTrain-v0",
            max_episode_steps=200,
            description="Open one grounded articulation such as a fridge or drawer.",
        )


class CloseSkill(AtomicSkill):
    def __init__(self, task: str, target: str) -> None:
        super().__init__(
            SkillType.CLOSE,
            task,
            target,
            SkillContract(
                parameters=(
                    SkillParameter("articulation", ParameterType.ARTICULATION),
                ),
                preconditions=("reachable({articulation})", "open({articulation})"),
                effects=("closed({articulation})",),
                invariants=("collision_safe()",),
                verification=("closed({articulation})",),
                failure_modes=("handle_not_grasped", "joint_blocked", "force_limit"),
            ),
            env_id="CloseSubtaskTrain-v0",
            max_episode_steps=200,
            description="Close one grounded articulation such as a fridge or drawer.",
        )


def _contract_dict(contract: SkillContract) -> Dict[str, Any]:
    return {
        "parameters": [
            {
                "name": parameter.name,
                "type": parameter.type.value,
                "description": parameter.description,
                "required": parameter.required,
            }
            for parameter in contract.parameters
        ],
        "preconditions": list(contract.preconditions),
        "effects": list(contract.effects),
        "invariants": list(contract.invariants),
        "verification": list(contract.verification),
        "failure_modes": list(contract.failure_modes),
    }


ATOMIC_SKILL_CLASSES = {
    SkillType.NAVIGATE: NavigateSkill,
    SkillType.PICK: PickSkill,
    SkillType.PLACE: PlaceSkill,
    SkillType.OPEN: OpenSkill,
    SkillType.CLOSE: CloseSkill,
}
