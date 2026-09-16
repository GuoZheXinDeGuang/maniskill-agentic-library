"""Object model for Layer-3 contracts and the Layer-4 policies that execute them.

The classes in this module describe *what* a contract promises separately from
*how* it is executed.  ``pick(apple)`` is one contract even when an RL, BC,
DP, or remote VLA policy executes it.

This module deliberately has no torch or ManiSkill imports.  Listing and
planning with the contract library must not allocate a GPU or load checkpoints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)


class ContractType(str, Enum):
    """Built-in contract vocabulary currently implemented by MS-HAB."""

    NAVIGATE = "navigate"
    PICK = "pick"
    PLACE = "place"
    OPEN = "open"
    CLOSE = "close"


class ParameterType(str, Enum):
    """Planner-facing parameter types used by contract terms."""

    ENTITY = "entity"
    LOCATION = "location"
    ARTICULATION = "articulation"
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"


class ExecutorType(str, Enum):
    """Kinds of low-level policy represented in the architecture diagram."""

    CHECKPOINT = "checkpoint"
    VLA = "vla"
    NAVIGATION = "navigation"
    CONTROLLER = "controller"
    SCRIPT = "script"
    STATE_MACHINE = "state_machine"


class ArtifactStatus(str, Enum):
    """Local availability of a policy's artifacts."""

    MISSING = "missing"
    PARTIAL = "partial"
    READY = "ready"


@dataclass(frozen=True)
class ContractParameter:
    name: str
    type: ParameterType
    description: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", ParameterType(self.type))
        if not self.name or not self.name.isidentifier():
            raise ValueError("contract parameter name must be a non-empty identifier")


@dataclass(frozen=True)
class BoundTerms:
    """Contract terms whose predicate templates have concrete arguments."""

    preconditions: Tuple[str, ...]
    effects: Tuple[str, ...]
    invariants: Tuple[str, ...]
    verification: Tuple[str, ...]
    failure_modes: Tuple[str, ...]
    deletes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "preconditions",
            "effects",
            "invariants",
            "verification",
            "failure_modes",
            "deletes",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        overlap = sorted(set(self.deletes) & set(self.effects))
        if overlap:
            raise ValueError(
                "grounded predicates cannot be both asserted and retracted: {}".format(
                    overlap
                )
            )
        invariant_overlap = sorted(set(self.deletes) & set(self.invariants))
        if invariant_overlap:
            raise ValueError(
                "a grounded contract cannot retract its own invariants: {}".format(
                    invariant_overlap
                )
            )

    def can_start(self, facts: Iterable[str]) -> bool:
        return set(self.preconditions).issubset(set(facts))

    def achieved(self, facts: Iterable[str]) -> bool:
        return set(self.effects).issubset(set(facts))

    def verified(self, facts: Iterable[str]) -> bool:
        """Whether the explicit success predicates hold after execution."""

        return set(self.verification).issubset(set(facts))

    def retracted(self, facts: Iterable[str]) -> bool:
        """Whether every predicate this contract retracts is actually gone."""

        return not (set(self.deletes) & set(facts))

    def apply_to(self, facts: Iterable[str]) -> FrozenSet[str]:
        """The symbolic state after a successful execution.

        Contracts reject an overlap between deletes and effects, so the order
        here is only an implementation detail.  Without delete effects, a
        state that records ``open(fridge)`` would keep ``closed(fridge)``
        alongside it, and ``holding(bowl)`` would survive placing the bowl.
        """

        return frozenset(set(facts) - set(self.deletes) | set(self.effects))


@dataclass(frozen=True)
class ContractTerms:
    """Pre/post-condition terms of one contract.

    Terms describe only whether the contract can physically start and what it
    changes in the current world state.  They do not encode task order or
    semantic plausibility; that belongs to the relations between Layer-2
    skill nodes.

    Predicates use ``str.format`` placeholders matching parameter names, for
    example ``holding({object})``.  They stay lightweight strings here; a
    simulator adapter is responsible for evaluating them against live state.
    """

    parameters: Tuple[ContractParameter, ...]
    preconditions: Tuple[str, ...]
    effects: Tuple[str, ...]
    invariants: Tuple[str, ...] = ()
    verification: Tuple[str, ...] = ()
    failure_modes: Tuple[str, ...] = ()
    deletes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "parameters",
            "preconditions",
            "effects",
            "invariants",
            "verification",
            "failure_modes",
            "deletes",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("contract terms have duplicate parameter names")
        if not self.effects:
            raise ValueError("contract terms must declare at least one effect")
        if not self.verification:
            object.__setattr__(self, "verification", tuple(self.effects))
        for group_name, predicates in (
            ("preconditions", self.preconditions),
            ("effects", self.effects),
            ("invariants", self.invariants),
            ("verification", self.verification),
            ("deletes", self.deletes),
        ):
            if any(not predicate.strip() for predicate in predicates):
                raise ValueError("{} cannot contain empty predicates".format(group_name))
        overlap = sorted(set(self.deletes) & set(self.effects))
        if overlap:
            raise ValueError(
                "predicates cannot be both asserted and retracted: {}".format(overlap)
            )
        invariant_overlap = sorted(set(self.deletes) & set(self.invariants))
        if invariant_overlap:
            raise ValueError(
                "a contract cannot retract its own invariants: {}".format(invariant_overlap)
            )

    def bind(self, arguments: Mapping[str, Any]) -> BoundTerms:
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
                "invalid contract arguments: missing={}, unknown={}".format(missing, unknown)
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

        return BoundTerms(
            preconditions=ground(self.preconditions),
            effects=ground(self.effects),
            invariants=ground(self.invariants),
            verification=ground(self.verification),
            failure_modes=self.failure_modes,
            deletes=ground(self.deletes),
        )


def _validate_parameter(parameter: ContractParameter, value: Any) -> None:
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


class Policy(ABC):
    """One low-level policy (checkpoint, VLA, controller, or script) that executes a contract."""

    def __init__(self, key: str, executor_type: ExecutorType) -> None:
        if not key:
            raise ValueError("policy key cannot be empty")
        self.key = key
        self.executor_type = executor_type

    @property
    @abstractmethod
    def status(self) -> ArtifactStatus:
        """Whether everything needed to execute this policy is available."""

    @abstractmethod
    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable library record."""


class CheckpointPolicy(Policy):
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
        super().__init__(key=key, executor_type=ExecutorType.CHECKPOINT)
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
    """One skill node grounded to a contract with concrete arguments."""

    contract: "Contract" = field(repr=False)
    arguments: Mapping[str, Any]
    terms: BoundTerms
    policy_key: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))

    @property
    def id(self) -> str:
        args = ",".join(
            "{}={}".format(key, self.arguments[key]) for key in sorted(self.arguments)
        )
        return "{}({})".format(self.contract.id, args)


class Contract(ABC):
    """Interface shared by every library-managed contract."""

    def __init__(
        self,
        name: str,
        task: str,
        terms: ContractTerms,
        description: str = "",
    ) -> None:
        if not name or not task:
            raise ValueError("contract name and task must be non-empty")
        self.name = name
        self.task = task
        self.terms = terms
        self.description = description

    @property
    def id(self) -> str:
        return "mshab.{}.{}".format(self.task, self.name)

    @property
    @abstractmethod
    def ready(self) -> bool:
        """Whether at least one complete execution path is available."""

    def bind(
        self, arguments: Mapping[str, Any], policy_key: Optional[str] = None
    ) -> SkillInvocation:
        return SkillInvocation(
            contract=self,
            arguments=dict(arguments),
            terms=self.terms.bind(arguments),
            policy_key=policy_key,
        )

    @abstractmethod
    def as_dict(self) -> Dict[str, Any]:
        pass


class AtomicContract(Contract):
    """A dispatchable contract together with the policies that can execute it."""

    def __init__(
        self,
        contract_type: Union[ContractType, str],
        task: str,
        target: str,
        terms: ContractTerms,
        env_id: str,
        max_episode_steps: int,
        description: str = "",
        target_parameter: Optional[str] = None,
    ) -> None:
        if not target:
            raise ValueError("atomic contract target must be non-empty")
        contract_type = _normalise_contract_type(contract_type)
        contract_type_name = (
            contract_type.value if isinstance(contract_type, ContractType) else contract_type
        )
        if target_parameter is None:
            target_parameter = _TARGET_PARAMETER.get(contract_type)
        if not target_parameter or not target_parameter.isidentifier():
            raise ValueError(
                "atomic contract target_parameter must be a non-empty identifier"
            )
        parameter_names = {parameter.name for parameter in terms.parameters}
        if target_parameter not in parameter_names:
            raise ValueError(
                "target parameter {!r} is not declared by the contract terms".format(
                    target_parameter
                )
            )
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        super().__init__(
            name="{}.{}".format(contract_type_name, target),
            task=task,
            terms=terms,
            description=description,
        )
        self.contract_type = contract_type
        self.target_parameter = target_parameter
        self.target = target
        self.env_id = env_id
        self.max_episode_steps = max_episode_steps
        self._policies: Dict[str, Policy] = {}

    @property
    def policies(self) -> Mapping[str, Policy]:
        return dict(self._policies)

    def add_policy(self, policy: Policy) -> None:
        if policy.key in self._policies:
            raise ValueError(
                "contract {} already has policy {!r}".format(self.id, policy.key)
            )
        self._policies[policy.key] = policy

    def policy(self, key: Optional[str] = None) -> Policy:
        if key is not None:
            try:
                return self._policies[key]
            except KeyError as exc:
                raise KeyError(
                    "contract {} has no policy {!r}; available={}".format(
                        self.id, key, sorted(self._policies)
                    )
                ) from exc
        ready = sorted(
            (
                policy
                for policy in self._policies.values()
                if policy.status == ArtifactStatus.READY
            ),
            key=lambda item: item.key,
        )
        if not ready:
            raise RuntimeError("contract {} has no ready policy".format(self.id))
        return ready[0]

    @property
    def ready(self) -> bool:
        return any(
            policy.status == ArtifactStatus.READY
            for policy in self._policies.values()
        )

    def bind(
        self, arguments: Mapping[str, Any], policy_key: Optional[str] = None
    ) -> SkillInvocation:
        values = dict(arguments)
        if self.target != "all":
            supplied = values.get(self.target_parameter)
            if supplied is not None and supplied != self.target:
                raise ValueError(
                    "contract {} is specialized for {!r}, not {!r}".format(
                        self.id, self.target, supplied
                    )
                )
            values[self.target_parameter] = self.target
        if policy_key is not None:
            self.policy(policy_key)
        return super().bind(values, policy_key=policy_key)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": "atomic",
            "task": self.task,
            "contract_type": self.contract_type_name,
            "target": self.target,
            "target_parameter": self.target_parameter,
            "description": self.description,
            "terms": _terms_dict(self.terms),
            "execution": {
                "env_id": self.env_id,
                "max_episode_steps": self.max_episode_steps,
                "policies": {
                    key: policy.as_dict()
                    for key, policy in sorted(self._policies.items())
                },
            },
            "ready": self.ready,
        }

    @property
    def contract_type_name(self) -> str:
        return (
            self.contract_type.value
            if isinstance(self.contract_type, ContractType)
            else self.contract_type
        )


_TARGET_PARAMETER = {
    ContractType.NAVIGATE: "goal",
    ContractType.PICK: "object",
    ContractType.PLACE: "object",
    ContractType.OPEN: "articulation",
    ContractType.CLOSE: "articulation",
}


def _normalise_contract_type(contract_type: Union[ContractType, str]) -> Union[ContractType, str]:
    if isinstance(contract_type, ContractType):
        return contract_type
    if not isinstance(contract_type, str) or not contract_type.strip():
        raise ValueError("contract_type must be a ContractType or non-empty string")
    value = contract_type.strip().lower()
    try:
        return ContractType(value)
    except ValueError:
        if not value.replace("_", "").isalnum():
            raise ValueError(
                "custom contract_type must contain only letters, numbers, or underscores"
            )
        return value


class NavigateContract(AtomicContract):
    def __init__(self, task: str, target: str = "all") -> None:
        super().__init__(
            ContractType.NAVIGATE,
            task,
            target,
            ContractTerms(
                parameters=(ContractParameter("goal", ParameterType.ENTITY),),
                preconditions=("present({goal})",),
                effects=("reachable({goal})",),
                verification=("reachable({goal})",),
                failure_modes=("navigation_timeout", "collision_limit"),
            ),
            env_id="NavigateSubtaskTrain-v0",
            max_episode_steps=1000,
            description="Move to a pose from which the grounded goal is reachable.",
        )


class PickContract(AtomicContract):
    def __init__(self, task: str, target: str) -> None:
        super().__init__(
            ContractType.PICK,
            task,
            target,
            ContractTerms(
                parameters=(ContractParameter("object", ParameterType.ENTITY),),
                preconditions=("reachable({object})", "gripper_empty()"),
                effects=("holding({object})",),
                invariants=("collision_safe()",),
                verification=("holding({object})",),
                failure_modes=("grasp_failed", "object_dropped", "force_limit"),
                deletes=("gripper_empty()",),
            ),
            env_id="PickSubtaskTrain-v0",
            max_episode_steps=200,
            description="Grasp and hold one grounded object.",
        )


class PlaceContract(AtomicContract):
    def __init__(self, task: str, target: str) -> None:
        super().__init__(
            ContractType.PLACE,
            task,
            target,
            ContractTerms(
                parameters=(
                    ContractParameter("object", ParameterType.ENTITY),
                    ContractParameter("destination", ParameterType.LOCATION),
                ),
                preconditions=("holding({object})", "reachable({destination})"),
                effects=("at({object},{destination})", "gripper_empty()"),
                invariants=("collision_safe()",),
                verification=("at({object},{destination})",),
                failure_modes=("placement_failed", "object_dropped", "force_limit"),
                deletes=("holding({object})",),
            ),
            env_id="PlaceSubtaskTrain-v0",
            max_episode_steps=200,
            description="Place a held object at a grounded destination.",
        )


class OpenContract(AtomicContract):
    def __init__(self, task: str, target: str) -> None:
        super().__init__(
            ContractType.OPEN,
            task,
            target,
            ContractTerms(
                parameters=(
                    ContractParameter("articulation", ParameterType.ARTICULATION),
                ),
                preconditions=("reachable({articulation})", "closed({articulation})"),
                effects=("open({articulation})",),
                invariants=("collision_safe()",),
                verification=("open({articulation})",),
                failure_modes=("handle_not_grasped", "joint_blocked", "force_limit"),
                deletes=("closed({articulation})",),
            ),
            env_id="OpenSubtaskTrain-v0",
            max_episode_steps=200,
            description="Open one grounded articulation such as a fridge or drawer.",
        )


class CloseContract(AtomicContract):
    def __init__(self, task: str, target: str) -> None:
        super().__init__(
            ContractType.CLOSE,
            task,
            target,
            ContractTerms(
                parameters=(
                    ContractParameter("articulation", ParameterType.ARTICULATION),
                ),
                preconditions=("reachable({articulation})", "open({articulation})"),
                effects=("closed({articulation})",),
                invariants=("collision_safe()",),
                verification=("closed({articulation})",),
                failure_modes=("handle_not_grasped", "joint_blocked", "force_limit"),
                deletes=("open({articulation})",),
            ),
            env_id="CloseSubtaskTrain-v0",
            max_episode_steps=200,
            description="Close one grounded articulation such as a fridge or drawer.",
        )


def _terms_dict(contract: ContractTerms) -> Dict[str, Any]:
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
        "deletes": list(contract.deletes),
    }


ATOMIC_CONTRACT_CLASSES = {
    ContractType.NAVIGATE: NavigateContract,
    ContractType.PICK: PickContract,
    ContractType.PLACE: PlaceContract,
    ContractType.OPEN: OpenContract,
    ContractType.CLOSE: CloseContract,
}
