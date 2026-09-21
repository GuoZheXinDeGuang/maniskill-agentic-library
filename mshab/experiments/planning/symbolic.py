"""A simulator-free environment and executor for the controller loop.

Facts in, facts out.  The environment holds a set of predicate strings.  A
policy executes a grounded contract by adding its effects and retracting its
deletes; a scripted failure replaces that with whatever the scenario says
happened instead.  Admission, invariant monitoring, and effect verification
are not re-implemented here: they run in the ordinary ``SkillRuntime``, so
the controller loop is exercised exactly as it will be on MS-HAB, with only
the adapter and the executor swapped.

Two world rules keep the fact set physically consistent where the contracts
are silent, because a contract can only retract predicates it names:

1. ``holding(x)`` retracts ``gripper_empty()`` and every ``at(x,...)``.
2. a step that adds ``reachable(y)`` retracts every other ``reachable(...)``:
   the robot is in one place at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from mshab.experiments.planning.documents import EntityDescription
from mshab.skills import schema
from mshab.skills.environment import (
    EnvironmentAdapter,
    EnvironmentDescription,
    EnvironmentEntity,
    EnvironmentSnapshot,
)
from mshab.skills.library import ContractLibrary
from mshab.skills.model import ArtifactStatus, GroundedSkill, Policy, PolicyKind
from mshab.skills.runtime import ContractMonitor, PolicyExecution, PolicyExecutor


class SymbolicPolicy(Policy):
    """The Layer-4 policy of the symbolic environment: applies contract effects."""

    ID = "symbolic.contract_effects"

    def __init__(self, policy_id: str = ID) -> None:
        super().__init__(policy_id, PolicyKind.SCRIPT, target="all")

    @property
    def status(self) -> ArtifactStatus:
        return ArtifactStatus.READY

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "target": self.target,
            "status": self.status.value,
        }


def bind_symbolic_policy(
    library: ContractLibrary, task: Optional[str] = None
) -> SymbolicPolicy:
    """Register one symbolic policy and bind it to every contract of ``task``.

    It is appended after the existing bindings.  Without downloaded
    checkpoints it is the only ready policy, so ``select_policy`` returns it;
    where checkpoints exist they stay preferred, and a controller can pin the
    symbolic policy by id instead.
    """

    policy = SymbolicPolicy()
    library.register_policy(policy)
    for contract in library.find(task=task):
        library.bind(contract.id, policy.id)
    return policy


@dataclass(frozen=True)
class ScriptedFailure:
    """What happens instead of the contract's effects on some attempts of a grounding.

    Keyed by contract type and grounded target rather than by node id, so a
    scenario also applies to graphs whose node ids a proposer chose.
    ``attempts`` lists the 1-based attempt numbers that fail; ``None`` means
    every attempt.  ``unless`` are facts that suspend the failure: an attempt
    made while every one of them holds succeeds normally.  That is how a
    scenario states a physical truth the proposer is not told, for example
    that an object inside a closed drawer cannot be picked until the drawer
    is open.  ``add``/``remove`` are the facts the failed attempt changes
    anyway, for example a dropped object.
    """

    contract_type: str
    target: str
    attempts: Optional[Tuple[int, ...]] = (1,)
    failure_mode: str = "scripted_failure"
    add: Tuple[str, ...] = ()
    remove: Tuple[str, ...] = ()
    unless: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.contract_type or not self.target or not self.failure_mode:
            raise ValueError("scripted failure needs a contract type, target, and mode")
        if self.attempts is not None:
            attempts = tuple(self.attempts)
            if not attempts or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in attempts
            ):
                raise ValueError("attempts must be positive integers or None")
            object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "add", tuple(self.add))
        object.__setattr__(self, "remove", tuple(self.remove))
        unless = tuple(self.unless)
        if any(not isinstance(fact, str) or not fact for fact in unless):
            raise ValueError("unless facts must be non-empty strings")
        object.__setattr__(self, "unless", unless)

    @property
    def key(self) -> Tuple[str, str]:
        return (self.contract_type, self.target)

    def applies(self, attempt: int, facts: Iterable[str] = ()) -> bool:
        """Whether this attempt fails: listed (or every attempt) and not suspended."""

        if self.attempts is not None and attempt not in self.attempts:
            return False
        if self.unless:
            held = set(facts)
            if all(fact in held for fact in self.unless):
                return False
        return True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "contract_type": self.contract_type,
            "target": self.target,
            "attempts": None if self.attempts is None else list(self.attempts),
            "failure_mode": self.failure_mode,
            "add": list(self.add),
            "remove": list(self.remove),
            "unless": list(self.unless),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScriptedFailure":
        where = "scripted_failure"
        payload = schema.require_mapping(payload, where=where)
        schema.require_keys(
            payload,
            where=where,
            required=("contract_type", "target"),
            optional=("attempts", "failure_mode", "add", "remove", "unless"),
        )
        attempts = payload.get("attempts", [1])
        if attempts is not None:
            if isinstance(attempts, (str, bytes)) or not hasattr(attempts, "__iter__"):
                raise schema.SchemaError("{}.attempts must be an array or null".format(where))
            attempts = tuple(attempts)
        return cls(
            contract_type=schema.require_str(payload, "contract_type", where=where),
            target=schema.require_str(payload, "target", where=where),
            attempts=attempts,
            failure_mode=schema.optional_str(
                payload, "failure_mode", where=where, default="scripted_failure"
            ),
            add=schema.require_str_tuple(payload, "add", where=where),
            remove=schema.require_str_tuple(payload, "remove", where=where),
            unless=schema.require_str_tuple(payload, "unless", where=where),
        )


def grounding_key(grounded: GroundedSkill) -> Tuple[str, str]:
    """``(contract type, grounded target)``: what scripted failures match on."""

    contract = grounded.contract
    return (
        contract.contract_type_name,
        str(grounded.arguments.get(contract.target_parameter)),
    )


class SymbolicEnvironmentAdapter(EnvironmentAdapter):
    """An environment whose whole state is a set of predicate facts.

    ``step`` takes an action of the form ``{"add": [...], "remove": [...]}``
    and applies the world rules afterwards.  Compatibility covers every
    contract environment id of ``task`` in the library.  ``entities`` are the
    scene's ``EntityDescription`` records; a proposer sees them again verbatim
    in its requests.
    """

    def __init__(
        self,
        library: ContractLibrary,
        task: str,
        initial_facts: Iterable[str],
        entities: Iterable[EntityDescription] = (),
        *,
        environment_id: str = "symbolic",
        scene_id: Optional[str] = None,
    ) -> None:
        self._initial_facts = frozenset(initial_facts)
        if any(not isinstance(fact, str) or not fact for fact in self._initial_facts):
            raise ValueError("facts must be non-empty strings")
        self._description = EnvironmentDescription(
            environment_id=environment_id,
            scene_id=scene_id,
            entities={
                item.name: EnvironmentEntity(item.name, item.kind, item.name)
                for item in entities
            },
            compatible_contract_env_ids=tuple(
                sorted({contract.env_id for contract in library.find(task=task)})
            ),
        )
        self._facts: Set[str] = set()
        self._step_index = 0
        self._snapshot: Optional[EnvironmentSnapshot] = None

    @property
    def description(self) -> EnvironmentDescription:
        return self._description

    @property
    def facts(self) -> frozenset:
        return frozenset(self._facts)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> EnvironmentSnapshot:
        self._facts = set(self._initial_facts)
        self._settle()
        self._step_index = 0
        self._snapshot = self._make_snapshot()
        return self._snapshot

    def snapshot(self) -> EnvironmentSnapshot:
        if self._snapshot is None:
            raise RuntimeError("symbolic environment must be reset before snapshot()")
        return self._snapshot

    def step(self, action: Any) -> EnvironmentSnapshot:
        if self._snapshot is None:
            raise RuntimeError("symbolic environment must be reset before step()")
        if not isinstance(action, Mapping) or set(action) - {"add", "remove"}:
            raise TypeError(
                "a symbolic action is a mapping with optional 'add' and 'remove' facts"
            )
        added = {str(fact) for fact in action.get("add", ())}
        self._facts -= {str(fact) for fact in action.get("remove", ())}
        self._facts |= added
        self._settle(added)
        self._step_index += 1
        self._snapshot = self._make_snapshot()
        return self._snapshot

    def _settle(self, added: Iterable[str] = ()) -> None:
        added = set(added)
        arrived = {fact for fact in added if fact.startswith("reachable(")}
        if arrived:
            self._facts -= {
                fact
                for fact in self._facts
                if fact.startswith("reachable(") and fact not in arrived
            }
        held = [
            fact[len("holding(") : -1]
            for fact in self._facts
            if fact.startswith("holding(") and fact.endswith(")")
        ]
        if held:
            self._facts.discard("gripper_empty()")
            for item in held:
                prefix = "at({},".format(item)
                self._facts -= {fact for fact in self._facts if fact.startswith(prefix)}

    def _make_snapshot(self) -> EnvironmentSnapshot:
        facts = frozenset(self._facts)
        return EnvironmentSnapshot(
            observation=tuple(sorted(facts)),
            info={"facts": sorted(facts)},
            facts=facts,
            step_index=self._step_index,
        )


class SymbolicPolicyExecutor(PolicyExecutor):
    """Execute a grounding by applying its effects, or a scripted failure instead.

    Several failures may address the same grounding; the first one that
    applies to the attempt, in the order given, is what happens.  A scenario
    lists a standing physical truth (``unless=...``) before a one-off mishap.
    """

    def __init__(self, failures: Iterable[ScriptedFailure] = ()) -> None:
        self._failures: Dict[Tuple[str, str], List[ScriptedFailure]] = {}
        for failure in failures:
            if not isinstance(failure, ScriptedFailure):
                raise TypeError("scripted failures must be ScriptedFailure instances")
            self._failures.setdefault(failure.key, []).append(failure)
        self.attempts: Dict[Tuple[str, str], int] = {}

    def execute(
        self,
        grounded: GroundedSkill,
        policy: Policy,
        environment: EnvironmentAdapter,
        monitor: ContractMonitor,
    ) -> PolicyExecution:
        key = grounding_key(grounded)
        attempt = self.attempts.get(key, 0) + 1
        self.attempts[key] = attempt
        facts = environment.snapshot().facts
        failure = next(
            (item for item in self._failures.get(key, ()) if item.applies(attempt, facts)),
            None,
        )
        if failure is not None:
            snapshot = environment.step({"add": failure.add, "remove": failure.remove})
            monitor(snapshot)
            return PolicyExecution(
                success=False,
                steps=1,
                failure_mode=failure.failure_mode,
                metadata={"attempt": attempt, "policy": policy.id, "scripted": True},
            )
        snapshot = environment.step(
            {"add": grounded.effects, "remove": grounded.deletes}
        )
        monitor(snapshot)
        return PolicyExecution(
            success=True, steps=1, metadata={"attempt": attempt, "policy": policy.id}
        )
