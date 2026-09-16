"""Registry and local checkpoint discovery for :mod:`mshab.skills`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from mshab.skills.model import (
    ATOMIC_CONTRACT_CLASSES,
    AtomicContract,
    CheckpointPolicy,
    Contract,
    ContractType,
)


class ContractLibrary:
    """In-memory owner of contracts and the policies registered to execute them."""

    def __init__(self, contracts: Iterable[Contract] = ()) -> None:
        self._contracts: Dict[str, Contract] = {}
        for contract in contracts:
            self.register(contract)

    def register(self, contract: Contract) -> None:
        if contract.id in self._contracts:
            raise ValueError("duplicate contract id {!r}".format(contract.id))
        self._contracts[contract.id] = contract

    def get(self, contract_id: str) -> Contract:
        try:
            return self._contracts[contract_id]
        except KeyError as exc:
            raise KeyError(
                "unknown contract {!r}; available={}".format(
                    contract_id, sorted(self._contracts)
                )
            ) from exc

    def find(
        self,
        task: Optional[str] = None,
        contract_type: Optional[Union[ContractType, str]] = None,
        target: Optional[str] = None,
        ready: Optional[bool] = None,
    ) -> List[Contract]:
        result = []
        for contract in self._contracts.values():
            if task is not None and contract.task != task:
                continue
            if ready is not None and contract.ready != ready:
                continue
            if contract_type is not None:
                requested_type = (
                    contract_type.value
                    if isinstance(contract_type, ContractType)
                    else str(contract_type)
                )
                if (
                    not isinstance(contract, AtomicContract)
                    or contract.contract_type_name != requested_type
                ):
                    continue
            if target is not None and (
                not isinstance(contract, AtomicContract) or contract.target != target
            ):
                continue
            result.append(contract)
        return sorted(result, key=lambda item: item.id)

    def to_dict(self) -> Dict[str, object]:
        contracts = [contract.as_dict() for contract in self.find()]
        return {
            "schema_version": "mshab.contract-library.v1",
            "count": len(contracts),
            "contracts": contracts,
        }

    def save_index(self, path: Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_checkpoint_root(cls, root: Path) -> "ContractLibrary":
        """Discover ``family/task/type/target/{config.yml,policy.pt}`` leaves.

        A leaf is registered even when just one artifact is present, allowing
        callers to report ``partial`` downloads instead of silently hiding them.
        Multiple policy families are folded into one AtomicContract as
        interchangeable policies.
        """

        checkpoint_root = Path(root)
        if not checkpoint_root.is_dir():
            raise FileNotFoundError(
                "checkpoint root does not exist: {}".format(checkpoint_root)
            )

        library = cls()
        leaves = sorted(
            {
                path.parent
                for artifact_name in ("policy.pt", "config.yml")
                for path in checkpoint_root.glob("*/*/*/*/{}".format(artifact_name))
            }
        )
        for leaf in leaves:
            family, task, raw_type, target = leaf.relative_to(checkpoint_root).parts
            try:
                contract_type = ContractType(raw_type)
            except ValueError:
                continue
            contract_id = "mshab.{}.{}.{}".format(task, contract_type.value, target)
            existing = library._contracts.get(contract_id)
            if existing is None:
                contract_class = ATOMIC_CONTRACT_CLASSES[contract_type]
                contract = contract_class(task=task, target=target)
                library.register(contract)
            elif isinstance(existing, AtomicContract):
                contract = existing
            else:
                raise TypeError("{} is not atomic".format(contract_id))

            policy_type = _policy_type(family, target)
            contract.add_policy(
                CheckpointPolicy(
                    key=family,
                    family=family,
                    checkpoint_path=leaf / "policy.pt",
                    config_path=leaf / "config.yml",
                    policy_type=policy_type,
                )
            )
        return library


def _policy_type(family: str, target: str) -> str:
    if family == "rl":
        return "rl_all_obj" if target == "all" else "rl_per_obj"
    return family
