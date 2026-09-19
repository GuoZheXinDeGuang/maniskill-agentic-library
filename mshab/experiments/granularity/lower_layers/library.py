"""The granularity lower layers as one ``ContractLibrary``.

Layer 3 is five generic contracts, one per :class:`ContractType`, all with
target ``all``.  Layer 4 is the 53 checkpoint policies of the manifest.  The
relation between them is the library's ordinary many-to-many binding: every
policy is bound to the one contract of its type, so ``pick.all`` is bound to
all 23 pick checkpoints, in sorted policy-id order.

Binding order alone would let the apple checkpoint execute a bowl pick.
Which bound policy runs a particular grounding is therefore decided at run
time by ``ContractLibrary.select_policy(contract_id, arguments=...)``, which
prefers a checkpoint trained for exactly the grounded object over an ``all``
checkpoint and never returns one trained for a different object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

from mshab.experiments.granularity.lower_layers.manifest import (
    POLICY_SPECS,
    spec_for,
)
from mshab.skills.library import ContractLibrary
from mshab.skills.model import CONTRACT_CLASSES, ContractType


EXPERIMENT_TASK = "granularity"
SCHEMA_VERSION = "mshab.granularity-library.v1"


def contract_id(contract_type: Union[ContractType, str]) -> str:
    """The id of the experiment's generic contract for one type."""

    return "mshab.{}.{}.all".format(
        EXPERIMENT_TASK, ContractType(contract_type).value
    )


def build_granularity_library(checkpoint_root: Path) -> ContractLibrary:
    """Five generic contracts, 53 checkpoint policies, one binding per policy.

    Policy readiness is read from ``checkpoint_root`` at run time and never
    changes what is registered or bound, so the library is identical on a
    clean clone and on a workstation with the full download.
    """

    library = ContractLibrary()
    for contract_type, contract_class in CONTRACT_CLASSES.items():
        library.register(contract_class(task=EXPERIMENT_TASK, target="all"))
    # POLICY_SPECS is sorted by id, so each contract's binding order is too.
    for spec in POLICY_SPECS:
        policy = spec.policy(checkpoint_root)
        library.register_policy(policy)
        library.bind(contract_id(spec.contract_type), policy.id)
    return library


def library_document(library: ContractLibrary) -> Dict[str, Any]:
    """A portable record of both layers and their bindings.

    Unlike ``ContractLibrary.to_dict()`` it carries no absolute checkpoint
    paths and no ready/missing state, so it is byte-identical on every
    machine.  Bindings are listed from both sides: each contract names its
    policies in preference order, each policy names its contracts.
    """

    contracts = []
    # ContractType order (navigate, pick, place, open, close) is the reading
    # order of the diagram; ``library.find`` would sort by id instead.
    for contract_type in ContractType:
        contract = library.get(contract_id(contract_type))
        record = contract.as_dict()
        record["policies"] = [
            policy.id for policy in library.policies_for(contract.id)
        ]
        contracts.append(record)
    policies = []
    for policy in library.policies:
        record = spec_for(policy.id).as_dict()
        record["contracts"] = [
            contract.id for contract in library.contracts_for(policy.id)
        ]
        policies.append(record)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "graph_granularity",
        "scope": "layers_3_and_4_only",
        "summary": {
            "contracts": len(contracts),
            "policies": len(policies),
            "bindings": sum(len(record["policies"]) for record in contracts),
            "policies_by_contract": {
                record["contract_type"]: len(record["policies"])
                for record in contracts
            },
        },
        "contracts": contracts,
        "policies": policies,
    }
