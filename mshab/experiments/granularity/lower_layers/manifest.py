"""The canonical inventory of the 53 downloaded MS-HAB RL checkpoint policies.

This manifest is the one thing the granularity experiment holds that
``mshab.skills`` does not: which ``<family>/<task>/<type>/<target>`` leaves
exist in the official download across the three long-horizon tasks.  A row
carries identity and checkpoint metadata only.  How a policy is bound to a
contract, and which bound policy executes a grounding, is decided by the
ordinary :class:`~mshab.skills.library.ContractLibrary` built in
:mod:`mshab.experiments.granularity.lower_layers.library`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from mshab.skills.model import CheckpointPolicy, ContractType


TASK_ABBREVIATIONS = {
    "prepare_groceries": "PG",
    "set_table": "ST",
    "tidy_house": "TH",
}

HOUSEHOLD_OBJECTS = (
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "024_bowl",
)

TASK_OBJECT_CATEGORIES = {
    "prepare_groceries": HOUSEHOLD_OBJECTS,
    "set_table": ("013_apple", "024_bowl"),
    "tidy_house": HOUSEHOLD_OBJECTS,
}


@dataclass(frozen=True)
class PolicySpec:
    """One row of the manifest: a checkpoint that exists in the download."""

    task_family: str
    contract_type: ContractType
    target: str
    family: str = "rl"

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract_type", ContractType(self.contract_type))
        if self.task_family not in TASK_ABBREVIATIONS:
            raise ValueError("unsupported task family {!r}".format(self.task_family))
        if not self.target:
            raise ValueError("policy target must be non-empty")
        if self.contract_type == ContractType.NAVIGATE:
            if self.target != "all":
                raise ValueError("navigation only has an all-target checkpoint")
        elif self.contract_type in (ContractType.PICK, ContractType.PLACE):
            allowed = TASK_OBJECT_CATEGORIES[self.task_family] + ("all",)
            if self.target not in allowed:
                raise ValueError(
                    "unsupported {} target {!r} for {}".format(
                        self.contract_type.value,
                        self.target,
                        self.task_family,
                    )
                )
        elif self.contract_type in (ContractType.OPEN, ContractType.CLOSE):
            if self.task_family != "set_table":
                raise ValueError("open/close checkpoints only exist for SetTable")
            if self.target not in ("fridge", "kitchen_counter"):
                raise ValueError(
                    "unsupported articulation target {!r}".format(self.target)
                )

    @property
    def id(self) -> str:
        return ".".join(
            (self.family, self.task_family, self.contract_type.value, self.target)
        )

    @property
    def short_id(self) -> str:
        return "{}.{}.{}".format(
            TASK_ABBREVIATIONS[self.task_family],
            self.contract_type.value,
            self.target,
        )

    @property
    def relative_leaf(self) -> str:
        return "/".join(
            (self.family, self.task_family, self.contract_type.value, self.target)
        )

    def policy(self, checkpoint_root: Path) -> CheckpointPolicy:
        """The stored policy for this row under ``checkpoint_root``.

        Readiness is computed from the filesystem at run time; the manifest
        itself never records it.
        """

        return CheckpointPolicy.from_leaf(
            Path(checkpoint_root),
            self.family,
            self.task_family,
            self.contract_type,
            self.target,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "short_id": self.short_id,
            "family": self.family,
            "task_family": self.task_family,
            "contract_type": self.contract_type.value,
            "target": self.target,
            "checkpoint_leaf": self.relative_leaf,
        }


def _policy_specs() -> Tuple[PolicySpec, ...]:
    specs = []
    for task_family in ("prepare_groceries", "set_table", "tidy_house"):
        specs.append(PolicySpec(task_family, ContractType.NAVIGATE, "all"))
        targets = TASK_OBJECT_CATEGORIES[task_family] + ("all",)
        for contract_type in (ContractType.PICK, ContractType.PLACE):
            specs.extend(
                PolicySpec(task_family, contract_type, target)
                for target in targets
            )
        if task_family == "set_table":
            for contract_type in (ContractType.OPEN, ContractType.CLOSE):
                specs.extend(
                    PolicySpec(task_family, contract_type, target)
                    for target in ("fridge", "kitchen_counter")
                )
    result = tuple(sorted(specs, key=lambda item: item.id))
    if len(result) != 53 or len({item.id for item in result}) != 53:
        raise AssertionError("the canonical RL policy manifest must contain 53 rows")
    return result


POLICY_SPECS = _policy_specs()
_SPECS_BY_ID = {spec.id: spec for spec in POLICY_SPECS}


def spec_for(policy_id: str) -> PolicySpec:
    """The manifest row behind one policy id."""

    try:
        return _SPECS_BY_ID[policy_id]
    except KeyError as exc:
        raise KeyError(
            "policy {!r} is not in the granularity manifest".format(policy_id)
        ) from exc
