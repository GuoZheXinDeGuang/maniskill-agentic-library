"""Copyable extension point for adding a new environment-specific contract."""

from __future__ import annotations

from typing import Optional

from mshab.skills.model import (
    AtomicContract,
    ParameterType,
    ContractTerms,
    ContractParameter,
)


class YourContract(AtomicContract):
    """Minimal example of a user-defined atomic contract.

    Replace ``your_contract`` and the default predicates with domain terms, then
    provide a ``Policy`` and a ``PolicyExecutor`` implementation.
    No change to the Layer-1/2 graph classes is required.
    """

    def __init__(
        self,
        task: str,
        target: str = "all",
        *,
        env_id: str = "YourContractEnv-v0",
        max_episode_steps: int = 200,
        terms: Optional[ContractTerms] = None,
    ) -> None:
        super().__init__(
            contract_type="your_contract",
            task=task,
            target=target,
            target_parameter="target",
            terms=terms
            or ContractTerms(
                parameters=(
                    ContractParameter(
                        "target",
                        ParameterType.ENTITY,
                        "Symbolic target resolved by the environment adapter.",
                    ),
                ),
                preconditions=("ready_for_your_contract({target})",),
                effects=("your_contract_done({target})",),
                invariants=("collision_safe()",),
                verification=("your_contract_done({target})",),
                failure_modes=(
                    "target_not_found",
                    "execution_timeout",
                    "force_limit",
                ),
            ),
            env_id=env_id,
            max_episode_steps=max_episode_steps,
            description="Template for a user-defined environment-specific contract.",
        )
