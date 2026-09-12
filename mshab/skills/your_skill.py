"""Copyable extension point for adding a new environment-specific skill."""

from __future__ import annotations

from typing import Optional

from mshab.skills.model import (
    AtomicSkill,
    ParameterType,
    SkillContract,
    SkillParameter,
)


class YourSkill(AtomicSkill):
    """Minimal example of a user-defined atomic skill.

    Replace ``your_skill`` and the default predicates with domain terms, then
    provide an ``ExecutionBackend`` and ``BackendExecutor`` implementation.
    No change to the Layer-1/2 graph classes is required.
    """

    def __init__(
        self,
        task: str,
        target: str = "all",
        *,
        env_id: str = "YourSkillEnv-v0",
        max_episode_steps: int = 200,
        contract: Optional[SkillContract] = None,
    ) -> None:
        super().__init__(
            skill_type="your_skill",
            task=task,
            target=target,
            target_parameter="target",
            contract=contract
            or SkillContract(
                parameters=(
                    SkillParameter(
                        "target",
                        ParameterType.ENTITY,
                        "Symbolic target resolved by the environment adapter.",
                    ),
                ),
                preconditions=("ready_for_your_skill({target})",),
                effects=("your_skill_done({target})",),
                invariants=("collision_safe()",),
                verification=("your_skill_done({target})",),
                failure_modes=(
                    "target_not_found",
                    "execution_timeout",
                    "force_limit",
                ),
            ),
            env_id=env_id,
            max_episode_steps=max_episode_steps,
            description="Template for a user-defined environment-specific skill.",
        )
