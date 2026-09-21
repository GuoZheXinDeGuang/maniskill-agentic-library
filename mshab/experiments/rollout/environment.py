"""MS-HAB behind the environment boundary: one SetTable episode on ``SkillRollout-v0``.

This module imports torch and ManiSkill; the controller never does.  It wraps
the official ``make_env`` chain (depth observations, frame stacking, video,
the Fetch action wrapper, the vector wrapper) around ``SkillRollout-v0`` and
puts ``MSHabEnvironmentAdapter`` in front of it with the episode's entities
and a fact extractor that reads the whole scene every step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from mshab.envs.make import EnvConfig, make_env
from mshab.envs.skill_rollout import SkillRolloutEnv
from mshab.experiments.rollout.episode import SceneMeasurements, SetTableEpisode
from mshab.skills.environment import EnvironmentSnapshot, MSHabEnvironmentAdapter
from mshab.skills.library import ContractLibrary


ROLLOUT_ENV_ID = "SkillRollout-v0"


def make_rollout_env(
    task_plan_path: Path,
    *,
    max_episode_steps: int,
    video_dir: Optional[Path] = None,
    info_on_video: bool = True,
    num_envs: int = 1,
    ignore_arm_checkers: bool = True,
    sim_backend: str = "gpu",
    env_kwargs: Optional[Mapping[str, Any]] = None,
):
    """The official evaluation wrapper chain around ``SkillRollout-v0``.

    ``max_episode_steps`` must exceed the whole rollout: the vector wrapper
    resets the environment when the time limit is reached, which would
    silently restart the episode under the controller.  ``video_dir`` turns
    on the official episode recorder, which writes one video of the run when
    the environment is closed.
    """

    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")
    kwargs = dict(env_kwargs or {})
    task_cfgs = dict(kwargs.pop("task_cfgs", {}))
    # The official runners ignore the arm-at-rest terms of navigation success.
    navigate = dict(task_cfgs.get("navigate", {}))
    navigate.setdefault("ignore_arm_checkers", ignore_arm_checkers)
    task_cfgs["navigate"] = navigate
    kwargs["task_cfgs"] = task_cfgs
    config = EnvConfig(
        env_id=ROLLOUT_ENV_ID,
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
        sim_backend=sim_backend,
        continuous_task=True,
        frame_stack=3,
        stack=None,
        task_plan_fp=str(task_plan_path),
        record_video=video_dir is not None,
        info_on_video=info_on_video,
        save_video_freq=None,
        env_kwargs=kwargs,
    )
    return make_env(config, video_path=None if video_dir is None else str(video_dir))


class RolloutEnvironmentAdapter(MSHabEnvironmentAdapter):
    """``MSHabEnvironmentAdapter`` over one ``SkillRollout-v0`` environment.

    The entities are the episode's, the facts come from
    :meth:`SkillRolloutEnv.scene_measurements` through
    :meth:`SetTableEpisode.facts`, and the compatible contract environments
    are those of the library's contracts, as in the symbolic adapter.  The
    seed is fixed at construction because the controller calls ``reset()``
    without arguments; the first reset also reconfigures the scene.
    """

    def __init__(
        self,
        env,
        episode: SetTableEpisode,
        library: ContractLibrary,
        task: str,
        *,
        seed: Optional[int] = None,
    ) -> None:
        base = env.unwrapped
        if not isinstance(base, SkillRolloutEnv):
            raise TypeError(
                "the rollout adapter drives {}, got {}".format(
                    ROLLOUT_ENV_ID, type(base).__name__
                )
            )
        if base.num_envs != 1:
            raise ValueError(
                "the rollout adapter drives one environment, got num_envs={}".format(
                    base.num_envs
                )
            )
        self.episode = episode
        self._seed = seed
        self._reconfigure = True
        super().__init__(
            env,
            ROLLOUT_ENV_ID,
            self._extract_facts,
            scene_id=episode.build_config_name,
            entities=episode.environment_entities(),
            compatible_contract_env_ids=tuple(
                sorted({contract.env_id for contract in library.find(task=task)})
            ),
            metadata={
                "plan_index": episode.plan_index,
                "dataset": episode.dataset,
                "init_config_name": episode.init_config_name,
                "seed": seed,
            },
        )

    @property
    def base_env(self) -> SkillRolloutEnv:
        return self.env.unwrapped

    @property
    def device(self) -> torch.device:
        return self.base_env.device

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> EnvironmentSnapshot:
        options = dict(options or {})
        if self._reconfigure:
            options.setdefault("reconfigure", True)
            self._reconfigure = False
        return super().reset(self._seed if seed is None else seed, options)

    def measurements(self) -> SceneMeasurements:
        raw = self.base_env.scene_measurements()
        return SceneMeasurements(
            grasped={index: values[0] for index, values in raw["grasped"].items()},
            at_goal={index: values[0] for index, values in raw["at_goal"].items()},
            near={index: values[0] for index, values in raw["near"].items()},
            opened={index: values[0] for index, values in raw.get("opened", {}).items()},
            closed={index: values[0] for index, values in raw.get("closed", {}).items()},
            collision_safe=raw["collision_safe"][0],
        )

    def _extract_facts(self, observation, info, description):
        return self.episode.facts(self.measurements())

    # -- what the executor needs ----------------------------------------------------

    def point_at(self, subtask_index: int) -> None:
        self.base_env.point_at(subtask_index)

    def subtask_succeeded(self, snapshot: EnvironmentSnapshot) -> bool:
        """MS-HAB's own success checker for the pointed subtask."""

        return bool(snapshot.info["subtask_success"][0].item())

    def noop_action(self) -> torch.Tensor:
        """Zero joint deltas and zero base velocity: hold still for one step."""

        base = self.base_env
        return torch.zeros(
            (base.num_envs, *base.single_action_space.shape), device=self.device
        )

    def clear_contact_force(self) -> EnvironmentSnapshot:
        """Start the cumulative contact force over and refresh the facts.

        MS-HAB counts contact force per subtask and starts every subtask at
        zero.  ``point_at`` does that when a node starts; this is for a node
        that ended by breaking the force limit, so that the next admission
        check, which reads the last snapshot, sees a clean slate.  Costs one
        step without moving.
        """

        self.base_env.robot_cumulative_force[:] = 0
        return self.step(self.noop_action())
