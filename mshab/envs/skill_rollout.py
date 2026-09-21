"""``SequentialTask-v0`` driven from outside, one subtask at a time.

The parent environment is a state machine over its task plan: it advances
``subtask_pointer`` when the pointed subtask's checkers pass, ends the episode
on the subtask horizon or the force limit, and reports whole-task success.
The skill library's runtime needs the physics, the observations, and the
checkers of that environment, but makes those decisions itself: which subtask
runs next is a planner decision, whether it succeeded is a contract
verification, and whether the task is done is judged on goal facts.

So this subclass keeps the plan, the merged actors, the observations, and the
per-subtask checkers, and changes three things:

- the pointer moves only through :meth:`point_at`;
- :meth:`evaluate` reports the pointed subtask's checkers as
  ``subtask_success`` and never advances or ends anything;
- :meth:`scene_measurements` checks every object and goal of the plan, not
  only the pointed one, so an adapter can state the whole scene as facts.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor

from mshab.envs.planner import (
    CloseSubtask,
    NavigateSubtask,
    OpenSubtask,
    PickSubtask,
    PlaceSubtask,
)
from mshab.envs.sequential_task import SequentialTaskEnv


@register_env("SkillRollout-v0")
class SkillRolloutEnv(SequentialTaskEnv):
    """See the module docstring."""

    # -- the plan -------------------------------------------------------------------

    def process_task_plan(self, env_idx, sampled_subtask_lists):
        super().process_task_plan(env_idx, sampled_subtask_lists)
        self.task_force_limits = torch.tensor(
            [
                float(self.task_cfgs[subtask.type].robot_cumulative_force_limit)
                for subtask in self.task_plan
            ],
            device=self.device,
            dtype=torch.float,
        )

    def point_at(self, subtask_index: int) -> None:
        """Make ``subtask_index`` the subtask observations and checkers describe.

        The cumulative contact force starts over, as the parent does when it
        advances the pointer, so the force limit is the subtask's own.
        """

        if not 0 <= subtask_index < len(self.task_plan):
            raise IndexError(
                "subtask {} is outside the {}-subtask plan".format(
                    subtask_index, len(self.task_plan)
                )
            )
        self.subtask_pointer[:] = subtask_index
        self.subtask_steps_left[:] = self.task_horizons[subtask_index]
        self.robot_cumulative_force[:] = 0

    # -- evaluation -----------------------------------------------------------------

    def evaluate(self):
        robot_force = (
            self.agent.robot.get_net_contact_forces(self.force_articulation_link_ids)
            .norm(dim=-1)
            .sum(dim=-1)
        )
        self.robot_cumulative_force += robot_force

        # The rest pose moves with the robot; the handle poses feed open/close
        # observations and the render.  Both as in the parent.
        self.ee_rest_world_pose = self.agent.base_link.pose * self.ee_rest_pos_wrt_base
        self.handle_world_poses = []
        for subtask, articulation in zip(self.task_plan, self.subtask_articulations):
            if isinstance(subtask, (OpenSubtask, CloseSubtask)):
                self.handle_world_poses.append(
                    articulation.links[subtask.articulation_handle_link_idx].pose
                    * subtask.articulation_relative_handle_pos
                )
            else:
                self.handle_world_poses.append(None)

        subtask_success, checkers = self._subtask_check_success()
        return dict(
            subtask_success=subtask_success,
            subtask=self.subtask_pointer.clone(),
            subtask_type=self.task_ids[self.subtask_pointer],
            robot_force=robot_force,
            robot_cumulative_force=self.robot_cumulative_force.clone(),
            **checkers,
        )

    # -- the whole scene --------------------------------------------------------------

    def _oriented_towards(self, goal: Actor, env_idx: torch.Tensor) -> torch.Tensor:
        """The parent's navigation orientation check for one goal."""

        goal_pose_wrt_base = self.agent.base_link.pose.inv() * goal.pose
        target = goal_pose_wrt_base.p[..., :2][env_idx]
        unit = target / torch.norm(target, dim=1).unsqueeze(-1).expand(*target.shape)
        rotation = torch.sign(unit[..., 1]) * torch.arccos(unit[..., 0])
        return torch.abs(rotation) <= self.navigate_cfg.navigated_successfully_rot

    def scene_measurements(self) -> Dict[str, Any]:
        """The checkers of every subtask, keyed by subtask index, per environment.

        ``grasped[i]`` for every pick subtask, ``at_goal[i]`` for every place
        subtask (the parent's place check without the arm and grasp terms),
        ``near[i]`` for every navigate subtask (close to and facing its
        goal), and ``collision_safe`` against the pointed subtask's force
        limit.  Values are Python lists with one boolean per environment; one
        device synchronisation for the whole scene.
        """

        with torch.device(self.device):
            env_idx = torch.arange(self.num_envs)
            keys: List[tuple] = []
            values: List[torch.Tensor] = []
            for index, subtask in enumerate(self.task_plan):
                if isinstance(subtask, PickSubtask):
                    keys.append(("grasped", index))
                    values.append(
                        self.agent.is_grasping(self.subtask_objs[index], max_angle=30)
                    )
                elif isinstance(subtask, PlaceSubtask):
                    keys.append(("at_goal", index))
                    values.append(
                        self._place_check_success(
                            self.subtask_objs[index],
                            self.subtask_goals[index],
                            subtask.goal_rectangle_corners,
                            env_idx,
                            check_progressive_completion=True,
                        )[0]
                    )
                elif isinstance(subtask, NavigateSubtask):
                    goal = self.subtask_goals[index]
                    articulation_type = (
                        subtask.articulation_config.articulation_type
                        if subtask.articulation_config is not None
                        else None
                    )
                    keys.append(("near", index))
                    values.append(
                        self._is_navigated_close(env_idx, goal, articulation_type)
                        & self._oriented_towards(goal, env_idx)
                    )
            keys.append(("collision_safe", None))
            values.append(
                self.robot_cumulative_force < self.task_force_limits[self.subtask_pointer]
            )
            table = torch.stack(values).cpu().tolist()
        result: Dict[str, Any] = {"grasped": {}, "at_goal": {}, "near": {}}
        for (group, index), row in zip(keys, table):
            if group == "collision_safe":
                result["collision_safe"] = [bool(item) for item in row]
            else:
                result[group][index] = [bool(item) for item in row]
        return result
