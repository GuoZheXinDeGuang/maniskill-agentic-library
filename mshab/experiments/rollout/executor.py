"""The RL checkpoints as Layer-4 policies: a ``PolicyExecutor`` for MS-HAB.

``CheckpointPolicyExecutor`` loads a ``CheckpointPolicy`` (SAC for pick and
place, PPO for navigation; the loaders mirror ``mshab.evaluate``) the first
time it is asked to run it, points the environment at the plan subtask the
grounded node stands for, and steps the policy until MS-HAB's own checker for
that subtask passes or the contract's horizon runs out.  Admission,
invariant monitoring, and effect verification are not done here; they run in
``SkillRuntime`` on the facts the adapter extracts, exactly as on the symbolic
environment.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
from gymnasium import spaces
from omegaconf import OmegaConf

from mshab.agents.ppo import Agent as PPOAgent
from mshab.agents.sac import Agent as SACAgent
from mshab.experiments.rollout.environment import RolloutEnvironmentAdapter
from mshab.experiments.rollout.episode import SetTableEpisode, UnsupportedGrounding
from mshab.skills.model import ArtifactStatus, CheckpointPolicy, GroundedSkill, Policy
from mshab.skills.runtime import (
    ContractMonitor,
    ContractViolation,
    PolicyExecution,
    PolicyExecutor,
)
from mshab.utils.array import to_tensor


ActFunction = Callable[[Any], torch.Tensor]
#: How the executor reports a node its episode has no subtask for.
NO_MATCHING_SUBTASK = "no_matching_subtask"
EXECUTION_TIMEOUT = "execution_timeout"


def _contact_force(snapshot) -> Optional[float]:
    """The cumulative contact force of the pointed subtask, from the env's info."""

    value = snapshot.info.get("robot_cumulative_force")
    if value is None:
        return None
    return round(float(value[0]), 1)


class CheckpointPolicyExecutor(PolicyExecutor):
    """See the module docstring.  ``executions`` records every run for the trace."""

    def __init__(
        self,
        episode: SetTableEpisode,
        *,
        device: Optional[str] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.episode = episode
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.log = log or (lambda line: None)
        self._policies: Dict[str, ActFunction] = {}
        self.executions: List[Dict[str, Any]] = []

    @property
    def simulator_steps(self) -> int:
        return sum(record["steps"] for record in self.executions)

    def execute(
        self,
        grounded: GroundedSkill,
        policy: Policy,
        environment: RolloutEnvironmentAdapter,
        monitor: ContractMonitor,
    ) -> PolicyExecution:
        if not isinstance(environment, RolloutEnvironmentAdapter):
            raise TypeError(
                "the checkpoint executor runs on a RolloutEnvironmentAdapter, got {}".format(
                    type(environment).__name__
                )
            )
        record: Dict[str, Any] = {
            "grounding": grounded.id,
            "policy": policy.id,
            "subtask_index": None,
            "subtask": None,
            "steps": 0,
            "success": False,
            "failure_mode": None,
        }
        self.executions.append(record)
        try:
            index = self.episode.subtask_for(
                grounded.contract.contract_type_name,
                grounded.arguments,
                environment.snapshot().facts,
            )
        except UnsupportedGrounding as exc:
            record["failure_mode"] = NO_MATCHING_SUBTASK
            record["reason"] = str(exc)
            self.log("  {}: {} ({})".format(grounded.id, NO_MATCHING_SUBTASK, exc))
            return PolicyExecution(
                success=False,
                steps=0,
                failure_mode=NO_MATCHING_SUBTASK,
                metadata={"reason": str(exc), "policy": policy.id},
            )
        record["subtask_index"] = index
        record["subtask"] = self.episode.subtask_label(index)
        act = self._act_function(policy, environment)
        horizon = grounded.contract.max_episode_steps
        self.log(
            "  {} -> subtask {} {} with {} (horizon {})".format(
                grounded.id, index, record["subtask"], policy.id, horizon
            )
        )

        environment.point_at(index)
        # The observation in hand describes the previously pointed subtask;
        # one step without moving refreshes it for this one.
        snapshot = environment.step(environment.noop_action())
        steps = 1
        record["steps"] = steps
        monitor(snapshot)
        policy_steps = 0
        try:
            with torch.no_grad():
                while True:
                    record["cumulative_force"] = _contact_force(snapshot)
                    if environment.subtask_succeeded(snapshot):
                        record.update(steps=steps, success=True)
                        self.log("    success after {} steps".format(steps))
                        return PolicyExecution(
                            success=True,
                            steps=steps,
                            metadata={
                                "policy": policy.id,
                                "subtask_index": index,
                                "policy_steps": policy_steps,
                            },
                        )
                    if policy_steps >= horizon:
                        break
                    observation = to_tensor(
                        snapshot.observation, device=self.device, dtype="float"
                    )
                    action = act(observation)
                    snapshot = environment.step(action)
                    steps += 1
                    policy_steps += 1
                    # Kept current every step: a monitor that raises on a
                    # violated invariant leaves the record with the steps taken.
                    record["steps"] = steps
                    monitor(snapshot)
        except ContractViolation as exc:
            # The runtime's invariant monitor stopped the skill: the contact
            # force limit of this subtask was broken.  MS-HAB starts every
            # subtask's count at zero and so does a retry here; the runtime
            # records the violation from the exception, not from the facts.
            record.update(
                steps=steps + 1,
                failure_mode="invariant_violated",
                reason=str(exc),
                cumulative_force=_contact_force(snapshot),
            )
            self.log("    invariant violated after {} steps: {}".format(steps, exc))
            environment.clear_contact_force()
            raise
        record.update(steps=steps, failure_mode=EXECUTION_TIMEOUT)
        self.log("    {} after {} steps".format(EXECUTION_TIMEOUT, steps))
        return PolicyExecution(
            success=False,
            steps=steps,
            failure_mode=EXECUTION_TIMEOUT,
            metadata={"policy": policy.id, "subtask_index": index, "policy_steps": policy_steps},
        )

    # -- loading ------------------------------------------------------------------

    def _act_function(
        self, policy: Policy, environment: RolloutEnvironmentAdapter
    ) -> ActFunction:
        if policy.id in self._policies:
            return self._policies[policy.id]
        if not isinstance(policy, CheckpointPolicy):
            raise TypeError(
                "the checkpoint executor runs CheckpointPolicy objects, got {} for {!r}".format(
                    type(policy).__name__, policy.id
                )
            )
        if policy.status != ArtifactStatus.READY:
            raise RuntimeError(
                "policy {!r} is {}: {} / {}".format(
                    policy.id, policy.status.value, policy.config_path, policy.checkpoint_path
                )
            )
        algo = OmegaConf.load(str(policy.config_path)).algo
        base = environment.base_env
        observation_space = base.single_observation_space
        action_shape = base.single_action_space.shape
        sample = to_tensor(environment.snapshot().observation, device=self.device, dtype="float")
        if algo.name == "ppo":
            agent = PPOAgent(sample, action_shape)
            agent.eval()
            agent.load_state_dict(
                torch.load(
                    str(policy.checkpoint_path), map_location=self.device, weights_only=False
                )["agent"]
            )
            agent.to(self.device)
            act: ActFunction = lambda obs: agent.get_action(obs, deterministic=True)
        elif algo.name == "sac":
            pixel_spaces: spaces.Dict = observation_space["pixels"]
            model_pixel_spaces = {}
            for key, space in pixel_spaces.items():
                shape, low, high, dtype = space.shape, space.low, space.high, space.dtype
                if len(shape) == 4:
                    shape = (shape[0] * shape[1], shape[-2], shape[-1])
                    low = low.reshape((-1, *low.shape[-2:]))
                    high = high.reshape((-1, *high.shape[-2:]))
                model_pixel_spaces[key] = spaces.Box(low, high, shape, dtype)
            agent = SACAgent(
                spaces.Dict(model_pixel_spaces),
                observation_space["state"].shape,
                action_shape,
                actor_hidden_dims=list(algo.actor_hidden_dims),
                critic_hidden_dims=list(algo.critic_hidden_dims),
                critic_layer_norm=algo.critic_layer_norm,
                critic_dropout=algo.critic_dropout,
                encoder_pixels_feature_dim=algo.encoder_pixels_feature_dim,
                encoder_state_feature_dim=algo.encoder_state_feature_dim,
                cnn_features=list(algo.cnn_features),
                cnn_filters=list(algo.cnn_filters),
                cnn_strides=list(algo.cnn_strides),
                cnn_padding=algo.cnn_padding,
                log_std_min=algo.actor_log_std_min,
                log_std_max=algo.actor_log_std_max,
                device=self.device,
            )
            agent.eval()
            agent.load_state_dict(
                torch.load(
                    str(policy.checkpoint_path), map_location=self.device, weights_only=False
                )["agent"]
            )
            agent.to(self.device)
            act = lambda obs: agent.actor(
                obs["pixels"], obs["state"], compute_pi=False, compute_log_pi=False
            )[0]
        else:
            raise NotImplementedError(
                "policy {!r} is a {!r} checkpoint; the executor loads the RL "
                "checkpoints (ppo, sac) only".format(policy.id, algo.name)
            )
        with torch.no_grad():
            act(sample)
        self.log("  loaded {} ({}) from {}".format(policy.id, algo.name, policy.checkpoint_path))
        self._policies[policy.id] = act
        return act
