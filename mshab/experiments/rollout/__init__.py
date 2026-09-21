"""Stage 6: the controller loop on MS-HAB.

The controller, validator, and proposers of ``mshab.experiments.planning`` are
reused unchanged; this package supplies what the plan listed as the simulator
follow-ups, split by what they import:

standard library only (tested on CPU)
    ``episode``   one official TidyHouse plan as the symbolic scene: entities,
                  goal, goal facts, grounded node -> plan subtask, facts from
                  the environment's measurements
    ``proposer``  ``TidyHouseRuleProposer``, the rule-based pseudo model that
                  plays the scripted proposer's role on a real episode
    ``run``       the command line; it imports the two modules below only
                  when it is about to start the simulator

torch and ManiSkill (the GPU runner)
    ``environment``  ``SkillRollout-v0`` behind ``MSHabEnvironmentAdapter``
    ``executor``     ``CheckpointPolicyExecutor``, the RL checkpoints as
                     Layer-4 policies

Run it with ``python -m mshab.experiments.rollout``; see the README.
"""

from mshab.experiments.rollout.episode import (
    EPISODE_SCHEMA_VERSION,
    OBJECT_KIND,
    RECEPTACLE_KIND,
    TRANSFER_SUBTASK_TYPES,
    SceneMeasurements,
    TidyHouseEpisode,
    Transfer,
    UnsupportedGrounding,
    habitat_instance,
    load_episode,
    object_category,
    plan_document,
    read_goal_receptacles,
    read_plan,
    sequential_plan_path,
)
from mshab.experiments.rollout.proposer import (
    DEFAULT_GIVE_UP_AFTER,
    TidyHouseRuleProposer,
)

__all__ = [
    "DEFAULT_GIVE_UP_AFTER",
    "EPISODE_SCHEMA_VERSION",
    "OBJECT_KIND",
    "RECEPTACLE_KIND",
    "SceneMeasurements",
    "TRANSFER_SUBTASK_TYPES",
    "TidyHouseEpisode",
    "TidyHouseRuleProposer",
    "Transfer",
    "UnsupportedGrounding",
    "habitat_instance",
    "load_episode",
    "object_category",
    "plan_document",
    "read_goal_receptacles",
    "read_plan",
    "sequential_plan_path",
]
