# Layers 3 and 4: contracts, environment, and policies

The simulator-specific half of the library: contracts and their predicates, the
environment adapter, the policies bound to a contract, how to add a contract
type, and how downloaded checkpoints are discovered.

Part of the [MS-HAB skill library guide](../README.md).

## Layer 3: contracts and environment facts

A skill node names one `Contract` by `contract_id`. Many nodes may name
the same contract: every `navigate_*` node in SetTable references
`mshab.set_table.navigate.all`. A `Contract` declares directly as attributes:

| Attribute | Meaning |
| --- | --- |
| `parameters` | Typed symbolic inputs |
| `preconditions` | Facts required before execution |
| `effects` | Facts promised after successful execution |
| `invariants` | Facts that must remain true during execution |
| `verification` | Facts used as explicit success evidence |
| `deletes` | Predicates this contract retracts (negative effects) |
| `failure_modes` | Named failures for recovery/fallback |

These predicates describe physical feasibility in the current world state
only. They are deliberately silent about task order; see
[Edges are about semantics](layers-1-2.md#edges-are-about-semantics-contracts-are-about-physics).

Layer-2 nodes do not ground themselves. `SkillGrounder` binds a node to its
contract and returns a `GroundedSkill`, whose predicates have the node's
arguments substituted:

```python
from mshab.skills import SkillGrounder

grounder = SkillGrounder(stack.library)
node = stack.skill_graph.nodes["pick_object_specialized"]
grounded = grounder.ground(node)

assert dict(grounded.arguments) == {"object": "013_apple"}
assert grounded.effects == ("holding(013_apple)",)
```

Grounding involves no policy. A `GroundedSkill` knows its contract and its
concrete predicates; which of the contract's bound policies runs it is a
Layer-4 decision made at execution time. Contract predicates only become
meaningful when an environment adapter produces matching facts.

## How Layer 3 and Layer 4 communicate with an environment

The environment boundary has four objects:

| Object | Responsibility |
| --- | --- |
| `EnvironmentEntity` | Map `013_apple` to an environment name such as `obj_0` |
| `EnvironmentDescription` | Environment id, scene id, entity catalog, compatibility, metadata |
| `EnvironmentSnapshot` | Observation, `info`, canonical facts, step index, termination state |
| `EnvironmentAdapter` | Reset, step, snapshot, entity resolution, compatibility |

`MSHabEnvironmentAdapter` wraps an environment returned by the existing
MS-HAB `make_env(...)`. It receives two MS-HAB-specific callbacks:

- `entity_extractor(env, observation, info)` discovers the current episode's
  object/articulation names and returns `EnvironmentEntity` values;
- `fact_extractor(observation, info, description)` translates MS-HAB success
  checker keys such as `is_grasped`, `navigated_close`,
  `articulation_open`, and `articulation_closed` into canonical predicates.

Minimal single-environment example:

```python
from mshab.skills import EnvironmentEntity, MSHabEnvironmentAdapter

def facts_from_set_table(observation, info, description):
    facts = {f"present({name})" for name in description.entities}
    # Production vector code must select an environment index before bool().
    if info.get("is_grasped", False):
        facts.add("holding(013_apple)")
    if info.get("articulation_open", False):
        facts.add("open(fridge)")
    if info.get("articulation_closed", False):
        facts.add("closed(fridge)")
    facts.add("collision_safe()")
    return facts

adapter = MSHabEnvironmentAdapter(
    env,
    environment_id="SequentialTask-v0",
    fact_extractor=facts_from_set_table,
    entities=(
        EnvironmentEntity("013_apple", "object", "obj_0"),
        EnvironmentEntity("fridge", "articulation", "articulation-0"),
        EnvironmentEntity("dining_table", "location", "goal_0"),
    ),
    compatible_contract_env_ids=(
        "NavigateSubtaskTrain-v0",
        "PickSubtaskTrain-v0",
        "PlaceSubtaskTrain-v0",
        "OpenSubtaskTrain-v0",
        "CloseSubtaskTrain-v0",
    ),
)
snapshot = adapter.reset()
```

For vectorized MS-HAB, use one selected environment index or return separate
snapshots per index. Do not collapse a batch of booleans into one global fact
set.

`SkillRuntime` coordinates both simulator-specific layers:

```text
Layer-2 SkillNode
    -> SkillGrounder binds the node to its contract (GroundedSkill)
    -> adapter snapshot supplies precondition/invariant facts
    -> ContractLibrary.select_policy picks a bound policy (explicit id, else first ready)
    -> PolicyExecutor loads the policy/controller and calls adapter.step(action)
    -> invariant monitor checks every step
    -> adapter supplies final effect/verification facts
    -> SkillExecutionResult records evidence and failure mode
```

Execution is an interface because PPO/BC/DP/VLA policies need different
loaders:

```python
from mshab.skills import PolicyExecution, PolicyExecutor

class YourPolicyExecutor(PolicyExecutor):
    def execute(self, grounded, policy, environment, monitor):
        model = self.load_or_get_cached_model(policy)
        for step in range(grounded.contract.max_episode_steps):
            action = model(environment.snapshot().observation)
            snapshot = environment.step(action)
            monitor(snapshot)
            if grounded.verified(snapshot.facts):
                return PolicyExecution(success=True, steps=step + 1)
        return PolicyExecution(
            success=False,
            steps=grounded.contract.max_episode_steps,
            failure_mode="execution_timeout",
        )
```

The repository defines this communication contract, but it does not yet
include a production PPO loader or complete vectorized SetTable fact extractor.

## Layer 4: policies

A policy is a low-level executable: an RL/BC/DP checkpoint, a VLA, a
controller, or a script. Policies never appear in the skill graph; they are
reached only through the contracts they are bound to. The downloaded
checkpoint layout currently provides one RL policy per SetTable contract:

```text
navigate.all
open.fridge                  close.fridge
open.kitchen_counter         close.kitchen_counter
pick.013_apple               place.013_apple
pick.024_bowl                place.024_bowl
pick.all                     place.all
```

### Contracts and policies are many-to-many

A `Contract` owns its environment id, horizon, and predicates. It does not
own policies. A `Policy` owns its artifacts and knows nothing about
contracts. `ContractLibrary` owns both and the *bindings* between them:

- **One contract, several policies.** `open.fridge` is executed by the RL,
  BC, and DP checkpoints trained for it. Bindings are ordered and
  `library.select_policy(contract_id)` returns the first ready one, so the
  binding order is the default preference; `select_policy(contract_id,
  policy_id)` picks one explicitly.
- **One policy, several contracts.** The `pick.all` checkpoint was trained
  over every SetTable object, so it is bound not only to `pick.all` but also
  to `pick.013_apple` and `pick.024_bowl`. `bind_generic_policies()` adds
  these bindings after each contract's own checkpoint; both
  `ContractLibrary.from_checkpoint_root()` and the SetTable manifest call it.

```text
ContractLibrary (SetTable manifest: 11 contracts, 11 policies, 15 bindings)

contract                        bound policies, in preference order
mshab.set_table.pick.013_apple  rl.set_table.pick.013_apple, rl.set_table.pick.all
mshab.set_table.pick.024_bowl   rl.set_table.pick.024_bowl,  rl.set_table.pick.all
mshab.set_table.pick.all        rl.set_table.pick.all
mshab.set_table.open.fridge     rl.set_table.open.fridge
...

policy                          executes
rl.set_table.pick.all           pick.013_apple, pick.024_bowl, pick.all
rl.set_table.pick.013_apple     pick.013_apple
```

A `CheckpointPolicy` id is `<family>.<task>.<type>.<target>`, mirroring the
checkpoint directory, so one checkpoint keeps one id however many contracts
it serves.

Policy choice and graph fallback live on different layers.
`pick_bowl_specialized` falling back to `pick_bowl_generic` is a Layer-2 edge
between two skill nodes that reference two contracts. `pick.024_bowl` being
executed by the `all` checkpoint when its own checkpoint is missing is a
Layer-4 binding; the skill node, its contract, and its grounded predicates do
not change.

```python
from mshab.skills import build_set_table_library

library = build_set_table_library(checkpoint_root)

library.policies_for("mshab.set_table.pick.024_bowl")   # own checkpoint, then pick.all
library.contracts_for("rl.set_table.pick.all")          # the three pick contracts
library.select_policy("mshab.set_table.pick.024_bowl")  # first ready in that order
library.select_policy("mshab.set_table.pick.024_bowl", "rl.set_table.pick.all")

library.register_policy(my_vla)                          # any Policy subclass
library.bind("mshab.set_table.pick.013_apple", my_vla.id)
library.bind("mshab.set_table.place.013_apple", my_vla.id)
```

## Add your own contract (`YourContract`)

`your_contract.py` is a copyable extension point. New contract types do not
require editing the `ContractType` enum: `Contract` accepts a validated
custom string plus an explicit target parameter.

```python
from mshab.skills import ContractLibrary, YourContract

library = ContractLibrary()
contract = YourContract(
    task="set_table",
    target="013_apple",
    env_id="YourContractEnv-v0",
)
library.register(contract)

assert contract.id == "mshab.set_table.your_contract.013_apple"
```

To make it executable:

1. replace the template predicates with real ones;
2. register a `Policy` (checkpoint artifacts or a controller) and bind it to
   the contract with `library.bind(contract.id, policy.id)`; a policy that is
   already registered can be bound to the new contract as well;
3. implement a `PolicyExecutor` that runs that policy;
4. add compatible entity/fact extraction to the environment adapter;
5. add a `SkillNode` through `SkillGraphPatch` and validate the contract id;
6. test admission, invariant monitoring, effects, and verification.

This is OOP extension: `YourContract` is a `Contract`, a new policy is a
`Policy`, and a runner is a `PolicyExecutor`. Relationships between the skill
nodes that reference this contract and other skill nodes remain graph edges.

## Discover downloaded policies

```python
import os
from pathlib import Path

from mshab.skills import ContractLibrary, ContractType

asset_root = Path(os.environ.get("MS_ASSET_DIR", "/root/.maniskill"))
library = ContractLibrary.from_checkpoint_root(
    asset_root / "data" / "mshab_checkpoints"
)

for contract in library.find(task="set_table", ready=True):
    print(contract.id, [p.id for p in library.policies_for(contract.id)])

apple_pick = library.find(
    task="set_table",
    contract_type=ContractType.PICK,
    target="013_apple",
)[0]
generic_pick_contracts = library.contracts_for("rl.set_table.pick.all")
```

Discovery registers one `CheckpointPolicy` per `family/task/type/target`
leaf, binds it to the contract the path names, and then calls
`bind_generic_policies()`. With the full download, `pick.013_apple` is bound
to its own RL checkpoint plus the three `pick.all` checkpoints.

Upper layers query by contract id and do not hard-code checkpoint paths.
`library.save_index(path)` exports contracts, policies, and bindings without
copying weights.

