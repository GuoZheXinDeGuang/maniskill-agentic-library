# Four-layer architecture

How the four layers own their objects, which way the dependencies point, and
what counts as the single source of truth for composition.

Part of the [MS-HAB skill library guide](../README.md).

## Layer ownership

| Layer | Main objects | Simulator-specific? | Responsibility |
| --- | --- | --- | --- |
| 1. Sub-goals | `SubGoalGraph`, `SubGoal` | No | Decompose the goal into ordered symbolic sub-goal predicates |
| 2. Skill graph | `SkillGraph`, `SkillSubgraph`, `SkillNode` | No | Give every sub-goal its own skill subgraph and connect those subgraphs |
| 3. Contracts | `Contract`, `GroundedSkill`, `SkillGrounder` | Yes | Bind a node's arguments to its contract and check its predicates against live facts |
| 4. Policies | `Policy`, `CheckpointPolicy`, `ContractLibrary` bindings, `PolicyExecutor` | Yes | Pick a policy bound to the contract, load it, and interact with the environment |

`SkillRuntime` sits on no single row: it coordinates Layers 3 and 4 by
grounding a node, checking admission and invariants against environment
snapshots, and dispatching the selected policy through a `PolicyExecutor`.

## Dependency direction

The dependency direction is one-way:

```text
goal (task text)
    │
    ▼
Layer 1: SubGoalGraph                           simulator-independent
    │ one sub-goal -> one skill subgraph
    ▼
Layer 2: SkillGraph / SkillSubgraph              simulator-independent
    │ owns SkillNode + internal SkillEdge
    │ CrossSubgraphEdge connects sub-goal subgraphs
    │ edges = semantic / logical order
    ▼
Layer 3: GroundedSkill                          simulator-specific (ManiSkill-HAB)
    │ node -> one contract; preconditions checked against current facts
    ▼
Layer 4: Policy / PolicyExecutor                simulator-specific (ManiSkill-HAB)
    │ contract <-> policies: many-to-many bindings, ordered by preference
    │ RL / BC / DP / VLA / controller / script policies
    ▼
EnvironmentAdapter                              reset / step / snapshot facts
```

Layer 2 never stores a checkpoint, simulator actor, raw observation, or
policy choice. A `SkillNode` contains only:

```text
id + contract_id + symbolic arguments + achieved sub-goal ids
```

## Object ownership

The OOP ownership hierarchy is:

```text
SubGoalGraph
└── SubGoal (Layer 1 definition)

SkillGraph (Layer 2 aggregate root)
├── SkillSubgraph[subgoal_id] (exactly one per implemented sub-goal)
│   ├── SkillNode (instrumental or sub-goal-achieving candidate)
│   └── SkillEdge (relations inside this sub-goal implementation)
└── CrossSubgraphEdge (relations between two sub-goal subgraphs)
```

## Composition has one source of truth

Skill-node composition has exactly one source of truth: graph relations.
There is no `SequentialSkill` or `AlternativeSkill` class. Sequence,
requirements, alternatives, and fallback are values of `SkillRelation`; their
ownership determines whether they are stored inside a subgraph or between
subgraphs.

