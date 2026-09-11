# MS-HAB Skill Library

This package is an object-oriented skill library for ManiSkill-HAB. It follows
the four-layer architecture below: task goals are realized by a graph of
candidate skills, every executable skill carries a contract, and every atomic
skill is connected to one or more execution backends.

<p align="center">
  <img src="../../docs/static/images/skill_library_architecture.svg"
       alt="Task-conditioned skill library architecture" width="100%" />
</p>

## Architecture

```text
1. FunctionalGoalGraph
   Task instruction -> desired world-state predicates
                         │ ACHIEVED_BY
                         ▼
2. SkillCompositionGraph
   SkillNode + SkillEdge + SkillRelation
   IS_A / ENABLES / REQUIRES / ALTERNATIVE_TO / FALLBACK_TO
                         │ REALIZED_AS
                         ▼
3. SkillContract
   parameters / preconditions / effects / invariants / verification / failures
                         │ EXECUTED_BY
                         ▼
4. AtomicSkill
   Navigate / Pick / Place / Open / Close
   each with RL / BC / DP / VLA / controller / script backends
```

Composition has exactly one source of truth: graph relations. There is no
`SequentialSkill` or `AlternativeSkill` class. A sequence is represented by
`ENABLES` or `REQUIRES` edges; an alternative or fallback is represented by
`ALTERNATIVE_TO` or `FALLBACK_TO` edges.

RL, BC, and DP are a different kind of choice. They are backend alternatives
for one semantic atomic skill and live in `AtomicSkill.backends`, not in the
skill-to-skill graph.

### Source layout

| File | Responsibility |
| --- | --- |
| `mshab/skills/model.py` | Contracts, atomic skill hierarchy, invocations, execution backends |
| `mshab/skills/graph.py` | Functional-goal graph and candidate skill-composition graph |
| `mshab/skills/library.py` | Registration, lookup, checkpoint discovery, JSON export |
| `mshab/skills/__init__.py` | Public imports |
| `scripts/test_skill_library.py` | Standalone local-checkpoint smoke test |
| `tests/test_skill_library.py` | Standard-library unit tests |

## Layer 1: functional goal graph

`FunctionalGoalGraph` represents what the task wants, independently of how a
robot achieves it.

### Attributes

| Object | Attribute | Meaning |
| --- | --- | --- |
| `FunctionalGoalGraph` | `instruction` | Original task instruction |
|  | `goals` | Goal id -> `FunctionalGoal` |
|  | `dependencies` | Required ordering between goals |
| `FunctionalGoal` | `id` | Stable graph-local id |
|  | `predicate` | Desired world-state fact, such as `holding(013_apple)` |
|  | `description` | Optional human-facing explanation |
| `GoalDependency` | `source` | Goal that must be achieved first |
|  | `target` | Goal enabled by the source |

### Example

```python
from mshab.skills import FunctionalGoal, FunctionalGoalGraph

goals = FunctionalGoalGraph("Retrieve the apple")
goals.add_goal(FunctionalGoal("apple_reachable", "reachable(013_apple)"))
goals.add_goal(FunctionalGoal("apple_retrieved", "holding(013_apple)"))
goals.add_dependency("apple_reachable", "apple_retrieved")

assert goals.execution_order() == (
    "apple_reachable",
    "apple_retrieved",
)
```

## Layer 2: candidate skill-composition graph

`SkillCompositionGraph` contains grounded candidate calls. A `SkillNode` wraps
one `SkillInvocation`; a `SkillEdge` is the only representation of a
skill-to-skill relation.

### Relation semantics

| Relation | Direction | Meaning |
| --- | --- | --- |
| `IS_A` | child -> parent | Source is a specialization/category child of target |
| `ENABLES` | first -> next | Completing source enables target |
| `REQUIRES` | consumer -> prerequisite | Source requires target to complete first |
| `ALTERNATIVE_TO` | symmetric | Source and target are peer candidates |
| `FALLBACK_TO` | primary -> fallback | Try target if source fails |

Pairwise `ALTERNATIVE_TO` edges form an undirected connected alternative
group. `FALLBACK_TO` stays directed because retry priority matters.

### Attributes and methods

| Object/API | Meaning |
| --- | --- |
| `SkillCompositionGraph.task` | Task namespace shared by every node |
| `SkillCompositionGraph.goal_graph` | Optional Layer-1 graph served by candidates |
| `SkillCompositionGraph.nodes` | Node id -> `SkillNode` |
| `SkillCompositionGraph.edges` | Tuple of `SkillEdge` relations |
| `SkillNode.id` | Graph-local id; allows the same skill to be called more than once |
| `SkillNode.invocation` | Grounded contract and optional backend selection |
| `SkillNode.achieves` | Functional goal ids achieved by this node |
| `relate(source, target, relation)` | Add and validate one skill relation |
| `prerequisites(node_id)` | Derive causal prerequisites from `REQUIRES/ENABLES` |
| `alternatives(node_id)` | Query the full alternative connected component |
| `fallbacks(node_id)` | Query directed fallback candidates |
| `candidates_for_goal(goal_id)` | Query the `ACHIEVED_BY` candidates for one functional goal |
| `uncovered_goals()` | Find goals that currently have no candidate skill |
| `ready_nodes(facts, completed)` | Contract- and dependency-admitted candidates |
| `execution_order()` | Stable causal topological order; cycles are rejected |

### Example: Navigate -> Pick with alternative and fallback relations

```python
from mshab.skills import SkillCompositionGraph, SkillNode, SkillRelation

navigate = library.get("mshab.set_table.navigate.all")
apple_pick = library.get("mshab.set_table.pick.013_apple")
generic_pick = library.get("mshab.set_table.pick.all")

graph = SkillCompositionGraph(task="set_table", goal_graph=goals)
graph.add_node(
    SkillNode(
        "navigate_to_apple",
        navigate.bind({"goal": "013_apple"}, backend_key="rl"),
        achieves=("apple_reachable",),
    )
)
graph.add_node(
    SkillNode(
        "pick_apple_specialized",
        apple_pick.bind({}, backend_key="rl"),
        achieves=("apple_retrieved",),
    )
)
graph.add_node(
    SkillNode(
        "pick_apple_generic",
        generic_pick.bind({"object": "013_apple"}, backend_key="rl"),
        achieves=("apple_retrieved",),
    )
)

# Sequence/dataflow is a relation, not a SequentialSkill object.
graph.relate(
    "navigate_to_apple",
    "pick_apple_specialized",
    SkillRelation.ENABLES,
)
graph.relate(
    "pick_apple_generic",
    "navigate_to_apple",
    SkillRelation.REQUIRES,
)

# The object-specific policy is a specialization of generic Pick.
graph.relate(
    "pick_apple_specialized",
    "pick_apple_generic",
    SkillRelation.IS_A,
)

# OR and retry semantics are also relations, not an AlternativeSkill object.
graph.relate(
    "pick_apple_specialized",
    "pick_apple_generic",
    SkillRelation.ALTERNATIVE_TO,
)
graph.relate(
    "pick_apple_specialized",
    "pick_apple_generic",
    SkillRelation.FALLBACK_TO,
)
```

## Layer 3: executable skill contracts

`SkillContract` describes what a skill needs, promises, preserves, verifies,
and may fail with. It does not load a policy or call `env.step()`.

### Attributes

| Attribute | Type | Meaning |
| --- | --- | --- |
| `parameters` | tuple of `SkillParameter` | Typed planner inputs |
| `preconditions` | tuple of predicate templates | Facts required before execution |
| `effects` | tuple of predicate templates | State changes promised after success |
| `invariants` | tuple of predicate templates | Safety/state facts monitored during execution |
| `verification` | tuple of predicate templates | Explicit success check after execution |
| `failure_modes` | tuple of `str` | Named failures for monitor/recovery logic |

Parameter types are `ENTITY`, `LOCATION`, `ARTICULATION`, `STRING`, `INTEGER`,
`FLOAT`, and `BOOLEAN`.

A contract starts as a template:

```text
precondition: reachable({object})
effect:       holding({object})
verify:       holding({object})
```

`bind()` validates parameters and grounds the template:

```python
call = apple_pick.bind({}, backend_key="rl")

assert dict(call.arguments) == {"object": "013_apple"}
assert call.contract.preconditions == (
    "reachable(013_apple)",
    "gripper_empty()",
)
assert call.contract.effects == ("holding(013_apple)",)
assert call.contract.verification == ("holding(013_apple)",)
```

`BoundContract` provides three admission/verification helpers:

```python
assert call.contract.can_start({
    "reachable(013_apple)",
    "gripper_empty()",
})
assert call.contract.achieved({"holding(013_apple)"})
assert call.contract.verified({"holding(013_apple)"})
```

### `SkillInvocation` attributes

| Attribute | Meaning |
| --- | --- |
| `skill` | Atomic semantic skill definition |
| `arguments` | Read-only grounded argument mapping |
| `contract` | Grounded `BoundContract` |
| `backend_key` | Optional explicit backend selection |
| `id` | Skill id plus canonical grounded arguments |

An object-specialized skill fills its fixed argument automatically:

```python
apple_pick = library.get("mshab.set_table.pick.013_apple")
call = apple_pick.bind({}, backend_key="rl")
assert dict(call.arguments) == {"object": "013_apple"}
```

A generic `all` policy requires explicit grounding:

```python
generic_pick = library.get("mshab.set_table.pick.all")
generic_call = generic_pick.bind(
    {"object": "013_apple"},
    backend_key="rl",
)
```

## Layer 4: atomic skills and execution backends

`Skill` is the abstract semantic interface. `AtomicSkill` is the executable
leaf type. MS-HAB currently supplies `NavigateSkill`, `PickSkill`, `PlaceSkill`,
`OpenSkill`, and `CloseSkill`.

### `Skill` attributes

| Attribute | Meaning |
| --- | --- |
| `name` | Task-local semantic name, such as `pick.013_apple` |
| `task` | Namespace such as `set_table` |
| `contract` | Layer-3 `SkillContract` |
| `description` | Human/planner-facing description |
| `id` | Stable id such as `mshab.set_table.pick.013_apple` |
| `ready` | Whether at least one execution backend is ready |

### `AtomicSkill` attributes

| Attribute | Meaning |
| --- | --- |
| `skill_type` | `NAVIGATE`, `PICK`, `PLACE`, `OPEN`, or `CLOSE` |
| `target` | Fixed target such as `013_apple`, `fridge`, or generic `all` |
| `env_id` | MS-HAB environment used for individual evaluation |
| `max_episode_steps` | Default per-invocation horizon |
| `backends` | Backend key -> `ExecutionBackend` |

### `CheckpointBackend` attributes

| Attribute | Meaning |
| --- | --- |
| `key` | Selector such as `rl`, `bc`, or `dp` |
| `executor_type` | `POLICY` for checkpoint backends |
| `status` | `MISSING`, `PARTIAL`, or `READY` |
| `family` | Checkpoint family |
| `policy_type` | MS-HAB routing value such as `rl_per_obj` |
| `checkpoint_path` | Path to `policy.pt` |
| `config_path` | Path to `config.yml` |
| `checkpoint_sha256` | Optional future model identity seal |

One semantic skill may have several execution implementations:

```text
mshab.set_table.pick.013_apple
├── rl -> CheckpointBackend
├── bc -> CheckpointBackend
└── dp -> CheckpointBackend
```

## Skill library registry

`SkillLibrary` is the read/query boundary used by planners and dispatchers.

| Method | Purpose |
| --- | --- |
| `register(skill)` | Register one semantic atomic skill |
| `get(skill_id)` | Exact lookup by stable id |
| `find(...)` | Filter by task, type, target, or readiness |
| `from_checkpoint_root(path)` | Discover checkpoints and group backend alternatives |
| `to_dict()` | Create a JSON-serializable snapshot |
| `save_index(path)` | Save metadata without copying `.pt` weights |

## How the library serves upper layers

```text
Task/VLM planner
    │ library.find() / get()
    ▼
FunctionalGoalGraph + SkillCompositionGraph
    │ graph.ready_nodes(facts, completed)
    ▼
SkillInvocation + BoundContract
    │ can_start(facts)
    ▼
Backend router
    │ skill.backend(invocation.backend_key)
    ▼
MS-HAB executor
    │ config_path + checkpoint_path + policy_type
    ▼
Monitor / verifier / recovery
      verified(facts) + failure_modes + graph.fallbacks(node)
```

Upper layers should never hard-code a checkpoint path. They ask the library for
a semantic skill, ground it, admit it through the contract, and then resolve a
ready backend:

```python
def prepare_pick(library, object_name, facts):
    skill = library.get(f"mshab.set_table.pick.{object_name}")
    invocation = skill.bind({}, backend_key="rl")
    if not invocation.contract.can_start(facts):
        raise RuntimeError(f"preconditions do not hold: {invocation.id}")

    backend = skill.backend(invocation.backend_key)
    return {
        "invocation": invocation,
        "policy_type": backend.policy_type,
        "config_path": backend.config_path,
        "checkpoint_path": backend.checkpoint_path,
    }
```

The current code implements goal/skill graph data structures, relation and
cycle validation, contract grounding, backend discovery/selection, and JSON
serialization. The live predicate adapter and policy execution loop are the
next layer; `SkillInvocation` does not yet actuate the simulator by itself.

## Installation

From a new checkout (the directory may be named `maniskill-hab`):

```bash
git clone https://github.com/GuoZheXinDeGuang/maniskill-agentic-library.git \
  maniskill-hab
cd maniskill-hab

conda create -n mshab python=3.9 -y
conda activate mshab

git clone https://github.com/haosulab/ManiSkill.git \
  -b mshab --single-branch ../ManiSkill-mshab
pip install -e ../ManiSkill-mshab
pip install -e .
pip install -U "huggingface_hub[cli]"
```

The graph/library modules are standard-library-only. Full simulation still
needs the dependencies and assets in the main
[README](../../README.md#setup-and-installation).

## Download checkpoints

From the ManiSkill-HAB repository root:

```bash
export MS_ASSET_DIR="$(cd .. && pwd)/mshab-assets"
mkdir -p "$MS_ASSET_DIR/data/mshab_checkpoints"
```

Download all released checkpoints:

```bash
hf download arth-shukla/mshab_checkpoints \
  --local-dir "$MS_ASSET_DIR/data/mshab_checkpoints"
```

Or only the 11 SetTable RL policies used by the starter library:

```bash
hf download arth-shukla/mshab_checkpoints \
  --include "rl/set_table/**" \
  --local-dir "$MS_ASSET_DIR/data/mshab_checkpoints"
```

Add `--dry-run` to inspect files before downloading. See the official
[Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli)
for authentication, revision, cache, and filtering options.

Expected layout:

```text
$MS_ASSET_DIR/data/mshab_checkpoints/
└── <family>/<task>/<skill-type>/<target>/
    ├── config.yml
    └── policy.pt
```

## Discover and query skills

```python
import os
from pathlib import Path

from mshab.skills import SkillLibrary, SkillType

asset_root = Path(os.environ.get("MS_ASSET_DIR", "../mshab-assets"))
library = SkillLibrary.from_checkpoint_root(
    asset_root / "data" / "mshab_checkpoints"
)

for skill in library.find(task="set_table", ready=True):
    print(skill.id, sorted(skill.backends))

apple_picks = library.find(
    task="set_table",
    skill_type=SkillType.PICK,
    target="013_apple",
)
apple_pick = apple_picks[0]
```

Export the metadata index without copying weights:

```python
library.save_index(Path("/tmp/mshab_skill_index.json"))
```

## Run tests

The standalone test is CPU-only. It checks the 11 downloaded SetTable
checkpoints, grounding, goal dependencies, skill-skill relations, fallback,
cycle rejection, and JSON serialization:

```bash
python scripts/test_skill_library.py --expected-count 11
```

With a custom checkpoint directory:

```bash
python scripts/test_skill_library.py \
  --checkpoint-root /path/to/mshab_checkpoints \
  --task set_table \
  --expected-count 11 \
  --export /tmp/mshab_skill_index.json
```

Run the unit tests without pytest:

```bash
python -m unittest discover -s tests -p test_skill_library.py -v
```

These tests do not load tensors or call `env.step()`. Use the existing MS-HAB
evaluation scripts for GPU policy rollouts.

## Design boundary and next steps

The OOP boundary is intentionally narrow:

- inheritance models true “is-a” relationships (`PickSkill` is an
  `AtomicSkill`, `CheckpointBackend` is an `ExecutionBackend`);
- ownership models “has-a” relationships (`AtomicSkill` has backends,
  `SkillNode` has an invocation);
- graph edges own every skill-to-skill relation;
- contracts remain declarative and independent of Torch/ManiSkill.

Next integrations should add:

1. an adapter from live MS-HAB state to symbolic facts;
2. an executor that loads/caches policies and runs one `SkillInvocation`;
3. evidence per `(skill, backend, task, split)`;
4. backend selection based on availability and measured success;
5. an expected-release manifest so completely absent checkpoints appear as
   `MISSING`, not merely undiscovered.
