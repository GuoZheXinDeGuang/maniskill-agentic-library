# Extending Layers 1 and 2

Adding sub-goals, subgraphs, and candidate skill nodes through the declarative
`SkillGraphPatch` boundary, plus the catalog format and the future insertion
VLM's output contract.

Part of the [MS-HAB skill library guide](../README.md).

## Adding to Layer 1 and Layer 2

New graph content is added through `SkillGraphPatch`. A patch is declarative,
JSON-friendly, reviewable, and validated by the normal graph methods.

```python
from mshab.skills import (
    SubGoal,
    SubGoalDependency,
    SkillSubgraph,
    SkillGraphPatch,
    SkillNode,
    SkillRelation,
    CrossSubgraphEdge,
)

inspect_subgraph = SkillSubgraph("apple_inspected", "set_table")
inspect_subgraph.add_node(
    SkillNode(
        "inspect_apple",
        "mshab.set_table.inspect.013_apple",
        {},
        achieves=("apple_inspected",),
    )
)

patch = SkillGraphPatch(
    subgoals=(SubGoal("apple_inspected", "inspected(013_apple)"),),
    subgoal_dependencies=(
        SubGoalDependency("object_retrieved", "apple_inspected"),
    ),
    skill_subgraphs=(inspect_subgraph,),
    cross_edges=(
        CrossSubgraphEdge(
            source_subgoal="object_retrieved",
            target_subgoal="apple_inspected",
            source_node="pick_object_specialized",
            target_node="inspect_apple",
            relation=SkillRelation.ENABLES,
        ),
    ),
)

patch.apply(stack.subgoal_graph, stack.skill_graph)
```

`apply()` builds and validates everything on staging copies and updates the two
live aggregate roots only after the complete patch passes, so a rejected patch
leaves Layer 1 and Layer 2 byte identical. Pass `library=stack.library` to make
contract-id validation part of the atomic apply; the example above omits it
because `mshab.set_table.inspect.013_apple` is not registered yet, and the
patch cannot pass Layer 3/4 grounding until that contract exists in the library.

### Extending a subgraph that is already registered

Registering a subgraph seals it, so a new candidate for an existing sub-goal is
added by immutable replacement rather than in-place mutation. A node declares
at most one fallback, so the new candidate joins the end of the existing
`FALLBACK_TO` chain:

```python
from mshab.skills import SkillEdge

patch = SkillGraphPatch().with_extension(
    "bowl_retrieved",
    nodes=(SkillNode("pick_bowl_bc", "mshab.set_table.pick.024_bowl", {},
                     achieves=("bowl_retrieved",)),),
    edges=(SkillEdge("pick_bowl_generic", "pick_bowl_bc",
                     SkillRelation.FALLBACK_TO),),
)
patch.apply(subgoal_graph, skill_graph, library=library)
```

`SkillSubgraphExtension.rebuild()` clones the sealed subgraph, applies the
addition, revalidates it, and the aggregate swaps it in under the usual
global-uniqueness and cycle checks. Existing cross-sub-goal relations keep
working because they are sub-goal-level.

### Parsing an untrusted patch document

`SkillGraphPatch.from_dict(payload, task=...)` is the entry point for a
human-authored or future VLM proposal. It rejects unknown keys, unknown
relations, identifiers outside `[A-Za-z0-9_.-]`, non-scalar node arguments,
and unsupported `schema_version` values. `task` is supplied by the caller,
never read from the payload, so a proposer cannot redirect a patch into another
task's namespace.

### Loading the checked-in catalog

The complete checked-in four-layer artifact uses `LibraryCatalog`. It validates
derived orders, flattened nodes/edges, plans, one grounded skill per node,
and contract/policy records with their bindings listed from both sides, then
reconstructs the authoritative Layer-1/2 objects:

```python
import json
from pathlib import Path

from mshab.skills import LibraryCatalog

payload = json.loads(Path("mshab/skills/catalogs/set_table.json").read_text())
catalog = LibraryCatalog.from_dict(payload)
assert catalog.as_dict() == payload
```

### Future insertion VLM: placing a new skill node

The initial SetTable graph is hand-authored by `SetTableGraphBuilder`; a VLM is
not required to generate it. The insertion VLM's role starts when a new
contract (and a policy that executes it) becomes available: inspect the
existing graph and propose which sub-goal subgraph should own the new skill
node and which relations should connect it. Its output boundary is the same
`SkillGraphPatch`:

```python
from mshab.skills import SkillGraphBuilder, SkillGraphPatch

class YourGraphBuilder(SkillGraphBuilder):
    def propose(self, goal, task, context) -> SkillGraphPatch:
        # Initial graph: manual rules.
        # Future insertion VLM: propose placement for a new skill node.
        return SkillGraphPatch(...)
```

The VLM never mutates graph internals or invents checkpoint paths; it proposes
a patch, and `apply()` decides whether it is accepted.

