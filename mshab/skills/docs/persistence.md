# Persistence: what is stored, and what a run leaves behind

Which layers can be written and read back, which artifacts are checked into
the repository, what a SetTable run does and does not record, and where the
gaps are.

Part of the [MS-HAB skill library guide](../README.md).

The short version: **Layers 1 and 2 are first-class persisted objects with a
validated round trip; Layers 3 and 4 are exported, not stored.** A run reads a
predefined catalog and writes no graph state of its own.

See [Data pipeline](./data-pipeline.md) for how these artifacts are produced
and consumed during a run.

## Serialization per layer

| Layer | Objects | Write | Read | Checked on read |
| --- | --- | --- | --- | --- |
| 1 | `SubGoal`, `SubGoalDependency`, `SubGoalGraph` | `as_dict()` | `from_dict()` | Strict schema, cycle rejection, `execution_order` staleness |
| 2 | `SkillNode`, `SkillEdge`, `SkillSubgraph`, `CrossSubgraphEdge`, `SkillGraph` | `as_dict()` | `from_dict()` | Strict schema, full `validate()`, derived `nodes`/`edges` view and `candidate_partial_order` staleness |
| 2 | `SkillPlan`, `SkillGraphPatch`, `SkillSubgraphExtension` | `as_dict()` | `from_dict()` | Strict schema; a patch additionally validates contract ids when a library is passed |
| 3 | `Contract` | `as_dict()` (complete: parameters, all predicate groups, `env_id`, horizon) | **none** | — |
| 3 | `GroundedSkill` | **none** | **none** | — |
| 4 | `Policy`, `CheckpointPolicy` | `as_dict()` | **none** | — |
| 4 | `ContractLibrary` | `to_dict()`, `save_index()` | **none** | — |

`LibraryCatalog.from_dict()` therefore rebuilds real Layer-1 and Layer-2
objects and revalidates them, but keeps the Layer-3 and Layer-4 sections as
opaque mappings. Nothing reconstructs a `Contract` or a `ContractLibrary` from
JSON.

Two consequences worth knowing:

- The catalog's contract records are **not** `Contract.as_dict()`. The
  generator builds a slimmer record — id, type, target, `env_id`,
  `max_episode_steps`, bound policies — so contract predicates reach disk only
  through `3_grounded_skills`, which is itself assembled by the generator
  rather than by a `GroundedSkill` method. The full `Contract.as_dict()` is
  used only by the granularity experiment.
- `ContractLibrary.save_index()` is implemented and has **no caller** anywhere
  in the repository.

## Artifacts checked into the repository

| Path | Produced by | Contents |
| --- | --- | --- |
| `mshab/skills/catalogs/set_table.json` (49 KB) | `scripts/generate_set_table_skill_graph.py` | Complete Layer 1 and Layer 2, grounded-skill predicates, contract and policy records with bindings, both execution plans, a summary |
| `docs/static/images/set_table_skill_graph.svg` | same script | Four-layer rendering of the same graph |
| `mshab/experiments/granularity/artifacts/library.json` | `python -m mshab.experiments.granularity.render` | Full `Contract.as_dict()` records for the five generic contracts, the 53-row policy manifest, and their 53 bindings listed from both sides |
| `mshab/experiments/granularity/artifacts/contract_policy_layers.svg` | same command | Rendering of those two layers |
| `mshab/experiments/granularity/graphs/<name>.json` | same command | One gold graph each: the `SkillGraphPatch` that builds it, metadata, a structural summary, and the nominal plan; `load_gold_graph()` rebuilds and revalidates Layers 1 and 2 from the patch |
| `mshab/experiments/granularity/graphs/<name>.svg` | same command | Rendering of that gold graph, one row per sub-goal |
| `mshab/experiments/docs/figures/extracts/*.json` | `python -m mshab.experiments.docs.figures.render --extract <run dir>` | Compact extracts of DeepSeek evaluation traces: the accepted responses, the request's entities and facts, the controller's decisions and replans, the agreement numbers |
| `mshab/experiments/docs/figures/*.svg` | `python -m mshab.experiments.docs.figures.render` | The pipeline, the Layer-1 chains side by side, and every extracted proposal drawn by the gold-graph renderer |

All three generators are deterministic and exclude machine-local state:
checkpoint `ready/missing/partial` status and absolute paths never enter an
artifact, so the files regenerate identically on a clean clone.

## What a SetTable run reads and writes

**Reads a predefined graph.** `scripts/build_set_table_graph_plan.py` loads the
committed catalog with `LibraryCatalog.from_dict()`. The Python builders are
not on the run path at all: `build_set_table_stack()` is called only by the
generator script and by tests. Running the task therefore replays the graph
version that is in git, not one rebuilt from the current source.

**Writes no graph state.** The run directory
`$MSHAB_EXPS_DIR/set_table-graph-plan/<run_name>/` holds `eval_videos/*.mp4`,
`output.txt`, `subtask_fail_counts.json`, tensorboard logs, and the runner's
own `console.log` and `exit_status`. There is no Layer-1/2 snapshot in it.

The only per-run file that mentions the graph is the grounded plan at
`$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/`
`set_table/custom/graph_<RUN_NAME>.json`, and it is a flattened trace rather
than a graph: 16 `skill_decisions` of `node_id` / `contract_id` / `target`,
plus the official scene data. Sub-goals, subgraphs, edges, and fallback chains
are gone. That file also lives in the assets volume rather than in the run
directory or the repository, is overwritten by the next run of the same name,
and is shared by every attempt of a multi-seed sweep because the grounded plan
is seed-independent.

## Gaps

1. **No contract or library loader.** `Contract.from_dict()` and
   `ContractLibrary.from_dict()` do not exist, so Layers 3 and 4 can only be
   rebuilt by executing Python — the hand-written contract classes plus the
   hard-coded SetTable manifest. This is the same gap that blocks a VLM from
   proposing a contract.
2. **No drift guard.** No test asserts that the committed
   `catalogs/set_table.json` equals what the current builders produce.
   `LibraryCatalog.from_dict()` only checks the file's *internal* consistency,
   so a builder change with no regeneration would pass the suite. A cheap
   guard is to build the stack with a non-existent checkpoint root — artifact
   status is excluded from the catalog anyway — and compare the generated
   document with the checked-in file.
3. **No run provenance.** Nothing in a run directory records which catalog
   version produced it. Recovering that today means correlating timestamps
   with git history. Either a content hash of the catalog inside the grounded
   plan's `selection` block, or a copy of the catalog into the run directory,
   would close it.
