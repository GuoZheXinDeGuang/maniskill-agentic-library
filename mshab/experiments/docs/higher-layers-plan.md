# Higher layers: from hand-authored graphs to a VLM proposer

Implementation plan for the next experiment: build a skill library that is
richer than the packaged SetTable toy, then let a VLM do the planning in the
two upper layers (Layer 1 sub-goal decomposition, Layer 2 skill subgraphs)
over a fixed lower half (Layer 3 contracts, Layer 4 policies), and measure
how the *granularity* of the VLM's decomposition affects task outcome.

Companion to the [skill-library guide](../../skills/README.md) and the
[granularity experiment](../granularity/README.md). Vocabulary follows the
guide exactly: *goal*, *sub-goal*, *skill node*, *contract*, *policy*.

## Research question

> Given a fixed contract inventory and a symbolic description of the scene,
> a VLM first decomposes the goal into an ordered sub-goal sequence, then
> plans one skill subgraph per sub-goal. How does the granularity of that
> decomposition (few coarse sub-goals versus many fine ones) change
> validity, execution success, recovery behaviour, and replanning cost?

The two upper layers are therefore two separate proposer calls:

```text
call 1  decompose:  goal + scene + contract inventory     ->  ordered sub-goals
call 2  per sub-goal (independent, one call each):
        sub-goal predicate + contract inventory + scene   ->  one SkillSubgraph
assembler: subgraphs + Layer-1 order                      ->  SkillGraphPatch
```

Granularity is the independent variable and is controlled at call 1. The
subgraph proposer in call 2 is the same for every granularity, so a
difference in outcome is attributable to the decomposition.

## Stages

Every stage is testable on CPU with the standard-library test suite. The only
object that changes between the scripted proposer and the real model is the
one that answers the two calls.

```text
stage 1  align lower layers   granularity contracts/policies  ->  ContractLibrary
stage 2  gold graphs          hand-authored L1/L2 over those contracts, coarse and fine, saved as JSON
stage 3  proposer boundary     Decomposition/Subgraph request and response documents + ScriptedProposer
stage 4  online loop          execute -> observe -> re-decide, symbolic environment, replan hook
stage 5  real model           DeepSeek API behind the same boundary, granularity sweep
stage 6  MS-HAB               the same controller on the simulator: TidyHouse adapter, checkpoint executor, GPU runner
```

## What the upper layers already provide

The code already has the interfaces the plan needs, so the work is mostly
composition rather than new abstractions:

| Need | Existing object | File |
| --- | --- | --- |
| Ordered sub-goal sequence from a proposer | `SubGoalGraph.from_sequence(goal, subgoals)` | `mshab/skills/graph.py` |
| Declarative, validated Layer-1/2 proposal | `SkillGraphPatch`, `SkillGraphPatch.from_dict(payload, task=...)` | `mshab/skills/extension.py` |
| One subgraph per sub-goal, sealed on registration | `SkillSubgraph`, `SkillSubgraph.from_dict` | `mshab/skills/graph.py` |
| GraphProposer interface | `SkillGraphBuilder.propose(goal, task, context) -> SkillGraphPatch` | `mshab/skills/extension.py` |
| One-node decision, fallback inside a subgraph | `SkillPlanner.decide()`, `NoViableCandidate` | `mshab/skills/plan.py` |
| Admission, monitoring, execution evidence | `SkillRuntime`, `SkillExecutionResult` | `mshab/skills/runtime.py` |
| Environment boundary | `EnvironmentAdapter`, `EnvironmentSnapshot` | `mshab/skills/environment.py` |
| Five generic contracts, 53 policies, 53 bindings | `build_granularity_library()` | `mshab/experiments/granularity/lower_layers/` |

## What was missing

The gaps this plan set out to close, each with the stage that closed it.
All six stages are implemented; the per-stage sections below record how.

1. **The granularity lower layers were not a `ContractLibrary`** (stage 1).
   `ConnectedLayers` was its own storage; `SkillGrounder` and `SkillRuntime`
   need a `ContractLibrary` with contracts, policies, and bindings.
   `build_granularity_library()` replaced it.
2. **Generic contracts lost policy specificity** (stage 1). With `pick.all`
   as the only pick contract, `select_policy()` returned the first ready
   policy regardless of the object, so `rl.set_table.pick.013_apple` could
   be selected for a bowl. `select_policy(..., arguments=...)` now filters by
   the grounded target.
3. **No online loop** (stage 4). Nothing executed a node, observed the
   result, and asked the proposer again; the SetTable runner replays a frozen
   plan. `TaskController` is that loop.
4. **No Layer-1 replan hook** (stage 4). `NoViableCandidate` ended the run,
   although the guide says a failed sub-goal triggers a fresh goal ->
   sub-goal decomposition. The controller now asks the proposer again with
   the failure, the history, and the current facts.
5. **No proposer request/response schema** (stage 3).
   `SkillGraphBuilder.propose()` takes a free-form `context`; the scripted
   proposer needed precise documents so that the real model can be prompted
   with the same content. `mshab/experiments/planning/documents.py` defines
   them.
6. **No simulator-free environment** (stage 4). Testing the loop on CPU
   needed an `EnvironmentAdapter` whose facts are a symbolic set that
   contract effects and deletes update: `SymbolicEnvironmentAdapter`.
7. **No real model behind the boundary, and no way to answer a rejection**
   (stage 5). The scripted proposer answers from gold graphs and cannot
   revise; a model needs the validator's reasons back and a second chance.
   `DeepSeekProposer` and the validator's retry rounds closed both.
8. **Nothing executed a skill node on MS-HAB** (stage 6). `MSHabEnvironmentAdapter`
   had no fact extractor and `PolicyExecutor` no implementation, and the
   official environment is a state machine over its own plan that advances
   and terminates by itself. `mshab/experiments/rollout/` supplies the
   TidyHouse episode (entities, facts, node -> plan subtask), an environment
   subclass whose pointer the runtime controls, and the checkpoint executor.

## Stage 1: align the granularity lower layers with `mshab.skills`

Package: `mshab/experiments/granularity/lower_layers/`.

Almost everything in this package other than the five `Contract` objects
re-implements `ContractLibrary`: `Layer3Contracts` is `register`/`find`,
`Layer4PolicyStore` is `register_policy`/`policies`, and
`ExecutesConnection`/`ConnectedLayers` are `bind`/`policies_for`. The
connection layer is also stricter than the settled model: it forces one
contract per policy and has no preference order, whereas `ContractLibrary`
is many-to-many and ordered. Stage 1 shrinks the package to a manifest plus
one library constructor.

- Keep `PolicySpec` and the 53 rows it enumerates as the canonical MS-HAB
  checkpoint inventory across the three official tasks. This is the one
  thing the package holds that `mshab.skills` does not.
- `build_granularity_library(checkpoint_root) -> ContractLibrary` registers
  the five generic contracts, one `CheckpointPolicy` per manifest row, and
  one binding per policy to the contract of its type. Binding order is
  sorted policy id, so the default policy is reproducible.
- Remove `Layer3Contracts`, `Layer4PolicyStore`, `ExecutesConnection`, and
  `ConnectedLayers`. The JSON and SVG artifacts are documentation only and
  are read by nothing; regenerate them from the library or drop them. No
  Layer-3/4 export is needed by any later stage: gold graphs persist Layers
  1 and 2 only, proposer requests carry `Contract.as_dict()`, which is
  already free of paths and machine state, and policies load from the
  checkpoint root.
- Policy applicability: `PolicySpec.target` already records what each
  checkpoint was trained on. Add an optional `arguments` parameter to
  `ContractLibrary.select_policy()` so a grounded `pick(024_bowl)` prefers a
  policy whose target is `024_bowl`, then `all`, and never a policy trained
  for a different object. Behaviour is unchanged when `arguments` is omitted,
  so SetTable is unaffected. `SkillRuntime` passes the grounded arguments
  through, so admission and execution use the same filter. These are the
  only changes to `mshab.skills`.
- Namespace: the contract ids stay `mshab.granularity.<type>.all`. The
  graphs built on them are test graphs for this experiment, not a library
  that later work extends, so a rename buys nothing.
- Test: `test_granularity_library.py` (which replaced
  `test_granularity_layer3_layer4.py`) builds the library from an empty
  checkpoint root, checks 5 contracts / 53 policies / 53 bindings, grounds
  `pick` with `024_bowl`, and checks the target-aware selection order.

## Stage 2: hand-authored gold graphs at two granularities

Package: `mshab/experiments/granularity/higher_layers/`.

A *gold graph* is a hand-authored, validated Layer-1/2 skill graph that
exists only for this experiment. It plays two roles: it is the answer
the scripted proposer returns in stages 3 and 4, and it is the reference
the real model's output is compared against in stage 5. Gold graphs are
not a library for later work to build on.

TidyHouse is an official MS-HAB long-horizon task (alongside PrepareGroceries
and SetTable): five object transfers, 20 atomic subtasks, no articulations,
RL checkpoints for all nine object categories. That makes it the primary
task here, exactly as `tasks/tidy_house/README.md` already proposes.

Builders follow the `SetTableGraphBuilder` pattern but reference the generic
contracts with explicit arguments. Each result is saved as a `SkillGraphPatch`
JSON document under `mshab/experiments/granularity/graphs/<name>.json`:

| Graph | Sub-goals | Nodes | Role |
| --- | --- | --- | --- |
| `tidy_house_coarse` | 5 | 20 | One `at(object_i,destination_i)` sub-goal per object |
| `tidy_house_fine` | 20 | 20 | `reachable -> holding -> reachable(dest) -> at` per object; same nodes, different ownership |
| `set_table_generic` | 8 | 16 | The packaged SetTable re-expressed with generic contracts; regression against the existing catalog |
| `prepare_groceries` | 7 to 9 | ~20 | Fridge open/close plus object transfers; exercises articulations (second priority, not built yet: the official transfer sequence is not in this checkout) |

Every skill node is one contract call, which is one MS-HAB atomic subtask:
the only unit a policy can execute and the only unit the environment
verifies. Node granularity is therefore fixed, and the experiment varies
only how many sub-goals own the nodes. With one generic contract per type
there is exactly one candidate node per role, so the gold graphs contain no
`FALLBACK_TO` chains; Layer-2 fallback stays covered by the packaged
SetTable graph.

Each graph is validated three ways before it is committed: `SkillGraph.
validate()` on load, `SkillPlanner.plan()` produces a full path, and every
node grounds against the stage-1 library. A generator script rebuilds the
JSON files and an SVG per graph, mirroring the existing `render.py`.

The coarse and fine TidyHouse graphs are the two ends of the granularity
axis. They serve as the scripted proposer's canned answers in stage 3 and as
the references the real model is compared against in stage 5.

## Stage 3: the two-call proposer boundary and the scripted proposer

Package: `mshab/experiments/planning/` (new; simulator-independent).

Four JSON documents with strict `from_dict` parsing through
`mshab.skills.schema`:

```text
DecompositionRequest
  task, goal
  contracts:   Contract.as_dict() records, verbatim                       (from ContractLibrary)
  entities:    [{name, kind}]                                               (from EnvironmentDescription)
  facts:       sorted current predicates                                    (from EnvironmentSnapshot)
  history:     achieved sub-goals, completed nodes, failed nodes
  failure:     null | {subgoal_id, node_id, failure_mode, missing_effects, missing_preconditions}
  granularity: "free" | "coarse" | "fine"                                   (the experimental knob)
  attempt:     integer, 0 for the initial plan

DecompositionResponse
  subgoals:    ordered [{id, predicate}]           -> SubGoalGraph.from_sequence
  rationale:   free text, stored for inspection only

SubgraphRequest
  task, goal
  subgoal:     {id, predicate}
  contracts, entities, facts                       (as above)
  neighbours:  {previous: predicate | null, next: predicate | null}   (context only)

SubgraphResponse
  subgraph:    {subgoal_id, nodes, edges}           -> SkillSubgraph.from_dict (derived views recomputed, never emitted)
  rationale:   free text
```

Each sub-goal's `SubgraphRequest` is answered independently. The
`assemble_patch` then builds the `SkillGraphPatch`: the sub-goals and their
consecutive dependencies from call 1, the subgraphs from call 2, and one
sub-goal-level `ENABLES` cross edge from every sub-goal to each root node of
the next one (source endpoint `None`, so a fallback achiever also enables
the successor). This is the same shape `SetTableGraphBuilder` produces by
hand, and it is what `SkillGraph.validate()` requires.

`GraphProposer` is a `SkillGraphBuilder` subclass with two extra methods,
`decompose(request) -> DecompositionResponse` and
`plan_subgraph(request) -> SubgraphResponse`; `propose()` is implemented on
top of them through the assembler so existing callers keep working. Two
implementations:

- `ScriptedProposer`: a lookup table from a request fingerprint to a canned
  response. For call 1 the fingerprint is `(task, goal, granularity, attempt,
  failure.subgoal_id)`; for call 2 it is `(task, subgoal.id, subgoal.predicate)`
  (the predicate alone collides: `at(obj,dest)` is a four-node subgraph in
  the coarse graph and a one-node subgraph in the fine one). The canned
  answers are cut out of the stage-2 gold graphs by `gold_proposer()`, the
  stage-4 scenarios add their replan answers through `scenario_proposer()`,
  and a table round-trips through JSON (`save()`/`load()`). Unknown
  fingerprints raise, so a test cannot silently pass on a default answer.
  This is the pseudo VLM.
- `DeepSeekProposer`: the real model, stage 5.

A `ProposalValidator` sits between any proposer and the graph: it parses the
responses, grounds every node against the library, assembles and applies the
patch to fresh graphs, runs `SkillPlanner.plan()`, and returns either a
`ValidatedProposal` (both layers, the patch, the plan, and every request and
response verbatim as the trace record) or raises `ProposalRejected` with
`Rejection(stage, subgoal_id, message)` entries. Subgraph rejections are
collected across all sub-goals before the round stops. The rejections are
what the real model will see on a retry, so they are produced here already.
Implemented in `mshab/experiments/planning/`; see its README.

Tests: round trip of all four documents; the scripted proposer reproduces the
coarse and fine gold graphs from their goals; a malformed subgraph response
is rejected with a message naming the sub-goal and leaves the graphs
untouched.

## Stage 4: online loop on a symbolic environment

Package: `mshab/experiments/planning/` (symbolic environment, controller) and
`mshab/experiments/granularity/higher_layers/scenarios.py` (scenarios). The
planning package depends on `mshab.skills` only; the experiment builds on it.

- `SymbolicEnvironmentAdapter`: facts are a set of predicate strings;
  `reset()` installs the initial facts; `step({"add", "remove"})` applies a
  change and two world rules the contracts cannot express (`holding(x)`
  retracts `gripper_empty()` and `at(x,...)`; arriving at `y` retracts every
  other `reachable(...)`). `SymbolicPolicyExecutor` applies a grounding's
  effects and deletes on success. A `ScriptedFailure` is keyed by contract
  type and grounded target, not node id, so it also hits graphs a proposer
  authored; it names the attempts that fail and the facts the failed attempt
  changes anyway (a dropped object). `bind_symbolic_policy` gives every
  contract a ready `SymbolicPolicy` after its checkpoint bindings.
- `TaskController.run(goal, proposer, environment, executor, granularity,
  goal_facts)` is the loop:

```text
proposal = validator.plan(proposer, goal, PlanningContext.initial(...))
loop:
    absorb facts: the next sub-goal in line whose predicate holds is achieved;
                  its nodes are skipped
    node = SkillPlanner.decide(completed, failed)      # one node, None when done
    result = SkillRuntime.execute_node(graph, node.id, executor)
    success -> completed; failure -> retry while attempts remain, else failed
    NoViableCandidate -> Failure(sub-goal, node, mode, missing effects/preconditions),
                         validator.plan(proposer, goal, context.replan(...)),
                         swap both layers, derive achievement from the facts again
```

- Achievement is judged only for the sub-goal next in line, and stays
  recorded once granted. Judging every sub-goal against the current facts
  would let a transient predicate such as `reachable(x)` mark a later
  sub-goal done and skip nodes that are still needed.
- Success is judged on the scenario's `goal_facts`, independently of the
  decomposition. Statuses: `success`, `goal_not_reached`,
  `proposal_rejected`, `replans_exhausted`.
- `RunResult` is the trace (`as_dict()`, `save()`): proposals with requests
  and responses verbatim or their rejections, decisions with facts added and
  removed, replans with their failures, initial and final facts, metrics.
- Scenarios, each run at both granularities: `nominal`; `pick_fails_once`
  (retry within the attempt budget; these graphs have one candidate per
  role, so Layer-2 fallback stays covered by SetTable); `pick_exhausted`
  (Layer-1 replan through the scripted proposer, which gives the object up);
  `object_dropped` (the retry cannot be admitted because nothing is held, so
  the proposer plans the transfer again); `object_already_delivered` (the
  coarse sub-goal is skipped whole, the fine ones pick the object up and put
  it back). A SetTable-generic run covers open and close.
- Metrics per run: `subgoals`, `subgraphs`, `nodes`, `mean_nodes_per_subgraph`,
  `node_executions`, `failed_executions`, `admission_failures`, `retries`,
  `skipped_nodes`, `redundant_executions`, `achieved_subgoals`,
  `goal_facts_achieved`, `replans`, `replanning_span`, `recovery_success`.

Passing these on CPU is the acceptance test for the pipeline. Only after
that does a real model enter.

## Stage 5: the real model (DeepSeek API)

Package: `mshab/experiments/planning/` (`deepseek.py`, `prompts.py`, the
validator's retry rounds) and `mshab/experiments/granularity/evaluate.py`.

`DeepSeekProposer` fills `decompose()` and `plan_subgraph()` with one model
call each, through the DeepSeek chat-completions API (OpenAI-compatible;
key from `DEEPSEEK_API_KEY`, model configurable, `deepseek-chat` by default).
Decisions that kept the swap small, and how each was implemented:

- The user message is the request JSON, verbatim; a fixed system prompt per
  call (`prompts.py`) states the vocabulary, the world model, the validator's
  rules, the relation semantics, all three granularity instructions, and the
  response schema with an example. JSON output is requested
  (`response_format` `json_object`), fences and prose around it are tolerated
  by `extract_json_object()`, and everything after that is the strict
  `from_dict`; the model's JSON shape is never trusted.
- Text only. `PlanningContext.images` and `SubgraphRequest.images` exist for
  a later vision-capable proposer; the symbolic environment never fills them
  and `DeepSeekProposer` refuses a request that carries one.
- Retries live in the validator, because only it sees the rejections:
  `ProposalValidator(retries=2)` runs up to three rounds. A rejection that
  names a sub-goal repeats that sub-goal's subgraph call with the rejections
  attached to the request (`SubgraphRequest.rejections`), keeping the
  decomposition and every answer that was not refused; a rejection that
  names none (the decomposition, the assembled graph) restarts at call 1.
  `DeepSeekProposer` keeps one conversation per request fingerprint and turns
  the attached rejections into a further user turn, so the model sees its own
  refused answer. Every round, with its requests, responses, and rejections,
  is in `ValidatedProposal.rounds` / `ProposalRejected.rounds` and therefore
  in the run trace; the proposer's `exchanges` add the raw replies and
  token usage.
- Two validator rules were added for a model that can be wrong in ways the
  gold graphs never are: an achiever's grounded effects must contain the
  sub-goal predicate (else the environment could never verify the sub-goal),
  and the plan-stage "two achievers, no `FALLBACK_TO` order" rejection is
  attributed to its sub-goal so a retry repeats only that call.
- The transport is a callable from messages to a reply; `DeepSeekChat` is
  the real one and imports the `openai` SDK lazily. The dependency is the
  `planning` extra in `pyproject.toml`, installed in the Docker image; the
  compose file passes `DEEPSEEK_API_KEY` through from the host or a `.env`
  file. Tests inject a recording fake and never call the network.

Evaluation (`python -m mshab.experiments.granularity.evaluate`), per goal
(TidyHouse with its five scenarios, SetTable-generic with its nominal
scenario), per granularity setting (`free`, `coarse`, `fine`), over several
samples:

- validity rate: accepted on the first try, accepted after retries, rejected;
  rounds and proposer calls per proposal;
- decomposition statistics: sub-goal count, predicate vocabulary used,
  agreement with the coarse or fine gold sequence (exact match, order
  similarity, predicate precision and recall; `free` is compared with both);
- subgraph agreement: node set and edge set match against the gold subgraph
  for the same predicate, nodes matched by contract type and arguments; and
  the whole graph's role precision and recall, which is granularity-free;
- controller outcome on the stage-4 scenarios with the same injected
  failures, including replanning span and proposal retries;
- the most frequent rejections by stage and rule, which show which validator
  rule the prompt explains badly.

`--proposer scripted` runs the same sweep with the gold graphs as the
proposer; that offline dry run is what the tests exercise.

## Stage 6: MS-HAB rollout

Package: `mshab/experiments/rollout/` (episode, rule proposer, runner, and
the torch-dependent adapter and executor) plus `mshab/envs/skill_rollout.py`
(`SkillRollout-v0`). `TaskController`, `ProposalValidator`, `SkillRuntime`,
and the granularity library are reused unchanged, as the plan required; the
stage swaps in an adapter and an executor. See the package README for the
details; the decisions were:

- **One official episode is the scene.** A TidyHouse sequential plan (20
  subtasks: navigate, pick, navigate, place per transfer) is read as plain
  JSON. Objects are named by category so `pick(024_bowl)` grounds to the
  bowl checkpoint (a second instance is `024_bowl_2`); receptacles take their
  scene names from the episode config's `goal_receptacles`
  (`frl_apartment_table_01`, ...), checked against the plan's objects first,
  else `receptacle_<k>` by distinct goal rectangle. The goal text pairs every
  object with its receptacle, which the stage-5 smoke test showed a model
  needs. `TidyHouseEpisode` is standard library only and tested on CPU.
- **The policies observe the pointed subtask, so executing a node means
  pointing the environment at its plan subtask.** `TidyHouseEpisode.
  subtask_for` maps a grounding to a subtask index: the transfer's own for
  pick and place (a place to another receptacle than the plan's is
  `UnsupportedGrounding`, reported as the failure mode `no_matching_subtask`),
  the navigation before it for `navigate(object)` and `navigate(receptacle)`
  (a shared receptacle resolves to the transfer whose object is held). The
  official `SequentialTask-v0` advances the pointer and ends the episode by
  itself, so `SkillRollout-v0` subclasses it: the pointer moves only through
  `point_at`, `evaluate()` reports the pointed subtask's checkers as
  `subtask_success` and never advances, and `scene_measurements()` runs the
  grasp, place, and navigation checks for every object and goal of the plan
  so the whole scene becomes facts.
- **Facts are MS-HAB's own checkers.** `holding(x)` is the grasp check,
  `at(x,r)` the object inside its goal and not grasped, `reachable(x)`
  navigation success for that target without the arm terms,
  `gripper_empty()` when nothing is grasped, `collision_safe()` the pointed
  subtask's cumulative-force limit. Nothing is re-implemented: admission,
  invariant monitoring, and effect verification run in `SkillRuntime` on
  these facts, exactly as on the symbolic environment.
- **Two checks decide a node.** MS-HAB's subtask checker ends the policy run
  (arm at rest, robot still, object grasped or placed), then the runtime
  verifies the contract's effects on the facts. The horizon is the
  contract's, which the data-pipeline note said would replace the scripts'
  constants once execution went through the runtime. When the invariant
  monitor stops a skill for the force limit, the executor clears the force
  count and refreshes the facts before re-raising, so a retry is admitted
  with the clean slate an MS-HAB subtask starts with.
- **Layer 4 is the library's choice, restricted to the task.**
  `build_granularity_library(root, task_families=("tidy_house",))` binds the
  21 TidyHouse checkpoints; without the filter the PrepareGroceries
  checkpoint of the same object sorts first and would be selected.
  `CheckpointPolicyExecutor` loads SAC and PPO as `mshab.evaluate` does.
- **A rule-based pseudo model plays the scripted proposer's role.** The
  scripted tables cannot answer a replan they were not given, and a real
  episode's entities are in no table. `TidyHouseRuleProposer` decomposes the
  undelivered transfers (a held one first) through `TidyHouseGraphBuilder`
  with the transfers' original numbers, retries a failed transfer once and
  then drops it, never drops a transfer whose object is in the gripper, and
  shortens a coarse subgraph to `navigate -> place` when the object is
  already held. On the gold graphs' own transfers it reproduces them exactly.
  The real model runs through the same command line.

The runner is `python -m mshab.experiments.rollout`; `--dry-run` validates
the proposal and maps every node to its plan subtask without a simulator.
The first GPU runs (rule proposer, five episodes) executed the official
checkpoints through the controller: on plan 6 the place policy broke the
contact-force limit on every attempt, and the official evaluator fails the
same episode at the same subtask; plans 10, 21, and 42 delivered two to
four of five objects, with a dropped object picked up again after a Layer-1
replan and the fine decomposition skipping the sub-goals a replan found
already achieved. The package README records the numbers. Runs write the
trace, the executions with their simulator steps, the one-plan document the
environment loaded, and a video.

## Deliverables and order

| # | Deliverable | Depends on | Test |
| --- | --- | --- | --- |
| 1 | 53-row manifest, `build_granularity_library`, target-aware `select_policy`; duplicate stores removed | — | `test_granularity_library.py` |
| 2 | Coarse and fine TidyHouse gold graphs, SetTable regression graph, builders, renderer | 1 | `test_higher_layer_graphs.py` |
| 3 | Four request/response documents, `GraphProposer`, `ScriptedProposer`, `assemble_patch`, `ProposalValidator` | 2 | `test_planning_boundary.py` |
| 4 | `SymbolicEnvironmentAdapter`, `SymbolicPolicyExecutor`, `TaskController`, five scenarios, metrics | 3 | `test_task_controller.py` |
| 5 | `DeepSeekProposer`, prompts, validator retries, evaluation script | 4 | `test_deepseek_proposer.py`, `test_granularity_evaluation.py` (offline, fake transport) |
| 6 | `TidyHouseEpisode`, `SkillRollout-v0`, `RolloutEnvironmentAdapter`, `CheckpointPolicyExecutor`, `TidyHouseRuleProposer`, the rollout runner | 1, 4, 5 | `test_rollout.py` (CPU: episode, facts, mapping, rule proposer, dry run); the GPU runner for the simulator |

## Decisions taken

- The VLM authors both upper layers, in two calls: one decomposition, then
  one independent subgraph per sub-goal. Granularity is controlled in the
  decomposition call and is the experimental variable.
- TidyHouse (official MS-HAB task) is the primary graph source; SetTable
  re-expressed with generic contracts is the regression case.
- Aligning the granularity lower layers with `mshab.skills` is stage 1.
- The first real model is reached through the DeepSeek API, text only. The
  retry budget is the validator's, not the model's: rejections travel back
  on the request document, and the model answers them as a further turn.
- Contract ids keep the `mshab.granularity.<type>.all` namespace; the gold
  graphs are experiment-local test graphs.
- A skill node is one contract call, one MS-HAB atomic subtask. The gold
  graphs have one candidate node per role; node failure is handled by a
  controller retry budget, sub-goal failure by a Layer-1 replan.
- No portable Layer-3/4 export is built; nothing on the path of stages 1
  to 6 reads one.
- Vocabulary: a *proposer* generates Layers 1 and 2 (the scripted pseudo
  model, later the real model); *planner* stays reserved for
  `SkillPlanner`, the one-node decision over an existing graph. The skills
  docs already used "proposer" for the VLM's role.
- Gold-graph documents and `LibraryCatalog` stay separate. Later work
  builds on the experiment's documents; `LibraryCatalog` remains the
  SetTable test artifact.
- On MS-HAB a skill node is executed by pointing the official environment at
  the plan subtask it stands for; the environment's own checkers are the
  facts, its subtask success ends the policy run, and the contract's effects
  are verified on top. The official environment is subclassed
  (`SkillRollout-v0`) rather than driven through its state machine, because
  the pointer, the horizon, and task success are the runtime's decisions.
- The stage-6 pseudo model is rule-based, not a table: a real episode's
  entities and failures are not known in advance. It is a validation device
  for the adapter and executor, not a baseline planner.
