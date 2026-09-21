# Tests

CPU only, standard library only: no pytest, no torch, no simulator, no
install step. Run from the repository root; the whole suite takes about two
seconds. Folders mirror the packages they test.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v                 # everything (what CI runs)
python -m unittest discover -s tests/planning -t . -p 'test_*.py' -v   # one folder
python -m unittest tests.skills.test_set_table_graph_decisions -v      # one file
```

| Folder | File | What it pins down |
| --- | --- | --- |
| `skills/` `mshab/skills` | `test_skill_library.py` | The four layer boundaries, `ContractLibrary`, the committed SetTable catalog and SVG |
| | `test_skill_graph_semantics.py` | Layer-1/2 semantics: fallback chains, follow-ups, sealing, atomicity, schema |
| | `test_contract_state_transitions.py` | Contracts as state transitions: effects and deletes |
| | `test_set_table_graph_decisions.py` | Manual SetTable graph → one skill node per decision; official scene data when present |
| | `test_skill_library_checkpoints.py` | The downloaded SetTable checkpoints; skips without them |
| | `test_task_packages.py` | Public and compatibility imports of the task packages |
| `granularity/` `mshab/experiments/granularity`, `experiments/docs/figures` | `test_granularity_library.py` | Five generic contracts, 53 policies, target-aware policy selection, the library artifacts |
| | `test_higher_layer_graphs.py` | SetTable gold graphs: validity, grounding, coarse/fine equivalence, follow-ups, builder steps, committed JSON and SVG |
| | `test_granularity_evaluation.py` | Stage-5 sweep: agreement metrics, rejection tally, the scripted dry run |
| | `test_experiment_figures.py` | The committed figures regenerate from the committed data (the DeepSeek check skips while `extracts/` is empty) |
| `planning/` `mshab/experiments/planning` | `test_planning_boundary.py` | Documents, scripted proposer, assembler, validator rejections |
| | `test_task_controller.py` | Symbolic environment (conditional failures included) and the `execute -> observe -> re-decide` controller on the seven SetTable scenarios |
| | `test_deepseek_proposer.py` | The real model behind a fake transport: prompts, parsing, retry rounds |
| `rollout/` `mshab/experiments/rollout` | `test_rollout.py` | Stage 6 on CPU: SetTable episode from an official plan, facts (articulations included), node → subtask, rule proposer, dry run |

Conventions: every folder is a package so discovery recurses; a test that
needs downloaded assets skips when they are absent; committed artifacts
(catalog, gold graphs, figures) are compared byte for byte with their
generators, so regenerate them when a renderer changes; nothing imports
another test module.
