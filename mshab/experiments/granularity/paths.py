"""Default locations shared by the experiment's generators."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ASSET_ROOT = Path(
    os.environ.get("MS_ASSET_DIR", str(REPOSITORY_ROOT.parent / "mshab-assets"))
)
DEFAULT_CHECKPOINT_ROOT = DEFAULT_ASSET_ROOT / "data" / "mshab_checkpoints"
# The official task plans, spawn data, and episode configs of the rearrange
# dataset; the stage-6 rollout reads one SetTable episode from here.
DEFAULT_REARRANGE_ROOT = (
    DEFAULT_ASSET_ROOT / "data" / "scene_datasets" / "replica_cad_dataset" / "rearrange"
)
DEFAULT_ARTIFACT_DIR = PACKAGE_DIR / "artifacts"
DEFAULT_GRAPH_DIR = PACKAGE_DIR / "graphs"
# Evaluation runs: under $MSHAB_EXPS_DIR (the Docker image sets it to the
# bind-mounted ./mshab_exps, which is gitignored), never in the package.
DEFAULT_EXPERIMENTS_DIR = Path(
    os.environ.get("MSHAB_EXPS_DIR", str(REPOSITORY_ROOT / "mshab_exps"))
)
DEFAULT_EVALUATION_DIR = DEFAULT_EXPERIMENTS_DIR / "planning"
# MS-HAB rollouts of the controller (stage 6): one directory per run with
# the trace, the executions, the grounded task plan, and the video.
DEFAULT_ROLLOUT_DIR = DEFAULT_EXPERIMENTS_DIR / "rollout"
