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
DEFAULT_ARTIFACT_DIR = PACKAGE_DIR / "artifacts"
DEFAULT_GRAPH_DIR = PACKAGE_DIR / "graphs"
