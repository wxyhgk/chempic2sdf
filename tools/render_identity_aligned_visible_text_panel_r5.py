"""Render the r5 measured-geometry panel into quarantine outputs."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import prepare_identity_aligned_visible_text_panel_r5  # noqa: F401,E402
from tools import render_identity_aligned_visible_text_panel as renderer  # noqa: E402


renderer.OUTPUT = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
renderer.STRICT_MOLECULES = True


if __name__ == "__main__":
    renderer.run()
