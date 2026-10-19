"""Run the frozen-initial raw reachability replay on the r5 panel."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_identity_aligned_raw_reachability as replay  # noqa: E402


replay.DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
replay.RENDER_AUDIT = ROOT / "outputs/audits/chemvector-identity-aligned-visible-text-render-20261017-r5/report.json"
replay.OUT = ROOT / "outputs/audits/chemvector-identity-aligned-raw-reachability-20261017-r5/report.json"


if __name__ == "__main__":
    replay.run()
