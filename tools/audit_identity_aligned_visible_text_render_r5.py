"""Run the read-only render integrity audit on the strict r5 panel."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_identity_aligned_visible_text_render as audit  # noqa: E402


audit.DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
audit.OUT = ROOT / "outputs/audits/chemvector-identity-aligned-visible-text-render-20261017-r5/report.json"
audit.ALLOWED_REJECTION = "molecule_quality_rejected: Typography clearance would erase a bond segment"


if __name__ == "__main__":
    audit.run()
