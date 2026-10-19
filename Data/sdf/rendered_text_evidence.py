"""Renderer diagnostics, independent of model and evaluation orchestration.

Re-rendering through noFreetype exposes SVG text nodes but does not reproduce
the original raster. These observations cannot authorize training revisions.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET


def no_freetype_text_diagnostic(mol, parameters: dict) -> dict:
    """Return atom-indexed SVG tokens with an explicit diagnostic-only status.

    ``labels`` maps integer RDKit atom indices to lists of text-node strings.
    Node order is retained, not normalized into a chemical formula. The input
    molecule is copied; this function does not modify it or load any datasets.
    """
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    prepared = rdMolDraw2D.PrepareMolForDrawing(
        Chem.Mol(mol), kekulize=parameters.get("kekulize", True),
        addChiralHs=False, wedgeBonds=True,
    )
    drawer = rdMolDraw2D.MolDraw2DSVG(768, 768, -1, -1, True)
    options = drawer.drawOptions()
    options.rotate = parameters.get("rotation_degrees", 0)
    options.bondLineWidth = parameters.get("bond_line_width", 2)
    options.fixedFontSize = parameters.get("font_size", 16)
    options.padding = .08
    options.multipleBondOffset = parameters.get("multiple_bond_offset", .16)
    options.addStereoAnnotation = False
    options.includeMetadata = False
    options.flagCloseContactsDist = -1
    drawer.DrawMolecule(prepared)
    drawer.FinishDrawing()
    labels = {}
    for element in ET.fromstring(drawer.GetDrawingText()).iter():
        identity = element.attrib.get("class", "")
        if element.tag.rsplit("}", 1)[-1] == "text" and identity.startswith("atom-"):
            labels.setdefault(int(identity[5:]), []).append("".join(element.itertext()).strip())
    return {
        "schema_version": "rendered-text-diagnostic-0.1",
        "evidence_kind": "nofreetype_diagnostic",
        "renderer": "rdkit-svg-noFreetype",
        "labels": labels,
        "same_source_raster": False,
        "revision_authorized": False,
        "limitation": "alternate text backend; not a pixel-identical source raster replay",
    }
