"""New font-capable renderer adapter; historical renderer files stay frozen.

Reuse vector attribution, stereo-aware layout and strict same-SVG masks. Only
the new adapter supports the four preregistered fonts. No old pixels are edited.
"""

from dataclasses import asdict
from hashlib import sha256
import io
import json
import math
import xml.etree.ElementTree as ET

import cairocffi
import cairosvg
import numpy as np
from PIL import Image, ImageChops
from rdkit import Chem, rdBase

from Data.sdf.chemvector_curriculum_v0_1.rendering import prepare_drawing
from Data.sdf.chemvector_curriculum_v0_1.stereo_typography import adjust
from Data.sdf.chemvector_curriculum_v0_1.functional_groups import annotate_groups
from Data.sdf.text_source_renderer.layers import json_bytes
from Data.sdf.text_source_renderer.svg_records import parse_style, local_name
from Data.sdf.text_source_renderer.v2.layers import compile_layers
from .profiles import font_record


def render(entry, profile):
    font = font_record(profile.font_family)
    if entry['split'] != 'quarantine' or entry['training_authorized'] or entry['external_overlap_status'] != 'unchecked':
        raise ValueError('New cases must remain quarantined')
    (mol, original_block, sdf, chemistry, directions, anchors,
     svg, glyphs, bonds, auxiliary) = prepare_drawing(entry, profile, typography='stereo-aware-v1')
    root = ET.fromstring(svg)
    for glyph in glyphs:
        node = root[glyph['node_index']]; style = parse_style(node.attrib['style'])
        style['font-family'] = profile.font_family
        node.set('style', ';'.join(f'{k}:{v}' for k, v in style.items()))
        glyph['svg_attributes'] = dict(node.attrib)
    svg = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    svg, typography = adjust(svg, glyphs, bonds, chemistry, anchors, directions, width=768, height=768)
    sample_key = entry['molecule_id'] + '__' + profile.profile_id
    provenance = dict(renderer_id='case-diversity-same-svg-1', config=asdict(profile), font=font,
        rdkit_version=rdBase.rdkitVersion, cairosvg_version=cairosvg.__version__, cairo_version=cairocffi.cairo_version_string(),
        layout='rdkit_compute2d_canonical_force_rdkit', typography=typography,
        resize_after_render=False, legacy_pixel_equivalent=False, scale_bond_width=False,
        bond_length_parameter_kind='RDKit fixedBondLength cap; see measured anchors')
    files = compile_layers(svg, glyphs, bonds, auxiliary, chemistry, anchors, directions,
                           width=768, height=768, sample_key=sample_key, provenance=provenance)
    layers = json.loads(files['.layers.json'])
    layers.update(split='quarantine', molecule_id=entry['molecule_id'], profile_id=profile.profile_id,
                  external_overlap_status='unchecked', training_authorized=False)
    files['.layers.json'] = json_bytes(layers)
    records = json.loads(files['.text-records.json'])
    distances = [math.dist(anchors[b['begin_atom']], anchors[b['end_atom']]) for b in chemistry['bonds']]
    if any(not 0 < x < 768 or not 0 < y < 768 for x, y in anchors) or any(not math.isfinite(d) or d <= 0 for d in distances):
        raise ValueError('Invalid anchor geometry')
    with Image.open(io.BytesIO(files['.png'])) as image:
        foreground = ImageChops.difference(image, Image.new('RGB', image.size, 'white')).getbbox()
        foreground_pixels = int((np.asarray(image).min(axis=2) < 255).sum())
    root = ET.fromstring(svg)
    measured = dict(glyph_ink_heights_px=[g['bbox'][3] - g['bbox'][1] for g in records['glyphs']],
        glyph_declared_font_sizes_px=[g['font_size'] for g in records['glyphs']],
        atom_text_ink_heights_px=[a['bbox'][3] - a['bbox'][1] for a in layers['atoms'] if a['bbox']],
        bond_anchor_distances_px=distances,
        stroke_widths_px=sorted({float(parse_style(n.attrib['style'])['stroke-width'].removesuffix('px'))
                                for n in root if local_name(n) == 'path'}),
        foreground_bbox=list(foreground), foreground_pixels=foreground_pixels,
        visible_atom_labels=sum(a['visible_text'] for a in layers['atoms']),
        implicit_carbon_atoms=sum(a['element'] == 'C' and not a['visible_text'] for a in layers['atoms']))
    groups = annotate_groups(mol)
    graph = dict(schema_version='curriculum-chemical-graph-0.1', molecule_id=entry['molecule_id'],
        canonical_smiles=entry['canonical_smiles'], atom_id_convention='zero-based SDF atom order',
        atoms=chemistry['atoms'], bonds=chemistry['bonds'], functional_groups=groups,
        functional_groups_role='overlapping audit metadata, not supernodes or model targets',
        pixel_geometry='see layers.json; SDF uses source unrotated 2D coordinates',
        split='quarantine', training_authorized=False, external_overlap_status='unchecked')
    files['.graph.json'], files['.sdf'] = json_bytes(graph), sdf
    meta = dict(schema_version='curriculum-sample-0.1', **entry, sample_key=sample_key,
        profile_id=profile.profile_id, profile=asdict(profile), measured=measured, functional_groups=groups,
        same_source_raster=True,
        validated=['sanitized_chemistry', 'sdf_identity_and_index_roundtrip', 'same_svg_foreground_coverage',
                   'no_glyph_or_text_geometry_overlap', 'no_attributed_ink_clipping'],
        sha256={suffix: sha256(data).hexdigest() for suffix, data in files.items()})
    files['.meta.json'] = json_bytes(meta)
    if Chem.MolToMolBlock(mol) != original_block: raise ValueError('Rendering mutated source chemistry')
    return files, meta
