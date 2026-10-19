"""One bound training-image review. Never an inference filter or label rewrite."""

from copy import deepcopy
from io import BytesIO
import math
import xml.etree.ElementTree as ET

import cairosvg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from chemvector.training.curriculum.pair_anchor_augmentation import OFFSETS

KEY = 'curr_d5ee3c0fcdc3c6c7__line_thick'
BOND_ID, ROW_ID = 12, 1523


def footprint(a, b, *, size=768):
    """Continuous ribbon raster footprint; NOT feature-grid receptive fields."""
    yy, xx = np.mgrid[:size, :size] + .5
    dx, dy = b[0]-a[0], b[1]-a[1]
    squared = dx*dx + dy*dy
    if squared <= 0: raise ValueError('Nonzero predicted segment required')
    fraction = ((xx-a[0])*dx + (yy-a[1])*dy)/squared
    distance = np.abs((xx-a[0])*dy-(yy-a[1])*dx)/math.sqrt(squared)
    return (fraction >= .04) & (fraction <= .96) & (distance <= 12.)


def polygon(a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    length = math.hypot(dx, dy)
    normal = (-dy/length*12, dx/length*12)
    begin, end = [(a[0]+dx*t, a[1]+dy*t) for t in (.04, .96)]
    return [(p[0]+sign*normal[0], p[1]+sign*normal[1])
            for p, sign in ((begin, 1), (end, 1), (end, -1), (begin, -1))]


def review(png, svg, layers, geometry, case):
    if case['sample_key'] != KEY or layers['sample_key'] != KEY:
        raise ValueError('Only the one previously flagged training case is in scope')
    row = next(r for r in case['sampled_rows'] if r['row_id'] == ROW_ID)
    if row['bond_id'] != BOND_ID or row['candidates'] != [39, 42] or row['owner_targets'] != [12, 13]:
        raise ValueError('Flagged candidate lineage changed')
    original = Image.open(BytesIO(png)).convert('RGB')
    if original.size != (768, 768): raise ValueError('Original 768px raster required')
    root = ET.fromstring(svg)
    if root.get('viewBox') != '0 0 768 768' or root.get('transform'):
        raise ValueError('Unexpected SVG coordinate contract')
    masks = {}
    for bond in layers['bonds']:
        # Re-render the original attributed vector nodes, including closed
        # wedge polygons. No segment approximation of thick/wedge strokes.
        selected = ET.Element(root.tag, root.attrib)
        ET.SubElement(selected, '{http://www.w3.org/2000/svg}rect',
                      {'width': '768', 'height': '768', 'fill': 'white'})
        for index in bond['svg_node_indices']:
            node = root[index]
            if f'bond-{bond["bond_id"]}' not in node.get('class', '').split() or node.get('transform'):
                raise ValueError('SVG/layer attribution mismatch')
            selected.append(deepcopy(node))
        rendered = cairosvg.svg2png(bytestring=ET.tostring(selected), output_width=768, output_height=768)
        gray = np.asarray(Image.open(BytesIO(rendered)).convert('L'))
        masks[bond['bond_id']] = gray < 192
    source_pixels = np.asarray(original.convert('L')) < 192
    raw = {c['slot']: c for c in geometry['candidates']}
    anchors = torch.tensor([raw[s]['geometry'][:2] for s in row['candidates']], dtype=torch.float32)
    targets = np.asarray(geometry['target_anchors'], dtype=np.float64) * 768
    own = masks[BOND_ID]
    views = []
    for index, offset in enumerate(OFFSETS):
        shifted = anchors + torch.tensor(offset, dtype=torch.float32)/768
        a, b = (shifted.double()*768).tolist()
        ribbon = footprint(a, b)
        delta = np.asarray(b)-a
        squared = float(delta @ delta)
        relative = targets-a
        fraction = relative @ delta/squared
        transverse = np.abs(relative[:, 0]*delta[1]-relative[:, 1]*delta[0])/math.sqrt(squared)
        nodes = [dict(atom_id=i, element=layers['atoms'][i]['element'],
                      longitudinal_fraction=float(fraction[i]), transverse_pixels=float(transverse[i]))
                 for i in range(len(targets)) if i not in row['owner_targets']
                 and .04 <= fraction[i] <= .96 and transverse[i] <= 12.]
        foreign = []
        for bond in layers['bonds']:
            bid = bond['bond_id']
            count = int((ribbon & masks[bid]).sum())
            if bid == BOND_ID or not count: continue
            foreign.append(dict(bond_id=bid, pixels=count, chemical_type=bond['chemical_type'],
                drawing_type=bond['drawing']['bond_type'], direction=bond['drawing']['direction'],
                shares_target_endpoint=bool(set((bond['begin_atom'], bond['end_atom'])) & set(row['owner_targets']))))
        eligible = ROW_ID in case['views'][index]['row_ids']
        if bool(nodes) != (ROW_ID in case['views'][index]['ribbon_other_node_row_ids']) and eligible:
            raise ValueError('Original geometry warning did not reproduce')
        views.append(dict(view_index=index, offset_pixels=list(offset), eligible=eligible,
            predicted_endpoints_pixels=[a, b], non_endpoint_nodes=nodes,
            target_ink_pixels_inside=int((ribbon & own).sum()), target_ink_pixels_total=int(own.sum()),
            input_ink_pixels_inside=int((ribbon & source_pixels).sum()), foreign_bond_intersections=foreign))

    # Original plus baseline and BOTH flagged shifts. The numerical report
    # retains all nine views; panels do not select a best scoring prediction.
    xy = anchors.double().numpy()*768
    box = (int(np.floor(xy[:, 0].min()-32)), int(np.floor(xy[:, 1].min()-32)),
           int(np.ceil(xy[:, 0].max()+32)), int(np.ceil(xy[:, 1].max()+32)))
    scale = 4
    crop = original.crop(box).resize(((box[2]-box[0])*scale, (box[3]-box[1])*scale), Image.Resampling.NEAREST)
    width, height = crop.size
    canvas = Image.new('RGB', (4*width, height+54), 'white')
    font = ImageFont.truetype('DejaVuSans.ttf', 14)
    for panel, view_index in enumerate((None, 0, 1, 4)):
        tile = crop.copy()
        title = 'Original raster'
        if view_index is not None:
            view = views[view_index]
            draw = ImageDraw.Draw(tile)
            transform = lambda p: ((p[0]-box[0])*scale, (p[1]-box[1])*scale)
            a, b = view['predicted_endpoints_pixels']
            draw.line([transform(p) for p in polygon(a, b)] + [transform(polygon(a, b)[0])], fill='#0066cc', width=2)
            draw.line([transform(a), transform(b)], fill='#dd3300', width=2)
            for node in view['non_endpoint_nodes']:
                px, py = transform(targets[node['atom_id']])
                draw.ellipse((px-5, py-5, px+5, py+5), outline='#aa00aa', width=2)
                draw.text((px+6, py), str(node['atom_id']), fill='#aa00aa', font=font)
            title = f'Offset {view["offset_pixels"]}; nodes {[n["atom_id"] for n in view["non_endpoint_nodes"]]}'
        canvas.paste(tile, (panel*width, 54))
        ImageDraw.Draw(canvas).text((panel*width+4, 10), title, fill='black', font=font)
    return dict(sample_key=KEY, row_id=ROW_ID, bond_id=BOND_ID, crop_box=list(box),
        footprint_convention='pixel centers at x+.5,y+.5; interior 4%-96%; transverse +/-12px',
        isolated_svg_ink_threshold=192, views=views,
        labels_changed=False, samples_removed=False, model_calls=0,
        limitations=['Continuous raster footprint is not exact discrete sampling or the learned feature receptive field',
            'Isolated SVG attribution supports visible-stroke review; antialiasing may differ from the saved raster',
            'Third-party stroke overlap is not proof that the pair label is wrong or the model is confused',
            'Original full raster, crop and every view are diagnostic only; no performance claim']), canvas
