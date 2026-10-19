"""Two fixed training cases: source-pixel visual review, never model inference."""

from collections import deque
from hashlib import sha256
import io
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/audits/chemvector-pair-visual-review2-20260921'
TRIAL = ROOT / 'outputs/experiments/chemvector-pair-only-matched-trial-20260921'
CASES = (
    dict(key='curr_4ef506af64e8f0b8__bond_short', pair=(15, 18), targets=(34, 31),
         candidates=(15, 18, 41, 38), gt_nodes=(34, 38, 39, 31),
         controls=((34, 38), (38, 39)), crop=(353, 230, 492, 387)),
    dict(key='curr_08d02946e879406f__font_large', pair=(4, 6), targets=(7, 14),
         candidates=(4, 6, 12, 10), gt_nodes=(7, 8, 14, 16),
         controls=((14, 16), (7, 8)), crop=(289, 178, 570, 400)),
)


def read(path, hashes, expected=None):
    path = Path(path).absolute()
    if path.resolve() != path or 'sealed_test' in path.parts:
        raise ValueError('Only explicit, nonsymlink training inputs')
    data = path.read_bytes()
    digest = sha256(data).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('Input hash mismatch: ' + str(path))
    if str(path) in hashes and hashes[str(path)] != digest:
        raise ValueError('Input changed')
    hashes[str(path)] = digest
    return data


def shortest_path(bonds, start, end):
    adjacency = {}
    for bond in bonds:
        a, b = bond['begin_atom'], bond['end_atom']
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    queue, seen = deque([[start]]), {start}
    while queue:
        path = queue.popleft()
        if path[-1] == end:
            return path
        for node in adjacency.get(path[-1], []):
            if node not in seen:
                seen.add(node); queue.append(path + [node])
    return None


def probability(record, pair):
    z = record['presence_logits'][record['pairs'].index(sorted(pair))]
    return 1 / (1 + math.exp(-z))


def ribbon(image, a, b):
    """RGB illustration only: NOT the learned feature-map ribbon/activation."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    nx, ny = -dy / length, dx / length
    w, h = 448, 80
    affine = (dx * .92 / (w - 1), nx * 24 / (h - 1), a[0] + .04 * dx - 12 * nx,
              dy * .92 / (w - 1), ny * 24 / (h - 1), a[1] + .04 * dy - 12 * ny)
    return image.transform((w, h), Image.Transform.AFFINE, affine,
                           Image.Resampling.BILINEAR, fillcolor='white')


def render(source, spec, predicted, targets, bonds, probabilities):
    sheet = Image.new('RGB', (1440, 670), 'white')
    font, small = ImageFont.load_default(size=18), ImageFont.load_default(size=15)
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), spec['key'], fill='black', font=font)
    draw.text((12, 34), f"False edge probability: {probabilities[0]:.3f} -> {probabilities[1]:.3f}",
              fill='#b91c1c', font=font)
    crop = spec['crop']; width, height = crop[2] - crop[0], crop[3] - crop[1]
    scale = min(440 / width, 390 / height)
    tile = source.crop(crop).resize((round(width * scale), round(height * scale)), Image.Resampling.NEAREST)
    titles = ('1. Original pixels (unmarked)', '2. Saved candidates + false edge', '3. Ground-truth bonds (source labels)')
    for column, title in enumerate(titles):
        origin = (column * 480 + (480 - tile.width) // 2, 90)
        sheet.paste(tile, origin)
        draw.text((column * 480 + 12, 65), title, fill='black', font=small)
        def p(xy):
            return (origin[0] + (xy[0] - crop[0]) * scale, origin[1] + (xy[1] - crop[1]) * scale)
        if column == 1:
            a, b = [predicted[i]['xy'] for i in spec['pair']]
            dx, dy = b[0] - a[0], b[1] - a[1]; length = math.hypot(dx, dy)
            nx, ny = -dy / length * 12, dx / length * 12
            corners = [p((a[0] + t*dx + side*nx, a[1] + t*dy + side*ny))
                       for t, side in ((.04, -1), (.96, -1), (.96, 1), (.04, 1), (.04, -1))]
            draw.line(corners, fill='#0891b2', width=2)
            for i in range(0, 40, 2):
                draw.line([p((a[0] + t*dx, a[1] + t*dy)) for t in (i/40, (i+1)/40)],
                          fill='#dc2626', width=2)
            for slot in spec['candidates']:
                xy = p(predicted[slot]['xy']); color = '#dc2626' if slot in spec['pair'] else '#1d4ed8'
                draw.ellipse((xy[0]-5, xy[1]-5, xy[0]+5, xy[1]+5), outline=color, width=2)
                suffix = f" [{predicted[slot]['text']}]" if predicted[slot]['text'] else ''
                draw.text((xy[0]+8, xy[1]-10), f'P{slot}{suffix}', fill=color, font=small,
                          stroke_width=1, stroke_fill='white')
        elif column == 2:
            for bond in bonds:
                a, b = bond['begin_atom'], bond['end_atom']
                if a in spec['gt_nodes'] and b in spec['gt_nodes']:
                    draw.line((p(targets[a]['center']), p(targets[b]['center'])), fill='#16a34a', width=2)
            for target in spec['gt_nodes']:
                xy = p(targets[target]['center'])
                draw.ellipse((xy[0]-4, xy[1]-4, xy[0]+4, xy[1]+4), outline='#15803d', width=2)
                draw.text((xy[0]+8, xy[1]-10), f'T{target}', fill='#15803d', font=small,
                          stroke_width=1, stroke_fill='white')
    pairs = [(f'False P{spec["pair"][0]}-P{spec["pair"][1]}',
              *[predicted[i]['xy'] for i in spec['pair']])]
    pairs += [(f'True SINGLE T{a}-T{b}', targets[a]['center'], targets[b]['center'])
              for a, b in spec['controls']]
    for column, (label, a, b) in enumerate(pairs):
        draw.text((column*480+16, 500), label, fill='black', font=font)
        sheet.paste(ribbon(source, a, b), (column*480+16, 530))
        draw.rectangle((column*480+16, 530, column*480+464, 610), outline='#aaaaaa')
    draw.text((12, 625), 'RGB strips: longitudinal 4%-96%, transverse +/-12 px. Illustrations, NOT model features.',
              fill='#444444', font=small)
    draw.text((12, 648), 'Red dashed = nonexistent direct edge; cyan = approximate ribbon; green = true connections. P IDs != T IDs.',
              fill='#444444', font=small)
    return sheet


def main():
    hashes = {}
    manifest = json.loads(read(ROOT/'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json',
        hashes, 'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111'))
    post = json.loads(read(TRIAL/'postcheck.json', hashes,
        'b09b82c98fe5280de0d79a6cca858a2854398fa66dca1d29205997f4cece1f54'))
    read(Path(__file__), hashes)
    OUTPUT.mkdir(exist_ok=False)
    cases = []
    for spec in CASES:
        key = spec['key']; row = next(r for r in manifest['training_samples'] if r['sample_key'] == key)
        if row['split'] != 'train': raise ValueError('Fixed training cases only')
        folder = Path(manifest['sources'][row['source']])/'samples'
        meta = json.loads(read(folder/(key+'.meta.json'), hashes, row['meta_sha256']))
        png = read(folder/(key+'.png'), hashes, meta['sha256']['.png'])
        source = Image.open(io.BytesIO(png)).convert('RGB')
        if source.size != (768, 768): raise ValueError('Expected same-source 768 square raster')
        layers = json.loads(read(folder/(key+'.layers.json'), hashes, meta['sha256']['.layers.json']))
        records = [json.loads(read(TRIAL/(name := f'check-{step:06d}--{key}.json'), hashes,
                    post['artifact_sha256'][name])) for step in (0, 50)]
        before, after = records
        if before['metrics']['candidate_label_sha256'] != after['metrics']['candidate_label_sha256']:
            raise ValueError('Candidate lineage changed')
        trace = before['trace']
        predicted = {slot: dict(xy=[v*768/1023 for v in atom['anchor']],
                               text=None if atom['text'] is None else atom['text']['value'])
                     for slot, atom in zip(trace['compiled_candidate_indices'], trace['scene_before_fallback']['atom_objects'])}
        targets = {a['atom_id']: a for a in layers['atoms']}
        bonds = layers['bonds']
        bond_pairs = {tuple(sorted((b['begin_atom'], b['end_atom']))): b for b in bonds}
        if tuple(sorted(spec['targets'])) in bond_pairs: raise ValueError('Focus must be a real nonedge')
        for control in spec['controls']:
            if bond_pairs[tuple(sorted(control))]['chemical_type'] != 'SINGLE':
                raise ValueError('Expected single-bond visual control')
        probabilities = [probability(r, spec['pair']) for r in records]
        sheet = render(source, spec, predicted, targets, bonds, probabilities)
        path = OUTPUT/(key+'.png'); sheet.save(path)
        cases.append(dict(sample_key=key, source_png=str(folder/(key+'.png')),
            false_pair=spec['pair'], ground_truth_endpoints=spec['targets'],
            target_shortest_path=shortest_path(bonds, *spec['targets']),
            candidate_annotations={k:predicted[k] for k in spec['candidates']},
            target_annotations={k:targets[k] for k in spec['gt_nodes']},
            ground_truth_controls=spec['controls'], probability_initial_final=probabilities,
            crop=spec['crop'], figure=str(path), figure_sha256=sha256(path.read_bytes()).hexdigest()))
    for path, digest in list(hashes.items()): read(path, hashes, digest)
    result = dict(schema_version='pair-visual-review2-0.1', cases=cases, input_sha256=hashes,
        model_forwards=0, checkpoint_loads=0, optimizer_steps=0, gpu_used=False,
        note='GT-assisted source-pixel interpretation only; not attention/feature attribution or changed labels')
    with (OUTPUT/'report.json').open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False); stream.write('\n')
    print(json.dumps(dict(output=str(OUTPUT), samples=2, cases=[dict(sample_key=c['sample_key'],
        target_shortest_path=c['target_shortest_path']) for c in cases]), indent=2))


if __name__ == '__main__':
    main()
