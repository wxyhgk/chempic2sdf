"""Human-readable proposal evidence; never a model input or data release."""

import html
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FOCUS = ('curr_8c1ec124155f2e0f__font_large', 'curr_ddc8766e5cc2a898__reference',
         'curr_85680a509563d789__font_large', 'curr_04a03f058eccd278__reference')


def overlay(output, case, png):
    with Image.open(png) as source:
        canvas = source.convert('RGB')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=13)
    by_slot = {c['slot']: c for c in case['candidates']}
    points = []
    for node in case['nodes']:
        for slot in node['unique_slots']:
            x, y = by_slot[slot]['pixel_xy']
            points.append((x, y))
            draw.ellipse((x-4, y-4, x+4, y+4), outline='#1565c0', width=2)
            draw.text((x+5, y+2), str(slot), fill='#1565c0', font=font)
    for pair in case['proposed_pairs']:
        x, y = by_slot[pair['fragment_slot']]['pixel_xy']
        points.append((x, y))
        rep = by_slot[pair['representative_slots'][0]]['pixel_xy']
        draw.line((x, y, *rep), fill='#986399', width=1)
        draw.rectangle((x-5, y-5, x+5, y+5), outline='#d01030', width=2)
        draw.text((x+6, y-15), str(pair['fragment_slot']), fill='#d01030', font=font)
    if points:
        crop = (max(0, int(min(x for x,y in points))-35), max(0, int(min(y for x,y in points))-35),
                min(canvas.width, int(max(x for x,y in points))+36), min(canvas.height, int(max(y for x,y in points))+36))
        canvas = canvas.crop(crop)
    # Preserve actual source pixels in the evidence; scale only for viewing.
    scale = min(3, 1000 / canvas.width)
    canvas = canvas.resize((round(canvas.width*scale), round(canvas.height*scale)))
    name = case['sample_key'] + '.png'
    canvas.save(Path(output) / name)
    return name


def page(output, cases):
    cards = []
    for case in cases:
        key = case['sample_key']
        picture = f'<img src="{key}.png" style="max-width:100%">' if key in FOCUS else ''
        rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in (
            p['fragment_slot'], p['owner_target'], ', '.join(p['kinds']),
            p['representative_slots'], p['acceptable_unique_slots'],
            [phase for phase, check in case['phase_checks'].items()
             if p['fragment_slot'] in check['removed_fragment_slots']])) + '</tr>' for p in case['proposed_pairs'])
        if not rows and not picture:
            continue
        cards.append(f'<section><h2>{html.escape(case["canonical_smiles"])}</h2>'
            f'<p>{html.escape(key)} · {html.escape(case["group"])}</p>{picture}'
            '<table><tr><th>片段槽</th><th>归属原子</th><th>位置证据</th><th>原始正例槽</th>'
            '<th>可代表该节点的唯一归属槽</th><th>原预测选中过的阶段</th></tr>' + rows + '</table>'
            f'<p><a href="{key}.json">全部候选、冲突和约束检查 JSON</a></p></section>')
    content = '<!doctype html><meta charset="utf-8"><title>节点归属监督提案</title>' \
        '<style>body{font:16px sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px}section{border-top:1px solid #aaa;margin-top:24px}</style>' \
        '<h1>化学节点归属：仅供复核，未授权训练</h1>' \
        '<p>只含原58张训练/回放图，不含64张开发图。提案来自真实目标和同源SVG，不是可部署的识别规则。</p>' \
        '<p>蓝圈=空间上可代表节点的唯一归属候选；红框=待复核笔画片段；紫线=与原始正例的归属配对。数字为候选槽号。</p>' \
        '<p>展示全部候选中的提案，包括原模型从未选中的低分点；数量不等于实际错误数。未决项不自动标负例。</p>' \
        '<p>各阶段只用于复核已有分数，没有新增预测；去掉片段的结果使用真实标注，不是模型准确率提升。</p>' + ''.join(cards)
    with (Path(output) / 'index.html').open('x', encoding='utf-8') as stream:
        stream.write(content)
