"""Local, source-PNG overlays only; not a renderer input to any model."""

import html
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FOCUS = ('curr_8c1ec124155f2e0f__font_large', 'curr_ddc8766e5cc2a898__reference',
         'curr_85680a509563d789__font_large', 'curr_04a03f058eccd278__reference',
         'curr_6828c4b28818df70__reference', 'curr_81d885d92a64ee47__line_thin',
         'curr_b3cdc70414238789__font_small')


def panel(image, anchors, states, label, crop):
    tile = image.crop(crop).convert('RGB')
    scale = min(440 / tile.width, 350 / tile.height)
    tile = tile.resize((max(1, round(tile.width * scale)), max(1, round(tile.height * scale))))
    canvas = Image.new('RGB', (460, 410), 'white')
    offset = ((460 - tile.width) // 2, 45 + (350 - tile.height) // 2)
    canvas.paste(tile, offset)
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), label, fill='black', font=ImageFont.load_default(size=16))
    def pixel(xy):
        return ((xy[0] - crop[0]) * scale + offset[0], (xy[1] - crop[1]) * scale + offset[1])
    for anchor in anchors:
        x, y = pixel([v * 768 for v in anchor])
        draw.line((x-5, y, x+5, y), fill='#008040', width=2)
        draw.line((x, y-5, x, y+5), fill='#008040', width=2)
    for row in states:
        x, y = pixel(row['pixel_xy'])
        if not row['selected']:
            color, radius = '#c07800', 8
            draw.rectangle((x-radius, y-radius, x+radius, y+radius), outline=color, width=2)
        elif row['selected_target'] >= 0:
            color, radius = '#1565c0', 5
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=color, width=2)
        else:
            color = '#d01030'
            draw.line((x-6,y-6,x+6,y+6), fill=color, width=2)
            draw.line((x-6,y+6,x+6,y-6), fill=color, width=2)
        draw.text((x+7, y-12), str(row['slot']), fill=color, font=ImageFont.load_default(size=12))
    return canvas


def render(output, case, source_png):
    image = Image.open(source_png).convert('RGB')
    points = [[v*768 for v in a] for a in case['target_anchors']]
    points += [r['pixel_xy'] for phase in case['phases'].values() for r in phase['candidates']]
    crop = (max(0, int(min(p[0] for p in points))-32), max(0, int(min(p[1] for p in points))-32),
            min(768, int(max(p[0] for p in points))+33), min(768, int(max(p[1] for p in points))+33))
    phases = list(case['phases'])
    canvas = Image.new('RGB', (460*len(phases), 440), 'white')
    for i, phase in enumerate(phases):
        value = case['phases'][phase]
        canvas.paste(panel(image, case['target_anchors'], value['candidates'], phase, crop), (i*460, 30))
    ImageDraw.Draw(canvas).text((8, 8), case['canonical_smiles'], fill='black', font=ImageFont.load_default(size=18))
    name = case['sample_key'] + '.png'
    canvas.save(Path(output) / name)
    return name


def index(output, entries):
    cards = ''.join(f'<article><h2>{html.escape(c["canonical_smiles"])}</h2>'
        f'<p>{html.escape(c["sample_key"])}</p><a href="{name}"><img src="{name}"></a>'
        f'<p><a href="{c["sample_key"]}.json">逐候选证据 JSON</a></p></article>' for c, name in entries)
    page = '<!doctype html><meta charset="utf-8"><title>化学节点与笔画候选审计</title>' \
        '<style>body{font:16px sans-serif;margin:24px;max-width:1900px}img{width:100%}article{border-top:1px solid #bbb}</style>' \
        '<h1>化学节点与笔画候选：固定反例</h1>' \
        '<p>绿色十字=目标原子锚点；蓝圈=选中且匹配；红叉=选中但未匹配；橙框=被拒绝的 raw 正例。数字为候选槽号。</p>' \
        '<p>此图使用真实目标及同源 SVG 作诊断解释，不是部署识别结果。raw 正例也可能受旧匹配契约影响而未匹配。</p>' \
        '<p>只有 retention-200 与 control-200 的训练/回放图可作等步比较；retention-400 与 control-200 开发图不可作等步因果比较。</p>' \
        '<p>文字框命中不等于笔画像素命中，也不意味着应删除该候选。未解析的 SVG 曲线保留为未知。</p>' + cards
    (Path(output) / 'index.html').write_text(page, encoding='utf-8')
