"""Display the exact frozen 200 development inputs; no inference or data reselection."""

import argparse
import base64
from collections import Counter
import hashlib
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
COMPARISON = Path('outputs/experiments/profile-comparison-20260914')
SNAPSHOT = ROOT / 'outputs/baselines/chemvector-v01016-reference-20260914/repository'
PROTOCOL_SHA = '565a236efd04de73b90f941ef8d19f373ade478bfaf175fd52dab6be6bcd04b7'
VARIANTS = ('coordgen', 'original', 'rdkit', 'stress')


def build(output):
    protocol_bytes = (SNAPSHOT / COMPARISON / 'protocol.json').read_bytes()
    if hashlib.sha256(protocol_bytes).hexdigest() != PROTOCOL_SHA:
        raise ValueError('Frozen input protocol changed')
    protocol = json.loads(protocol_bytes)
    keys = protocol['sample_order']
    if len(keys) != 200 or len(set(keys)) != 200:
        raise ValueError('Expected exactly 200 distinct input samples')
    molecules = Counter(protocol['sample_metadata'][key]['molecule_id'] for key in keys)
    if len(molecules) != 50 or set(molecules.values()) != {4}:
        raise ValueError('Expected exactly 50 molecules with four images each')
    for start in range(0, 200, 4):
        group = [protocol['sample_metadata'][key] for key in keys[start:start + 4]]
        if tuple(row['variant'] for row in group) != VARIANTS:
            raise ValueError('Unexpected input variant ordering')
        if len({row['molecule_id'] for row in group}) != 1:
            raise ValueError('Row must contain one molecule')

    records, pictures, figures = [], [], []
    for number, key in enumerate(keys, 1):
        if Path(key).name != key:
            raise ValueError('Unsafe sample key')
        source = SNAPSHOT / COMPARISON / 'images' / f'{key}.png'
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        metadata = protocol['sample_metadata'][key]
        if digest != metadata['png_sha256']:
            raise ValueError(f'Input image changed: {key}')
        with Image.open(io.BytesIO(raw)) as opened:
            rgba = opened.convert('RGBA')
            picture = Image.new('RGBA', rgba.size, 'white')
            picture.alpha_composite(rgba)
            pictures.append(picture.convert('RGB'))
        label = f'{number:03d} | {key}'
        uri = 'data:image/png;base64,' + base64.b64encode(raw).decode('ascii')
        figures.append(
            f'<figure><button type="button" aria-label="放大 {html.escape(label)}">'
            f'<img loading="lazy" src="{uri}" alt="{html.escape(label)}"></button>'
            f'<figcaption>{html.escape(label)}<br>{picture.width} × {picture.height} px · '
            f'{html.escape(metadata["molecule_id"])}</figcaption></figure>'
        )
        records.append({'number': number, 'sample_key': key, 'source': str(source.relative_to(ROOT)),
                        'sha256': digest, 'width': picture.width, 'height': picture.height,
                        'molecule_id': metadata['molecule_id'], 'variant': metadata['variant']})

    output.mkdir(parents=True, exist_ok=False)
    font_path = '/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf'
    font = ImageFont.truetype(font_path, 17)
    title_font = ImageFont.truetype(font_path, 27)
    sheet_names = []
    cell_width, cell_height, margin, top = 400, 360, 24, 100
    for page in range(10):
        sheet = Image.new('RGB', (margin * 2 + cell_width * 4, top + cell_height * 5 + margin), '#edf0f4')
        draw = ImageDraw.Draw(sheet)
        draw.text((margin, 15), f'Frozen development inputs | {page * 20 + 1:03d}-{page * 20 + 20:03d} / 200',
                  font=title_font, fill='#142132')
        draw.text((margin, 52), 'Each row: one molecule. Actual input PNGs; no predictions; no cropping.', font=font, fill='#435166')
        for column, variant in enumerate(VARIANTS):
            draw.text((margin + column * cell_width + 8, 77), variant, font=font, fill='#142132')
        for slot in range(20):
            index = page * 20 + slot
            x = margin + (slot % 4) * cell_width
            y = top + (slot // 4) * cell_height
            draw.rectangle((x + 3, y + 3, x + cell_width - 5, y + cell_height - 5), fill='white')
            thumbnail = ImageOps.contain(pictures[index], (cell_width - 24, cell_height - 56), Image.Resampling.LANCZOS)
            sheet.paste(thumbnail, (x + (cell_width - thumbnail.width) // 2, y + 8 + (cell_height - 56 - thumbnail.height) // 2))
            draw.text((x + 10, y + cell_height - 38), f'{index + 1:03d} | {keys[index]}', font=font, fill='#142132')
        name = f'page-{page + 1:02d}.png'
        sheet.save(output / name)
        sheet_names.append(name)

    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>固定开发集：200 张实际输入图</title>
<style>
body{margin:24px;font:16px system-ui,sans-serif;color:#142132;background:#edf0f4}
h1{font-size:24px}header{max-width:1100px}a{color:#185ea8}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
figure{margin:0;padding:10px;background:white;border-radius:8px;min-width:0}
figure button{display:block;width:100%;padding:0;background:white;border:0;cursor:zoom-in}
figure img{display:block;width:100%;height:320px;object-fit:contain}
figcaption{font-size:12px;overflow-wrap:anywhere;margin-top:8px}
dialog{max-width:95vw;max-height:95vh;border:1px solid #ccd3dd;border-radius:8px}
dialog::backdrop{background:#000b}dialog img{display:block;max-width:88vw;max-height:80vh;object-fit:contain}
dialog button{float:right;padding:8px 16px}dialog p{clear:both}
.scroll{overflow:auto}.grid{min-width:900px}
</style><header><h1>200 张实际输入图（50 个分子 × 4 个版本）</h1>
<p>每行是同一个分子；从左到右：CoordGen、original、RDKit、stress。
编号沿用固定评测顺序。这里是输入图，不是模型预测，也不是 200 个不同分子。</p>
<p>点击图片查看完整分辨率；预览不裁剪。此页内嵌全部原 PNG，可离线浏览。
这些图片来自已反复使用的开发对照，不是独立测试集。</p>
<nav>分页总览：NAVIGATION</nav></header><hr><div class="scroll"><main class="grid">FIGURES</main></div>
<dialog><button type="button">关闭（Esc）</button><p></p><img alt=""></dialog>
<script>
const dialog=document.querySelector('dialog');
document.querySelectorAll('figure button').forEach(button=>button.addEventListener('click',()=>{
 const source=button.querySelector('img');
 const target=dialog.querySelector('img');target.src=source.src;target.alt=source.alt;
 dialog.querySelector('p').textContent=button.parentElement.querySelector('figcaption').textContent;
 dialog.showModal();
}));
dialog.querySelector('button').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close();});
</script></html>'''
    navigation = ' · '.join(f'<a href="{name}">{i * 20 + 1:03d}–{i * 20 + 20:03d}</a>' for i, name in enumerate(sheet_names))
    document = document.replace('NAVIGATION', navigation).replace('FIGURES', '\n'.join(figures))
    (output / 'index.html').write_text(document, encoding='utf-8')
    manifest = {'protocol_sha256': PROTOCOL_SHA, 'input_count': len(records), 'molecule_count': len(molecules),
                'source': 'frozen comparison input PNGs, not predictions', 'inputs': records, 'sheets': sheet_names}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'gallery': str(output / 'index.html'), 'images_verified': len(records), 'pages': len(sheet_names)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    build(parser.parse_args().output_dir)
