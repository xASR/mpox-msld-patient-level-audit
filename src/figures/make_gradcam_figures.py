"""Rebuild manuscript Figure 6 and seven Figure S1 pages from saved Grad-CAM outputs.

This performs layout only; it does not train a model, recompute maps or crop
source photographs. Use the recorded Arial font files for identical PNG output.
"""
from pathlib import Path
import argparse
import math
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gradcam',type=Path,required=True,help='Complete saved gradcam folder')
    parser.add_argument('--out',type=Path,required=True,help='Destination for Figure 6 and Figure_S1_pages')
    parser.add_argument('--font-regular',type=Path,required=True,help='Arial regular TTF used for the manuscript')
    parser.add_argument('--font-bold',type=Path,required=True,help='Arial bold TTF used for the manuscript')
    args=parser.parse_args()
    SRC=args.gradcam
    OUT=args.out
    OUT.mkdir(parents=True,exist_ok=True)
    sel = pd.read_csv(SRC / 'selected_images.csv')
    idx = pd.read_csv(SRC / 'gradcam_index.csv')
    names = ['Healthy', 'Cowpox', 'Mpox', 'Measles', 'Chickenpox', 'HFMD']
    font_bold = ImageFont.truetype(str(args.font_bold), 34)
    font_text = ImageFont.truetype(str(args.font_regular), 29)
    font_small = ImageFont.truetype(str(args.font_regular), 27)

    def choose(group, ascending=False):
        return group.sort_values(['conf', 'filename'], ascending=[ascending, True]).iloc[0]

    cases = [
        (choose(sel[(sel['true'] == 0) & sel['correct']], ascending=True), 'lowest-confidence correct Healthy image'),
        (choose(sel[(sel['true'] == 1) & sel['correct']]), 'highest-confidence correct Cowpox image'),
        (choose(sel[(sel['true'] == 2) & (sel['pred'] == 4)]), 'highest-confidence mpox-to-chickenpox error'),
        (choose(sel[(sel['true'] == 3) & sel['correct']]), 'highest-confidence correct Measles image'),
        (choose(sel[(sel['true'] == 4) & (sel['pred'] == 2)]), 'highest-confidence chickenpox-to-mpox error'),
        (choose(sel[(sel['true'] == 5) & sel['correct']]), 'highest-confidence correct HFMD image'),
    ]

    TILE = 360
    COL_GAP = 20
    GROUP_WIDTH = 3 * TILE + 2 * COL_GAP
    SIDE = 40
    GROUP_GAP = 70
    GROUP_HEIGHT = 515
    WIDTH = 2 * SIDE + 2 * GROUP_WIDTH + GROUP_GAP

    def centered(draw, text, box, font, fill='black'):
        x0, y0, x1, y1 = box
        lines = text.split('\n')
        boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        line_height = max(b[3] - b[1] for b in boxes) + 12
        total = len(lines) * line_height
        for i, (line, b) in enumerate(zip(lines, boxes)):
            draw.text(((x0 + x1 - (b[2] - b[0])) / 2, y0 + (y1 - y0 - total) / 2 + i * line_height),
                      line, fill=fill, font=font)

    def draw_case(canvas, row, position, label):
        column, grid_row = position % 2, position // 2
        left = SIDE + column * (GROUP_WIDTH + GROUP_GAP)
        top = 80 + grid_row * GROUP_HEIGHT
        draw = ImageDraw.Draw(canvas)
        title = f'({label}) {Path(row["filename"]).stem}'
        draw.text((left, top), title, font=font_bold, fill='black')
        subtitle = f'Dataset: {names[int(row["true"])]} | Predicted: {names[int(row["pred"])]} | p = {row["conf"]:.3f}'
        draw.text((left, top + 46), subtitle, font=font_text, fill='#333333')
        headers = ['Original image', 'Dataset-class map', 'Predicted-class map']
        targets = ['original', 'true', 'pred']
        for c, (header, target) in enumerate(zip(headers, targets)):
            x = left + c * (TILE + COL_GAP)
            centered(draw, header, (x, top + 91, x + TILE, top + 132), font_small)
            y = top + 139
            if target == 'original':
                filename = f'fold{int(row["fold"])}_{Path(row["filename"]).stem}_original.png'
            elif target == 'pred' and row['correct']:
                draw.rectangle((x, y, x + TILE - 1, y + TILE - 1), fill='#f6f6f6', outline='#dddddd')
                centered(draw, 'Same class\nSee middle panel', (x + 12, y + 12, x + TILE - 12, y + TILE - 12), font_text, '#666666')
                continue
            else:
                matches = idx[(idx['filename'] == row['filename']) & (idx['explained'] == target)]
                assert len(matches) == 1
                filename = matches.iloc[0]['overlay']
            with Image.open(SRC / filename) as source:
                # Layout only: preserve the whole source image and its aspect ratio.
                image = ImageOps.contain(source.convert('RGB'), (TILE, TILE), Image.Resampling.LANCZOS)
                canvas.paste(image, (x + (TILE - image.width) // 2, y + (TILE - image.height) // 2))

    def montage(rows, title, destination, labels=None):
        grid_rows = math.ceil(len(rows) / 2)
        canvas = Image.new('RGB', (WIDTH, 80 + grid_rows * GROUP_HEIGHT + 55), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((SIDE, 20), title, font=font_bold, fill='black')
        for i, row in enumerate(rows):
            draw_case(canvas, row, i, labels[i] if labels else str(i + 1))
        footer = 'ConvNeXt-Tiny | Relative Grad-CAM intensity normalised separately for each map'
        draw.text((SIDE, canvas.height - 44), footer, font=font_small, fill='#444444')
        canvas.save(destination, dpi=(300, 300))

    montage([row for row, why in cases], 'Figure 6. Selected ConvNeXt-Tiny Grad-CAM examples',
            OUT / 'Figure6_gradcam_examples.png', labels=list('abcdef'))
    manifest = []
    for label, (row, why) in zip('abcdef', cases):
        manifest.append(dict(panel=label, filename=row['filename'], patient_id=row['patient_id'], fold=int(row['fold']),
                             dataset_class=names[int(row['true'])], predicted_class=names[int(row['pred'])],
                             predicted_probability=float(row['conf']), subset_rule=why,
                             original_gallery_reason=row['reason']))
    pd.DataFrame(manifest).to_csv(OUT / 'Figure6_panel_manifest.csv', index=False)

    pages = OUT / 'Figure_S1_pages'
    pages.mkdir(exist_ok=True)
    page_count = math.ceil(len(sel) / 6)
    for page in range(page_count):
        frame = sel.iloc[page * 6:(page + 1) * 6]
        montage([row for _, row in frame.iterrows()], f'Figure S1. Full selected gallery, page {page + 1} of {page_count}',
                pages / f'Figure_S1_page_{page + 1:02}.png', labels=[str(i + 1) for i in frame.index])
    print('Rendered Figure 6 and seven readable Figure S1 pages from unchanged source images and overlays.')
    print(pd.DataFrame(manifest).to_string(index=False))

if __name__=='__main__':
    main()
