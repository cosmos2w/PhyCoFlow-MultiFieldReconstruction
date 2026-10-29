#!/usr/bin/env python3
"""Collect the final vector masters without changing page geometry or content."""
from pathlib import Path
import json
import fitz

root = Path(__file__).resolve().parents[1]
figures = json.loads((root / 'metadata/report_figures.json').read_text())
atlas = fitz.open()
toc = []
for figure in figures:
    path = root / 'figures' / (figure['stem'] + '.pdf')
    with fitz.open(path) as source:
        assert len(source) == 1, path
        toc.append([1, f"Figure {figure['number']}: {figure['stem']}", len(atlas)+1])
        atlas.insert_pdf(source)
atlas.set_toc(toc)
atlas.save(root / 'FIGURE_ATLAS.pdf', garbage=4, deflate=True)
print(f'Collected {len(atlas)} unchanged vector master pages')
atlas.close()
