#!/usr/bin/env python3
"""Render a report preview with PyMuPDF from the optional paper dependencies."""
import sys
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from PIL import Image
import fitz

html=Path(sys.argv[1]).resolve()
output=Path(sys.argv[2]).resolve()
source=html.read_text()
def fit_image(match):
    tag = match.group(0)
    src = re.search(r'src="([^"]+)"', tag)
    if not src:
        return tag
    with Image.open(html.parent / unquote(src.group(1))) as image:
        width, height = image.size
    # Story can omit content following an image taller than its page box.
    # Fit each aspect ratio, reserving room for the complete caption.
    pixels = min(650, 480 * width / height)
    return tag.replace('<img', f'<img style="width:{pixels:.2f}px;height:auto"', 1)
source = re.sub(r'<img\b[^>]*>', fit_image, source)
css='body{font-size:10pt} table{font-size:8pt;width:100%;table-layout:fixed} th,td{font-size:8pt;padding:2px} img{height:auto}'
story=fitz.Story(html=source,user_css=css,archive=fitz.Archive(str(html.parent)))
page=fitz.Rect(0,0,842,595)
def layout(number,filled):
    return page,page+(30,30,-30,-30),None
pdf=story.write_with_links(layout)
if not len(pdf): raise RuntimeError('PDF has no pages')
class Cells(HTMLParser):
    def __init__(self):
        super().__init__(); self.active=False; self.cells=[]; self.value=''
    def handle_starttag(self,tag,attrs):
        if tag in ('th','td'): self.active=True; self.value=''
    def handle_data(self,data):
        if self.active: self.value+=data
    def handle_endtag(self,tag):
        if tag in ('th','td'): self.active=False; self.cells.append(self.value)
def normalize(value): return re.sub(r'\s+','',unicodedata.normalize('NFKC',value))
cells=Cells();cells.feed(source)
text=''.join(p.get_text() for p in pdf); normalized=normalize(text)
missing=[c for c in cells.cells if normalize(c) and normalize(c) not in normalized]
if missing: raise RuntimeError(f'PDF omitted or clipped table cells: {missing[:10]}')
numbers=re.findall(r'<strong>Figure\s+(\d+)\.</strong>',source)
missing_captions=[n for n in numbers if f'Figure {n}.' not in text]
if missing_captions: raise RuntimeError(f'PDF omitted figure captions: {missing_captions}')
pdf.save(output, garbage=4, deflate=True, deflate_images=True)
print(f'Rendered {len(pdf)} pages to {output}; checked {len(cells.cells)} table cells and {len(numbers)} figure captions')
pdf.close()
