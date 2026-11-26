"""Round 7 PDF rendering with bounded tables for Writer."""
import html
import re
import subprocess
import tempfile
from pathlib import Path

def render(output,captions):
    import mistune
    from PIL import Image
    class Renderer(mistune.HTMLRenderer):
        def heading(self,text,level,**attrs):
            anchor=re.sub(r'[^a-z0-9 -]','',re.sub(r'<[^>]*>','',text).lower()).replace(' ','-')
            return f'<h{level} id="{anchor}">{text}</h{level}>\n'
    markdown=(output/'report.md').read_text()
    rendered=mistune.create_markdown(renderer=Renderer(),plugins=['table'])(markdown)
    # Writer can split even a row marked page-break-inside:avoid. Place bounded
    # groups on fresh pages with their own headings so cells stay together.
    def bounded_table(match):
        table=match.group(0)
        head=re.search(r'<thead>.*?</thead>',table,flags=re.DOTALL)
        body=re.search(r'<tbody>(.*?)</tbody>',table,flags=re.DOTALL)
        if not head or not body: raise ValueError('unexpected report table structure')
        rows=re.findall(r'<tr>.*?</tr>',body.group(1),flags=re.DOTALL)
        if not rows: raise ValueError('empty report table')
        return ''.join('<p style="page-break-before:always"><br></p><table>'+head.group(0)+'<tbody>'+''.join(rows[i:i+6])+'</tbody></table>' for i in range(0,len(rows),6))
    rendered=re.sub(r'<table>.*?</table>',bounded_table,rendered,flags=re.DOTALL)
    css='img {max-width: 100%;height:auto} table {border-collapse:collapse;font-size:9pt} td,th {padding:3px;border-bottom:1px solid #bbb} h2 {page-break-after:avoid} img {page-break-inside:avoid}'
    html_text='<html><head><meta charset="utf-8"><title>Round 7 scientific report</title><style>'+css+'</style></head><body>'+rendered+'</body></html>'
    (output/'report.html').write_text(html_text)
    for name in captions:
        image=output/'figures'/f'{name}.png';pdf=output/'figures'/f'{name}.pdf'
        with Image.open(image) as im:im.verify()
        subprocess.run(['pdfinfo',str(pdf)],check=True,capture_output=True,text=True)
        anchor='figure-'+name.replace('_','-')
        if (f'id="{anchor}"' not in rendered or f'href="#{anchor}"' not in rendered
                or f'figures/{name}.png' not in rendered or f'figures/{name}.pdf' not in rendered):
            raise ValueError(f'figure preview, index or PDF link missing: {name}')
    if re.search(r'!\[[^\]]*\]\([^)]*\.pdf\)',markdown):raise ValueError('PDF masters must be links')
    # Writer ignores CSS max-width: size each raster explicitly for one A4 page.
    # Group each figure heading, raster and caption in one table row so Writer
    # keeps them together when the row fits, as established in the Round 2 packager.
    def size_image(match):
        tag=match.group(0)
        found=re.search(r'src="([^"]+)"',tag)
        if not found:raise ValueError('rendered figure has no source')
        src=html.unescape(found.group(1));path=(output/src).resolve(strict=True)
        if output not in path.parents:raise ValueError('figure path escapes report')
        with Image.open(path) as im:width,height=im.size
        scale=min(1.,650/width,590/height)
        tag=re.sub(r'src="[^"]+"','src="'+path.as_uri()+'"',tag)
        return re.sub(r'\s*/?>$',f' width="{max(1,round(width*scale))}" height="{max(1,round(height*scale))}">',tag)
    absolute=re.sub(r'<img\b[^>]*>',size_image,html_text)
    absolute=re.sub(r'href="((?:figures/|\.\./)[^\"]+)"',lambda m:'href="'+(output/m.group(1)).resolve(strict=True).as_uri()+'"',absolute)
    # First body title is a one-cell table to avoid Writer's empty first page.
    title=re.search(r'<h1\b[^>]*>.*?</h1>',absolute,flags=re.DOTALL|re.IGNORECASE)
    if title is None:raise ValueError('report title missing')
    absolute=absolute[:title.start()]+'<table><tr><td>'+title.group(0)+'</td></tr></table>'+absolute[title.end():]
    for name in captions:
        anchor='figure-'+name.replace('_','-')
        pattern=r'(<h2 id="'+re.escape(anchor)+r'">.*?</h2>\s*<p><img\b[^>]*>\s*</p>\s*<p>.*?</p>)'
        absolute,count=re.subn(pattern,lambda m:'<table width="100%" style="page-break-inside:avoid"><tr><td>'+m.group(1)+'</td></tr></table>',absolute,flags=re.DOTALL,count=1)
        if count!=1:raise ValueError(f'cannot group complete figure section: {name}')
    absolute=absolute.replace('<table>', '<table width="100%" cellspacing="0" cellpadding="3">')
    absolute=absolute.replace('<tr>', '<tr style="page-break-inside:avoid">')
    for index in (1,2):
        marker=f'<h3 id="field-table-{index}">'
        absolute=absolute.replace(marker,'<p style="page-break-before:always"><br></p>'+marker)
    with tempfile.TemporaryDirectory(prefix='car_round7_render_') as temp:
        tmp=Path(temp);source=tmp/'report_render_source.html';source.write_text(absolute)
        subprocess.run(['libreoffice','-env:UserInstallation='+ (tmp/'profile').as_uri(), '--headless','--convert-to','pdf:writer_pdf_Export','--outdir',str(tmp),str(source)],capture_output=True,text=True,check=True,timeout=180)
        pdf=tmp/'report_render_source.pdf'
        if not pdf.is_file():raise ValueError('LibreOffice did not produce the report PDF')
        (output/'report.pdf').write_bytes(pdf.read_bytes())
    info=subprocess.run(['pdfinfo',str(output/'report.pdf')],check=True,capture_output=True,text=True).stdout
    found=re.search(r'^Pages:\s*(\d+)',info,re.MULTILINE)
    if not found or int(found.group(1))<1:raise ValueError('invalid rendered PDF')
    pages=int(found.group(1)); preview=output/'page_previews';preview.mkdir()
    subprocess.run(['pdftoppm','-png','-r','60',str(output/'report.pdf'),str(preview/'page')],check=True,capture_output=True,timeout=180)
    files=sorted(preview.glob('page-*.png'))
    if len(files)!=pages:raise ValueError('rendered page preview count differs')
    for path in files:
        with Image.open(path) as im:im.verify()
    if captions:
        urls=subprocess.run(['pdfinfo','-url',str(output/'report.pdf')],check=True,capture_output=True,text=True).stdout
        for name in captions:
            if f'{name}.pdf' not in urls: raise ValueError(f'PDF master link missing from rendered PDF: {name}')
        listing=subprocess.run(['pdfimages','-list',str(output/'report.pdf')],check=True,capture_output=True,text=True).stdout
        if len([line for line in listing.splitlines()[2:] if line.strip()]) < len(captions):
            raise ValueError('rendered PDF does not contain every preview image')
    report_text=subprocess.run(['pdftotext',str(output/'report.pdf'),'-'],check=True,capture_output=True,text=True).stdout
    if 'Coherence Adaptive Repair' not in report_text.split('\f',1)[0]:raise ValueError('report title missing from first PDF page')
    return pages
