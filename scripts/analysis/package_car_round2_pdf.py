"""Package a completed Round 2 HTML report as an independently timed PDF."""
import time

START_UNIX = time.time()
CPU_START = time.process_time()

import argparse
import hashlib
import html
import json
import re
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

FIGURES = ('training_losses', 'nominations', 'core_trajectory', 'fidelity', 'paired_blocks',
           'task_deployment_cost', 'native_ffm', 'native_senseiver')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images, self.links, self.ids = [], [], set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if 'id' in values:
            self.ids.add(values['id'])
        if tag == 'img':
            self.images.append(values)
        if tag == 'a':
            self.links.append(values.get('href', ''))


def inside(root, value):
    path = (root / value).resolve(strict=True)
    if root not in path.parents:
        raise ValueError(f'asset escapes report: {value}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    report = args.report_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    if campaign not in report.parents or campaign not in output.parents or output.exists():
        raise ValueError('report and new output must belong to campaign')
    output.mkdir(parents=True)
    receipt = {'schema': 'car_round2_pdf_packaging_v1', 'status': 'running', 'start_unix': START_UNIX,
               'independent_cpu': True, 'scope': 'CPU packaging from process initialization through PDF validation',
               'report_dir': str(report), 'helper_sha256': digest(Path(__file__).resolve())}
    try:
        from PIL import Image
        source = report / 'report.html'
        markdown = report / 'report.md'
        text = source.read_text(encoding='utf-8')
        refs = References(); refs.feed(text)
        expected = {f'figures/{name}.png' for name in FIGURES}
        if {item.get('src') for item in refs.images} != expected or len(refs.images) != len(expected):
            raise ValueError('exactly eight expected report figures required')
        hashes = {'report_html': digest(source), 'report_md': digest(markdown)}
        dimensions = {}
        for item in refs.images:
            name = item['src']; path = inside(report, name)
            if not item.get('alt'):
                raise ValueError(f'figure alt text missing: {name}')
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
            if min(width, height) <= 0:
                raise ValueError('invalid figure dimensions')
            # A4 page at 96 CSS pixels/inch with 0.6-inch margins; reserve
            # 0.8 inch vertically for heading/caption and keep aspect ratio.
            factor = min(1., 678 / width, 853 / height)
            dimensions[name] = [max(1, round(width * factor)), max(1, round(height * factor))]
            hashes[name] = digest(path)
            master = name[:-4] + '.pdf'
            master_path = inside(report, master)
            if master not in refs.links:
                raise ValueError(f'PDF master link missing: {master}')
            hashes[master] = digest(master_path)
            anchor = 'figure-' + Path(name).stem.replace('_', '-')
            if anchor not in refs.ids or '#' + anchor not in refs.links:
                raise ValueError(f'figure index link missing: {anchor}')
        for link in refs.links:
            if link.startswith('#') and link[1:] not in refs.ids:
                raise ValueError(f'broken internal link: {link}')
            parsed = urlsplit(link)
            if link and not link.startswith('#') and not parsed.scheme:
                asset = inside(report, unquote(parsed.path))
                hashes[unquote(parsed.path)] = digest(asset)
        def replace_img(match):
            tag = match.group(0)
            src = html.unescape(re.search(r'src="([^"]+)"', tag).group(1))
            width, height = dimensions[src]
            tag = re.sub(r'src="[^"]+"', 'src="' + inside(report, src).as_uri() + '"', tag)
            return re.sub(r'\s*/?>$', f' width="{width}" height="{height}">', tag)
        converted = re.sub(r'<img\b[^>]*>', replace_img, text)
        def local_link(match):
            link = html.unescape(match.group(1))
            parsed = urlsplit(link)
            if not link or link.startswith('#') or parsed.scheme:
                return match.group(0)
            return 'href="' + inside(report, unquote(parsed.path)).as_uri() + '"'
        converted = re.sub(r'href="([^"]*)"', local_link, converted)
        # Writer ignores inherited table sizing from the stylesheet. Explicit
        # cell formatting keeps numeric columns and headers readable in PDF.
        converted = converted.replace('<table>', '<table width="100%" border="1" cellspacing="0" cellpadding="3">')
        def format_cell(match):
            value = re.sub(r'(?<!\w)[+-]?\d+(?:\.\d+)?[eE][+-]\d+',
                           lambda number: number.group(0).replace('-', '\u2060-\u2060').replace('+', '\u2060+\u2060'),
                           match.group(2))
            return match.group(1) + '<font size="1">' + value + '</font>' + match.group(3)
        converted = re.sub(r'(<t[dh]\b[^>]*>)(.*?)(</t[dh]>)', format_cell, converted, flags=re.DOTALL)
        # Writer's HTML importer drops the first body element and inserts a
        # blank page when a later table follows ordinary body content. Make
        # the existing title the first table's content, preserving its text.
        title = re.search(r'<h1\b[^>]*>(.*?)</h1>', converted, flags=re.DOTALL | re.IGNORECASE)
        if title is None:
            raise ValueError('report title missing')
        title_text = html.unescape(re.sub(r'<[^>]+>', '', title.group(1))).strip()
        if not title_text:
            raise ValueError('report title empty')
        converted = converted[:title.start()] + '<table><tr><td>' + title.group(0) + '</td></tr></table>' + converted[title.end():]
        style = '<style>@page { size: A4; margin: 0.6in; } body { font-family: sans-serif; font-size: 10pt; } img { max-width: 100%; } h2 { page-break-after: avoid; }</style>'
        converted = converted.replace('<title>Round 2 report</title>', '<title>Round 2 report</title>' + style)
        with tempfile.TemporaryDirectory(prefix='car_round2_pdf_') as temporary:
            temp = Path(temporary)
            render_source = output / 'report_render_source.html'
            render_source.write_text(converted, encoding='utf-8')
            command = ['libreoffice', '-env:UserInstallation=' + (temp / 'profile').as_uri(), '--headless',
                       '--convert-to', 'pdf:writer_pdf_Export', '--outdir', str(output), str(render_source)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=True)
        pdf = output / 'report_render_source.pdf'
        if pdf.is_file():
            pdf.rename(output / 'report.pdf')
        pdf = output / 'report.pdf'
        if not pdf.is_file():
            raise ValueError('renderer did not produce report.pdf: ' + result.stdout + result.stderr)
        info = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True, check=True).stdout
        pages = int(re.search(r'^Pages:\s+(\d+)', info, re.MULTILINE).group(1))
        if pages < 1:
            raise ValueError('empty PDF')
        rendered_text = subprocess.run(['pdftotext', '-layout', str(pdf), '-'], capture_output=True, text=True, check=True).stdout
        if re.sub(r'\s+', ' ', title_text) not in re.sub(r'\s+', ' ', rendered_text.split('\f', 1)[0]):
            raise ValueError('report title absent from first PDF page')
        for name in FIGURES:
            if 'Figure ' + name.replace('_', '-') not in rendered_text:
                raise ValueError(f'figure heading absent from PDF: {name}')
        normalized_text = re.sub(r'\s+', ' ', rendered_text)
        for item in refs.images:
            caption = re.sub(r'\s+', ' ', item['alt']).strip()
            if caption not in normalized_text:
                raise ValueError(f'figure caption absent from PDF: {item["src"]}')
        url_info = subprocess.run(['pdfinfo', '-url', str(pdf)], capture_output=True, text=True, check=True).stdout
        linked = set()
        for line in url_info.splitlines()[1:]:
            parts = line.split(None, 2)
            if len(parts) == 3 and parts[2].endswith('.pdf'):
                linked.add((output / unquote(parts[2])).resolve(strict=True))
        masters = {inside(report, f'figures/{name}.pdf') for name in FIGURES}
        if not masters.issubset(linked):
            raise ValueError('PDF master annotations are missing or broken')
        listing = subprocess.run(['pdfimages', '-list', str(pdf)], capture_output=True, text=True, check=True).stdout
        image_rows = [line for line in listing.splitlines() if re.match(r'^\s*\d+\s+\d+\s+', line)]
        if len(image_rows) < len(FIGURES):
            raise ValueError('not all report figures were embedded')
        receipt.update(status='completed', pages=pages, embedded_image_rows=len(image_rows), verified_pdf_master_links=len(masters),
                       figure_dimensions_css_pixels=dimensions, input_sha256=hashes,
                       pdf_sha256=digest(pdf), pdf='report.pdf', renderer='LibreOffice Writer PDF export')
    except BaseException as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        end = time.time()
        receipt.update(end_unix=end, wall_seconds=end - START_UNIX, cpu_seconds=time.process_time() - CPU_START)
        with (output / 'pdf_packaging_receipt.json').open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
