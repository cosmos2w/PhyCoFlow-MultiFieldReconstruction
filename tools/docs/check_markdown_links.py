"""Check Markdown links using URL normalization before filesystem resolution.

Install the project's docs extra. Portable publications must not depend on
machine-specific symlinks or paths outside the checkout.
"""
import argparse
import json
import os
import posixpath
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import mistune


def url_path(document, destination):
    """Return a local URL-normalized path, never resolving symlinks first."""
    url = urlsplit(destination)
    if url.netloc and url.scheme != "file":
        return None
    if url.scheme and url.scheme != "file":
        return None
    document = os.path.abspath(document)
    raw = unquote(url.path)
    if not raw:
        return Path(document)
    return Path(posixpath.normpath(posixpath.join(posixpath.dirname(document), raw)))


class HTMLReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []
        self.anchors = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.anchors.add(attrs["id"])
        for name in ("href", "src"):
            if attrs.get(name):
                self.urls.append(attrs[name])


def references(text):
    tokens = mistune.create_markdown(renderer="ast", plugins=["table"])(text)
    urls, anchors, counts = [], set(), {}

    def plain(children):
        return "".join(x.get("raw", "") + plain(x.get("children", [])) for x in children)

    def visit(items):
        for token in items:
            kind = token["type"]
            if kind in {"link", "image"}:
                urls.append(token["attrs"]["url"])
            if kind == "heading":
                slug = re.sub(r"[^\w\s-]", "", plain(token.get("children", [])).lower())
                slug = re.sub(r"\s", "-", slug)
                count = counts.get(slug, 0)
                counts[slug] = count + 1
                anchors.add(slug + (f"-{count}" if count else ""))
            if kind in {"inline_html", "block_html"}:
                html = HTMLReferences()
                html.feed(token.get("raw", ""))
                urls.extend(html.urls)
                anchors.update(html.anchors)
            visit(token.get("children", []))

    visit(tokens)
    return urls, anchors


def check_document(document, root, portable=False):
    document = Path(os.path.abspath(document))
    root = Path(os.path.abspath(root))
    if portable and (not document.is_relative_to(root) or document.resolve() != document):
        return {"document": str(document), "local_links": 0, "issues": [{
            "document": str(document), "destination": str(document),
            "normalized": str(document),
            "reason": "portable Markdown document is outside the checkout or uses a symlink",
        }]}
    urls, _ = references(document.read_text())
    issues = []
    local = 0
    for destination in urls:
        target = url_path(document, destination)
        if target is None:
            continue
        local += 1
        reason = None
        if not target.exists():
            reason = "URL-normalized destination is missing"
        elif portable and (urlsplit(destination).scheme == "file"
                           or unquote(urlsplit(destination).path).startswith("/")):
            reason = "portable document uses an absolute local URL"
        elif portable and (not target.is_relative_to(root)
                           or target.resolve() != target):
            reason = "portable document depends on an external path or symlink"
        else:
            fragment = unquote(urlsplit(destination).fragment)
            if fragment and target.suffix.lower() == ".md":
                _, anchors = references(target.read_text())
                if fragment not in anchors:
                    reason = "Markdown heading fragment is missing"
        if reason:
            issues.append({"document": str(document), "destination": destination,
                           "normalized": str(target), "reason": reason})
    return {"document": str(document), "local_links": local, "issues": issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--portable", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    files = sorted({file for path in args.paths for file in
                    (path.rglob("*.md") if path.is_dir() else [path])})
    results = [check_document(path, args.root, args.portable) for path in files]
    result = {"documents": len(results), "local_links": sum(x["local_links"] for x in results),
              "issues": [issue for item in results for issue in item["issues"]]}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Checked {result['documents']} documents and {result['local_links']} local links; "
              f"{len(result['issues'])} issues")
        for issue in result["issues"]:
            print(f"{issue['document']}: {issue['destination']}: {issue['reason']}")
    raise SystemExit(bool(result["issues"]))


if __name__ == "__main__":
    main()
