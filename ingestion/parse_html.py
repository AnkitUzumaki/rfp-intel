"""HTML bid-page parsing: strip site chrome, keep headings, turn label/value markup into 'Label: Value'."""
from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

# not "form": some portals (ASP.NET) wrap the whole page in one <form>
_CHROME = [
    "script", "style", "noscript", "svg", "nav", "header", "footer",
    "button", "iframe", "input", "select", "textarea",
]
_NOISE_CLASS = re.compile(r"cookie|modal|dialog|banner|share|print-options|breadcrumb", re.I)


def _text(el: Tag) -> str:
    return " ".join(el.get_text(" ", strip=True).split())


def _value(label: Tag) -> str:
    """All siblings after the label (e.g. phone + email under one contact label)."""
    return " ".join(t for t in (_text(v) for v in label.find_next_siblings()) if t)


def _collect_fields(root: Tag) -> dict[str, str]:
    """Generic label/value patterns: <dl>, 2-cell table rows, and elements whose class mentions 'label'."""
    fields: dict[str, str] = {}
    for dl in root.find_all("dl"):
        for dt in dl.find_all("dt"):
            dd = dt.find_next_sibling("dd")
            if dd:
                fields.setdefault(_text(dt), _text(dd))
    for tr in root.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if len(cells) == 2:
            fields.setdefault(_text(cells[0]), _text(cells[1]))
    for label in root.find_all(class_=re.compile(r"label", re.I)):
        if _text(label) and _value(label):
            fields.setdefault(_text(label), _value(label))
    return {k.rstrip(":"): v for k, v in fields.items() if k and v and len(k) < 80}


def _render(root: Tag) -> str:
    """Markdown-ish text: headings as '## X', label/value pairs collapsed onto one line."""
    for label in root.find_all(class_=re.compile(r"label", re.I)):
        field = label.parent
        if field is None or field.parent is None:  # already replaced as part of an outer field
            continue
        if _text(label) and _value(label):
            field.replace_with(f"\n{_text(label)}: {_value(label)}\n")
    for h in root.find_all(re.compile(r"^h[1-6]$")):
        h.replace_with(f"\n## {_text(h)}\n")
    lines = [ln.strip() for ln in root.get_text("\n").splitlines()]
    out: list[str] = []
    for ln in lines:
        if ln and (not out or ln != out[-1]):  # drop blanks + consecutive duplicate UI labels
            out.append(ln)
    return "\n".join(out)


def parse_html(path: Path) -> tuple[str, str | None, dict[str, str], list[str]]:
    """Return (text, title, fields, errors). Never raises."""
    try:
        soup = BeautifulSoup(path.read_bytes(), "lxml")
    except Exception as exc:
        return "", None, {}, [f"unreadable html: {exc}"]
    title = _text(soup.title) if soup.title else None
    for tag in soup(_CHROME):
        tag.decompose()
    for tag in soup.find_all(class_=_NOISE_CLASS):
        tag.decompose()
    found = soup.select_one("main, [role=main]") or soup.body
    root: Tag = found if isinstance(found, Tag) else soup
    try:
        fields = _collect_fields(root)
        text = _render(root)
    except Exception as exc:  # malformed markup: fall back to plain text
        fields, text = {}, root.get_text("\n", strip=True)
        return text, title, fields, [f"html structure error, plain text used: {exc}"]
    errors = [] if text.strip() else ["html has no readable text"]
    return text, title, fields, errors
