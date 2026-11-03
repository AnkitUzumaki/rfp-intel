"""Text cleaning: repeated headers/footers, hyphenation, broken lines, whitespace."""
from __future__ import annotations

import re
from collections import Counter

_DIGITS = re.compile(r"\d+")


_MAX_EDGE_LEN = 120  # headers/footers are short; longer lines are body text
_PAGE_LINE = re.compile(r"\bpage\b|^\W*\d+\W*$", re.I)


def _key(line: str) -> str:
    # "Page 3 of 40" and "Page 4 of 40" count as the same line; other numbers must match exactly
    line = line.strip().lower()
    return _DIGITS.sub("#", line) if _PAGE_LINE.search(line) else line


def remove_headers_footers(pages: list[str], ratio: float = 0.5, zone: int = 3) -> list[str]:
    """Drop lines that repeat in the top/bottom `zone` lines of >= `ratio` of pages.

    Page 1 keeps its copy: the first occurrence is often the document title.
    """
    if len(pages) < 3:
        return pages

    def edges(lines: list[str], learn: bool) -> list[int]:
        """Short lines in the top/bottom zone. While learning which lines repeat, the zone shrinks on
        short pages so body text is never counted as a header; removal then uses the full zone."""
        nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
        z = min(zone, len(nonblank) // 3) if learn else min(zone, len(nonblank))
        return [i for i in nonblank[:z] + nonblank[len(nonblank) - z:] if len(lines[i].strip()) <= _MAX_EDGE_LEN]

    counts: Counter[str] = Counter()
    for text in pages:
        lines = text.splitlines()
        counts.update({_key(lines[i]) for i in edges(lines, learn=True)})
    repeated = {k for k, n in counts.items() if n >= max(2, ratio * len(pages))}
    out = [pages[0]]
    for text in pages[1:]:
        lines = text.splitlines()
        drop = {i for i in edges(lines, learn=False) if _key(lines[i]) in repeated}
        out.append("\n".join(ln for i, ln in enumerate(lines) if i not in drop))
    return out


def clean_text(text: str) -> str:
    text = text.replace(" ", " ").replace("​", "")
    text = re.sub(r"(\w)-\n\s*([a-z])", r"\1\2", text)  # hyphenation: "require-\nment" -> "requirement"
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    # broken lines: join when the line doesn't end a sentence and the next starts lowercase
    text = re.sub(r"(?<=[^.:;!?\n])\n(?=[a-z(])", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
