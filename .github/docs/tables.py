"""The results page embeds the generated tables two heading levels down, without their own titles, so its
table of contents lists the page's sections rather than every group inside the tables."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMBED = re.compile(r'^--8<-- "(results/[\w.-]+\.md)"$', re.MULTILINE)


def _embed(match: re.Match) -> str:
    table = (ROOT / match[1]).read_text(encoding="utf-8")
    table = re.sub(r"\A# .*\n+", "", table)
    return re.sub(r"^(#+) ", r"\1## ", table, flags=re.MULTILINE)


def on_page_markdown(markdown: str, **kwargs) -> str:
    return EMBED.sub(_embed, markdown)
