"""Guard the published pages against the ways a written file has silently lost characters here.

Three times a backslash escape was swallowed on the way to disk -- `\frac` became a form feed,
`\text` a tab, `\right` a newline -- and each time the page shipped with a broken formula that
nothing failed on. These checks are cheap and catch that class before it publishes.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted(ROOT.glob("docs/**/*.md")) + sorted(ROOT.glob("*.md"))
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_no_control_characters_survived_writing(page):
    """A form feed or a vertical tab in a page is a `\f` or `\v` that never made it."""
    found = CONTROL.findall(page.read_text(encoding="utf-8"))
    assert not found, f"{page.name} holds control characters {[hex(ord(c)) for c in found]}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_formulas_are_balanced_and_on_one_line(page):
    r"""`\left` needs its `\right`, and a display formula that gained a newline lost an escape."""
    for body in re.findall(r"\$\$(.+?)\$\$", page.read_text(encoding="utf-8"), re.S):
        assert body.count(r"\left") == body.count(r"\right"), f"{page.name}: {body.strip()[:60]}"
        assert "\n" not in body.strip(), f"{page.name}: newline inside {body.strip()[:60]}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_no_line_opens_with_the_tail_of_a_latex_command(page):
    """`ight)`, `rac{` and friends at the start of a line are what an eaten escape leaves behind."""
    tails = ("ight)", "rac{", "ext{", "eft(", "abla", "elta")
    for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
        assert not line.startswith(tails), f"{page.name}:{number} starts with a broken escape: {line[:40]}"
