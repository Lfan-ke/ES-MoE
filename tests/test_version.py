"""One version, written in three places: the package, its build metadata and its citation."""

import re
from pathlib import Path

import esmoe

ROOT = Path(__file__).resolve().parents[1]


def test_the_package_the_build_and_the_citation_name_one_version():
    built = re.search(r'^version = "(.+)"$', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.MULTILINE)
    cited = re.search(r'^version: "(.+)"$', (ROOT / "CITATION.cff").read_text(encoding="utf-8"), re.MULTILINE)
    assert built and cited
    assert esmoe.__version__ == built.group(1) == cited.group(1)
