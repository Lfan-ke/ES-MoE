"""Keep the addresses of the unversioned site answering, by pointing each one at `latest/`.

    python .github/docs/stubs.py <gh-pages checkout>

Run after `mike deploy` and `mike set-default`. Whatever the unversioned site left at the root goes,
and every page under `latest/` gets a redirect at the same path without the version, so links
already out in the world (README, PyPI, the wiki, discussions) keep working.
"""

import json
import shutil
import sys
from pathlib import Path

CONFIG = Path(__file__).with_name("mkdocs.yml")
KEEP = {".git", ".nojekyll", "index.html", "versions.json"}

STUB = """<!doctype html>
<html><head><meta charset="utf-8">
<title>ES-MoE</title>
<link rel="canonical" href="{target}">
<meta name="robots" content="noindex,follow">
<meta http-equiv="refresh" content="0; url={target}">
<script>location.replace("{target}" + location.search + location.hash)</script>
</head><body><a href="{target}">ES-MoE</a></body></html>
"""


def site_url() -> str:
    # Read as text: the config carries a python tag that a safe YAML loader refuses.
    line = next(line for line in CONFIG.read_text(encoding="utf-8").splitlines() if line.startswith("site_url:"))
    return line.split(":", 1)[1].strip().rstrip("/")


def main(pages: Path) -> None:
    latest = pages / "latest"
    if not latest.is_dir():
        raise SystemExit(f"{latest} is missing: deploy a release with the `latest` alias first")
    versions = json.loads((pages / "versions.json").read_text(encoding="utf-8"))
    named = {v["version"] for v in versions} | {alias for v in versions for alias in v["aliases"]}

    for entry in pages.iterdir():
        if entry.name not in KEEP | named:
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()

    base = site_url()
    for page in latest.rglob("index.html"):
        parts = page.relative_to(latest).parts[:-1]
        if not parts or parts[0] in named:
            continue
        stub = pages.joinpath(*parts, "index.html")
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text(STUB.format(target=f"{base}/latest/{'/'.join(parts)}/"), encoding="utf-8")
    # GitHub Pages answers every missing address with the root 404; latest's own resolves its assets absolutely.
    shutil.copyfile(latest / "404.html", pages / "404.html")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
