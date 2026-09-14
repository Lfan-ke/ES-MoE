"""Addresses the site has to answer but does not generate: `/zh/`, because Chinese is the default
language and lives at the root, and a sitemap.xml in every page directory, because i18n writes the
language links relative to the current page and the theme fetches the sitemap the same way."""

from pathlib import Path

# Four point-release notes were merged into one page; their old addresses keep answering.
MOVED = {f"RELEASE_v0.1.{patch}": "RELEASE" for patch in range(4)}
# Retired pages keep their address and land on the page that carries the content now.
MOVED.update({"MIDTERM": "results", "BASELINE": "SELECTION", "PR_DRAFT": "RELEASE"})

ALIAS = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>ES-MoE</title>
<link rel="canonical" href="{target}">
<meta name="robots" content="noindex,follow">
<meta http-equiv="refresh" content="0; url={target}">
<script>location.replace("{target}" + location.search + location.hash)</script>
</head><body><a href="{target}">ES-MoE</a></body></html>
"""

SITEMAP = """<?xml version="1.0" encoding="utf-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{entries}
</urlset>
"""


def _sitemap(base: str, pages: list[tuple[str, ...]]) -> str:
    entries = [f"  <url><loc>{base}/{''.join(part + '/' for part in parts)}</loc></url>" for parts in pages]
    return SITEMAP.format(entries="\n".join(entries))


def on_post_build(config) -> None:
    # i18n runs post_build once per language, the English pass with site_dir at site/en; both write
    # from the site root, and the last pass completes it.
    site = Path(config["site_dir"])
    site = site.parent if site.name == "en" else site
    base = config["site_url"].rstrip("/")
    pages = sorted(page.relative_to(site).parts[:-1] for page in site.rglob("index.html"))
    chinese = [parts for parts in pages if parts[:1] != ("en",) and parts[:1] != ("zh",)]
    english = [parts for parts in pages if parts[:1] == ("en",)]

    for parts in chinese:
        alias = site.joinpath("zh", *parts, "index.html")
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text(ALIAS.format(target=f"{base}/{'/'.join(parts)}{'/' if parts else ''}"), encoding="utf-8")

    for old, new in MOVED.items():
        for language in ("", "zh/", "en/"):
            target = f"{base}/{'en/' if language == 'en/' else ''}{new}/"
            stub = site / language.rstrip("/") / old / "index.html" if language else site / old / "index.html"
            stub.parent.mkdir(parents=True, exist_ok=True)
            stub.write_text(ALIAS.format(target=target), encoding="utf-8")

    (site / "en").mkdir(exist_ok=True)
    for pages_of_language in (chinese, english):
        for page in pages_of_language:
            if not page:  # the root sitemap is mkdocs' own and already lists both languages
                continue
            (site.joinpath(*page) / "sitemap.xml").write_text(_sitemap(base, pages_of_language), encoding="utf-8")
