"""Addresses the site has to answer but does not generate: `/zh/`, because Chinese is the default
language and lives at the root, and a sitemap.xml in every page directory, because i18n writes the
language links relative to the current page and the theme fetches the sitemap the same way."""

import gzip
import re
from pathlib import Path
from urllib.parse import urlsplit

# Four point-release notes were merged into one page; their old addresses keep answering.
MOVED = {f"RELEASE_v0.1.{patch}": "RELEASE" for patch in range(4)}
# Retired pages keep their address and land on the page that carries the content now.
MOVED.update({"MIDTERM": "results", "BASELINE": "SELECTION", "PR_DRAFT": "RELEASE", "limitations": "design"})

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


def relink(page: Path, links: dict[str, str], base: str | None = None) -> None:
    """Point a page's language links, and optionally the theme's base, where they should go."""
    if not page.exists():
        return
    html = page.read_text(encoding="utf-8")
    for language, href in links.items():
        html = re.sub(rf'href="[^"]*" hreflang="{language}"', f'href="{href}" hreflang="{language}"', html)
    if base:
        html = re.sub(r'"base": *"[^"]*"', f'"base": "{base}"', html)
    page.write_text(html, encoding="utf-8")


def fix_entries(site: Path, path: str) -> None:
    """The two home pages and the 404 page, for a version whose address path is `path`."""
    # i18n writes the home pages' language links as absolute paths into this version, which a copied
    # alias such as `latest` then points at the wrong version; subpages already get relative ones.
    relink(site / "index.html", {"zh": "./", "en": "en/"})
    relink(site / "en" / "index.html", {"zh": "../", "en": "./"})
    # One 404 page answers missing addresses at any depth: search and the version list resolve against
    # its base, which mike leaves without the trailing slash.
    relink(site / "404.html", {"zh": f"{path}/", "en": f"{path}/en/"}, base=f"{path}/")


CHINESE_404: list[str] = []


def on_post_template(output: str, template_name: str, config) -> str:
    # The English pass renders the root 404 last; the site's default language is Chinese.
    if template_name == "404.html" and str(config["theme"]["language"]).startswith("zh"):
        CHINESE_404[:] = [output]
    return output


def on_post_build(config) -> None:
    # i18n runs post_build once per language, the English pass with site_dir at site/en; both write
    # from the site root, and the last pass completes it.
    site = Path(config["site_dir"])
    site = site.parent if site.name == "en" else site
    if CHINESE_404:
        (site / "404.html").write_text(CHINESE_404[0], encoding="utf-8")
    base = config["site_url"].rstrip("/")
    pages = sorted(page.relative_to(site).parts[:-1] for page in site.rglob("index.html"))
    pages = [parts for parts in pages if not set(parts) & MOVED.keys()]
    chinese = [parts for parts in pages if parts[:1] != ("en",) and parts[:1] != ("zh",)]
    english = [parts for parts in pages if parts[:1] == ("en",)]

    # Relative targets: the same stub inside `latest/` must land in `latest/`, not the numbered copy.
    for parts in chinese:
        alias = site.joinpath("zh", *parts, "index.html")
        alias.parent.mkdir(parents=True, exist_ok=True)
        target = "../" * (len(parts) + 1) + "".join(part + "/" for part in parts)
        alias.write_text(ALIAS.format(target=target), encoding="utf-8")

    for old, new in MOVED.items():
        for language in ("", "zh/", "en/"):
            stub = site / language / old / "index.html"
            stub.parent.mkdir(parents=True, exist_ok=True)
            up = "../" * (2 if language else 1)
            stub.write_text(ALIAS.format(target=f"{up}{'en/' if language == 'en/' else ''}{new}/"), encoding="utf-8")

    (site / "en").mkdir(exist_ok=True)
    for pages_of_language in (chinese, english):
        for page in pages_of_language:
            if not page:  # the root sitemap is mkdocs' own and already lists both languages
                continue
            (site.joinpath(*page) / "sitemap.xml").write_text(_sitemap(base, pages_of_language), encoding="utf-8")

    # The root sitemap's language links glue the page path onto site_url, which mike sets without a
    # trailing slash, so `dev` and `API/` come out as `devAPI/`. site_url itself stays: i18n reruns
    # mike's urljoin for the English pass, and a slash there would double the version.
    root = site / "sitemap.xml"
    if root.exists():
        text = re.sub(re.escape(base) + r'(?=[^/"<])', base + "/", root.read_text(encoding="utf-8"))
        root.write_text(text, encoding="utf-8")
        with gzip.open(root.with_suffix(".xml.gz"), "wb") as packed:
            packed.write(text.encode("utf-8"))

    fix_entries(site, urlsplit(base).path)
