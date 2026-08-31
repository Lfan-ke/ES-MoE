"""站点该应答却不会自己生成的地址:/zh/(中文是默认语言,住在根目录),以及每个页面目录下的 sitemap.xml
——i18n 把语言切换链接写成相对当前页的,主题就按相对路径去取同名 sitemap。"""

from pathlib import Path

SKIP = {"en", "zh", "assets", "search", "stylesheets", "javascripts"}

# 四份小版本说明并成了一份,老地址继续应答。
MOVED = {f"RELEASE_v0.1.{patch}": "RELEASE" for patch in range(4)}

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
    # i18n 每种语言各跑一遍 post_build,英文那遍的 site_dir 是 site/en;两遍都写向站点根,最后一遍补齐。
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
            if not page:  # 站点根的 sitemap 由 mkdocs 自己写,里面两种语言都有,别覆盖
                continue
            (site.joinpath(*page) / "sitemap.xml").write_text(_sitemap(base, pages_of_language), encoding="utf-8")
