"""站点该应答却不会自己生成的两个地址:/zh/(中文是默认语言,住在根目录)与 en/sitemap.xml(主题按语言去取)。"""

from pathlib import Path

SKIP = {"en", "zh", "assets", "search", "stylesheets", "javascripts"}

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


def on_post_build(config) -> None:
    # i18n 每种语言各跑一遍 post_build,英文那遍的 site_dir 是 site/en;两遍都写向站点根,最后一遍补齐。
    site = Path(config["site_dir"])
    site = site.parent if site.name == "en" else site
    base = config["site_url"].rstrip("/")
    pages = sorted(page.relative_to(site).parts[:-1] for page in site.rglob("index.html"))

    for parts in pages:
        if parts and parts[0] in SKIP:
            continue
        alias = site.joinpath("zh", *parts, "index.html")
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text(ALIAS.format(target=f"{base}/{'/'.join(parts)}{'/' if parts else ''}"), encoding="utf-8")

    english = [f"  <url><loc>{base}/{'/'.join(parts)}/</loc></url>" for parts in pages if parts[:1] == ("en",)]
    (site / "en").mkdir(exist_ok=True)
    (site / "en" / "sitemap.xml").write_text(SITEMAP.format(entries="\n".join(english)), encoding="utf-8")
