"""中文是默认语言,住在站点根目录;但 /zh/ 是所有人都会去猜的路径,不能 404。"""

from pathlib import Path

SKIP = {"en", "zh", "assets", "search", "stylesheets", "javascripts"}

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>ES-MoE</title>
<link rel="canonical" href="{target}">
<meta name="robots" content="noindex,follow">
<meta http-equiv="refresh" content="0; url={target}">
<script>location.replace("{target}" + location.search + location.hash)</script>
</head><body><a href="{target}">ES-MoE</a></body></html>
"""


def on_post_build(config) -> None:
    site = Path(config["site_dir"])
    base = config["site_url"].rstrip("/")
    for page in site.rglob("index.html"):
        parts = page.relative_to(site).parts[:-1]
        if parts and parts[0] in SKIP:
            continue
        alias = site.joinpath("zh", *parts, "index.html")
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_text(PAGE.format(target=f"{base}/{'/'.join(parts)}{'/' if parts else ''}"), encoding="utf-8")
