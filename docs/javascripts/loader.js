// The chart, diagram and formula libraries weigh megabytes; a page fetches only the ones it uses.
(function () {
  const CDN = {
    echarts: "https://cdn.jsdelivr.net/npm/echarts@5.6.0/dist/echarts.min.js",
    mermaid: "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js",
    mathjax: "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js",
  };
  const pending = {};

  function load(name) {
    pending[name] = pending[name] || new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = CDN[name];
      script.async = true;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
    return pending[name];
  }

  window.MathJax = {
    tex: { inlineMath: [["\\(", "\\)"]], displayMath: [["\\[", "\\]"]], processEscapes: true, processEnvironments: true },
    options: { ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex" },
    startup: { typeset: false },
  };

  // A page outside the latest x.y says which docs it is, and links to the same page in the latest.
  function outdated() {
    const match = location.pathname.match(/^(.*?\/)(dev|latest|\d+\.\d+(?:\.\d+)?)\/(.*)$/);
    if (!match || match[2] === "latest" || document.querySelector(".es-outdated")) return;
    fetch(match[1] + "versions.json")
      .then((response) => (response.ok ? response.json() : []))
      .then((versions) => {
        const latest = versions.find((v) => (v.aliases || []).includes("latest"));
        if (!latest || latest.version === match[2]) return;
        const zh = (document.documentElement.lang || "").startsWith("zh");
        const banner = document.createElement("div");
        banner.className = "es-outdated";
        const dev = match[2] === "dev";
        const zhText = dev ? "这是开发版文档，随 main 更新，含小版本改动。" : "这是 " + match[2] + " 的文档，不是最新版本。";
        const enText = dev ? "Development docs, updated with main, including patch releases. " : "These are the " + match[2] + " docs, not the latest. ";
        banner.innerHTML = (zh ? zhText : enText) +
          '<a href="' + match[1] + "latest/" + match[3] + '">' + (zh ? "查看 " : "See ") + latest.version + (zh ? " 文档" : " docs") + "</a>";
        const main = document.querySelector(".md-main");
        if (main) main.parentNode.insertBefore(banner, main);
      })
      .catch(() => {});
  }

  document$.subscribe(() => {
    outdated();
    const site = window.ESMOE || {};
    if (document.querySelector("[data-figure], #esmoe-effect")) {
      load("echarts").then(() => {
        if (site.figures) site.figures();
        if (site.effect) site.effect();
      });
    }
    if (document.querySelector(".mermaid-chart")) load("mermaid").then(() => site.diagrams && site.diagrams());
    if (document.querySelector(".arithmatex")) {
      load("mathjax")
        .then(() => window.MathJax.startup.promise)
        .then(() => {
          window.MathJax.typesetClear();
          return window.MathJax.typesetPromise();
        });
    }
  });
})();
