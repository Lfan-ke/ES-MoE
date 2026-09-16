// Material renders mermaid by fetching it from unpkg, which is unreachable from parts of the world
// this project is read in. The fences carry their own class so Material leaves them alone and we
// render them from a mirror that is reachable, in the site's own palette for either scheme.
(function () {
  const FONT = '"Source Sans 3", "Noto Sans SC", system-ui, sans-serif';
  const SYSTEM = window.matchMedia("(prefers-color-scheme: dark)");
  const PALETTE = {
    light: {
      background: "transparent", mainBkg: "#eef4f6", primaryColor: "#eef4f6", primaryTextColor: "#1d2433",
      primaryBorderColor: "#9dbcc4", nodeBorder: "#9dbcc4", lineColor: "#98a2b3", textColor: "#1d2433",
      secondaryColor: "#fbf1de", tertiaryColor: "#f5f7fa", clusterBkg: "#f7f9fb", clusterBorder: "#d6dde5",
      edgeLabelBackground: "#ffffff", titleColor: "#1d2433",
    },
    dark: {
      background: "transparent", mainBkg: "#1b2536", primaryColor: "#1b2536", primaryTextColor: "#e2e6ee",
      primaryBorderColor: "#3f6874", nodeBorder: "#3f6874", lineColor: "#6f7a8c", textColor: "#e2e6ee",
      secondaryColor: "#2a2519", tertiaryColor: "#161d2b", clusterBkg: "#172031", clusterBorder: "#2c3749",
      edgeLabelBackground: "#141b29", titleColor: "#e2e6ee",
    },
  };

  function dark() {
    const scheme = document.body.getAttribute("data-md-color-scheme");
    if (scheme === "slate") return true;
    if (scheme === "default") return false;
    return SYSTEM.matches;
  }

  function render() {
    if (!window.mermaid) return;
    const charts = [...document.querySelectorAll(".mermaid-chart")];
    if (!charts.length) return;
    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: "base",
      themeVariables: Object.assign({ fontFamily: FONT, fontSize: "14px" }, PALETTE[dark() ? "dark" : "light"]),
      // At natural size the labels stay readable; a wide diagram scrolls inside its box instead of shrinking.
      flowchart: { curve: "basis", padding: 12, nodeSpacing: 36, rankSpacing: 44, useMaxWidth: false },
    });
    charts.forEach((chart) => {
      if (!chart.dataset.source) chart.dataset.source = chart.textContent.trim();
      chart.innerHTML = chart.dataset.source;
      chart.removeAttribute("data-processed");
    });
    window.mermaid.run({ nodes: charts });
  }

  (window.ESMOE = window.ESMOE || {}).diagrams = render;
  // Switching the scheme repaints the page but not the SVG mermaid already drew.
  new MutationObserver(render).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
  SYSTEM.addEventListener("change", render);
})();
