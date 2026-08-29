// Material renders mermaid by fetching it from unpkg, which is unreachable from parts of the world
// this project is read in. The fences carry their own class so Material leaves them alone and we
// render them from a mirror that is reachable.
document$.subscribe(() => {
  if (!window.mermaid) return;
  const dark = document.body.dataset.mdColorScheme === "slate";
  window.mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: dark ? "dark" : "neutral" });
  const charts = [...document.querySelectorAll(".mermaid-chart")];
  charts.forEach((chart) => {
    if (!chart.dataset.source) chart.dataset.source = chart.textContent.trim();
    chart.innerHTML = chart.dataset.source;
    chart.removeAttribute("data-processed");
  });
  if (charts.length) window.mermaid.run({ nodes: charts });
});
