# Effect chart

Per-seed paired deltas of the default and rewire arms in the seven-generation matrix: backbone on the x axis, the same-seed paired delta on the y axis. Above the zero line the grafted block won that seed.

<figure class="es-fig" id="fig-1">
<figcaption><b>Fig. 1</b><span>Paired deltas across seven generations</span><em>dots for seeds, bar for the mean, line for the range</em></figcaption>
<div id="esmoe-effect" class="esmoe-chart"></div>
<small>Data: <code>results/summary.md</code></small>
</figure>

The bar is the mean, drawn only with three seeds or more (the v10n default arm has five); the vertical line is the range and the dots are the seeds. Switch between mAP50 and mAP50-95 above the chart, hover a cell for its details, and save a PNG from the corner.

## How to read it

- Read the range, not only the bar. Three seeds often straddle zero even when the mean is positive; the judgment lines said up front that three seeds support no significance test, and even 3/3 gives a sign-test p of 0.125.
- The trend of the means is the main finding: the default wiring is positive on SPPF-ended backbones (v5n, v8n, v9t), sits on zero once the backbone ends in attention (v10n, 11n), and turns negative with area attention and the E2E head (12n, 26n).
- An arm with fewer than three seeds gets no mean, only its dots.
- The two arms have the same parameter count (3,327,330 on YOLOv8n); only the wiring differs.

## Area buckets

<figure class="es-fig" id="fig-2">
<figcaption><b>Fig. 2</b><span>Split by object size</span><em>paired deltas in COCO area buckets, bars for the mean of three seeds</em></figcaption>
<div class="esmoe-figure" data-figure="buckets-generations" data-arms="both"></div>
<small>Data: <code>results/buckets.md</code></small>
</figure>

The buckets show no consistent size trade-off: with the default wiring on v8n large objects get worse on all three seeds (APl −0.0104), and `rewire` turns that positive; on 26n it is small objects that lose (APs −0.0045, 0/3).

## Alignment arms

<figure class="es-fig es-fig--tall" id="fig-3">
<figcaption><b>Fig. 3</b><span>Alignment arms</span><em>upstream's settings inside the block, four blocks and balance terms, each against the same-seed baseline</em></figcaption>
<div class="esmoe-figure" data-figure="alignment"></div>
<small>Data: <code>results/summary.md</code></small>
</figure>

These arms compare block configurations on one backbone, which Fig. 1's generation axis cannot carry. Output normalisation and dense training are both 3/3 on v5n; four blocks are 0/3 on both v10n and v5n. Round-by-round verdicts are on [Experiments](experiments.md) and [Judgment lines](JUDGMENT.md).

## Recompute

    uv run python scripts/report.py   # paired tables
    uv run python scripts/charts.py   # figures and data

`scripts/charts.py` computes the numbers from `results/*.json` into `docs/javascripts/data.js`, the same source as [Results](results.md) and [Judgment lines](JUDGMENT.md). The SVGs under `docs/assets/` are static versions of the same data, used by the README and the wiki.
