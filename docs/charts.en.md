# Effect chart

The chart below plots every protocol run in `results/`: backbone on the x axis, the same-seed paired delta on the y axis. Above the zero line the grafted block won that seed. The bar is the mean of three seeds, the vertical line their range, and the dots the seeds themselves. Switch between mAP50 and mAP50-95 above the chart, hover a cell for its mean, wins and per-seed values, and save a PNG from the corner.

<div id="esmoe-effect" class="esmoe-chart"></div>

`scripts/charts.py` recomputes the numbers from `results/*.json` into `docs/javascripts/data.js`, so this figure shares a source with the tables on [Results](results.md) and [Judgment lines](JUDGMENT.md) and cannot drift away from them.

## How to read it

- **The range bar matters more than the mean.** Three seeds routinely straddle the zero line even where the mean is positive, which is exactly why the range is drawn. The judgment lines said it up front: three seeds support no significance test, and even 3/3 gives a sign-test p of 0.125.
- **The trend of the means is the project's main finding**: the default graft is positive on SPPF-ended backbones (v5n/v8n/v9t), sits on zero once the backbone ends in attention (v10n/11n), and turns negative on area attention and the E2E head (12n/26n).
- **An arm with fewer than three seeds gets no mean**, only its dots, so an unfinished cell cannot be read as a settled one.
- **The gap between the arms is wiring, not backbone.** `rewire` has exactly the same parameter count as the default arm (3,327,330); only whether consumers read the block's output differs.

## Recompute

    uv run python scripts/report.py   # paired tables
    uv run python scripts/charts.py   # figure and data

The same script also writes `docs/assets/effect.svg`. README and the wiki use that static figure, because GitHub strips scripts in both places and an interactive chart would not render.
