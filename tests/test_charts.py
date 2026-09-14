"""The scripts that turn results/ into the docs site's figures, and the audit's declaration rule.

The site draws every experiment from docs/javascripts/data.js, so a stale file or a figure without a
builder ships a page that is wrong or blank, and nothing else would fail.
"""

import re
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import charts  # noqa: E402
import closure  # noqa: E402
import dataset  # noqa: E402
import report  # noqa: E402


def test_the_committed_chart_data_is_current():
    committed = (ROOT / "docs" / "javascripts" / "data.js").read_text(encoding="utf-8")
    assert committed == charts.data_module(), "docs/javascripts/data.js is stale: run scripts/charts.py"


@pytest.mark.parametrize("values", [[0.0017, 0.0015, 0.0029], [0.0104, -0.0011, 0.0044], [0.002, -0.001], [0.5]])
def test_chart_intervals_match_the_tables(values):
    drawn = charts.summary(values)
    written = report.interval(values)
    if len(values) < 2:
        assert written == "-" and drawn["lo"] is None and drawn["hi"] is None
        return
    lo, hi = (float(v) for v in re.findall(r"[+-]\d+\.\d+", written))
    assert drawn["lo"] == pytest.approx(lo, abs=1e-4) and drawn["hi"] == pytest.approx(hi, abs=1e-4)


def test_every_figure_on_the_site_has_a_builder():
    source = (ROOT / "docs" / "javascripts" / "figures.js").read_text(encoding="utf-8")
    builders = set(re.findall(r'^\s*"?([A-Za-z][\w-]*)"?\s*:\s*function\b', source, re.MULTILINE))
    pages = [*(ROOT / "docs").glob("*.md")]
    used = {name for page in pages for name in re.findall(r'data-figure="([^"]+)"', page.read_text(encoding="utf-8"))}
    assert used, "no page places a figure"
    assert used <= builders, f"figures without a builder: {sorted(used - builders)}"


def _tree(root: Path) -> None:
    for kind in ("images", "labels"):
        for split in dataset.SPLITS:
            (root / kind / split).mkdir(parents=True)
    Image.new("RGB", (1600, 900)).save(root / "images" / "train" / "a.jpg")
    (root / "labels" / "train" / "a.txt").write_text("3 0.5 0.5 1.0 1.0\n0 0.5 0.5 0.01 0.01\n")
    # A square image with a label a hair past its edge lands past the last bin edge once letterboxed.
    Image.new("RGB", (640, 640)).save(root / "images" / "val" / "b.jpg")
    (root / "labels" / "val" / "b.TXT").write_text("2 0.5 0.5 1.002 1.002\n")


def test_dataset_statistics_read_a_directory_and_a_zip_alike(tmp_path):
    tree = tmp_path / "ds"
    _tree(tree)
    archive = tmp_path / "ds.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for path in tree.rglob("*"):
            if path.is_file():
                z.write(path, "VisDrone/" + path.relative_to(tree).as_posix())
        z.writestr("__MACOSX/VisDrone/images/train/._a.jpg", b"a resource fork, not an image")
    for source in (tree, archive):
        splits = dataset.describe(source)
        train, val, test = splits["train"], splits["val"], splits["test"]
        assert (train["images"], train["boxes"], train["resolutions"]) == (1, 2, [[1600, 900, 1]])
        assert train["classes"] == [1, 0, 0, 1]
        assert train["area"] == {"small": 1, "medium": 0, "large": 1}
        assert train["side_at_800"]["counts"][0] == 1 and train["side_at_800"]["counts"][-1] == 1
        assert val["boxes"] == 1 and val["side_at_800"]["counts"][-1] == 1
        assert test["images"] == 0 and test["median_side"] == {"original": None, "at_800": None}


def test_class_names_come_as_a_list_or_a_mapping(tmp_path):
    listed, mapped = tmp_path / "listed.yaml", tmp_path / "mapped.yaml"
    listed.write_text("names: [car, bus]\n")
    mapped.write_text("names:\n  1: bus\n  0: car\n")
    assert dataset.class_names(listed) == dataset.class_names(mapped) == ["car", "bus"]


def test_a_thin_cell_is_declared_only_by_its_exact_name_on_both_pages():
    thin = ["yolov5n-e4k2w0.01@e120f1i800[metaxc500/metax3.7]"]
    exact, longer = "`yolov5n-e4k2w0.01` has seed 0 only", "`yolov5n-e4k2w0.01-rewire` has seed 0 only"
    assert closure.declared(thin, [exact, exact])
    assert not closure.declared(thin, [longer, longer])
    assert not closure.declared(thin, [exact, "not named here"])
    assert closure.declared([], [longer])
