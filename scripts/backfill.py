"""Fill the appendix-B record fields into runs recorded before `scripts/train.py` wrote them.

Config hash, dataset split sizes, GPU-hours and artifact checksum. Every value is derived from
what a record already states or from files still on disk; a field that cannot be derived is left
absent rather than guessed.

    uv run python scripts/backfill.py --check
    uv run python scripts/backfill.py --splits train=6471,val=548,test=1610 --hashes host-a.txt
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REQUIRED = ("config.sha256", "dataset.splits", "budget.gpu_hours", "artifact.sha256")
# Checkpoints pulled off the training boxes, newest mirror first.
MIRRORS = (ROOT.parent / "e1-artifacts", Path("D:/e1-backup/extracted"), Path("D:/e1-backup/ckpt-repo/weights"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoints(listings: list[str] | None) -> dict[str, tuple[str, str]]:
    """Run name -> (sha256, where it was hashed).

    A checkpoint that never left the training box is hashed there; `sha256sum runs/*/weights/best.pt`
    piped into a file is all a listing is. Local mirrors cover the rest.
    """
    found: dict[str, tuple[str, str]] = {}
    for listing in listings or ():
        for line in Path(listing).read_text(encoding="utf-8").splitlines():
            value, _, path = line.partition("  ")
            if path.endswith("best.pt"):
                found.setdefault(PurePosixPath(path).parts[-3], (value, Path(listing).name))
    for mirror in MIRRORS:
        for path in mirror.rglob("*-best.pt") if mirror.is_dir() else ():
            found.setdefault(path.name.removesuffix("-best.pt"), (digest(path), path.name))
    return found


def splits(spec: str | None) -> dict[str, int] | None:
    """Split sizes: counted here when the dataset is complete, else taken from `--splits`.

    Runs happened on a rented box; the images are not always on the machine doing the backfill,
    so the counts read off that box can be passed in instead of being invented here.
    """
    import yaml

    config = yaml.safe_load((ROOT / "configs" / "visdrone.yaml").read_text(encoding="utf-8"))
    for base in (ROOT.parent / "ds" / "VisDrone_dataset", Path(config["path"])):
        counted = {
            name: sum(1 for _ in (base / config[name]).glob("*.jpg"))
            for name in ("train", "val", "test")
            if name in config and (base / config[name]).is_dir()
        }
        if all(counted.get(name) for name in ("train", "val", "test")):
            return counted
    if spec:
        return {k: int(v) for k, v in (part.split("=") for part in spec.split(","))}
    return None


def facts(spec: str | None) -> dict | None:
    import yaml

    config = yaml.safe_load((ROOT / "configs" / "visdrone.yaml").read_text(encoding="utf-8"))
    counts = splits(spec)
    if counts is None:
        return None
    return {"name": Path(config["path"]).name, "classes": len(config["names"]), "splits": counts}


def missing(record: dict) -> list[str]:
    absent = []
    for field in REQUIRED:
        head, tail = field.split(".")
        section = record.get(head)
        if not isinstance(section, dict) or not (section.get(tail) or section.get("unavailable")):
            absent.append(field)
    return absent


def fill(record: dict, dataset: dict | None, weights: dict[str, tuple[str, str]], reason: str = "") -> bool:
    changed = False

    config = record.setdefault("config", {})
    if not config.get("sha256"):
        # The grafted yaml was written on the training box and is not here, so the hash covers the
        # settings that produced it -- reproducible from the record, and distinct per arm.
        canonical = json.dumps({k: v for k, v in config.items() if k != "sha256"}, sort_keys=True)
        config["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        config["sha256_of"] = "settings"
        changed = True

    if dataset and not record.setdefault("dataset", {}).get("splits"):
        record["dataset"].update(dataset)
        changed = True

    budget = record.setdefault("budget", {})
    if not budget.get("gpu_hours") and budget.get("wall_seconds"):
        budget["gpu_hours"] = round(budget["wall_seconds"] / 3600, 3)
        changed = True

    artifact = record.get("artifact")
    if isinstance(artifact, str):
        artifact = record["artifact"] = {"path": artifact}
        changed = True
    if isinstance(artifact, dict) and not artifact.get("sha256"):
        found = weights.get(PurePosixPath(artifact["path"]).parts[-3])
        if found:
            artifact["sha256"], artifact["sha256_of"] = found
            artifact.pop("unavailable", None)
            changed = True
        elif reason and "unavailable" not in artifact:
            artifact["unavailable"] = reason
            changed = True
    return changed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="report gaps, write nothing")
    p.add_argument("--splits", help="e.g. train=6471,val=548,test=1610, when the images are elsewhere")
    p.add_argument("--hashes", action="append", help="a `sha256sum runs/*/weights/best.pt` listing from a host")
    p.add_argument("--unavailable", default="", help="why a checkpoint cannot be hashed, for the ones left over")
    args = p.parse_args()

    records = [path for path in sorted(RESULTS.glob("*.json")) if "experiment_id" in json.loads(path.read_text())]
    if args.check:
        gaps: dict[str, int] = {}
        for path in records:
            for field in missing(json.loads(path.read_text(encoding="utf-8"))):
                gaps[field] = gaps.get(field, 0) + 1
        print(f"{len(records)} records")
        for field in REQUIRED:
            print(f"  {field:<20} missing in {gaps.get(field, 0)}")
        return 1 if gaps else 0

    dataset, weights, written = facts(args.splits), checkpoints(args.hashes), 0
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        if fill(record, dataset, weights, args.unavailable):
            path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            written += 1
    print(f"filled {written} records from {len(weights)} checkpoints" + ("" if dataset else "; no split sizes"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
