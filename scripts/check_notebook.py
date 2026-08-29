"""Execute the quick start end to end so it cannot rot as ultralytics moves."""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main(path: str = "notebooks/quickstart.ipynb") -> int:
    notebook = nbformat.read(ROOT / path, as_version=4)
    NotebookClient(notebook, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    print(f"{path} executed cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
