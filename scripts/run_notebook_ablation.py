#!/usr/bin/env python
"""
Utility script to execute the `11_traditional_ablation.ipynb` notebook end to end.

Usage:
    python scripts/run_notebook_ablation.py
    python scripts/run_notebook_ablation.py --notebook notebooks/11_traditional_ablation.ipynb
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

try:  # nbclient moved CellExecutionError in newer releases
    from nbclient import CellExecutionError  # type: ignore
except ImportError:  # pragma: no cover
    from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DEFAULT_NOTEBOOK = REPO_ROOT / "notebooks" / "11_traditional_ablation.ipynb"
OUTPUT_DIR = REPO_ROOT / "outputs" / "executed_notebooks"


def ensure_src_on_pythonpath() -> None:
    src_str = str(SRC_DIR)
    current = os.environ.get("PYTHONPATH", "")
    paths = current.split(os.pathsep) if current else []
    if src_str in paths:
        return
    new_paths = [src_str] + paths
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, new_paths))


def execute_notebook(notebook_path: Path, output_path: Path, timeout: int) -> None:
    print(f"[info] Loading notebook: {notebook_path}")
    nb = nbformat.read(notebook_path.open("r", encoding="utf-8"), as_version=4)

    client = NotebookClient(
        nb,
        kernel_name="python3",
        timeout=timeout,
        allow_errors=False,
    )

    try:
        print("[info] Executing notebook ... this may take a while.")
        client.execute()
    except CellExecutionError as exc:
        print("[error] Notebook execution failed.")
        raise exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, output_path.open("w", encoding="utf-8"))
    print(f"[done] Executed notebook saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the traditional ablation notebook automatically.")
    parser.add_argument(
        "--notebook",
        type=Path,
        default=DEFAULT_NOTEBOOK,
        help="Path to the notebook to execute.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-cell timeout in seconds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the executed notebook. Defaults to outputs/executed_notebooks/<timestamp>.ipynb",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    notebook_path = args.notebook if args.notebook.is_absolute() else (REPO_ROOT / args.notebook)
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = OUTPUT_DIR / f"{notebook_path.stem}_executed_{timestamp}.ipynb"
    output_path = (
        args.output if args.output is not None else default_output
    )
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    ensure_src_on_pythonpath()
    execute_notebook(notebook_path, output_path, args.timeout)


if __name__ == "__main__":
    main()


