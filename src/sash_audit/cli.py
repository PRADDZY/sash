"""Local analysis CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import read_predictions, write_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/analysis"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    write_analysis(read_predictions(args.predictions), args.output)


if __name__ == "__main__":
    main()

