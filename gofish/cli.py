"""Command-line interface for GoFish.

Usage::

    gofish config.ini                  # full forecast
    gofish config.ini --quiet          # suppress per-bin printout
    gofish config.ini --no-write       # don't write output files
    python -m gofish config.ini        # equivalent
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ForecastConfig
from .pipeline import run_forecast


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gofish",
        description="Fisher forecast tool for galaxy clustering surveys (DESI-style).",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the .ini config file.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write per-bin cov/data text files.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-bin progress printing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ForecastConfig.from_ini(args.config)
    run_forecast(config, write_outputs=not args.no_write, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
