"""Download authorized public sources and build local season Parquet datasets.

This script does not commit raw data. Files are stored under data/cache and
data/parquet, both excluded from Git.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yardline_intelligence.data_foundation.multiseason import (  # noqa: E402
    build_all_seasons_parquet,
    ensure_seasons,
)


def parse_seasons(value: str) -> list[int]:
    seasons = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not seasons:
        raise argparse.ArgumentTypeError("Informe ao menos uma temporada.")
    return seasons


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Yardline Intelligence datasets.")
    parser.add_argument(
        "--seasons",
        type=parse_seasons,
        default=parse_seasons("2018,2019,2020,2021,2022,2023,2024,2025"),
        help="Comma-separated NFL seasons (default: 2018-2025).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild Parquet datasets even when they already exist.",
    )
    args = parser.parse_args()

    print(f"Preparing seasons: {', '.join(map(str, args.seasons))}")
    ingestion = ensure_seasons(args.seasons)

    failures = []
    for season, source_results in ingestion.items():
        if not isinstance(source_results, dict):
            continue
        for source, result in source_results.items():
            if isinstance(result, dict) and result.get("error"):
                failures.append(f"{season}/{source}: {result['error']}")

    if failures:
        print("Source ingestion failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    builds = build_all_seasons_parquet(
        args.seasons,
        force_rebuild=args.force_rebuild,
    )
    failed_builds = [
        f"{season}: {result.get('message', 'dataset not created')}"
        for season, result in builds.items()
        if result.get("status") != "ok" or not result.get("exists")
    ]
    if failed_builds:
        print("Dataset build failed:")
        for failure in failed_builds:
            print(f"  - {failure}")
        return 1

    print("Data preparation completed.")
    for season, result in builds.items():
        print(f"  {season}: {result['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
