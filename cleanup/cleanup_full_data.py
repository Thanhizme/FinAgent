"""Delete generated data files across the project.

Rules:
- Delete only CSV files under data/quant_outputs (recursively).
- Delete all files under data/raw and data/processed (recursively).
- Keep directory structure and preserve .gitkeep files.

Default mode is dry-run. Use --force to actually delete.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
QUANT_OUTPUTS_DIR = DATA_DIR / "quant_outputs"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
KEEP_FILES = {".gitkeep"}


def iter_files(root: Path):
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def should_skip(path: Path) -> bool:
    return path.name in KEEP_FILES


def collect_targets() -> list[Path]:
    targets: list[Path] = []

    if QUANT_OUTPUTS_DIR.exists():
        for path in iter_files(QUANT_OUTPUTS_DIR):
            if path.suffix.lower() == ".csv" and not should_skip(path):
                targets.append(path)

    for base_dir in (RAW_DIR, PROCESSED_DIR):
        if base_dir.exists():
            for path in iter_files(base_dir):
                if not should_skip(path):
                    targets.append(path)

    seen: set[Path] = set()
    unique_targets: list[Path] = []
    for path in targets:
        if path not in seen:
            seen.add(path)
            unique_targets.append(path)
    return unique_targets


def delete_targets(targets: list[Path], dry_run: bool) -> int:
    deleted = 0
    for path in targets:
        print(f"{'DRY-RUN would delete' if dry_run else 'Deleting'}: {path}")
        if not dry_run:
            path.unlink(missing_ok=True)
        deleted += 1
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean full data outputs.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Actually delete files. Without this flag the script only prints what it would remove.",
    )
    args = parser.parse_args()

    targets = collect_targets()
    if not targets:
        print("No matching files found.")
        return 0

    print(f"Found {len(targets)} file(s) to remove.")
    deleted = delete_targets(targets, dry_run=not args.force)

    if args.force:
        print(f"\nDeleted {deleted} file(s).")
    else:
        print("\nDry run finished. Re-run with --force to delete these files.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())