"""``data clean`` — copy a dataset, dropping filesystem junk.

Junk means OS/editor artifacts that have nothing to do with the dataset
itself: Finder/Explorer metadata, AppleDouble shadow files, editor swap/temp
files. Real dataset content (images, YOLO labels, ``data.yaml``) is never
touched by the junk rules — only copied straight through.

Because the output is built fresh at ``dst`` from the files judged worth
keeping, directories that would otherwise end up empty (all their contents
were junk) are simply never created — no separate "prune empty dirs" pass
is needed.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

# Exact junk filenames, wherever they appear.
JUNK_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}

# Directories that are junk in their entirety (Finder/Explorer/archiver
# metadata folders) — skipped wholesale, contents never inspected.
JUNK_DIR_NAMES = {"__MACOSX", ".Spotlight-V100", ".fseventsd", ".Trashes", ".TemporaryItems"}

# Junk filename suffixes (editor swap/backup/temp files).
JUNK_SUFFIXES = {".swp", ".swo", ".tmp", ".bak"}


def is_junk_dir(name: str) -> bool:
    return name in JUNK_DIR_NAMES


def is_junk_file(name: str) -> bool:
    if name in JUNK_NAMES:
        return True
    if name.startswith("._"):  # AppleDouble shadow file
        return True
    if name.endswith("~"):  # editor backup
        return True
    return Path(name).suffix.lower() in JUNK_SUFFIXES


@dataclass
class CleanPlan:
    keep: list[Path] = field(default_factory=list)          # paths relative to src
    junk_files: list[Path] = field(default_factory=list)     # relative to src
    junk_dirs: list[Path] = field(default_factory=list)      # relative to src


def build_plan(src: Path) -> CleanPlan:
    """Walk SRC and classify every entry as kept or junk. Read-only."""
    plan = CleanPlan()
    for dirpath in sorted(p for p in src.rglob("*") if p.is_dir()):
        rel_dir = dirpath.relative_to(src)
        if is_junk_dir(dirpath.name):
            plan.junk_dirs.append(rel_dir)

    junk_dir_set = {src / d for d in plan.junk_dirs}

    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        if any(parent in junk_dir_set for parent in path.parents):
            continue  # already accounted for under a junk directory
        rel = path.relative_to(src)
        if is_junk_file(path.name):
            plan.junk_files.append(rel)
        else:
            plan.keep.append(rel)

    return plan


def apply_plan(plan: CleanPlan, src: Path, dst: Path) -> None:
    """Materialize PLAN at DST by copying every kept file from SRC."""
    for rel in plan.keep:
        dst_path = dst / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, dst_path)


def clean_dataset(src: Path, dst: Path, dry_run: bool) -> CleanPlan:
    """Build the clean plan for SRC and, unless DRY_RUN, write it to DST."""
    plan = build_plan(src)
    if not dry_run:
        apply_plan(plan, src, dst)
    return plan
