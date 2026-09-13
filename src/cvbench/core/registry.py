"""Generic name-to-path resolution shared by the experiment, augmentation,
data, and workspace stores (`core/exp_store.py`, `core/aug_store.py`,
`core/data_store.py`, `core/work_store.py`).

Each of those stores has the same shape: a base directory holding named
entries, where a bare name (`my_run`) resolves to `base_dir/my_run` the same
way a literal path is used as-is, and — for stores backed by a single file
per entry — `base_dir/my_run<ext>` is tried too. `Registry` implements that
resolution once so each store only has to instantiate it with its own
base directory, labels, and mode (directory vs. file).
"""
from __future__ import annotations

from pathlib import Path

import click

from cvbench.core._names import validate_slug


class Registry:
    """Resolve bare names and literal paths against a base directory.

    Directory mode (`ext=None`, the default): entries are subdirectories of
    `base_dir`, e.g. `experiments/my_run`.

    File mode (`ext=".yaml"`): entries are files with that extension under
    `base_dir`, e.g. `workspace/augmentations/standard.yaml`. A name that
    already includes the extension is tried as-is before it's appended.
    """

    def __init__(
        self,
        base_dir: str,
        *,
        not_found_label: str = "Entry",
        entity_name: str = "entry",
        param_hint: str = "NAME",
        ext: str | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.not_found_label = not_found_label
        self.entity_name = entity_name
        self.param_hint = param_hint
        self.ext = ext

    def _exists(self, path: Path) -> bool:
        return path.is_file() if self.ext else path.exists()

    def validate_name(self, name: str, what: str = "Name") -> str:
        """Raise ValueError if name is not safe to use as an entry name."""
        return validate_slug(name, what=what)

    def resolve(self, name_or_path: str) -> str:
        """Resolve a literal path or bare name to an existing entry.

        Accepts a full/relative path, a bare name under `base_dir`, or (in
        file mode) a bare name with `ext` already appended.
        """
        p = Path(name_or_path)
        if self._exists(p):
            return str(p)

        candidate = Path(self.base_dir) / name_or_path
        if self._exists(candidate):
            return str(candidate)

        tried = candidate
        if self.ext:
            with_ext = candidate.with_suffix(self.ext)
            if with_ext.is_file():
                return str(with_ext)
            tried = with_ext

        raise click.BadParameter(
            f"{self.not_found_label} not found: '{name_or_path}' (also tried '{tried}')",
            param_hint=self.param_hint,
        )

    def assert_available(self, new_name: str, current_dir: Path | None = None) -> None:
        """Raise ValueError if new_name conflicts with an existing directory entry."""
        base = Path(self.base_dir)
        if not base.is_dir():
            return
        new_lower = new_name.lower()
        for existing in base.iterdir():
            if not existing.is_dir():
                continue
            if current_dir and existing.resolve() == current_dir.resolve():
                continue
            if existing.name.lower() == new_lower:
                raise ValueError(
                    f"A {self.entity_name} named '{existing.name}' already exists."
                )

    def unique_path(self, name: str) -> Path:
        """Return base_dir/name, appending _2, _3, etc. if the path already exists."""
        base = Path(self.base_dir)
        candidate = base / name
        if not candidate.exists():
            return candidate
        n = 2
        while (base / f"{name}_{n}").exists():
            n += 1
        return base / f"{name}_{n}"

    def list_entries(self) -> list[str]:
        """Return sorted entry names under base_dir.

        Subdirectory names in directory mode, file stems in file mode.
        """
        base = Path(self.base_dir)
        if not base.is_dir():
            return []
        if self.ext:
            return sorted(p.stem for p in base.glob(f"*{self.ext}"))
        return sorted(p.name for p in base.iterdir() if p.is_dir())
