import click
import pytest

from cvbench.core.augmentations_store import (
    list_saved_augmentations,
    resolve_aug_file,
    save_augmentation_config,
    validate_aug_name,
)
from cvbench.core.config import load_aug_file


def test_resolve_aug_file_literal_path(tmp_path):
    f = tmp_path / "spec.yaml"
    f.write_text("transforms: []\n")
    assert resolve_aug_file(str(f)) == str(f)


def test_resolve_aug_file_bare_name(tmp_path, monkeypatch):
    aug_dir = tmp_path / "augs"
    aug_dir.mkdir()
    (aug_dir / "standard.yaml").write_text("transforms: []\n")

    import cvbench.core.augmentations_store as store
    monkeypatch.setattr(store, "AUGMENTATIONS_DIR", str(aug_dir))
    assert resolve_aug_file("standard") == str(aug_dir / "standard.yaml")
    assert resolve_aug_file("standard.yaml") == str(aug_dir / "standard.yaml")


def test_resolve_aug_file_not_found_raises_bad_parameter(tmp_path, monkeypatch):
    import cvbench.core.augmentations_store as store
    monkeypatch.setattr(store, "AUGMENTATIONS_DIR", str(tmp_path / "augs"))
    with pytest.raises(click.BadParameter):
        resolve_aug_file("nope")


def test_validate_aug_name_rejects_unsafe_names():
    with pytest.raises(ValueError):
        validate_aug_name("../etc/passwd")
    with pytest.raises(ValueError):
        validate_aug_name("")
    assert validate_aug_name("my-config_1") == "my-config_1"


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    import cvbench.core.augmentations_store as store
    monkeypatch.setattr(store, "AUGMENTATIONS_DIR", str(tmp_path / "augs"))

    transforms = [{"name": "keras_flip", "prob": 1.0, "mode": "horizontal"}]
    path = store.save_augmentation_config("standard", transforms)
    assert path.exists()

    cfg = load_aug_file(str(path))
    assert len(cfg.transforms) == 1
    assert cfg.transforms[0].name == "keras_flip"


def test_list_saved_augmentations_empty(tmp_path, monkeypatch):
    import cvbench.core.augmentations_store as store
    monkeypatch.setattr(store, "AUGMENTATIONS_DIR", str(tmp_path / "does-not-exist"))
    assert list_saved_augmentations() == []


def test_list_saved_augmentations_newest_first_and_tolerates_missing_meta(tmp_path, monkeypatch):
    import cvbench.core.augmentations_store as store
    aug_dir = tmp_path / "augs"
    aug_dir.mkdir()
    monkeypatch.setattr(store, "AUGMENTATIONS_DIR", str(aug_dir))

    # No `meta:` block at all — must not crash, falls back to mtime.
    (aug_dir / "handwritten.yaml").write_text("transforms:\n- name: keras_flip\n  prob: 1.0\n")

    store.save_augmentation_config("newer", [{"name": "keras_flip", "prob": 1.0}])

    entries = list_saved_augmentations()
    names = [e["name"] for e in entries]
    assert "handwritten" in names and "newer" in names

    handwritten = next(e for e in entries if e["name"] == "handwritten")
    assert handwritten["n_transforms"] == 1
