from pathlib import Path

import pytest
from click.testing import CliRunner

from cvbench.cli.data import data
from cvbench.core.config import load_aug_file


@pytest.fixture(autouse=True)
def _in_tmp_dir(tmp_path, monkeypatch):
    """The default output path is relative ('workspace/augmentation.yaml') — sandbox it."""
    monkeypatch.chdir(tmp_path)


def _run(args):
    return CliRunner().invoke(data, ["aug", *args])


@pytest.mark.parametrize("preset", ["light", "standard", "heavy"])
def test_preset_writes_loadable_config_to_default_path(preset):
    result = _run(["--preset", preset])
    assert result.exit_code == 0, result.output

    cfg = load_aug_file("workspace/augmentation.yaml")
    assert cfg.transforms
    assert "Saved" in result.output


def test_reference_is_default_and_loads_as_empty_config():
    result = _run([])
    assert result.exit_code == 0, result.output

    text = Path("workspace/augmentation.yaml").read_text()
    assert "# - name: keras_flip" in text
    assert "# - name: aug_blur" in text
    assert load_aug_file("workspace/augmentation.yaml").transforms == []


def test_generated_config_has_descriptions_and_param_notes():
    _run(["--preset", "standard"])
    text = Path("workspace/augmentation.yaml").read_text()
    assert "# Gaussian blur." in text
    assert "# chance this fires per image" in text


def test_output_option_creates_parent_dirs():
    result = _run(["--preset", "light", "-o", "nested/dir/my.yaml"])
    assert result.exit_code == 0, result.output
    assert load_aug_file("nested/dir/my.yaml").transforms


def test_refuses_to_overwrite_without_force():
    Path("aug.yaml").write_text("transforms: []\n")
    result = _run(["--preset", "light", "-o", "aug.yaml"])
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert Path("aug.yaml").read_text() == "transforms: []\n"


def test_force_overwrites():
    Path("aug.yaml").write_text("transforms: []\n")
    result = _run(["--preset", "light", "-o", "aug.yaml", "--force"])
    assert result.exit_code == 0, result.output
    assert load_aug_file("aug.yaml").transforms


def test_unknown_preset_rejected():
    result = _run(["--preset", "nope"])
    assert result.exit_code != 0


def test_upsample_rejects_missing_augmentation_file(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    result = CliRunner().invoke(
        data,
        ["upsample", str(src), str(tmp_path / "dst"), "--augmentation", "no-such.yaml", "--target", "5"],
    )
    assert result.exit_code != 0
    assert "no-such.yaml" in result.output
