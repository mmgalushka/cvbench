import pytest
from click.testing import CliRunner

from cvbench.cli.augmentations import augmentations


@pytest.fixture(autouse=True)
def _in_tmp_dir(tmp_path, monkeypatch):
    """AUGMENTATIONS_DIR is relative ('workspace/augmentations') — sandbox it per test."""
    monkeypatch.chdir(tmp_path)


def _run(args, input=None):
    return CliRunner().invoke(augmentations, args, input=input)


def test_transforms_lists_keras_and_custom_functions():
    """Regression test: _aug_function_defaults() used to `import augmentations`
    (the wrong module), crashing this command outside the Docker container."""
    result = _run(["transforms"])
    assert result.exit_code == 0, result.output
    assert "keras_flip" in result.output
    assert "aug_blur" in result.output


def test_generate_blank_slate_add_one_transform():
    # Add another? y -> pick "1" (keras_flip) -> prob default -> mode default -> add another? n
    result = _run(["generate", "--name", "blank1"], input="y\n1\n\n\nn\n")
    assert result.exit_code == 0, result.output
    assert "Saved" in result.output

    show = _run(["show", "blank1"])
    assert show.exit_code == 0
    assert "keras_flip" in show.output


def test_generate_zero_transforms_raises():
    result = _run(["generate", "--name", "empty"], input="n\n")
    assert result.exit_code != 0
    assert "No transforms selected" in result.output


def test_generate_from_preset_keep_drop_customize():
    # standard preset transforms, in order: keras_flip, keras_rotation,
    # keras_brightness, keras_contrast, aug_blur.
    answers = "\n".join([
        "y", "n",             # keras_flip: keep, don't customize
        "n",                  # keras_rotation: drop
        "y", "y", "0.9", "",  # keras_brightness: keep, customize prob=0.9, factor=default
        "y", "n",             # keras_contrast: keep, don't customize
        "y", "n",             # aug_blur: keep, don't customize
        "n",                  # don't add another
    ]) + "\n"
    result = _run(["generate", "--preset", "standard", "--name", "customized"], input=answers)
    assert result.exit_code == 0, result.output
    assert "4 transform(s)" in result.output

    show = _run(["show", "customized"])
    assert "keras_rotation" not in show.output
    assert "prob: 0.9" in show.output


def test_generate_reference_preset_bypasses_wizard():
    result = _run(["generate", "--preset", "reference", "--name", "ref"])
    assert result.exit_code == 0, result.output

    show = _run(["show", "ref"])
    assert "Augmentation reference" in show.output
    assert "preset: reference" in show.output


def test_list_show_delete_round_trip():
    # "a": blank slate, no transforms added -> generate fails, nothing saved.
    _run(["generate", "--name", "a"], input="n\n")
    _run(["generate", "--preset", "light", "--name", "b"], input="y\nn\ny\nn\nn\n")

    listing = _run(["list"])
    assert listing.exit_code == 0
    assert "b" in listing.output

    deleted = _run(["delete", "b", "--yes"])
    assert deleted.exit_code == 0
    assert "Deleted" in deleted.output

    listing_after = _run(["list"])
    assert "No augmentation configs found" in listing_after.output


def test_list_empty_shows_hint():
    result = _run(["list"])
    assert result.exit_code == 0
    assert "No augmentation configs found" in result.output


def test_delete_missing_name_reports_friendly_error():
    result = _run(["delete", "does-not-exist", "--yes"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()
