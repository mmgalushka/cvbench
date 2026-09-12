from unittest import mock

import pytest
from click.testing import CliRunner

import cvbench.cli.augmentations as aug_mod
from cvbench.cli.augmentations import augmentations
from cvbench.core.config import load_aug_file


@pytest.fixture(autouse=True)
def _in_tmp_dir(tmp_path, monkeypatch):
    """AUGMENTATIONS_DIR is relative ('workspace/augmentations') — sandbox it per test."""
    monkeypatch.chdir(tmp_path)


def _run(args, input=None):
    return CliRunner().invoke(augmentations, args, input=input)


def _checked(names):
    """Patch the checkbox screen to return NAMES, bypassing the real terminal UI."""
    return mock.patch.object(aug_mod, "_checklist_prompt", return_value=list(names))


def test_checklist_prompt_preselects_and_preserves_catalogue_order():
    catalogue = [("keras_flip", {"mode": "horizontal"}), ("keras_rotation", {"factor": 0.1}),
                 ("aug_blur", {"radius": 1.0})]
    with mock.patch.object(aug_mod.questionary, "checkbox") as checkbox:
        checkbox.return_value.ask.return_value = ["aug_blur", "keras_flip"]  # out of order
        result = aug_mod._checklist_prompt(catalogue, preselected={"keras_flip"})

    assert result == ["keras_flip", "aug_blur"]  # restored to catalogue order
    choices = checkbox.call_args.kwargs["choices"]
    assert [c.checked for c in choices] == [True, False, False]


def test_checklist_prompt_returns_none_on_abort():
    with mock.patch.object(aug_mod.questionary, "checkbox") as checkbox:
        checkbox.return_value.ask.return_value = None
        assert aug_mod._checklist_prompt([("keras_flip", {})], preselected=set()) is None


def test_checklist_prompt_shows_descriptions():
    catalogue = [("keras_flip", {"mode": "horizontal"})]
    with mock.patch.object(aug_mod.questionary, "checkbox") as checkbox:
        checkbox.return_value.ask.return_value = []
        aug_mod._checklist_prompt(catalogue, preselected=set())
    choices = checkbox.call_args.kwargs["choices"]
    assert "flip" in choices[0].title.lower()


def test_transforms_lists_keras_and_custom_functions():
    """Regression test: _aug_function_defaults() used to `import augmentations`
    (the wrong module), crashing this command outside the Docker container."""
    result = _run(["transforms"])
    assert result.exit_code == 0, result.output
    assert "keras_flip" in result.output
    assert "aug_blur" in result.output


def test_generate_zero_transforms_raises():
    with _checked([]):
        result = _run(["generate", "--name", "empty"])
    assert result.exit_code != 0
    assert "No transforms selected" in result.output


def test_generate_aborted_checklist_raises_abort():
    with mock.patch.object(aug_mod, "_checklist_prompt", return_value=None):
        result = _run(["generate", "--name", "aborted"])
    assert result.exit_code != 0


def test_generate_no_customize_prompts_needed():
    """Selection alone is enough — no per-item 'customize?' round trip."""
    with _checked(["keras_flip", "aug_blur"]):
        result = _run(["generate", "--preset", "standard", "--name", "quick"])
    assert result.exit_code == 0, result.output
    assert "2 transform(s)" in result.output


def test_generate_preserves_preset_tuned_values():
    """A selected preset transform keeps the preset's own prob/params."""
    with _checked(["aug_blur"]):
        result = _run(["generate", "--preset", "standard", "--name", "asis"])
    assert result.exit_code == 0, result.output

    cfg = load_aug_file("workspace/augmentations/asis.yaml")
    assert cfg.transforms[0].prob == 0.3          # standard preset's aug_blur prob
    assert cfg.transforms[0].params["radius"] == 1.0

    show = _run(["show", "asis"])
    assert "Gaussian blur" in show.output          # description comment present


def test_generate_non_preset_transform_gets_usable_default_not_placeholder():
    """A transform picked outside any preset (or with no --preset at all) must
    get a real, usable default — never the bare '<required>' placeholder."""
    with _checked(["aug_fog"]):  # 'strength' has no signature default
        result = _run(["generate", "--name", "fogonly"])
    assert result.exit_code == 0, result.output

    cfg = load_aug_file("workspace/augmentations/fogonly.yaml")
    assert cfg.transforms[0].params["strength"] != "<required>"
    assert isinstance(cfg.transforms[0].params["strength"], float)


def test_generate_writes_param_notes_from_registry():
    with _checked(["aug_fade_horizontal"]):
        result = _run(["generate", "--name", "faded"])
    assert result.exit_code == 0, result.output

    show = _run(["show", "faded"])
    assert "left | right | both" in show.output    # side's choices, from registry._RANGES


def test_generate_reference_preset_bypasses_wizard():
    result = _run(["generate", "--preset", "reference", "--name", "ref"])
    assert result.exit_code == 0, result.output

    show = _run(["show", "ref"])
    assert "Augmentation reference" in show.output


def test_generate_preset_from_saved_config():
    """--preset also accepts the name of any config saved earlier, not just built-ins."""
    with _checked(["keras_flip", "aug_blur"]):
        _run(["generate", "--preset", "standard", "--name", "base"])

    with _checked(["aug_blur"]):
        result = _run(["generate", "--preset", "base", "--name", "derived"])
    assert result.exit_code == 0, result.output

    cfg = load_aug_file("workspace/augmentations/derived.yaml")
    assert cfg.transforms[0].name == "aug_blur"
    assert cfg.transforms[0].prob == 0.3          # carried over from 'base', not a fresh default


def test_generate_unknown_preset_reports_friendly_error():
    result = _run(["generate", "--preset", "does-not-exist", "--name", "x"])
    assert result.exit_code != 0
    assert "Unknown preset" in result.output


def test_list_show_delete_round_trip():
    with _checked([]):
        _run(["generate", "--name", "a"])  # nothing checked -> fails, nothing saved
    with _checked(["keras_flip"]):
        _run(["generate", "--preset", "light", "--name", "b"])

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


def test_edit_writes_back_editor_output():
    with _checked(["keras_flip"]):
        _run(["generate", "--name", "editme"])

    with mock.patch.object(aug_mod.click, "edit", return_value="transforms: []\n"):
        result = _run(["edit", "editme"])
    assert result.exit_code == 0, result.output
    assert "Saved" in result.output

    cfg = load_aug_file("workspace/augmentations/editme.yaml")
    assert cfg.transforms == []


def test_edit_no_changes_leaves_file_untouched():
    with _checked(["keras_flip"]):
        _run(["generate", "--name", "untouched"])
    before = open("workspace/augmentations/untouched.yaml").read()

    with mock.patch.object(aug_mod.click, "edit", return_value=None):
        result = _run(["edit", "untouched"])
    assert result.exit_code == 0
    assert "No changes made" in result.output
    assert open("workspace/augmentations/untouched.yaml").read() == before


def test_edit_missing_name_reports_friendly_error():
    result = _run(["edit", "does-not-exist"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()
