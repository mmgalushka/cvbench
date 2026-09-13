import click
import pytest

from cvbench.core.registry import Registry

# ---------------------------------------------------------------------------
# resolve — directory mode
# ---------------------------------------------------------------------------

def test_resolve_literal_path_directory_mode(tmp_path):
    d = tmp_path / "somewhere"
    d.mkdir()
    reg = Registry(str(tmp_path / "base"))
    assert reg.resolve(str(d)) == str(d)


def test_resolve_bare_name_directory_mode(tmp_path):
    base = tmp_path / "base"
    (base / "my_run").mkdir(parents=True)
    reg = Registry(str(base))
    assert reg.resolve("my_run") == str(base / "my_run")


def test_resolve_not_found_raises_bad_parameter_directory_mode(tmp_path):
    reg = Registry(str(tmp_path / "base"), not_found_label="Run directory", param_hint="EXPERIMENT")
    with pytest.raises(click.BadParameter) as exc_info:
        reg.resolve("nope")
    msg = str(exc_info.value)
    assert "Run directory not found: 'nope'" in msg
    assert "also tried" in msg


# ---------------------------------------------------------------------------
# resolve — file mode
# ---------------------------------------------------------------------------

def test_resolve_literal_path_file_mode(tmp_path):
    f = tmp_path / "spec.yaml"
    f.write_text("transforms: []\n")
    reg = Registry(str(tmp_path / "base"), ext=".yaml")
    assert reg.resolve(str(f)) == str(f)


def test_resolve_bare_name_file_mode_appends_ext(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "standard.yaml").write_text("transforms: []\n")
    reg = Registry(str(base), ext=".yaml")
    assert reg.resolve("standard") == str(base / "standard.yaml")


def test_resolve_bare_name_file_mode_with_ext_already_given(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "standard.yaml").write_text("transforms: []\n")
    reg = Registry(str(base), ext=".yaml")
    assert reg.resolve("standard.yaml") == str(base / "standard.yaml")


def test_resolve_not_found_raises_bad_parameter_file_mode(tmp_path):
    reg = Registry(
        str(tmp_path / "base"), ext=".yaml",
        not_found_label="Augmentation config", param_hint="--augmentation",
    )
    with pytest.raises(click.BadParameter) as exc_info:
        reg.resolve("nope")
    msg = str(exc_info.value)
    assert "Augmentation config not found: 'nope'" in msg
    assert "also tried" in msg


# ---------------------------------------------------------------------------
# validate_name
# ---------------------------------------------------------------------------

def test_validate_name_rejects_unsafe_names(tmp_path):
    reg = Registry(str(tmp_path))
    with pytest.raises(ValueError):
        reg.validate_name("../etc/passwd")
    with pytest.raises(ValueError):
        reg.validate_name("")
    assert reg.validate_name("my-config_1") == "my-config_1"


# ---------------------------------------------------------------------------
# assert_available
# ---------------------------------------------------------------------------

def test_assert_available_ok_when_base_dir_missing(tmp_path):
    reg = Registry(str(tmp_path / "does-not-exist"))
    reg.assert_available("anything")  # no-op, must not raise


def test_assert_available_ok_for_unique_name(tmp_path):
    base = tmp_path / "base"
    (base / "existing").mkdir(parents=True)
    reg = Registry(str(base), entity_name="experiment")
    reg.assert_available("new_name")  # no-op, must not raise


def test_assert_available_raises_on_case_insensitive_conflict(tmp_path):
    base = tmp_path / "base"
    (base / "MyRun").mkdir(parents=True)
    reg = Registry(str(base), entity_name="experiment")
    with pytest.raises(ValueError, match="MyRun"):
        reg.assert_available("myrun")


def test_assert_available_skips_non_directory_entries(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "not_a_dir.txt").write_text("x")  # same stem as the name being checked
    reg = Registry(str(base))
    reg.assert_available("not_a_dir.txt")  # must not raise — files aren't entries


def test_assert_available_skips_current_dir(tmp_path):
    base = tmp_path / "base"
    current = base / "my_run"
    current.mkdir(parents=True)
    reg = Registry(str(base))
    reg.assert_available("my_run", current_dir=current)  # renaming to itself is fine


# ---------------------------------------------------------------------------
# unique_path
# ---------------------------------------------------------------------------

def test_unique_path_no_conflict(tmp_path):
    reg = Registry(str(tmp_path))
    assert reg.unique_path("exp_01") == tmp_path / "exp_01"


def test_unique_path_with_conflicts(tmp_path):
    (tmp_path / "exp_01").mkdir()
    (tmp_path / "exp_01_2").mkdir()
    reg = Registry(str(tmp_path))
    assert reg.unique_path("exp_01") == tmp_path / "exp_01_3"


# ---------------------------------------------------------------------------
# list_entries
# ---------------------------------------------------------------------------

def test_list_entries_missing_base_dir(tmp_path):
    reg = Registry(str(tmp_path / "nope"))
    assert reg.list_entries() == []


def test_list_entries_directory_mode(tmp_path):
    (tmp_path / "b_run").mkdir()
    (tmp_path / "a_run").mkdir()
    (tmp_path / "not_a_dir.txt").write_text("x")
    reg = Registry(str(tmp_path))
    assert reg.list_entries() == ["a_run", "b_run"]


def test_list_entries_file_mode(tmp_path):
    (tmp_path / "b.yaml").write_text("transforms: []\n")
    (tmp_path / "a.yaml").write_text("transforms: []\n")
    (tmp_path / "ignored.txt").write_text("x")
    reg = Registry(str(tmp_path), ext=".yaml")
    assert reg.list_entries() == ["a", "b"]
