import pytest

from cvbench.core.errors import ConfigError, ConflictError, NotFoundError
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
    reg = Registry(
        str(tmp_path / "base"), not_found_label="Run", param_hint="EXPERIMENT", list_hint="runs list",
    )
    with pytest.raises(NotFoundError) as exc_info:
        reg.resolve("nope")
    msg = str(exc_info.value)
    assert "Run not found: 'nope'." in msg
    assert "See `runs list`." in msg
    assert "also tried" not in msg.lower()


def test_resolve_not_found_suggests_close_match(tmp_path):
    base = tmp_path / "base"
    (base / "resnet_run").mkdir(parents=True)
    reg = Registry(str(base), not_found_label="Run", list_hint="runs list")
    with pytest.raises(NotFoundError) as exc_info:
        reg.resolve("resnet_rum")
    assert "Did you mean 'resnet_run'?" in str(exc_info.value)


def test_resolve_not_found_suggests_extra_entries(tmp_path):
    reg = Registry(str(tmp_path / "base"), extra_entries=lambda: ["trial_001"])
    with pytest.raises(NotFoundError) as exc_info:
        reg.resolve("trial_01")
    assert "Did you mean 'trial_001'?" in str(exc_info.value)


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
    with pytest.raises(NotFoundError) as exc_info:
        reg.resolve("nope")
    msg = str(exc_info.value)
    assert "Augmentation config not found: 'nope'" in msg
    assert "Also tried" in msg


# ---------------------------------------------------------------------------
# validate_name
# ---------------------------------------------------------------------------

def test_validate_name_rejects_unsafe_names(tmp_path):
    reg = Registry(str(tmp_path))
    with pytest.raises(ConfigError):
        reg.validate_name("../etc/passwd")
    with pytest.raises(ConfigError):
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
    with pytest.raises(ConflictError, match="MyRun"):
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
