import click
import pytest

from cvbench.core.data_store import resolve_data_dir, validate_data_name


def test_resolve_data_dir_literal_path(tmp_path):
    d = tmp_path / "my_dataset"
    d.mkdir()
    assert resolve_data_dir(str(d)) == str(d)


def test_resolve_data_dir_bare_name(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    (data_dir / "my_dataset").mkdir(parents=True)

    import cvbench.core.data_store as store
    monkeypatch.setattr(store, "DATA_DIR", str(data_dir))
    assert resolve_data_dir("my_dataset") == str(data_dir / "my_dataset")


def test_resolve_data_dir_not_found_raises_bad_parameter_naming_both_paths(tmp_path, monkeypatch):
    import cvbench.core.data_store as store
    monkeypatch.setattr(store, "DATA_DIR", str(tmp_path / "data"))
    with pytest.raises(click.BadParameter) as exc_info:
        resolve_data_dir("not_a_real_dataset")
    msg = str(exc_info.value)
    assert "not_a_real_dataset" in msg
    assert str(tmp_path / "data" / "not_a_real_dataset") in msg


def test_validate_data_name_rejects_unsafe_names():
    with pytest.raises(ValueError):
        validate_data_name("../etc/passwd")
    assert validate_data_name("my-dataset_1") == "my-dataset_1"
