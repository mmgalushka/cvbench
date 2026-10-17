"""`train` rejects bad options and unusable datasets up front: exit 2, no traceback, no files."""
import pytest
from click.testing import CliRunner

from cvbench.cli import train as train_mod
from cvbench.cli.train import train
from cvbench.core.config import BACKBONE_NAMES
from cvbench.datasets.layout import classification_layout_errors, detection_layout_errors


@pytest.fixture
def data_dir(tmp_path):
    for cls in ("cat", "dog"):
        d = tmp_path / "data" / "train" / cls
        d.mkdir(parents=True)
        for i in range(2):
            (d / f"{i}.jpg").write_bytes(b"x")
    return tmp_path / "data"


@pytest.fixture(autouse=True)
def no_training(monkeypatch, tmp_path):
    """Any option that slips past validation would reach run_training; fail loudly if so."""
    monkeypatch.chdir(tmp_path)

    def boom(**kwargs):
        raise AssertionError("run_training reached")

    monkeypatch.setattr("cvbench.services.training.run_training", boom)


@pytest.mark.parametrize("args, fragment", [
    (["--backbone", "nope"], "--backbone"),
    (["--epochs", "-1"], "--epochs"),
    (["--epochs", "0"], "--epochs"),
    (["--lr", "0"], "--lr"),
    (["--batch-size", "0"], "--batch-size"),
    (["--input-size", "0"], "--input-size"),
    (["--dropout", "1"], "--dropout"),
    (["--val-split", "1.5"], "--val-split"),
    (["--fine-tune-from-layer", "-2"], "--fine-tune-from-layer"),
    (["--seed", "-3"], "--seed"),
    (["--resume", "missing.keras"], "--resume"),
    (["--loss", "focal:gamma"], "--loss"),
    (["--loss", "huber"], "--loss"),
    (["--loss", "focal:gamma=abc"], "--loss"),
    (["--loss", "focal:bogus=1"], "--loss"),
    (["--loss", "focal:label_smoothing=2"], "--loss"),
    (["--optimizer", "rmsprop"], "--optimizer"),
    (["--optimizer", "sgd:momentum=x"], "--optimizer"),
    (["--optimizer", "sgd:"], "--optimizer"),
    (["--lr-scheduler", "foo"], "--lr-scheduler"),
    (["--lr-scheduler", "patience=2,factor=1.5"], "--lr-scheduler"),
    (["--class-weight", "{oops"], "--class-weight"),
    (["--class-weight", '{"cat": -1}'], "--class-weight"),
])
def test_bad_option_exits_2(data_dir, tmp_path, args, fragment):
    result = CliRunner().invoke(train, [str(data_dir), *args])
    assert result.exit_code == 2, result.output
    assert fragment in result.output
    assert "Traceback" not in result.output
    assert not (tmp_path / "experiments").exists()


def test_error_for_key_value_options_includes_an_example(data_dir):
    result = CliRunner().invoke(train, [str(data_dir), "--loss", "focal:gamma"])
    assert "focal:gamma=2.0" in result.output


@pytest.mark.parametrize("raw, expected", [
    ("sgd", ("sgd", 0.0, 0.9)),
    ("sgd:momentum=0.5,weight_decay=1e-4", ("sgd", 1e-4, 0.5)),
    ("adam:weight_decay=0.01", ("adam", 0.01, 0.9)),
])
def test_optimizer_parser_accepts_valid_specs(raw, expected):
    opt = train_mod._parse_optimizer(raw)
    assert (opt.type, opt.weight_decay, opt.momentum) == expected


def test_loss_and_scheduler_parsers_accept_valid_specs():
    loss = train_mod._parse_loss("focal:gamma=3,label_smoothing=0.1")
    assert (loss.type, loss.focal_gamma, loss.label_smoothing) == ("focal", 3.0, 0.1)
    assert train_mod._parse_loss("crossentropy").type == "crossentropy"
    sched = train_mod._parse_lr_scheduler("patience=3,factor=0.3,min=1e-6")
    assert (sched.patience, sched.factor, sched.min_lr) == (3, 0.3, 1e-6)


def test_valid_command_reaches_run_training(data_dir):
    result = CliRunner().invoke(train, [str(data_dir), "--loss", "focal:gamma=2", "--epochs", "1"])
    assert isinstance(result.exception, AssertionError)  # the stub, i.e. validation passed


def test_unsplit_dataset_points_at_data_split(tmp_path):
    flat = tmp_path / "flat"
    (flat / "cat").mkdir(parents=True)
    result = CliRunner().invoke(train, [str(flat)])
    assert result.exit_code == 2
    assert "train/" in result.output
    assert "data split" in result.output
    assert not (tmp_path / "experiments").exists()


def test_missing_from_config_is_a_usage_error(data_dir, tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    result = CliRunner().invoke(train, [str(data_dir), "--from", str(baseline)])
    assert result.exit_code == 2
    assert "--from" in result.output


def test_backbone_names_match_presets():
    from cvbench.core.backbone import _BACKBONE_PRESETS

    assert set(BACKBONE_NAMES) == set(_BACKBONE_PRESETS)


# ---- layout validators ------------------------------------------------------

def test_classification_layout_accepts_split_dataset(data_dir):
    assert classification_layout_errors(data_dir) == []


def test_classification_layout_not_a_directory(tmp_path):
    assert "not a directory" in classification_layout_errors(tmp_path / "nope")[0]


def test_classification_layout_empty_train(tmp_path):
    (tmp_path / "train").mkdir()
    assert "no class subfolders" in classification_layout_errors(tmp_path)[0]


def test_classification_layout_empty_class(data_dir):
    (data_dir / "train" / "bird").mkdir()
    assert "bird" in classification_layout_errors(data_dir)[0]


def test_classification_layout_single_class(tmp_path):
    cls = tmp_path / "train" / "only"
    cls.mkdir(parents=True)
    (cls / "a.jpg").write_bytes(b"x")
    (cls / "b.jpg").write_bytes(b"x")
    assert any("only one class" in e for e in classification_layout_errors(tmp_path))


def test_classification_layout_val_without_classes(data_dir):
    (data_dir / "val").mkdir()
    assert any("val/" in e for e in classification_layout_errors(data_dir))


def test_classification_layout_auto_split_needs_two_images_per_class(data_dir):
    (data_dir / "train" / "cat" / "1.jpg").unlink()
    assert any("cat" in e and "2 images" in e for e in classification_layout_errors(data_dir))


def test_detection_layout_requires_train_and_val(tmp_path):
    errors = detection_layout_errors(tmp_path)
    assert any("images/train" in e for e in errors)
    assert any("images/val" in e for e in errors)
