"""The Task extension hooks, proven with a test-only autoencoder task (tests/autoencoder_task.py).

The autoencoder is unlabeled, has two artifacts (encoder, decoder) and its own fit loop; it is
reached only through the existing registries (tasks, layouts, exporters), with nothing in
services/, cli/ or web/ knowing about it. The default behaviour of the built-in tasks is
pinned alongside.
"""
from __future__ import annotations

import os
import signal
import types
from pathlib import Path

import pytest

from cvbench import tasks as tasks_mod
from cvbench.core.config import build_config, load_config, save_config
from cvbench.core.errors import ConfigError, NotFoundError
from cvbench.core.task import LABEL_KINDS, LabelSpec
from cvbench.tasks import get_task

AE_EP = types.SimpleNamespace(
    name="autoencoder", module="tests.autoencoder_task", attr="AutoEncoderTask",
    value="tests.autoencoder_task:AutoEncoderTask",
)


@pytest.fixture
def ae_installed(monkeypatch):
    real = tasks_mod.entry_points
    monkeypatch.setattr(
        tasks_mod, "entry_points",
        lambda group: [*real(group=group), AE_EP] if group == "cvbench.tasks" else real(group=group),
    )


OVERRIDES = {
    "model.backbone": "efficientnet_b0", "model.weights": "none", "training.epochs": 2,
    "data.batch_size": 2, "model.input_size": 32, "training.seed": 3,
}


# ---- LabelSpec ---------------------------------------------------------------

def test_label_spec_kinds_and_properties():
    assert LABEL_KINDS == ("class", "boxes", "masks", "none")
    assert LabelSpec().kind == "class"
    assert [LabelSpec(k).has_classes for k in LABEL_KINDS] == [True, True, False, False]
    assert [LabelSpec(k).joint_targets for k in LABEL_KINDS] == [False, False, True, False]
    with pytest.raises(ValueError, match="Unknown label kind"):
        LabelSpec("depth")


def test_builtin_tasks_declare_their_label_kind():
    assert get_task("classification").label_spec.kind == "class"
    assert get_task("detection").label_spec.kind == "boxes"


def test_the_hook_modules_import_without_tensorflow():
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import cvbench.core.fit, cvbench.core.layout, cvbench.core.task, cvbench.datasets.adapters\n"
        "from cvbench.tasks import get_task\n"
        "t = get_task('classification'); t.artifacts(); t.export_targets(); t.label_spec; t.layout_adapter()\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('tensorflow', 'keras'))\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


# ---- artifacts ---------------------------------------------------------------

@pytest.mark.parametrize("name", ["classification", "detection"])
def test_builtin_tasks_have_the_single_default_artifact(name):
    task = get_task(name)
    assert task.artifacts() == ("model",) and task.primary_artifact == "model"
    assert task.artifact_path("run/best.keras", "model") == Path("run/best.keras")
    assert task.export_targets() is None


def test_artifact_paths_sit_next_to_the_primary_checkpoint(ae_installed):
    task = get_task("autoencoder")
    assert task.artifacts() == ("encoder", "decoder")
    assert task.artifact_path("run/best.keras", "encoder") == Path("run/best.keras")
    assert task.artifact_path("run/best.keras", "decoder") == Path("run/best.decoder.keras")
    assert task.artifact_path("run/interrupt_epoch003.keras", "decoder") == Path(
        "run/interrupt_epoch003.decoder.keras"
    )
    with pytest.raises(ValueError, match="no artifact 'critic'"):
        task.artifact_path("run/best.keras", "critic")


def test_load_models_reports_a_missing_artifact(ae_installed, tmp_path, monkeypatch):
    task = get_task("autoencoder")
    monkeypatch.setattr(type(task), "load_model", lambda self, path: path)
    (tmp_path / "best.keras").touch()
    with pytest.raises(NotFoundError, match="Artifact 'decoder'"):
        task.load_models(str(tmp_path / "best.keras"))
    (tmp_path / "best.decoder.keras").touch()
    assert task.load_models(str(tmp_path / "best.keras")) == {
        "encoder": str(tmp_path / "best.keras"), "decoder": str(tmp_path / "best.decoder.keras"),
    }


def test_prune_removes_every_artifact_file_of_an_old_epoch(tmp_path):
    from cvbench.core.checkpoint import prune_checkpoints
    from cvbench.core.config import CVBenchConfig

    for e in (1, 2, 3):
        (tmp_path / f"epoch_{e:03d}.keras").touch()
        (tmp_path / f"epoch_{e:03d}.decoder.keras").touch()
    (tmp_path / "best.keras").touch()
    (tmp_path / "best.decoder.keras").touch()
    cfg = CVBenchConfig()
    cfg.training.checkpoints.keep_last_n = 2

    prune_checkpoints(str(tmp_path), cfg)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "best.decoder.keras", "best.keras", "epoch_002.decoder.keras", "epoch_002.keras",
        "epoch_003.decoder.keras", "epoch_003.keras",
    ]


# ---- layout adapters ---------------------------------------------------------

def test_layout_registry_holds_the_builtin_adapters_and_rejects_unknown_names():
    from cvbench.core.layout import get_layout, layout_names

    get_task("classification")  # importing a task registers the built-in layouts
    assert {"class_folders", "yolo"} <= set(layout_names())
    with pytest.raises(ConfigError, match="Unknown dataset layout 'nope'.*class_folders"):
        get_layout("nope")


def test_builtin_layout_adapters_sniff_and_report_like_the_old_task_methods(tiny_cls_root, tiny_yolo_root):
    cls, det = get_task("classification"), get_task("detection")
    assert cls.layout_adapter().name == "class_folders" and det.layout_adapter().name == "yolo"
    assert cls.matches_layout(tiny_cls_root) and not cls.matches_layout(tiny_yolo_root)
    assert det.matches_layout(tiny_yolo_root) and not det.matches_layout(tiny_cls_root)
    assert cls.layout_errors(tiny_cls_root) == [] and det.layout_errors(tiny_yolo_root) == []
    assert any("images/val" in e for e in det.layout_errors(tiny_cls_root))


def test_a_task_without_a_layout_matches_nothing_and_cannot_resolve(tmp_path):
    from cvbench.core.task import Task

    class _NoLayout(get_task("classification").__class__):
        layout = None

        @classmethod
        def matches_layout(cls, data_dir):
            return Task.matches_layout.__func__(cls, data_dir)

        def layout_errors(self, data_dir):
            return Task.layout_errors(self, data_dir)

        def resolve_layout(self, cfg):
            return Task.resolve_layout(self, cfg)

    task = _NoLayout()
    assert task.layout_adapter() is None and not task.matches_layout(tmp_path)
    assert task.layout_errors(tmp_path) == []
    with pytest.raises(NotImplementedError, match="must set `layout`"):
        task.resolve_layout(build_config(str(tmp_path)))


def test_flat_layout_is_sniffed_through_layout_priority(ae_installed, tiny_flat_root, tiny_cls_root):
    assert tasks_mod.detect_task_name(tiny_flat_root) == "autoencoder"
    assert tasks_mod.detect_task_name(tiny_cls_root) == "classification"
    adapter = get_task("autoencoder").layout_adapter()
    assert adapter.errors(tiny_flat_root) == []
    assert adapter.errors(tiny_flat_root / "nope")[0].startswith("No train/")
    (tiny_flat_root / "val").rename(tiny_flat_root / "valid")
    assert "No val/" in adapter.errors(tiny_flat_root)[0]
    for f in (tiny_flat_root / "train").iterdir():
        f.unlink()
    assert "contains no images" in adapter.errors(tiny_flat_root)[0]


# ---- export targets and calibration ------------------------------------------

def test_export_targets_are_checked_against_the_exporter_registry(ae_installed, monkeypatch):
    from cvbench.services.export import EXPORTERS, export_targets

    assert export_targets(get_task("classification")) == tuple(EXPORTERS)
    assert export_targets(get_task("autoencoder")) == ("onnx", "tflite")

    task = get_task("autoencoder")
    monkeypatch.setattr(type(task), "export_targets", lambda self: ("onnx", "coreml"))
    with pytest.raises(ConfigError, match="unknown export target.*coreml.*Registered: tflite"):
        export_targets(task)


def test_run_export_refuses_a_format_the_task_does_not_offer(ae_installed, exps, write_exp):
    from cvbench.services.export import run_export

    run_dir = write_exp(exps, "ae_run")
    cfg = load_config(str(run_dir))
    cfg.task = "autoencoder"
    save_config(cfg, str(run_dir))

    with pytest.raises(ConfigError, match="'hailo' is not supported by autoencoder runs.*onnx, tflite"):
        run_export("ae_run", "hailo")
    with pytest.raises(ConfigError, match="Unknown export format 'nope'"):
        run_export("ae_run", "nope")


def test_calibration_defaults_to_class_folders_and_a_task_can_supply_its_own(
    ae_installed, tiny_cls_root, tiny_flat_root,
):
    from cvbench.services.export.calibration import build_calibration_set, calibration_groups

    cls_cfg = build_config(str(tiny_cls_root))
    groups = calibration_groups(cls_cfg)
    assert set(groups) == {p.name for p in (tiny_cls_root / "train").iterdir()}

    ae_cfg = build_config(str(tiny_flat_root))
    assert ae_cfg.task == "autoencoder"
    ae_cfg.data.train_dir = str(tiny_flat_root / "train")
    assert set(calibration_groups(ae_cfg)) == {"_all"}

    ae_cfg.model.input_size = 16
    out = tiny_flat_root / "calib.npy"
    n, buckets = build_calibration_set(ae_cfg, out, total=4, strategy="proportional")
    assert buckets == 1 and n == 4 and out.is_file()

    ae_cfg.data.train_dir = str(tiny_flat_root / "missing")
    ae_cfg.data.val_dir = str(tiny_flat_root / "missing")
    with pytest.raises(Exception, match="No images found"):
        build_calibration_set(ae_cfg, out, total=4)


# ---- the autoencoder through the real services -------------------------------

@pytest.mark.tf
def test_train_checkpoint_resume_and_evaluate_a_two_artifact_run(ae_installed, exps, tiny_flat_root):
    import numpy as np

    from cvbench.services.evaluation import run_evaluation
    from cvbench.services.training import run_training

    exp_dir = Path(run_training(
        str(tiny_flat_root), overrides={**OVERRIDES, "autoenc.latent_dim": 4}, listener=None,
    ))

    # one best checkpoint per artifact, the custom loop's CSV log, and the standard run record
    assert (exp_dir / "best.keras").is_file() and (exp_dir / "best.decoder.keras").is_file()
    assert (exp_dir / "training_log.csv").read_text().splitlines()[0] == "epoch,loss,val_loss"
    cfg = load_config(str(exp_dir))
    assert cfg.task == "autoencoder" and cfg.run.status == "done" and cfg.run.epochs_run == 2
    assert cfg.run.primary_metric == "latent_score" and cfg.run.metrics["val_loss"] > 0
    assert cfg.autoenc.latent_dim == 4 and cfg.data.classes == []

    task = get_task("autoencoder")
    models = task.load_models(str(exp_dir / "best.keras"))
    assert models["encoder"].output_shape == (None, 4) and models["decoder"].input_shape == (None, 4)
    recon = models["decoder"].predict(models["encoder"].predict(np.zeros((1, 32, 32, 3), "float32"), verbose=0),
                                      verbose=0)
    assert recon.shape == (1, 32, 32, 3)

    # evaluation goes through the primary artifact only
    report = run_evaluation(str(exp_dir))
    assert report["task"] == "autoencoder" and 0.0 <= report["overall"]["value"] <= 1.0

    # resume restores both artifacts and continues from the recorded epoch
    saved_encoder = models["encoder"].get_weights()
    resumed = Path(run_training(
        str(tiny_flat_root), overrides={**OVERRIDES, "training.epochs": 3, "autoenc.latent_dim": 4},
        output_dir=str(exp_dir), resume=str(exp_dir / "best.keras"), overwrite=True,
    ))
    assert resumed == exp_dir
    cfg = load_config(str(exp_dir))
    assert cfg.run.epochs_run == 3 and cfg.run.status == "done"
    rows = (exp_dir / "training_log.csv").read_text().splitlines()
    assert [r.split(",")[0] for r in rows] == ["epoch", "0", "1", "2"]
    assert (exp_dir / "best.decoder.keras").is_file() and saved_encoder


@pytest.mark.tf
def test_a_resume_without_the_decoder_checkpoint_fails_cleanly(ae_installed, exps, tiny_flat_root):
    from cvbench.services.training import run_training

    exp_dir = Path(run_training(str(tiny_flat_root), overrides=OVERRIDES))
    (exp_dir / "best.decoder.keras").unlink()
    with pytest.raises(NotFoundError, match="Artifact 'decoder'"):
        run_training(
            str(tiny_flat_root), overrides=OVERRIDES, output_dir=str(exp_dir),
            resume=str(exp_dir / "best.keras"), overwrite=True,
        )


@pytest.mark.tf
def test_an_interrupted_run_saves_an_interrupt_checkpoint_per_artifact(ae_installed, exps, tiny_flat_root):
    from cvbench.core.events import EpochEnded
    from cvbench.services.training import run_training

    def listener(event):
        if isinstance(event, EpochEnded):
            os.kill(os.getpid(), signal.SIGINT)  # the trainer's handler asks for a graceful stop

    previous = signal.getsignal(signal.SIGINT)
    try:
        exp_dir = Path(run_training(
            str(tiny_flat_root), overrides={**OVERRIDES, "training.epochs": 5}, listener=listener,
        ))
    finally:
        signal.signal(signal.SIGINT, previous)

    cfg = load_config(str(exp_dir))
    assert cfg.run.status == "interrupted" and cfg.run.epochs_run == 1 and cfg.run.resumable
    assert Path(cfg.run.resume_checkpoint).name == "interrupt_epoch001.keras"
    assert (exp_dir / "interrupt_epoch001.keras").is_file()
    assert (exp_dir / "interrupt_epoch001.decoder.keras").is_file()


@pytest.mark.tf
def test_every_epoch_strategy_saves_and_prunes_all_artifacts(ae_installed, exps, tiny_flat_root):
    from cvbench.services.training import run_training

    exp_dir = Path(run_training(str(tiny_flat_root), overrides={
        **OVERRIDES, "training.epochs": 3, "training.checkpoints.strategy": "every_epoch",
        "training.checkpoints.keep_last_n": 1,
    }))
    assert sorted(p.name for p in exp_dir.glob("epoch_*")) == ["epoch_003.decoder.keras", "epoch_003.keras"]


@pytest.mark.tf
def test_the_checkpointer_applies_the_monitor_mode_and_rejects_unknown_strategies(tmp_path):
    from cvbench.core.checkpoint import ArtifactCheckpointer, load_best_history
    from cvbench.core.config import CVBenchConfig

    class _Model:
        def __init__(self):
            self.saved = []

        def save(self, path):
            self.saved.append(Path(path).name)

    class _Task:
        def artifact_path(self, checkpoint, name):
            return Path(checkpoint).with_name(f"x.{name}")

    cfg = CVBenchConfig()
    cfg.training.checkpoints.monitor, cfg.training.checkpoints.mode = "acc", "max"
    model = _Model()
    ckpt = ArtifactCheckpointer(cfg, tmp_path, _Task(), {"a": model})
    for epoch, value in enumerate([0.5, 0.4, 0.9]):
        ckpt.on_epoch_end(epoch, {"acc": value})
    ckpt.on_epoch_end(3, {"other": 1.0})  # monitor missing: nothing happens
    assert len(model.saved) == 2 and ckpt.best == 0.9
    assert [r["epoch"] for r in load_best_history(str(tmp_path))] == [1, 3]

    cfg.training.checkpoints.strategy = "sometimes"
    with pytest.raises(ConfigError, match="Unknown checkpoint strategy"):
        ckpt.on_epoch_end(4, {"acc": 1.0})


# ---- joint augmentation ------------------------------------------------------

def _write_pair(directory: Path, i: int):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(i)
    img = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[:, :6] = 255  # left strip — a flip moves it to the right
    Image.fromarray(img).save(directory / f"i{i}.png")
    Image.fromarray(mask).save(directory / f"m{i}.png")
    return str(directory / f"i{i}.png"), str(directory / f"m{i}.png")


@pytest.mark.tf
def test_geometric_augmentation_moves_image_and_mask_together(tmp_path):
    import numpy as np

    from cvbench.core.config import TransformConfig
    from cvbench.core.input_pipeline import build_image_dataset

    pairs = [_write_pair(tmp_path, i) for i in range(4)]
    images, masks = [p[0] for p in pairs], [p[1] for p in pairs]
    flip = [TransformConfig(name="keras_flip", params={"mode": "horizontal"}, prob=1.0)]

    ds = build_image_dataset(images, np.array(masks), 16, 4, training=True, seed=1,
                             transforms=flip, joint_targets=True)
    x, y = next(iter(ds))
    plain = build_image_dataset(images, np.array(masks), 16, 4, training=False, joint_targets=True)
    x0, y0 = next(iter(plain))

    assert y.shape == (4, 16, 16, 1) and set(np.unique(y.numpy())) <= {0.0, 255.0}
    # RandomFlip flips about half the samples; whichever way an image went, its mask went too
    flipped = 0
    for xi, yi in zip(x.numpy(), y.numpy(), strict=True):
        if any(np.allclose(xi, xo) for xo in x0.numpy()):
            assert yi[:, :6].min() == 255 and yi[:, 6:].max() == 0
        else:
            assert any(np.allclose(xi, xo[:, ::-1]) for xo in x0.numpy())
            assert yi[:, -6:].min() == 255 and yi[:, :-6].max() == 0
            flipped += 1
    assert 0 < flipped < 4
    assert np.all(y0.numpy()[:, :, :6] == 255) and np.all(y0.numpy()[:, :, 6:] == 0)


@pytest.mark.tf
def test_photometric_and_custom_transforms_leave_the_mask_untouched(tmp_path):
    import numpy as np

    from cvbench.core.config import TransformConfig
    from cvbench.core.input_pipeline import build_image_dataset

    pairs = [_write_pair(tmp_path, i) for i in range(2)]
    transforms = [
        TransformConfig(name="keras_brightness", params={}, prob=1.0),
        TransformConfig(name="aug_blur", params={"radius": 1.0}, prob=1.0),
    ]
    ds = build_image_dataset([p[0] for p in pairs], np.array([p[1] for p in pairs]), 16, 2,
                             training=True, seed=2, transforms=transforms, joint_targets=True)
    _, y = next(iter(ds))
    assert np.all(y.numpy()[:, :, :6] == 255) and np.all(y.numpy()[:, :, 6:] == 0)

    only_custom = build_image_dataset(
        [p[0] for p in pairs], np.array([p[1] for p in pairs]), 16, 2, training=True, seed=2,
        transforms=[TransformConfig(name="aug_blur", params={"radius": 1.0}, prob=1.0)], joint_targets=True,
    )
    assert next(iter(only_custom))[1].shape == (2, 16, 16, 1)


@pytest.mark.tf
def test_crop_resizes_the_mask_back_with_nearest_neighbour(tmp_path):
    import numpy as np

    from cvbench.core.config import TransformConfig
    from cvbench.core.input_pipeline import build_image_dataset

    pairs = [_write_pair(tmp_path, i) for i in range(2)]
    crop = [TransformConfig(name="keras_crop", params={"height": 8, "width": 8}, prob=1.0)]
    ds = build_image_dataset([p[0] for p in pairs], np.array([p[1] for p in pairs]), 16, 2,
                             training=True, seed=2, transforms=crop, joint_targets=True)
    x, y = next(iter(ds))
    assert x.shape == (2, 16, 16, 3) and y.shape == (2, 16, 16, 1)
    assert set(np.unique(y.numpy())) <= {0.0, 255.0}


@pytest.mark.tf
def test_a_joint_pipeline_without_keras_transforms_is_a_plain_one(tmp_path):
    import numpy as np

    from cvbench.augmentations.pipeline import build_joint_keras_aug_fn
    from cvbench.core.config import TransformConfig

    assert build_joint_keras_aug_fn([]) is None
    assert build_joint_keras_aug_fn([TransformConfig(name="aug_blur", params={"radius": 1.0}, prob=1.0)]) is None
    seeded = build_joint_keras_aug_fn(
        [TransformConfig(name="keras_flip", params={}, prob=1.0)], seed=5,
    )
    assert seeded is not None
    pairs = [_write_pair(tmp_path, 0)]
    from cvbench.core.input_pipeline import build_image_dataset

    plain = build_image_dataset([pairs[0][0]], np.array([pairs[0][1]]), 16, 1, training=True, seed=3,
                                joint_targets=True)
    assert next(iter(plain))[1].shape == (1, 16, 16, 1)
