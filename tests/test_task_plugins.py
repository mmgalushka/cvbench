"""Task-owned config sections, layout priority and entry-point plugins (tasks and backbones)."""
import types
import warnings

import pytest
import yaml

from cvbench import tasks as tasks_mod
from cvbench.core import backbone as backbone_mod
from cvbench.core.backbone import BACKBONES, BackboneSpec
from cvbench.core.config import (
    CVBenchConfig,
    apply_overrides,
    build_config,
    config_from_dict,
    config_to_dict,
    load_config,
    resolve_key,
    save_config,
)
from cvbench.core.errors import CVBenchError
from cvbench.detection.config import DetectionConfig

PLUGIN_EP = types.SimpleNamespace(
    name="plugin_cls", module="tests.plugin_task", attr="PluginTask",
    value="tests.plugin_task:PluginTask",
)


def _install_task_eps(monkeypatch, *eps):
    real = tasks_mod.entry_points
    monkeypatch.setattr(
        tasks_mod, "entry_points",
        lambda group: list(eps) if group == "cvbench.tasks" else real(group=group),
    )


# ---- task-owned sections ---------------------------------------------------------------

DETECTION_YAML = """\
task: detection
data:
  data_dir: data
  train_dir: data/train
  val_dir: data/val
  test_dir: data/test
  classes:
  - cat
  - dog
  batch_size: 16
  val_split: 0.2
  val_split_explicit: false
model:
  backbone: resnet_18
  weights: imagenet
  input_size: 224
  num_classes: 2
  dropout: 0.2
  fine_tune_from_layer: 0
augmentation:
  transforms: []
training:
  epochs: 10
  learning_rate: 0.0001
detection:
  strides:
  - 8
  - 16
  anchors:
  - - - 0.1
      - 0.2
  anchors_per_scale: 1
  ignore_iou_threshold: 0.4
  noobj_weight: 0.3
  conf_threshold: 0.35
  iou_threshold: 0.6
  max_detections: 50
"""


def test_detection_yaml_round_trips_byte_for_byte(tmp_path):
    # Start from what save_config writes today, so the file is canonical, then load -> save.
    first = tmp_path / "a"
    cfg = config_from_dict(yaml.safe_load(DETECTION_YAML))
    save_config(cfg, str(first))
    written = (first / "config.yaml").read_text()
    assert "detection:\n  strides:\n  - 8" in written
    assert written.index("\ndetection:") < written.index("\nrun:")  # section sits before run

    second = tmp_path / "b"
    save_config(load_config(str(first)), str(second))
    assert (second / "config.yaml").read_bytes() == (first / "config.yaml").read_bytes()
    assert load_config(str(second)).detection == DetectionConfig(
        strides=[8, 16], anchors=[[[0.1, 0.2]]], anchors_per_scale=1,
        ignore_iou_threshold=0.4, noobj_weight=0.3, conf_threshold=0.35,
        iou_threshold=0.6, max_detections=50,
    )


def test_section_is_stored_in_sections_not_as_a_field():
    cfg = CVBenchConfig(task="detection")
    assert isinstance(cfg.sections["detection"], DetectionConfig)
    assert cfg.detection is cfg.sections["detection"]
    assert "detection" not in config_to_dict(CVBenchConfig())  # classification runs have none
    assert "sections" not in config_to_dict(cfg)
    with pytest.raises(AttributeError):
        CVBenchConfig().detection  # noqa: B018


def test_foreign_section_is_dropped_and_missing_section_gets_defaults():
    cls_cfg = config_from_dict({"task": "classification", "detection": {"conf_threshold": 0.9}})
    assert cls_cfg.sections == {}
    det_cfg = config_from_dict({"task": "detection"})
    assert det_cfg.detection == DetectionConfig()


def test_sections_key_in_yaml_is_rejected():
    with pytest.raises(CVBenchError, match="Unknown config key 'sections'"):
        config_from_dict({"sections": {}})


def test_set_detection_key_via_dotted_override():
    cfg = build_config("data", overrides={"task": "detection", "detection.conf_threshold": "0.3"})
    assert cfg.detection.conf_threshold == 0.3
    assert resolve_key("detection.max_detections") == "detection.max_detections"
    with pytest.raises(CVBenchError, match="Did you mean 'detection.conf_threshold'"):
        resolve_key("detection.conf_treshold")
    with pytest.raises(CVBenchError, match="expected float"):
        apply_overrides(cfg, {"detection.conf_threshold": "high"})
    with pytest.raises(CVBenchError, match="only applies to detection runs"):
        build_config("data", overrides={"detection.conf_threshold": 0.3})


def test_cli_set_option_resolves_detection_keys():
    from cvbench.cli.train import parse_set_option

    assert parse_set_option("detection.conf_threshold=0.3") == {"detection.conf_threshold": "0.3"}


def test_changing_task_resets_sections():
    cfg = build_config("data", overrides={"task": "detection", "detection.conf_threshold": 0.9})
    apply_overrides(cfg, {"task": "classification"})
    assert cfg.sections == {}
    apply_overrides(cfg, {"task": "detection"})
    assert cfg.detection.conf_threshold == DetectionConfig().conf_threshold


def test_detection_sweep_axis_expands_and_applies():
    from cvbench.core.sweep_store import SweepError, expand_grid, split_axis_values

    values = split_axis_values("detection.conf_threshold", "0.1,0.5")
    assert values == ["0.1", "0.5"]
    with pytest.raises(SweepError, match="ranges are not supported"):
        split_axis_values("detection.conf_threshold", "0.1:0.5")
    with pytest.raises(SweepError, match="Unknown config key"):
        split_axis_values("detection.nope", "1,2")

    trials = expand_grid({"detection.conf_threshold": values, "detection.max_detections": ["10", "20"]})
    assert len(trials) == 4
    cfgs = [build_config("data", overrides={"task": "detection", **{k: v for k, v in t.items()}}) for t in trials]
    assert [(c.detection.conf_threshold, c.detection.max_detections) for c in cfgs] == [
        (0.1, 10), (0.1, 20), (0.5, 10), (0.5, 20),
    ]


# ---- layout priority ---------------------------------------------------------------------

def _make_yolo(root):
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels").mkdir()


def test_detect_task_name_honours_priority(tmp_path, monkeypatch):
    _install_task_eps(monkeypatch, PLUGIN_EP)
    _make_yolo(tmp_path)
    (tmp_path / "PLUGIN_MARKER").touch()
    # detection (10) outranks the plugin (5), which outranks the catch-all classification (0)
    assert tasks_mod.detect_task_name(tmp_path) == "detection"
    other = tmp_path / "plain"
    other.mkdir()
    (other / "PLUGIN_MARKER").touch()
    assert tasks_mod.detect_task_name(other) == "plugin_cls"
    bare = tmp_path / "bare"
    bare.mkdir()
    assert tasks_mod.detect_task_name(bare) == "classification"


def test_raising_a_plugin_priority_makes_it_win(tmp_path, monkeypatch):
    from tests.plugin_task import PluginTask

    _install_task_eps(monkeypatch, PLUGIN_EP)
    _make_yolo(tmp_path)
    (tmp_path / "PLUGIN_MARKER").touch()
    monkeypatch.setattr(PluginTask, "layout_priority", 20)
    assert tasks_mod.detect_task_name(tmp_path) == "plugin_cls"


def test_layout_priorities_of_builtin_tasks():
    assert tasks_mod.get_task("detection").layout_priority == 10
    assert tasks_mod.get_task("classification").layout_priority == 0
    assert not hasattr(tasks_mod, "_SNIFF_ORDER")


# ---- entry-point tasks -------------------------------------------------------------------

def test_plugin_task_is_registered_lazily_and_owns_its_section(monkeypatch):
    import sys

    sys.modules.pop("tests.plugin_task", None)
    _install_task_eps(monkeypatch, PLUGIN_EP)
    assert tasks_mod.task_names() == ("classification", "detection", "plugin_cls")
    assert "tests.plugin_task" not in sys.modules  # listing names imports nothing
    task = tasks_mod.get_task("plugin_cls")
    assert task.name == "plugin_cls"

    cfg = build_config("data", overrides={"task": "plugin_cls", "plugin.strength": "0.9"})
    assert cfg.plugin.strength == 0.9 and cfg.plugin.label == "x"
    assert resolve_key("plugin.label") == "plugin.label"


def test_plugin_section_round_trips_through_config_yaml(tmp_path, monkeypatch):
    _install_task_eps(monkeypatch, PLUGIN_EP)
    cfg = build_config("data", overrides={"task": "plugin_cls", "plugin.strength": 0.25})
    save_config(cfg, str(tmp_path))
    text = (tmp_path / "config.yaml").read_text()
    assert "plugin:\n  strength: 0.25\n  label: x\n" in text
    reloaded = load_config(str(tmp_path))
    assert reloaded.task == "plugin_cls" and reloaded.plugin.strength == 0.25
    assert config_to_dict(reloaded) == config_to_dict(cfg)


def test_builtin_task_name_beats_plugin_with_warning(monkeypatch):
    clash = types.SimpleNamespace(
        name="detection", module="tests.plugin_task", attr="PluginTask", value="tests.plugin_task:PluginTask",
    )
    _install_task_eps(monkeypatch, clash)
    with pytest.warns(UserWarning, match="built-in task has that name"):
        task = tasks_mod.get_task("detection")
    assert type(task).__module__ == "cvbench.detection"


def test_unknown_task_error_lists_plugin_names(monkeypatch):
    _install_task_eps(monkeypatch, PLUGIN_EP)
    with pytest.raises(CVBenchError, match="plugin_cls"):
        tasks_mod.get_task("nope")
    with pytest.raises(CVBenchError, match="Unknown task 'nope'"):
        build_config("data", overrides={"task": "nope"})


# ---- entry-point backbones ---------------------------------------------------------------

@pytest.fixture
def isolated_backbones(monkeypatch):
    monkeypatch.setattr(backbone_mod, "BACKBONES", dict(BACKBONES))
    monkeypatch.setattr(backbone_mod, "_plugins_loaded", False)


def _install_backbone_eps(monkeypatch, *eps):
    monkeypatch.setattr(
        backbone_mod, "entry_points", lambda group: list(eps) if group == "cvbench.backbones" else []
    )


def _ep(name, value):
    return types.SimpleNamespace(name=name, value=f"pkg:{name}", load=lambda: value)


def test_backbone_plugins_load_once_on_first_use(isolated_backbones, monkeypatch):
    calls = []
    spec = BackboneSpec("plug_a", factory="pkg.mod:build")

    def many():
        calls.append(1)
        return [BackboneSpec("plug_b", factory="pkg.mod:b"), BackboneSpec("plug_c", factory="pkg.mod:c")]

    _install_backbone_eps(monkeypatch, _ep("a", spec), _ep("many", many))
    assert "plug_a" not in backbone_mod.BACKBONES  # nothing happens at import / registration time
    names = backbone_mod.backbone_names()
    assert names[-3:] == ["plug_a", "plug_b", "plug_c"]
    backbone_mod.backbone_names()
    assert calls == [1]  # loaded once


def test_backbone_plugins_load_before_resolving_a_backbone(isolated_backbones, monkeypatch):
    built = []
    _install_backbone_eps(monkeypatch, _ep("a", BackboneSpec("plug_a", factory="tests.plugin_task:make")))
    import tests.plugin_task as mod

    def make(cfg, load_weights):
        built.append(load_weights)
        raise RuntimeError("stop here")

    monkeypatch.setattr(mod, "make", make, raising=False)
    cfg = CVBenchConfig()
    cfg.model.backbone = "plug_a"
    cfg.model.weights = "none"
    with pytest.raises(RuntimeError, match="stop here"):
        backbone_mod._load_backbone(cfg)
    assert built == [False]


def test_builtin_backbone_beats_plugin_with_warning(isolated_backbones, monkeypatch):
    clash = BackboneSpec("resnet_18", factory="pkg.mod:build")
    _install_backbone_eps(monkeypatch, _ep("resnet_18", clash))
    with pytest.warns(UserWarning, match="Ignoring plugin backbone 'resnet_18'"):
        backbone_mod.backbone_names()
    assert backbone_mod.BACKBONES["resnet_18"].factory is None  # the built-in spec survived


def test_no_warning_without_plugins(isolated_backbones, monkeypatch):
    _install_backbone_eps(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert backbone_mod.backbone_names() == list(BACKBONES)
