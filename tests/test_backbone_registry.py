"""The backbone registry: names, registration rules and how a spec is resolved."""
import pytest

from cvbench.core import backbone as backbone_mod
from cvbench.core.backbone import BACKBONES, BackboneSpec, backbone_names, register_backbone
from cvbench.core.config import CVBenchConfig
from cvbench.core.errors import ConfigError


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    monkeypatch.setattr(backbone_mod, "BACKBONES", dict(BACKBONES))


def test_registry_lists_the_shipped_backbones():
    assert backbone_names() == [
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
        "efficientnet_b4", "efficientnet_b5", "resnet_18", "resnet_50",
    ]
    quant = [n for n, s in BACKBONES.items() if s.quant_friendly]
    assert quant == ["resnet_18", "resnet_50"]


def test_unknown_backbone_raises_config_error():
    cfg = CVBenchConfig()
    cfg.model.backbone = "resnet_99"
    with pytest.raises(ConfigError, match="Unknown backbone 'resnet_99'.*resnet_18"):
        backbone_mod._load_backbone(cfg)


def test_empty_backbone_is_unknown():
    with pytest.raises(ConfigError, match="Unknown backbone ''"):
        backbone_mod._load_backbone(CVBenchConfig())


def test_duplicate_registration_raises_config_error():
    with pytest.raises(ConfigError, match="already registered"):
        register_backbone(BackboneSpec("resnet_18", "keras_hub.models.ResNetBackbone", "resnet_18_imagenet"))


@pytest.mark.parametrize("spec", [
    BackboneSpec("empty"),
    BackboneSpec("no_preset", hub_class="keras_hub.models.ResNetBackbone"),
    BackboneSpec("both", hub_class="a.B", preset="p", factory="m:f"),
], ids=lambda s: s.name)
def test_incomplete_or_ambiguous_spec_is_rejected(spec):
    with pytest.raises(ConfigError, match=spec.name):
        register_backbone(spec)
    assert spec.name not in backbone_names()


class _FakeBackbone:
    trainable = True
    layers: list = []


_factory_calls = []


def _fake_factory(cfg, load_weights):
    _factory_calls.append((cfg.model.backbone, load_weights))
    return _FakeBackbone()


def test_factory_path_is_used_and_freeze_applied():
    _factory_calls.clear()
    register_backbone(BackboneSpec("custom", factory=f"{__name__}:_fake_factory"))
    assert "custom" in backbone_names()

    cfg = CVBenchConfig()
    cfg.model.backbone = "custom"
    cfg.model.weights = "none"
    model = backbone_mod._load_backbone(cfg)

    assert isinstance(model, _FakeBackbone)
    assert _factory_calls == [("custom", False)]
    assert model.trainable is False  # fine_tune_from_layer == 0 freezes it


_hub_calls = []


class _Hub:
    @staticmethod
    def from_preset(preset, **kwargs):
        _hub_calls.append((preset, kwargs))
        return _FakeBackbone()


def test_hub_class_is_resolved_by_import_path():
    _hub_calls.clear()
    register_backbone(BackboneSpec("hubbed", hub_class=f"{__name__}._Hub", preset="p1"))

    cfg = CVBenchConfig()
    cfg.model.backbone = "hubbed"
    backbone_mod._load_backbone(cfg, name="backbone")

    assert _hub_calls == [("p1", {"load_weights": True, "name": "backbone"})]
