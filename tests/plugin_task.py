"""A minimal external task, used to prove plugins are picked up through entry points."""
from dataclasses import dataclass
from pathlib import Path

from cvbench.classification import ClassificationTask


@dataclass
class PluginConfig:
    strength: float = 0.5
    label: str = "x"


class PluginTask(ClassificationTask):
    """Classification behaviour under another name, with its own config section."""

    name = "plugin_cls"
    short_name = "plg"
    layout_priority = 5
    config_section = ("plugin", PluginConfig)

    @classmethod
    def matches_layout(cls, data_dir: Path) -> bool:
        return (Path(data_dir) / "PLUGIN_MARKER").exists()

    def evaluate(self, *args, **kwargs):
        report = super().evaluate(*args, **kwargs)
        report["task"] = self.name  # the classification evaluator stamps its own name
        return report
