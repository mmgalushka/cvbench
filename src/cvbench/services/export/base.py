"""Shared export machinery: the Exporter interface and the steps every target repeats."""
from __future__ import annotations

import abc
import functools
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from cvbench.core import console
from cvbench.tasks import resolve_task

HAILO_HW_ARCHS = ("hailo8", "hailo8l", "hailo8r")


@dataclass
class ExportOptions:
    quantize: str = "none"
    output_dir: str | None = None
    calib_total: int = 1024
    calib_strategy: str = "proportional"
    hw_arch: str = "hailo8l"


def quiet_tf() -> None:
    """Silence TensorFlow and absl logging for the duration of an export."""
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    import absl.logging

    absl.logging.set_verbosity(absl.logging.ERROR)


def quiet_converter(fn):
    """Mute native converter stderr (TensorSpec/LLVM chatter); it is replayed if the call fails."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with console.quiet_native_stderr():
            return fn(*args, **kwargs)

    return wrapper


def load_model(cfg, checkpoint: Path):
    return resolve_task(cfg).load_model(str(checkpoint))


def build_export_info(
    cfg, checkpoint: Path, format: str, quantize: str | None, extra: dict
) -> dict:
    info = {
        "source_checkpoint": str(checkpoint),
        "format": format,
        "quantize": quantize,
        "input_shape": [1, cfg.model.input_size, cfg.model.input_size, 3],
        "input_dtype": "float32",
        "normalization": "internal",
        "classes": cfg.data.classes,
        "backbone": cfg.model.backbone,
        "val_accuracy": cfg.run.metrics.get("val_accuracy"),
        "val_loss": cfg.run.metrics.get("val_loss"),
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }
    info.update(extra)
    return info


def write_export_info(export_dir: Path, info: dict) -> None:
    (export_dir / "export_info.json").write_text(json.dumps(info, indent=2))


def print_banner(title: str) -> None:
    print(console.rule())
    print(f" {title}")
    print(console.rule())


def print_footer(message: str, export_dir: Path) -> None:
    print(console.rule())
    print(console.green(f" {message} → {export_dir}"))
    print(console.rule())


class Exporter(abc.ABC):
    """One export target. Register an instance in ``EXPORTERS`` to expose it."""

    name: ClassVar[str]
    footer: ClassVar[str] = "Package ready"

    def validate(self, opts: ExportOptions) -> None:
        """Reject bad options before any work starts; raise ConfigError."""
        return None

    def export_dir(self, run_dir: Path, opts: ExportOptions) -> Path:
        return run_dir / "export" / self.name

    def quantize_value(self, opts: ExportOptions) -> str | None:
        return None

    def banner(self, run_dir: Path, export_dir: Path, opts: ExportOptions) -> str:
        return f"{self.name} package — '{run_dir.name}'"

    @abc.abstractmethod
    def export(self, run_dir: Path, cfg, export_dir: Path, opts: ExportOptions) -> dict:
        """Write artifacts into ``export_dir``; return extra export_info fields."""

    def print_instructions(
        self, run_dir: Path, export_dir: Path, cfg, opts: ExportOptions
    ) -> None:
        """Print target-specific follow-up steps."""
        return None


class ConvertExporter(Exporter):
    """Exporter that converts the loaded model into a single file in a user-chosen folder."""

    footer = "Export complete"
    model_info: ClassVar[dict] = {}

    def export_dir(self, run_dir: Path, opts: ExportOptions) -> Path:
        base = Path(opts.output_dir) if opts.output_dir else run_dir / "export"
        return base / self.subfolder(opts)

    def subfolder(self, opts: ExportOptions) -> str:
        return self.name

    def banner(self, run_dir: Path, export_dir: Path, opts: ExportOptions) -> str:
        return f"Exporting '{run_dir.name}' → {export_dir.name}"

    @abc.abstractmethod
    def model_filename(self, opts: ExportOptions) -> str: ...

    @abc.abstractmethod
    def convert(self, model, path: Path, cfg, opts: ExportOptions) -> None: ...

    def export(self, run_dir: Path, cfg, export_dir: Path, opts: ExportOptions) -> dict:
        checkpoint = run_dir / "best.keras"
        print(console.dim(f"  Loading   {checkpoint}"))
        model = load_model(cfg, checkpoint)
        model_path = export_dir / self.model_filename(opts)
        self.convert(model, model_path, cfg, opts)
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  Written   {model_path.name}  ({size_mb:.1f} MB)")
        return dict(self.model_info)
