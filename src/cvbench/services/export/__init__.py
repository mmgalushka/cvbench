"""Export a run's best checkpoint to a deployment target (see ``EXPORTERS``)."""
from __future__ import annotations

from pathlib import Path

from cvbench.core.config import load_config
from cvbench.core.errors import ConfigError, NotFoundError
from cvbench.core.exp_store import resolve_run_dir

from .base import (
    HAILO_HW_ARCHS,
    Exporter,
    ExportOptions,
    build_export_info,
    print_banner,
    print_footer,
    quiet_tf,
    write_export_info,
)
from .calibration import build_calibration_set
from .hailo import HailoExporter, prepare_hailo_package, print_hailo_instructions
from .onnx import OnnxExporter, print_onnx_snippet
from .plan import PlanExporter, print_plan_instructions
from .tflite import TfliteExporter

EXPORTERS: dict[str, Exporter] = {
    e.name: e for e in (TfliteExporter(), OnnxExporter(), PlanExporter(), HailoExporter())
}

__all__ = [
    "EXPORTERS",
    "HAILO_HW_ARCHS",
    "Exporter",
    "ExportOptions",
    "build_calibration_set",
    "prepare_hailo_package",
    "print_hailo_instructions",
    "print_onnx_snippet",
    "print_plan_instructions",
    "run_export",
]


def run_export(
    experiment: str,
    format: str,
    quantize: str = "none",
    output_dir: str | None = None,
    calib_total: int = 1024,
    calib_strategy: str = "proportional",
    hw_arch: str = "hailo8l",
) -> Path | None:
    """Export the best checkpoint of a run to TFLite, ONNX, or print Jetson plan instructions.

    Exported models take float32 NHWC input in the 0-255 range; normalisation
    happens inside the model. ``output_dir`` applies to tflite/onnx only (a
    ``<format>`` subfolder is created inside it); plan and hailo always write
    to ``<run>/export/<format>/``.

    Returns the path to the export subfolder.
    """
    exporter = EXPORTERS.get(format)
    if exporter is None:
        raise ConfigError(
            f"Unknown export format '{format}'. Choose from: {', '.join(EXPORTERS)}"
        )
    opts = ExportOptions(quantize, output_dir, calib_total, calib_strategy, hw_arch)
    exporter.validate(opts)

    run_dir = Path(resolve_run_dir(experiment))
    quiet_tf()
    cfg = load_config(str(run_dir))
    checkpoint = run_dir / "best.keras"
    if not checkpoint.exists():
        raise NotFoundError(f"No best.keras found in: {run_dir}")

    export_dir = exporter.export_dir(run_dir, opts)
    export_dir.mkdir(parents=True, exist_ok=True)
    print_banner(exporter.banner(run_dir, export_dir, opts))

    extra = exporter.export(run_dir, cfg, export_dir, opts)
    info = build_export_info(cfg, checkpoint, exporter.name, exporter.quantize_value(opts), extra)
    write_export_info(export_dir, info)

    print("  Written   export_info.json")
    print_footer(exporter.footer, export_dir)
    exporter.print_instructions(run_dir, export_dir, cfg, opts)
    return export_dir
