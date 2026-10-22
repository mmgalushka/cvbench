from __future__ import annotations

from pathlib import Path

from cvbench.core import console
from cvbench.core.errors import ConfigError

from . import tflite as tflite_mod
from .base import HAILO_HW_ARCHS, Exporter, ExportOptions, load_model
from .calibration import build_calibration_set


def write_alls(output_path: Path) -> None:
    if output_path.exists():
        print(console.dim("  Reusing   model.alls"))
        return
    output_path.write_text(
        "model_optimization_flavor(optimization_level=2, compression_level=1)\n"
        "quantization_param(avgpool1, precision_mode=a16_w16)\n"
        "quantization_param(avgpool3, precision_mode=a16_w16)\n"
        "quantization_param(fc11, precision_mode=a16_w16)\n"
    )
    print(console.dim("  Written   model.alls"))


def prepare_hailo_package(
    run_dir: Path, cfg, calib_total: int = 1024, calib_strategy: str = "proportional"
) -> tuple[Path, dict]:
    import numpy as np

    export_dir = run_dir / "export" / "hailo"
    export_dir.mkdir(parents=True, exist_ok=True)

    tflite_path = export_dir / "model.tflite"
    if tflite_path.exists():
        print(console.dim("  Reusing   model.tflite"))
    else:
        model = load_model(cfg, run_dir / "best.keras")
        print(console.dim("  Converting to TFLite (float32)..."))
        tflite_mod.convert_tflite(model, tflite_path, quantize="none")

    calib_path = export_dir / "calib_set.npy"
    if calib_path.exists():
        print(console.dim("  Reusing   calib_set.npy"))
        shape = np.load(str(calib_path), mmap_mode="r").shape
        calib_meta = {"calib_set_images": shape[0], "calib_set_shuffled": True}
        if shape[0] < calib_total:
            print(
                console.yellow(
                    f"  Warning   existing calib_set.npy has {shape[0]} images, fewer than"
                    f" the {calib_total} requested; delete it to rebuild."
                )
            )
    else:
        n_total, n_classes = build_calibration_set(
            cfg, calib_path, total=calib_total, strategy=calib_strategy
        )
        calib_meta = {
            "calib_set_images": n_total,
            "calib_set_classes": n_classes,
            "calib_set_total": calib_total,
            "calib_set_strategy": calib_strategy,
            "calib_set_shuffled": True,
        }
    write_alls(export_dir / "model.alls")

    return export_dir, calib_meta


def print_hailo_instructions(
    run_name: str, export_dir: Path, hw_arch: str = "hailo8l"
) -> None:
    rel = f"experiments/{run_name}/export/hailo"
    print(console.rule())
    print(f" Hailo HEF — {hw_arch} deployment")
    print(console.rule())
    print()
    print(
        console.blue(
            " The Hailo conversion commands must be run inside the Hailo Docker container."
        )
    )
    print(
        console.blue(
            f" Mount or copy the export folder into the container: {rel}/"
        )
    )
    print()
    print(f" {console.bold('Step 1')} {console.dim('— parse TFLite to HAR:')}")
    print()
    print("   hailo parser tf model.tflite")
    print()
    print(
        f" {console.bold('Step 2')} {console.dim('— optimize with calibration data:')}"
    )
    print()
    print("   hailo optimize \\")
    print(f"       --hw-arch {hw_arch} \\")
    print("       --calib-set-path calib_set.npy \\")
    print("       --model-script model.alls \\")
    print("       --output-har-path model_optimized.har \\")
    print("       model.har")
    print()
    print(f" {console.bold('Step 3')} {console.dim('— compile to HEF:')}")
    print()
    print(f"   hailo compiler --hw-arch {hw_arch} model_optimized.har")
    print()
    print(console.rule())


class HailoExporter(Exporter):
    name = "hailo"

    def validate(self, opts: ExportOptions) -> None:
        if opts.hw_arch not in HAILO_HW_ARCHS:
            raise ConfigError(
                f"Unknown --hw-arch '{opts.hw_arch}'. Choose from: {', '.join(HAILO_HW_ARCHS)}"
            )

    def banner(self, run_dir: Path, export_dir: Path, opts: ExportOptions) -> str:
        return f"Hailo package — '{run_dir.name}'"

    def export(self, run_dir: Path, cfg, export_dir: Path, opts: ExportOptions) -> dict:
        _, calib_meta = prepare_hailo_package(
            run_dir, cfg, calib_total=opts.calib_total, calib_strategy=opts.calib_strategy
        )
        return {"hw_arch": opts.hw_arch, **calib_meta}

    def print_instructions(self, run_dir, export_dir, cfg, opts: ExportOptions) -> None:
        print_hailo_instructions(run_dir.name, export_dir, opts.hw_arch)
