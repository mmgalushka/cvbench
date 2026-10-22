from __future__ import annotations

from pathlib import Path

from cvbench.core import console

from . import onnx as onnx_mod
from .base import Exporter, ExportOptions, load_model
from .onnx import JETSON_TRTEXEC


def print_plan_instructions(run_name: str, onnx_exists: bool) -> None:
    print(console.rule())
    print(" TensorRT Engine Plan — Jetson deployment")
    print(console.rule())
    print()
    print(
        console.blue(
            " A TensorRT .plan file must be built on the target Jetson device itself,"
        )
    )
    print(console.blue(" since it is compiled for a specific GPU architecture."))
    print()
    print(
        f" {console.bold('Step 1')} {console.dim('— copy the ONNX model to your Jetson:')}"
    )
    print()
    print(f"   scp experiments/{run_name}/export/plan/model.onnx \\")
    print("       user@jetson:/home/user/model.onnx")
    print()
    print(
        f" {console.bold('Step 2')} {console.dim('— on the Jetson, convert to TensorRT engine plan:')}"
    )
    print()
    print("   trtexec --onnx=model.onnx --saveEngine=model.plan --noTF32")
    print()
    print(
        f" {console.bold('Step 3')} {console.dim('— on the Jetson, save infer.py and run it:')}"
    )
    print()
    print(f"   predict {run_name} --format plan      # prints infer.py")
    print("   ./infer.py model.plan image.jpg")
    print()
    print(
        console.dim(
            " Or use the TensorRT Python API / DeepStream directly. Input is float32"
            " NHWC 0–255; normalisation is inside the model."
        )
    )
    print()
    if not onnx_exists:
        print(
            console.yellow(
                " Note: ONNX export not found. Generate it first with:"
            )
        )
        print()
        print(f"   runs export {run_name} --format onnx")
        print()
    print(console.rule())


class PlanExporter(Exporter):
    name = "plan"

    def banner(self, run_dir: Path, export_dir: Path, opts: ExportOptions) -> str:
        return f"TensorRT Plan package — '{run_dir.name}'"

    def export(self, run_dir: Path, cfg, export_dir: Path, opts: ExportOptions) -> dict:
        onnx_path = export_dir / "model.onnx"
        if onnx_path.exists():
            print(console.dim("  Reusing   model.onnx"))
        else:
            model = load_model(cfg, run_dir / "best.keras")
            print(console.dim("  Converting to ONNX (opset=13)..."))
            onnx_mod.convert_onnx(model, onnx_path, cfg.model.input_size)
        return {"jetson_trtexec": JETSON_TRTEXEC}

    def print_instructions(self, run_dir, export_dir, cfg, opts: ExportOptions) -> None:
        print_plan_instructions(run_dir.name, onnx_exists=True)
