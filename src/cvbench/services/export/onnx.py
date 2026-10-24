from __future__ import annotations

from pathlib import Path

from cvbench.core import console
from cvbench.core.errors import ExportError

from .base import ConvertExporter, ExportOptions, quiet_converter

JETSON_TRTEXEC = "trtexec --onnx=model.onnx --saveEngine=model.plan --noTF32"


@quiet_converter
def convert_onnx(model, output_path: Path, input_size: int) -> None:
    try:
        import tensorflow as tf
        import tf2onnx
    except ImportError:
        raise ExportError(
            "tf2onnx is required for ONNX export. "
            "Install it with: uv pip install 'cvbench[export]'"
        ) from None

    input_spec = (
        tf.TensorSpec([1, input_size, input_size, 3], tf.float32, name="image"),
    )
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=input_spec, opset=13
    )

    import onnx

    onnx.save(model_proto, str(output_path))


def print_onnx_snippet(model_name: str, input_size: int) -> None:
    print(console.dim(" Run it with onnxruntime (input: float32 NHWC, values 0–255):"))
    print()
    print("   import numpy as np, onnxruntime as ort")
    print(f'   sess = ort.InferenceSession("{model_name}")')
    print(f"   x = np.zeros((1, {input_size}, {input_size}, 3), dtype=np.float32)  # your image, 0-255")
    print('   probs = sess.run(None, {"image": x})[0]')
    print()


class OnnxExporter(ConvertExporter):
    name = "onnx"
    model_info = {"jetson_trtexec": JETSON_TRTEXEC}

    def model_filename(self, opts: ExportOptions) -> str:
        return "model.onnx"

    def convert(self, model, path: Path, cfg, opts: ExportOptions) -> None:
        print(console.dim("  Converting to ONNX (opset=13)..."))
        convert_onnx(model, path, cfg.model.input_size)

    def print_instructions(self, run_dir, export_dir, cfg, opts: ExportOptions) -> None:
        print_onnx_snippet("model.onnx", cfg.model.input_size)
