from __future__ import annotations

from pathlib import Path

from cvbench.core import console

from .base import ConvertExporter, ExportOptions, quiet_converter

QUANTIZE_SUFFIX = {"none": "", "float16": "_float16", "int8": "_int8"}


@quiet_converter
def convert_tflite(model, output_path: Path, quantize: str) -> None:
    import tempfile

    import tensorflow as tf

    # TFLiteConverter.from_keras_model() fails with Keras 3 models due to an LLVM
    # type inference bug. The workaround is to export to SavedModel first, then
    # convert from that.
    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_model_path = Path(tmp_dir) / "saved_model"
        # Keras prints the SavedModel signature (~50 TensorSpec lines) via print_msg.
        import keras

        was_enabled = keras.config.is_interactive_logging_enabled()
        keras.config.disable_interactive_logging()
        try:
            model.export(str(saved_model_path))
        finally:
            if was_enabled:
                keras.config.enable_interactive_logging()

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
        if quantize in ("float16", "int8"):
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if quantize == "float16":
            converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

    output_path.write_bytes(tflite_model)


class TfliteExporter(ConvertExporter):
    name = "tflite"

    def subfolder(self, opts: ExportOptions) -> str:
        return f"tflite{QUANTIZE_SUFFIX.get(opts.quantize, '')}"

    def quantize_value(self, opts: ExportOptions) -> str | None:
        return opts.quantize

    def model_filename(self, opts: ExportOptions) -> str:
        return f"model{QUANTIZE_SUFFIX.get(opts.quantize, '')}.tflite"

    def convert(self, model, path: Path, cfg, opts: ExportOptions) -> None:
        print(console.dim(f"  Converting to TFLite (quantize={opts.quantize})..."))
        convert_tflite(model, path, opts.quantize)

    def print_instructions(self, run_dir, export_dir, cfg, opts: ExportOptions) -> None:
        if opts.quantize == "int8":
            print(
                console.dim(
                    " Note: int8 is dynamic-range quantization (int8 weights, float32"
                    " input/output), not full-integer."
                )
            )
