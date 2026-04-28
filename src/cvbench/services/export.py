from __future__ import annotations

import json
import os
import random
import warnings
from datetime import datetime
from pathlib import Path

import keras

from cvbench.core import _fmt
from cvbench.core.config import load_config
from cvbench.core.runs import resolve_run_dir

_TFLITE_QUANTIZE_SUFFIX = {"none": "", "float16": "_float16", "int8": "_int8"}


def _subfolder_name(format: str, quantize: str) -> str:
    if format == "tflite":
        suffix = _TFLITE_QUANTIZE_SUFFIX.get(quantize, "")
        return f"tflite{suffix}"
    return format


def _export_tflite(model, output_path: Path, quantize: str) -> None:
    import tempfile
    import tensorflow as tf

    # TFLiteConverter.from_keras_model() fails with Keras 3 models due to an LLVM
    # type inference bug. The workaround is to export to SavedModel first, then
    # convert from that.
    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_model_path = Path(tmp_dir) / "saved_model"
        model.export(str(saved_model_path))

        converter = tf.lite.TFLiteConverter.from_saved_model(
            str(saved_model_path)
        )
        if quantize in ("float16", "int8"):
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if quantize == "float16":
            converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

    output_path.write_bytes(tflite_model)


def _export_onnx(model, output_path: Path, input_size: int) -> None:
    try:
        import tf2onnx
        import tensorflow as tf
    except ImportError:
        raise RuntimeError(
            "tf2onnx is required for ONNX export. "
            "Install it with: pip install 'cvbench[export]'"
        )

    input_spec = (
        tf.TensorSpec(
            [1, input_size, input_size, 3], tf.float32, name="image"
        ),
    )
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=input_spec, opset=13
    )

    import onnx

    onnx.save(model_proto, str(output_path))


def _print_plan_instructions(run_name: str, onnx_exists: bool) -> None:
    print(_fmt.rule())
    print(f" TensorRT Engine Plan — Jetson deployment")
    print(_fmt.rule())
    print()
    print(
        _fmt.blue(
            " A TensorRT .plan file must be built on the target Jetson device itself,"
        )
    )
    print(_fmt.blue(" since it is compiled for a specific GPU architecture."))
    print()
    print(
        f" {_fmt.bold('Step 1')} {_fmt.dim('— copy the ONNX model to your Jetson:')}"
    )
    print()
    print(f"   scp experiments/{run_name}/export/onnx/model.onnx \\")
    print("       user@jetson:/home/user/model.onnx")
    print()
    print(
        f" {_fmt.bold('Step 2')} {_fmt.dim('— on the Jetson, convert to TensorRT engine plan:')}"
    )
    print()
    print("   trtexec --onnx=model.onnx --saveEngine=model.plan --noTF32")
    print()
    print(
        f" {_fmt.bold('Step 3')} {_fmt.dim('— run inference using the TensorRT Python API or DeepStream.')}"
    )
    print()
    if not onnx_exists:
        print(
            _fmt.yellow(
                " Note: ONNX export not found. Generate it first with:"
            )
        )
        print()
        print(f"   runs export {run_name} --format onnx")
        print()
    print(_fmt.rule())


def _collect_images(directory: Path) -> list[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for ext in exts:
        files.extend(directory.rglob(ext))
    return files


def _build_calibration_set(cfg, output_path: Path, images_per_class: int) -> None:
    import numpy as np
    from PIL import Image

    source_dir = Path(
        cfg.data.train_dir
        if os.path.isdir(cfg.data.train_dir)
        else cfg.data.val_dir
    )
    size = cfg.model.input_size
    rng = random.Random(42)

    # Collect images grouped by class subdirectory.
    class_dirs = sorted(p for p in source_dir.iterdir() if p.is_dir())
    if class_dirs:
        per_class: dict[str, list[Path]] = {
            d.name: _collect_images(d) for d in class_dirs
        }
        per_class = {k: v for k, v in per_class.items() if v}
    else:
        # Flat dataset — treat as a single bucket.
        all_files = _collect_images(source_dir)
        per_class = {"_all": all_files}

    if not per_class:
        raise RuntimeError(f"No images found in: {source_dir}")

    n_classes = len(per_class)

    selected: list[Path] = []
    for files in per_class.values():
        take = min(images_per_class, len(files))
        chosen = rng.sample(files, take)
        selected.extend(chosen)

    print(
        _fmt.dim(
            f"  Building  calib_set.npy ({len(selected)} images, "
            f"{images_per_class}/class across {n_classes} class(es), {size}×{size})..."
        )
    )

    normalize = getattr(cfg.model, "normalization", "internal") == "external"
    calib_data = []
    for img_path in sorted(selected):
        img = Image.open(str(img_path)).convert("RGB")
        img = img.resize((size, size))
        arr = np.array(img, dtype=np.float32)
        calib_data.append(arr / 255.0 if normalize else arr)

    calib_set = np.array(calib_data)
    np.save(str(output_path), calib_set)
    print(_fmt.dim(f"  Written   calib_set.npy  shape={calib_set.shape}"))


def _write_alls(output_path: Path) -> None:
    output_path.write_text(
        "model_optimization_flavor(optimization_level=2, compression_level=1)\n"
    )
    print(_fmt.dim("  Written   model.alls"))


def _prepare_hailo_package(run_dir: Path, cfg, images_per_class: int = 32) -> Path:
    export_dir = run_dir / "export" / "hailo"
    export_dir.mkdir(parents=True, exist_ok=True)

    tflite_path = export_dir / "model.tflite"
    if tflite_path.exists():
        print(_fmt.dim(f"  Reusing   model.tflite"))
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Skipping variable loading for optimizer"
            )
            model = keras.saving.load_model(str(run_dir / "best.keras"))
        print(_fmt.dim("  Converting to TFLite (float32)..."))
        _export_tflite(model, tflite_path, quantize="none")

    _build_calibration_set(cfg, export_dir / "calib_set.npy", images_per_class)
    _write_alls(export_dir / "model.alls")

    return export_dir


def _print_hailo_instructions(run_name: str, export_dir: Path) -> None:
    rel = f"experiments/{run_name}/export/hailo"
    print(_fmt.rule())
    print(f" Hailo HEF — hailo8l deployment")
    print(_fmt.rule())
    print()
    print(
        _fmt.blue(
            " The Hailo conversion commands must be run inside the Hailo Docker container."
        )
    )
    print(
        _fmt.blue(
            f" Mount or copy the export folder into the container: {rel}/"
        )
    )
    print()
    print(f" {_fmt.bold('Step 1')} {_fmt.dim('— parse TFLite to HAR:')}")
    print()
    print("   hailo parser tf model.tflite")
    print()
    print(
        f" {_fmt.bold('Step 2')} {_fmt.dim('— optimize with calibration data:')}"
    )
    print()
    print("   hailo optimize \\")
    print("       --hw-arch hailo8l \\")
    print("       --calib-set-path calib_set.npy \\")
    print("       --model-script model.alls \\")
    print("       --output-har-path model_optimized.har \\")
    print("       model.har")
    print()
    print(f" {_fmt.bold('Step 3')} {_fmt.dim('— compile to HEF:')}")
    print()
    print("   hailo compiler --hw-arch hailo8l model_optimized.har")
    print()
    print(_fmt.rule())


def run_export(
    experiment: str,
    format: str,
    quantize: str = "none",
    output_dir: str | None = None,
    images_per_class: int = 32,
) -> Path | None:
    """Export the best checkpoint of a run to TFLite, ONNX, or print Jetson plan instructions.

    Returns the path to the export subfolder (None for plan format).
    """
    run_dir = Path(resolve_run_dir(experiment))

    if format == "plan":
        onnx_path = run_dir / "export" / "onnx" / "model.onnx"
        _print_plan_instructions(run_dir.name, onnx_exists=onnx_path.exists())
        return None

    if format == "hailo":
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)

        cfg = load_config(str(run_dir))
        checkpoint = run_dir / "best.keras"
        if not checkpoint.exists():
            raise FileNotFoundError(f"No best.keras found in: {run_dir}")

        print(_fmt.rule())
        print(f" Hailo package — '{run_dir.name}'")
        print(_fmt.rule())

        export_dir = _prepare_hailo_package(run_dir, cfg, images_per_class)

        export_info = {
            "source_checkpoint": str(checkpoint),
            "format": "hailo",
            "quantize": None,
            "input_shape": [1, cfg.model.input_size, cfg.model.input_size, 3],
            "input_dtype": "float32",
            "classes": cfg.data.classes,
            "backbone": cfg.model.backbone,
            "val_accuracy": cfg.run.val_accuracy,
            "val_loss": cfg.run.val_loss,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
        }
        (export_dir / "export_info.json").write_text(
            json.dumps(export_info, indent=2)
        )
        print(f"  Written   export_info.json")
        print(_fmt.rule())
        print(_fmt.green(f" Package ready → {export_dir}"))
        print(_fmt.rule())
        _print_hailo_instructions(run_dir.name, export_dir)
        return export_dir

    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    import absl.logging

    absl.logging.set_verbosity(absl.logging.ERROR)

    cfg = load_config(str(run_dir))

    checkpoint = run_dir / "best.keras"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No best.keras found in: {run_dir}")

    subfolder = _subfolder_name(format, quantize)
    export_base = Path(output_dir) if output_dir else run_dir / "export"
    export_dir = export_base / subfolder
    export_dir.mkdir(parents=True, exist_ok=True)

    print(_fmt.rule())
    print(f" Exporting '{run_dir.name}' → {subfolder}")
    print(_fmt.rule())
    print(_fmt.dim(f"  Loading   {checkpoint}"))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Skipping variable loading for optimizer"
        )
        model = keras.saving.load_model(str(checkpoint))

    if format == "tflite":
        model_filename = (
            f"model{_TFLITE_QUANTIZE_SUFFIX.get(quantize, '')}.tflite"
        )
        model_path = export_dir / model_filename
        print(_fmt.dim(f"  Converting to TFLite (quantize={quantize})..."))
        _export_tflite(model, model_path, quantize)
    elif format == "onnx":
        model_path = export_dir / "model.onnx"
        print(_fmt.dim(f"  Converting to ONNX (opset=13)..."))
        _export_onnx(model, model_path, cfg.model.input_size)

    size_mb = model_path.stat().st_size / (1024 * 1024)

    export_info = {
        "source_checkpoint": str(checkpoint),
        "format": format,
        "quantize": quantize if format == "tflite" else None,
        "input_shape": [1, cfg.model.input_size, cfg.model.input_size, 3],
        "input_dtype": "float32",
        "classes": cfg.data.classes,
        "backbone": cfg.model.backbone,
        "val_accuracy": cfg.run.val_accuracy,
        "val_loss": cfg.run.val_loss,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }
    if format == "onnx":
        export_info["jetson_trtexec"] = (
            "trtexec --onnx=model.onnx --saveEngine=model.plan --noTF32"
        )
    info_path = export_dir / "export_info.json"
    info_path.write_text(json.dumps(export_info, indent=2))

    print(f"  Written   {model_path.name}  ({size_mb:.1f} MB)")
    print(f"  Written   export_info.json")
    print(_fmt.rule())
    print(_fmt.green(f" Export complete → {export_dir}"))
    print(_fmt.rule())

    return export_dir
