from __future__ import annotations

import functools
import json
import os
import random
from datetime import datetime
from pathlib import Path

from cvbench.core import console
from cvbench.core.config import load_config
from cvbench.core.errors import ConfigError, DatasetError, ExportError, NotFoundError
from cvbench.core.exp_store import resolve_run_dir
from cvbench.datasets.layout import list_images
from cvbench.tasks import resolve_task

_TFLITE_QUANTIZE_SUFFIX = {"none": "", "float16": "_float16", "int8": "_int8"}

HAILO_HW_ARCHS = ("hailo8", "hailo8l", "hailo8r")


def _subfolder_name(format: str, quantize: str) -> str:
    if format == "tflite":
        suffix = _TFLITE_QUANTIZE_SUFFIX.get(quantize, "")
        return f"tflite{suffix}"
    return format


def _quiet_converter(fn):
    """Mute native converter stderr (TensorSpec/LLVM chatter); it is replayed if the call fails."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with console.quiet_native_stderr():
            return fn(*args, **kwargs)

    return wrapper


@_quiet_converter
def _export_tflite(model, output_path: Path, quantize: str) -> None:
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

        converter = tf.lite.TFLiteConverter.from_saved_model(
            str(saved_model_path)
        )
        if quantize in ("float16", "int8"):
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if quantize == "float16":
            converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

    output_path.write_bytes(tflite_model)


@_quiet_converter
def _export_onnx(model, output_path: Path, input_size: int) -> None:
    try:
        import tensorflow as tf
        import tf2onnx
    except ImportError:
        raise ExportError(
            "tf2onnx is required for ONNX export. "
            "Install it with: pip install 'cvbench[export]'"
        ) from None

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


def _prepare_plan_package(run_dir: Path, cfg) -> Path:
    export_dir = run_dir / "export" / "plan"
    export_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = export_dir / "model.onnx"
    if onnx_path.exists():
        print(console.dim("  Reusing   model.onnx"))
    else:
        model = resolve_task(cfg).load_model(str(run_dir / "best.keras"))
        print(console.dim("  Converting to ONNX (opset=13)..."))
        _export_onnx(model, onnx_path, cfg.model.input_size)

    return export_dir


def _print_plan_instructions(run_name: str, onnx_exists: bool) -> None:
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


def _print_onnx_snippet(model_name: str, input_size: int) -> None:
    print(console.dim(" Run it with onnxruntime (input: float32 NHWC, values 0–255):"))
    print()
    print("   import numpy as np, onnxruntime as ort")
    print(f'   sess = ort.InferenceSession("{model_name}")')
    print(f"   x = np.zeros((1, {input_size}, {input_size}, 3), dtype=np.float32)  # your image, 0-255")
    print('   probs = sess.run(None, {"image": x})[0]')
    print()


def _build_calibration_set(
    cfg, output_path: Path, total: int = 1024, strategy: str = "stratified"
) -> tuple[int, int]:
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
            d.name: list_images(d, recursive=True) for d in class_dirs
        }
        per_class = {k: v for k, v in per_class.items() if v}
    else:
        # Flat dataset — treat as a single bucket.
        all_files = list_images(source_dir, recursive=True)
        per_class = {"_all": all_files}

    if not per_class:
        raise DatasetError(f"No images found in: {source_dir}")

    n_classes = len(per_class)
    selected: list[Path] = []

    if strategy == "stratified":
        from sklearn.cluster import MiniBatchKMeans

        per_class_quota = max(1, round(total / n_classes))
        for files in per_class.values():
            quota = min(per_class_quota, len(files))
            n_clusters = min(quota, len(files))
            if n_clusters < len(files):
                feats = []
                for p in files:
                    img = Image.open(str(p)).convert("L").resize((8, 8), Image.Resampling.LANCZOS)
                    feats.append(np.array(img, dtype=np.float32).flatten() / 255.0)
                feats_np = np.stack(feats)
                labels = MiniBatchKMeans(
                    n_clusters=n_clusters, random_state=42, n_init=3
                ).fit_predict(feats_np)
                clusters: dict[int, list[Path]] = {}
                for path, label in zip(files, labels, strict=True):
                    clusters.setdefault(int(label), []).append(path)
                for bucket in clusters.values():
                    selected.append(rng.choice(bucket))
            else:
                selected.extend(files)
    elif strategy == "equal":
        per_class_quota = max(1, round(total / n_classes))
        for files in per_class.values():
            take = min(per_class_quota, len(files))
            selected.extend(rng.sample(files, take))
    elif strategy == "diverse":
        from sklearn.cluster import MiniBatchKMeans

        flat_files: list[Path] = []
        for files in per_class.values():
            flat_files.extend(files)

        # Extract compact feature: 8×8 grayscale thumbnail, flattened and normalised.
        features = []
        for p in flat_files:
            img = Image.open(str(p)).convert("L").resize((8, 8), Image.Resampling.LANCZOS)
            features.append(np.array(img, dtype=np.float32).flatten() / 255.0)
        features_np = np.stack(features)

        N_CLUSTERS = min(32, len(flat_files))
        kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=3)
        labels = kmeans.fit_predict(features_np)

        buckets: list[list[Path]] = [[] for _ in range(N_CLUSTERS)]
        for path, label in zip(flat_files, labels, strict=True):
            buckets[label].append(path)

        non_empty = [c for c in buckets if c]
        per_cluster_quota = max(1, round(total / len(non_empty)))
        for cluster in non_empty:
            take = min(per_cluster_quota, len(cluster))
            selected.extend(rng.sample(cluster, take))

        n_classes = len(non_empty)
    else:
        total_available = sum(len(v) for v in per_class.values())
        for files in per_class.values():
            proportion = len(files) / total_available
            quota = max(1, round(proportion * total))
            take = min(quota, len(files))
            selected.extend(rng.sample(files, take))

    # Shuffle so the array is not ordered by class — critical for Hailo
    # calibration algorithms that process images in sequential mini-batches.
    rng.shuffle(selected)

    unit = "cluster(s)" if strategy == "diverse" else "class(es)"  # stratified also uses class(es)
    print(
        console.dim(
            f"  Building  calib_set.npy ({len(selected)} images, "
            f"{strategy} across {n_classes} {unit}, {size}×{size})..."
        )
    )

    calib_data = []
    for img_path in selected:
        img = Image.open(str(img_path)).convert("RGB")
        img = img.resize((size, size))
        arr = np.array(img, dtype=np.float32)
        calib_data.append(arr)

    calib_set = np.array(calib_data)
    np.save(str(output_path), calib_set)
    print(console.dim(f"  Written   calib_set.npy  shape={calib_set.shape}"))
    if len(selected) < total:
        print(
            console.yellow(
                f"  Warning   calibration set has {len(selected)} images, fewer than the"
                f" {total} requested (not enough distinct training images for"
                f" '{strategy}'); quantization may be less accurate."
            )
        )
    return len(selected), n_classes


def _write_alls(output_path: Path) -> None:
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


def _prepare_hailo_package(
    run_dir: Path, cfg, calib_total: int = 1024, calib_strategy: str = "proportional"
) -> tuple[Path, dict]:
    import numpy as np

    export_dir = run_dir / "export" / "hailo"
    export_dir.mkdir(parents=True, exist_ok=True)

    tflite_path = export_dir / "model.tflite"
    if tflite_path.exists():
        print(console.dim("  Reusing   model.tflite"))
    else:
        model = resolve_task(cfg).load_model(str(run_dir / "best.keras"))
        print(console.dim("  Converting to TFLite (float32)..."))
        _export_tflite(model, tflite_path, quantize="none")

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
        n_total, n_classes = _build_calibration_set(
            cfg, calib_path, total=calib_total, strategy=calib_strategy
        )
        calib_meta = {
            "calib_set_images": n_total,
            "calib_set_classes": n_classes,
            "calib_set_total": calib_total,
            "calib_set_strategy": calib_strategy,
            "calib_set_shuffled": True,
        }
    _write_alls(export_dir / "model.alls")

    return export_dir, calib_meta


def _print_hailo_instructions(
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
    if hw_arch not in HAILO_HW_ARCHS:
        raise ConfigError(
            f"Unknown --hw-arch '{hw_arch}'. Choose from: {', '.join(HAILO_HW_ARCHS)}"
        )
    run_dir = Path(resolve_run_dir(experiment))

    if format == "plan":
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)

        cfg = load_config(str(run_dir))
        checkpoint = run_dir / "best.keras"
        if not checkpoint.exists():
            raise NotFoundError(f"No best.keras found in: {run_dir}")

        print(console.rule())
        print(f" TensorRT Plan package — '{run_dir.name}'")
        print(console.rule())

        export_dir = _prepare_plan_package(run_dir, cfg)

        export_info = {
            "source_checkpoint": str(checkpoint),
            "format": "plan",
            "quantize": None,
            "input_shape": [1, cfg.model.input_size, cfg.model.input_size, 3],
            "input_dtype": "float32",
            "normalization": "internal",
            "classes": cfg.data.classes,
            "backbone": cfg.model.backbone,
            "val_accuracy": cfg.run.metrics.get("val_accuracy"),
            "val_loss": cfg.run.metrics.get("val_loss"),
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "jetson_trtexec": "trtexec --onnx=model.onnx --saveEngine=model.plan --noTF32",
        }
        (export_dir / "export_info.json").write_text(
            json.dumps(export_info, indent=2)
        )
        print("  Written   export_info.json")
        print(console.rule())
        print(console.green(f" Package ready → {export_dir}"))
        print(console.rule())
        _print_plan_instructions(run_dir.name, onnx_exists=True)
        return export_dir

    if format == "hailo":
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)

        cfg = load_config(str(run_dir))
        checkpoint = run_dir / "best.keras"
        if not checkpoint.exists():
            raise NotFoundError(f"No best.keras found in: {run_dir}")

        print(console.rule())
        print(f" Hailo package — '{run_dir.name}'")
        print(console.rule())

        export_dir, calib_meta = _prepare_hailo_package(
            run_dir, cfg, calib_total=calib_total, calib_strategy=calib_strategy
        )

        export_info = {
            "source_checkpoint": str(checkpoint),
            "format": "hailo",
            "quantize": None,
            "input_shape": [1, cfg.model.input_size, cfg.model.input_size, 3],
            "input_dtype": "float32",
            "normalization": "internal",
            "classes": cfg.data.classes,
            "backbone": cfg.model.backbone,
            "val_accuracy": cfg.run.metrics.get("val_accuracy"),
            "val_loss": cfg.run.metrics.get("val_loss"),
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "hw_arch": hw_arch,
            **calib_meta,
        }
        (export_dir / "export_info.json").write_text(
            json.dumps(export_info, indent=2)
        )
        print("  Written   export_info.json")
        print(console.rule())
        print(console.green(f" Package ready → {export_dir}"))
        print(console.rule())
        _print_hailo_instructions(run_dir.name, export_dir, hw_arch)
        return export_dir

    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    import absl.logging

    absl.logging.set_verbosity(absl.logging.ERROR)

    cfg = load_config(str(run_dir))

    checkpoint = run_dir / "best.keras"
    if not checkpoint.exists():
        raise NotFoundError(f"No best.keras found in: {run_dir}")

    subfolder = _subfolder_name(format, quantize)
    export_base = Path(output_dir) if output_dir else run_dir / "export"
    export_dir = export_base / subfolder
    export_dir.mkdir(parents=True, exist_ok=True)

    print(console.rule())
    print(f" Exporting '{run_dir.name}' → {subfolder}")
    print(console.rule())
    print(console.dim(f"  Loading   {checkpoint}"))

    model = resolve_task(cfg).load_model(str(checkpoint))

    if format == "tflite":
        model_filename = (
            f"model{_TFLITE_QUANTIZE_SUFFIX.get(quantize, '')}.tflite"
        )
        model_path = export_dir / model_filename
        print(console.dim(f"  Converting to TFLite (quantize={quantize})..."))
        _export_tflite(model, model_path, quantize)
    elif format == "onnx":
        model_path = export_dir / "model.onnx"
        print(console.dim("  Converting to ONNX (opset=13)..."))
        _export_onnx(model, model_path, cfg.model.input_size)

    size_mb = model_path.stat().st_size / (1024 * 1024)

    export_info = {
        "source_checkpoint": str(checkpoint),
        "format": format,
        "quantize": quantize if format == "tflite" else None,
        "input_shape": [1, cfg.model.input_size, cfg.model.input_size, 3],
        "input_dtype": "float32",
        "normalization": "internal",
        "classes": cfg.data.classes,
        "backbone": cfg.model.backbone,
        "val_accuracy": cfg.run.metrics.get("val_accuracy"),
        "val_loss": cfg.run.metrics.get("val_loss"),
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }
    if format == "onnx":
        export_info["jetson_trtexec"] = (
            "trtexec --onnx=model.onnx --saveEngine=model.plan --noTF32"
        )
    info_path = export_dir / "export_info.json"
    info_path.write_text(json.dumps(export_info, indent=2))

    print(f"  Written   {model_path.name}  ({size_mb:.1f} MB)")
    print("  Written   export_info.json")
    print(console.rule())
    print(console.green(f" Export complete → {export_dir}"))
    print(console.rule())
    if format == "tflite" and quantize == "int8":
        print(
            console.dim(
                " Note: int8 is dynamic-range quantization (int8 weights, float32"
                " input/output), not full-integer."
            )
        )
    if format == "onnx":
        _print_onnx_snippet(model_path.name, cfg.model.input_size)

    return export_dir
