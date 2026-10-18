from pathlib import Path

import click

from cvbench.cli import _help
from cvbench.core import _console
from cvbench.tasks import get_task

_MODEL_SUFFIXES = {".keras": "keras", ".onnx": "onnx", ".tflite": "tflite"}


@_help.command(
    examples=[
        ("predict cls_effnet_b0_2026_01_21 photo.jpg",
         "Classify one image with the trained Keras model"),
        ("predict cls_effnet_b0_2026_01_21 images/ --format tflite",
         "Run a folder through the exported TFLite model"),
        ("predict cls_effnet_b0_2026_01_21 images/ --format all",
         "Compare every exported format side by side to spot conversion drift"),
        ("predict cls_effnet_b0_2026_01_21 photo.jpg --top-k 3",
         "Show the three most likely classes per image"),
        ("predict cls_effnet_b0_2026_01_21 images/ --model out/model_int8.tflite",
         "Run a specific model file, e.g. a quantized or relocated export"),
        ("predict det_yolo_2026_01_21 photo.jpg --conf 0.4",
         "Detect objects, keeping only boxes scoring at least 0.4"),
        ("predict --format plan",
         "Print the Jetson inference script (no run or images needed)"),
    ],
    see_also=[
        ("runs export <run> --format tflite", "create the tflite/onnx model first"),
    ],
)
@click.argument("experiment", metavar="RUN", required=False, default=None)
@click.argument(
    "input_path", metavar="INPUT", type=click.Path(exists=True), required=False
)
@click.option(
    "--format",
    "fmt",
    default="keras",
    type=click.Choice(["keras", "onnx", "tflite", "plan", "all"]),
    show_default=True,
    help="Model format to use for inference.",
)
@click.option(
    "--conf",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Detection runs only: minimum score to report a box "
    "(default: the run's detection.conf_threshold).",
)
@click.option(
    "--top-k",
    "top_k",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Classification runs only: classes to show per image.",
)
@click.option(
    "--probs",
    is_flag=True,
    help="Classification runs only: show the full probability of every class.",
)
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Run this model file instead of the run's default export "
    "(format is taken from its extension: .keras, .onnx, .tflite).",
)
def predict(experiment, input_path, fmt, conf, top_k, probs, model_path):
    """Run inference on INPUT using a trained RUN.

    RUN is a run name or full path to a run directory.
    INPUT is an image file or folder of images.

    Both arguments are optional when --format plan is used.

    Classification runs print the top class per image (--top-k N or --probs
    for more). Unreadable files are skipped and listed. Detection runs print
    every decoded box (class, score, normalized x y w h); use --conf to
    change the score threshold.

    Use --format all to run every available format side by side and spot
    conversion differences (useful for debugging false positives after export).
    Use --format plan to print the Jetson inference script and run instructions.
    """
    from cvbench.services.prediction import run_experiment_prediction

    if fmt == "plan":
        print(_console.rule(thick=True))
        print(" CVBench — predict  [plan]")
        print(_console.rule(thick=True))
        _print_plan_predict(experiment or "")
        return

    if not experiment:
        raise click.UsageError("RUN is required.")
    if not input_path:
        raise click.UsageError("INPUT is required.")

    if model_path is not None:
        source = click.get_current_context().get_parameter_source("fmt")
        suffix_fmt = _MODEL_SUFFIXES.get(Path(model_path).suffix.lower())
        if fmt == "all":
            raise click.UsageError("--model cannot be combined with --format all.")
        if suffix_fmt is None:
            raise click.UsageError(
                "--model must end in .keras, .onnx or .tflite "
                f"(got {Path(model_path).name})."
            )
        if source is not click.core.ParameterSource.DEFAULT and fmt != suffix_fmt:
            raise click.UsageError(
                f"--model is a {suffix_fmt} file but --format is {fmt}."
            )
        fmt = suffix_fmt

    try:
        result = run_experiment_prediction(
            experiment,
            input_path,
            fmt,
            conf,
            top_k=None if probs else top_k,
            model_path=model_path,
        )
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e)) from e

    exp_name = result["experiment"]

    fmt_label = f"[{fmt}]"
    print(_console.rule(thick=True))
    print(f" CVBench — predict  {exp_name}  {fmt_label}")
    print(_console.rule(thick=True))

    formats_run = result["formats_run"]
    formats_skipped = result["formats_skipped"]

    print()
    if not formats_run:
        print(_console.yellow(" No models available to run inference."))
    else:
        get_task(result["task"]).render_predictions(formats_run, fmt == "all")
    print()

    print(_console.rule())
    if formats_run:
        n = len(formats_run[0]["results"])
        print(f" {n} image{'s' if n != 1 else ''}")
        for f in formats_run:
            print(f" {_console.dim('model:')}   {f['format']:<8}  {f['model_path']}")
    for s in result["inputs_skipped"]:
        print(
            f" {_console.dim('unreadable:')} {s['filename']}  "
            f"Cannot read image ({s['reason']})"
        )
    if formats_skipped:
        print()
        for s in formats_skipped:
            print(f" {_console.dim('skipped:')} {s['format']:<8}  {s['reason']}")
    if fmt == "all":
        print()
        print(
            _console.dim(
                " plan: Jetson only — for more information use: predict --format plan"
            )
        )


def _print_plan_predict(run_name: str) -> None:
    print()
    print(
        _console.blue(
            " .plan inference runs on Jetson only — the engine is compiled for a specific GPU."
        )
    )
    print()
    print(" To get full deployment instructions run:")
    print()
    print(f"   runs export {run_name} --format plan")
    print()
    print(_console.rule())
    print(
        " Once you have model.plan on the Jetson, save the script below as infer.py,"
    )
    print(" make it executable and run it:")
    print()
    print("   chmod +x infer.py")
    print("   ./infer.py model.plan image.jpg")
    print("   ./infer.py model.plan images/")
    print()
    print(_console.rule())
    print()
    print("   #!/usr/bin/env python3")
    print('   """TensorRT inference — single image or folder."""')
    print("   import argparse, os, sys")
    print("   import numpy as np")
    print("   import tensorrt as trt")
    print("   from PIL import Image")
    print()
    print("   IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}")
    print("   INPUT_SIZE = 224  # match the size used during training")
    print()
    print("   def load_image(path):")
    print(
        "       img = Image.open(path).convert('RGB').resize((INPUT_SIZE, INPUT_SIZE))"
    )
    print(
        "       return np.array(img, dtype=np.float32)[None]  # (1, H, W, 3)"
    )
    print()
    print("   def collect_images(path):")
    print("       p = os.path.abspath(path)")
    print("       if os.path.isfile(p):")
    print("           return [p]")
    print("       return sorted(")
    print("           os.path.join(r, f)")
    print("           for r, _, files in os.walk(p)")
    print(
        "           for f in files if os.path.splitext(f)[1].lower() in IMG_EXTS"
    )
    print("       )")
    print()
    print("   def load_engine(plan_path):")
    print("       logger = trt.Logger(trt.Logger.WARNING)")
    print("       with open(plan_path, 'rb') as f, trt.Runtime(logger) as rt:")
    print("           return rt.deserialize_cuda_engine(f.read())")
    print()
    print("   def main():")
    print("       ap = argparse.ArgumentParser()")
    print("       ap.add_argument('plan',  help='path to model.plan')")
    print("       ap.add_argument('input', help='image file or folder')")
    print("       args = ap.parse_args()")
    print()
    print("       engine = load_engine(args.plan)")
    print("       images = collect_images(args.input)")
    print("       if not images:")
    print("           sys.exit(f'No images found at {args.input}')")
    print()
    print("       with engine.create_execution_context() as ctx:")
    print("           for path in images:")
    print("               arr = load_image(path)")
    print("               probs = ctx.execute_v2([arr])[0][0]")
    print("               idx = int(np.argmax(probs))")
    print(
        "               print(f'{os.path.basename(path):<40} class {idx}  ({probs[idx]*100:.1f}%)')"
    )
    print()
    print("   if __name__ == '__main__':")
    print("       main()")
