import click

from cvbench.core import _fmt
from cvbench.core.runs import (
    scan_experiments,
    best_experiment,
    resolve_run_dir,
    EXPERIMENTS_DIR,
)
from cvbench.core.config import load_config
from cvbench.services.export import run_export


def _fit(s: str, width: int) -> str:
    """Fit string to width using a middle ellipsis, preserving start and end."""
    if len(s) <= width:
        return s
    keep = width - 1  # 1 char for the ellipsis
    head = keep // 2
    tail = keep - head
    return s[:head] + "…" + s[-tail:]


_DEFAULT_EXPERIMENTS_DIR = EXPERIMENTS_DIR


@click.group()
def runs():
    """Manage and inspect experiment runs."""


@runs.command("list")
@click.argument("experiments_dir", default=_DEFAULT_EXPERIMENTS_DIR)
@click.option(
    "--sort",
    default="date",
    type=click.Choice(["val_accuracy", "val_loss", "date", "backbone"]),
    show_default=True,
)
def list_runs(experiments_dir, sort):
    """List experiments in EXPERIMENTS_DIR (default: experiments/)."""
    entries = scan_experiments(experiments_dir, sort_by=sort)
    if not entries:
        print(f" No experiments found in '{experiments_dir}'.")
        return

    tr = _fmt.rule(76, "white")
    print(tr)
    print(f" {'Run':<45} {'Status':<12} {'Val Loss':>9}  {'Epochs':>6}")
    print(tr)
    for r in entries:
        loss = r.get("val_loss")
        loss_str = f"{loss:.4f}" if loss is not None else "   —   "
        name = _fit(r["name"], 45)
        print(
            f" {name:<45} {r.get('status', '?'):<12} {loss_str:>9}  {r.get('epochs_run', '?'):>6}"
        )
    print(tr)


@runs.command()
@click.argument("experiment_a")
@click.argument("experiment_b")
def compare(experiment_a, experiment_b):
    """Compare two experiments side by side.

    EXPERIMENT_A and EXPERIMENT_B are run names (e.g. effnet_b3_lr5e5_trial_2024_01_21)
    or full paths to run directories. Bare names are resolved under experiments/.
    """
    run_a = resolve_run_dir(experiment_a)
    run_b = resolve_run_dir(experiment_b)
    try:
        a_cfg = load_config(run_a)
    except FileNotFoundError:
        raise click.ClickException(f"No config.yaml in: {run_a}")
    try:
        b_cfg = load_config(run_b)
    except FileNotFoundError:
        raise click.ClickException(f"No config.yaml in: {run_b}")

    from cvbench.core.runs import _read_entry
    from pathlib import Path

    a = _read_entry(Path(run_a))
    b = _read_entry(Path(run_b))

    fields = [
        "backbone",
        "lr",
        "epochs",
        "val_loss",
        "val_accuracy",
        "test_accuracy",
        "epochs_run",
        "status",
        "date",
    ]
    name_a = a.get("name", run_a)
    name_b = b.get("name", run_b)

    col_w = 26
    tr = _fmt.rule(79, "white")
    print(tr)
    print(
        f" {'Field':<22}  {_fit(name_a, col_w):<{col_w}}  {_fit(name_b, col_w):<{col_w}}"
    )
    print(tr)
    for f in fields:
        va = str(a.get(f, "—"))
        vb = str(b.get(f, "—"))
        diff = " ◀" if va != vb else ""
        print(f" {f:<22}  {va:<26}  {vb:<26}{diff}")
    print(tr)


@runs.command()
@click.argument("experiment")
@click.option(
    "--format",
    "fmt",
    required=True,
    type=click.Choice(["tflite", "onnx", "plan"]),
    help="Export format (plan prints Jetson TensorRT instructions).",
)
@click.option(
    "--quantize",
    default="none",
    type=click.Choice(["none", "float16", "int8"]),
    show_default=True,
    help="TFLite quantization mode (ignored for ONNX and plan).",
)
@click.option(
    "--output",
    "output_dir",
    default=None,
    help="Output directory (default: <experiment>/export/). Ignored for plan.",
)
def export(experiment, fmt, quantize, output_dir):
    """Export the best checkpoint of EXPERIMENT to TFLite or ONNX, or print Jetson deployment instructions (plan).

    EXPERIMENT is a run name or full path to a run directory.
    """
    try:
        run_export(
            experiment, format=fmt, quantize=quantize, output_dir=output_dir
        )
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except RuntimeError as e:
        raise click.ClickException(str(e))


@runs.command()
@click.argument("experiments_dir", default=_DEFAULT_EXPERIMENTS_DIR)
@click.option(
    "--metric",
    default="val_loss",
    type=click.Choice(["val_loss", "val_accuracy", "test_accuracy"]),
    show_default=True,
)
def best(experiments_dir, metric):
    """Show the best experiment in EXPERIMENTS_DIR by a given metric."""
    run = best_experiment(experiments_dir, metric)
    if run is None:
        print(
            f" No experiments with metric '{metric}' found in '{experiments_dir}'."
        )
        return

    print(_fmt.rule())
    print(f" {_fmt.bold(f'CVBench — best run by {metric}')}")
    print(_fmt.rule())
    for k, v in run.items():
        print(f" {k:<22}: {v}")
    print(_fmt.rule())


@runs.command()
@click.argument("experiment", required=False, default=None)
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
def predict(experiment, input_path, fmt):
    """Run inference on INPUT using a trained EXPERIMENT.

    EXPERIMENT is a run name or full path to a run directory.
    INPUT is an image file or folder of images.

    Both arguments are optional when --format plan is used.

    Use --format all to run every available format side by side and spot
    conversion differences (useful for debugging false positives after export).
    Use --format plan to print the Jetson inference script and run instructions.
    """
    from cvbench.services.prediction import run_experiment_prediction

    if fmt == "plan":
        print(_fmt.rule(thick=True))
        print(f" CVBench — predict  [plan]")
        print(_fmt.rule(thick=True))
        _print_plan_predict(experiment or "")
        return

    if not experiment:
        raise click.UsageError("EXPERIMENT is required.")
    if not input_path:
        raise click.UsageError("INPUT is required.")

    try:
        result = run_experiment_prediction(experiment, input_path, fmt)
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))

    exp_name = result["experiment"]

    fmt_label = f"[{fmt}]"
    print(_fmt.rule(thick=True))
    print(f" CVBench — predict  {exp_name}  {fmt_label}")
    print(_fmt.rule(thick=True))

    formats_run = result["formats_run"]
    formats_skipped = result["formats_skipped"]

    print()
    if not formats_run:
        print(_fmt.yellow(" No models available to run inference."))
    elif fmt == "all":
        _print_all_formats(formats_run)
    else:
        _print_single_format(formats_run[0]["results"])
    print()

    print(_fmt.rule())
    n = sum(len(f["results"]) for f in formats_run[:1])
    print(f" {n} image{'s' if n != 1 else ''}")
    if formats_skipped:
        print()
        for s in formats_skipped:
            print(f" {_fmt.dim('skipped:')} {s['format']:<8}  {s['reason']}")
    if fmt == "all":
        print()
        print(
            _fmt.dim(
                f" plan: Jetson only — for more information use: cvbench runs predict --format plan"
            )
        )


def _print_single_format(results: list[dict]) -> None:
    for r in results:
        print(
            f" {r['filename']:<40} {r['class_name']:<20} {r['confidence'] * 100:5.1f}%"
        )


def _print_all_formats(formats_run: list[dict]) -> None:
    fmt_names = [f["format"] for f in formats_run]
    col_w = 20

    header = f" {'image':<38}" + "".join(f"  {n:<{col_w}}" for n in fmt_names)
    print(header)
    print(_fmt.dim(" " + "─" * (len(header) - 1)))

    # Use the first format (keras if present) as the reference for disagreement
    ref_results = {
        r["filename"]: r["class_name"] for r in formats_run[0]["results"]
    }
    filenames = [r["filename"] for r in formats_run[0]["results"]]

    for filename in filenames:
        ref_class = ref_results[filename]
        row = f" {filename:<38}"
        for fmt_data in formats_run:
            res = next(
                (r for r in fmt_data["results"] if r["filename"] == filename),
                None,
            )
            if res is None:
                cell = "—"
            else:
                label = f"{res['class_name']} ({res['confidence'] * 100:.1f}%)"
                is_ref = fmt_data is formats_run[0]
                differs = not is_ref and res["class_name"] != ref_class
                if differs:
                    label = f"{label}  ⚠️"
                cell = label
            row += f"  {cell:<{col_w}}"
        print(row)


def _print_plan_predict(run_name: str) -> None:
    print()
    print(
        _fmt.blue(
            " .plan inference runs on Jetson only — the engine is compiled for a specific GPU."
        )
    )
    print()
    print(f" To get full deployment instructions run:")
    print()
    print(f"   cvbench runs export {run_name} --format plan")
    print()
    print(_fmt.rule())
    print(
        " Once you have model.plan on the Jetson, save the script below as infer.py,"
    )
    print(" make it executable and run it:")
    print()
    print("   chmod +x infer.py")
    print("   ./infer.py model.plan image.jpg")
    print("   ./infer.py model.plan images/")
    print()
    print(_fmt.rule())
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
