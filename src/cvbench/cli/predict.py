import click

from cvbench.core import _fmt


@click.command()
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
                f" plan: Jetson only — for more information use: cvbench predict --format plan"
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
