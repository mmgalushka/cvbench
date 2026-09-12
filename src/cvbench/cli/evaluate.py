import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import click

from cvbench.cli import _help

# NOTE: cvbench.services.evaluation is imported inside evaluate() — it pulls in
# TensorFlow, which would otherwise make `evaluate --help` slow to start.


@_help.command(
    examples=[
        ("evaluate cls_effnet_b0_2026_01_21",
         "Evaluate a run by name (resolved under experiments/)"),
        ("evaluate experiments/cls_effnet_b0_2026_01_21 --output-dir /tmp/eval",
         "Evaluate by path and write the report elsewhere"),
    ],
    see_also=[
        ("runs list", "see every run and its scores"),
        ("runs export <run> --format tflite", "package the model for a device"),
    ],
)
@click.argument("experiment")
@click.option("--output-dir", default=None, help="Where to write eval outputs (default: run dir).")
def evaluate(experiment, output_dir):
    """Evaluate a trained model on the held-out test split.

    EXPERIMENT is the run name (e.g. effnet_b3_lr5e5_cutmix_trial_2024_01_21)
    or a full path to the run directory. If a bare name is given, it is resolved
    under experiments/.
    """
    from cvbench.services.evaluation import run_evaluation  # deferred: pulls in TensorFlow

    run_evaluation(experiment=experiment, output_dir=output_dir)
