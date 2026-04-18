import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import click

from cvbench.services.evaluation import run_evaluation


@click.command()
@click.argument("experiment")
@click.option("--output-dir", default=None, help="Where to write eval outputs (default: run dir).")
def evaluate(experiment, output_dir):
    """Evaluate a trained model on the held-out test split.

    EXPERIMENT is the run name (e.g. effnet_b3_lr5e5_cutmix_trial_2024_01_21)
    or a full path to the run directory. If a bare name is given, it is resolved
    under experiments/.
    """
    run_evaluation(experiment=experiment, output_dir=output_dir)
