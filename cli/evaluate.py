import click
import keras

from core.config import load_config
from core.data import build_dataset, get_class_names
from core import evaluator as _evaluator


@click.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--split", default="val", type=click.Choice(["val", "test"]),
              show_default=True, help="Dataset split to evaluate on.")
@click.option("--output-dir", default=None, help="Where to write eval outputs (default: run_dir).")
def evaluate(run_dir, split, output_dir):
    """Evaluate a trained model in RUN_DIR."""
    cfg = load_config(run_dir)

    data_dir = cfg.data.val_dir if split == "val" else cfg.data.test_dir
    class_names = get_class_names(cfg.data.train_dir)
    dataset = build_dataset(data_dir, class_names, cfg, training=False)

    model = keras.saving.load_model(f"{run_dir}/best.keras")

    _evaluator.evaluate(
        model=model,
        dataset=dataset,
        class_names=class_names,
        run_dir=run_dir,
        split=split,
        output_dir=output_dir,
    )
