import click

from cvbench.services.prediction import run_prediction


@click.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True),
              help="Path to .keras model file.")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="Image file or folder of images.")
def predict(checkpoint, input_path):
    """Run inference on an image or folder of images."""
    w = 55
    print("━" * w)
    print(" CVBench — predict")
    print("━" * w)

    try:
        results = run_prediction(checkpoint=checkpoint, input_path=input_path)
    except ValueError as e:
        raise click.ClickException(str(e))

    for r in results:
        print(f" {r['filename']:<35} class {r['class_index']}  ({r['confidence'] * 100:.1f}%)")

    print("━" * w)
