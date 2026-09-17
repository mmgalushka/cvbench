# Your First Model

A guided walkthrough from a freshly-started container to a trained,
evaluated, exported model — with pointers to the detail pages for anything
you want to customize along the way.

## 1. Open a session and start a tmux session

Training can take a while, so start it inside `tmux` — this way it keeps
running even if you close your terminal or lose your SSH connection.

```bash
docker exec -it cvbench bash
tm new train
```

## 2. Generate a dataset

For a first run, skip finding a real dataset and generate a synthetic one —
it's ready in seconds and exercises the full pipeline:

```bash
data generate
```

This writes a 4-class classification dataset to `data/synthetic/`. See
[Generating Synthetic Data](../data/synthetic.md) for the YOLO variant and
folder layouts, or [Preparing Real Datasets](../data/prepare.md) once you're
ready to bring your own images.

!!! note "Classification or detection — CVBench supports both"
    `train` detects which task you're running automatically from `DATA_DIR`'s
    layout: a dataset with `images/` and `labels/` subdirectories (YOLO-style
    boxes) is trained as **detection**; anything else (one subfolder per
    class) is trained as **classification**. No extra flag is needed — just
    point `train` at the right kind of dataset. See
    [Generating Synthetic Data](../data/synthetic.md) for both folder layouts.

## 3. Train

```bash
train data/synthetic --epochs 5 --backbone efficientnet_b0
```

Training prints the run name when it finishes — you'll need it for every
step that follows. Detach from tmux with `Ctrl+B D`; the run keeps going.

This smoke-test run uses the defaults everywhere else — optimizer, loss,
learning-rate schedule, and fine-tuning strategy are all worth exploring once
you move past a first run: see [Training Basics](../training/basics.md),
[Optimizer & Loss](../training/optimizer-loss.md),
[Learning Rate Scheduling](../training/lr-scheduling.md), and
[Two-Phase Training](../training/two-phase.md).

## 4. Find your run

```bash
runs list
```

Lists every run, newest first, with backbone, status, validation loss, and
epochs run.

## 5. Evaluate

```bash
evaluate <run-name>
```

Scores the run on its held-out test split and writes `eval_report.json` into
the run directory. See [Evaluating a Run](../evaluation/evaluate.md) for what
gets reported.

## 6. Predict

```bash
predict <run-name> data/synthetic/test/circle/0000.jpg
```

Runs inference on a single image (or a whole folder) with the trained model.
See [Running Predictions](../evaluation/predict.md) for the export-format
comparison and Jetson options.

## 7. Serve

```bash
serve --host 0.0.0.0 --port 8000
```

Browse everything you just did in the WebUI at `http://<server-ip>:8000` —
training curves, the evaluation report with its interactive confusion
matrix, side-by-side run comparison, and one-click export. See
[Browsing Runs & Datasets](../webui/overview.md) and
[Interactive Confusion Matrix](../webui/confusion-matrix.md).

## What's next

- Package the model for a device: [Export Formats (TFLite/ONNX)](../deployment/export.md),
  [Hailo HEF Export](../deployment/hailo.md), or [Jetson Deployment](../deployment/jetson.md).
- Fix class imbalance or add augmentation before your next run:
  [Augmentation](../data/augmentation.md).
- Reshape a real dataset into train/val/test:
  [Preparing Real Datasets](../data/prepare.md).
