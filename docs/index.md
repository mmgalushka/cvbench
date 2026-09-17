---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# CVBench

### One GPU-enabled container bundling Keras, TensorFlow, JupyterLab, and a WebUI for computer vision work.

No more stitching together a training script, a notebook, an experiment
tracker, and a deployment story by hand. CVBench packages all of it into one
container, so you can train, evaluate, predict, track, and deploy computer
vision models — and focus on the problem instead of managing an ML
environment.

[Get started](getting-started/quickstart.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/mmgalushka/cvbench){ .md-button }

</div>

<div class="grid cards" markdown>

-   :material-brain:{ .lg .middle } [**Train**](training/basics.md)

    ---

    Train classification and detection models with augmentation,
    optimizer/loss options, and two-phase fine-tuning built in.

-   :material-chart-box-outline:{ .lg .middle } [**Evaluate**](evaluation/evaluate.md)

    ---

    Score any run on its held-out test split — per-class metrics, confusion
    matrices, and detection outcome tables.

-   :material-target:{ .lg .middle } [**Predict**](evaluation/predict.md)

    ---

    Run inference on new images with a trained experiment, from the CLI or
    the WebUI.

-   :material-package-variant-closed:{ .lg .middle } [**Deploy**](deployment/export.md)

    ---

    Export a trained model to TFLite, ONNX, or a Hailo HEF package, or get
    step-by-step Jetson deployment instructions — all from one command.

-   :material-history:{ .lg .middle } [**Explore**](tools/experiment-tracker.md)

    ---

    Every run is recorded automatically — browse, compare, and revisit past
    experiments at any time.

-   :material-notebook-outline:{ .lg .middle } [**Customize**](tools/jupyter-notebook.md)

    ---

    Need something the CLI doesn't cover? JupyterLab is right there in the
    same container for custom, one-off experimental work.

</div>

## How it works

<div class="grid cards" markdown>

-   :material-console:{ .lg .middle } **1. Work from the CLI**

    ---

    Prepare data, train, evaluate, and predict — the CLI covers the full
    day-to-day workflow with a single command per step.

-   :material-monitor-dashboard:{ .lg .middle } **2. Dig deeper in the WebUI**

    ---

    Need a closer look at results? Switch to the built-in WebUI to browse
    runs, compare experiments side by side, and explore an interactive
    confusion matrix.

-   :material-notebook-outline:{ .lg .middle } **3. Go custom in JupyterLab**

    ---

    For one-off experiments or putting an already-trained model to work in
    code, JupyterLab is right there in the same container.

</div>

<div class="hero-footer" markdown>

Full CLI reference and guides: use the navigation above, or run `commands`
inside the container for the complete picture.

</div>
