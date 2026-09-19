<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mmgalushka/cvbench/main/docs/assets/images/logo-horizontal-dark.svg">
  <img src="https://raw.githubusercontent.com/mmgalushka/cvbench/main/docs/assets/images/logo-horizontal-light.svg" alt="CVBench" width="380">
</picture>

[![CI](https://github.com/mmgalushka/cvbench/actions/workflows/ci.yaml/badge.svg)](https://github.com/mmgalushka/cvbench/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/mmgalushka/cvbench/graph/badge.svg?token=2tAfSTBylU)](https://codecov.io/gh/mmgalushka/cvbench)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Docker Pulls](https://img.shields.io/docker/pulls/mmgalushka/cvbench.svg)](https://hub.docker.com/r/mmgalushka/cvbench)
[![GitHub release](https://img.shields.io/github/v/release/mmgalushka/cvbench)](https://github.com/mmgalushka/cvbench/releases)

GPU-enabled computer vision training sandbox. [Keras](https://keras.io/) +
[TensorFlow](https://www.tensorflow.org/) + [JupyterLab](https://jupyterlab.readthedocs.io/)
in one container — generate or bring your own data, train and evaluate
classification/detection models, track experiments, and export to
[TFLite](https://ai.google.dev/edge/litert)/[ONNX](https://onnx.ai/)/[Hailo](https://hailo.ai/)/[Jetson](https://developer.nvidia.com/embedded-computing),
all from one CLI and WebUI.

</div>

## 30-second quick start

```bash
mkdir -p ~/cvbench/{data,workspace,experiments}
docker run -d \
  --name cvbench \
  -p 0.0.0.0:8000:8000 -p 0.0.0.0:8888:8888 \
  -v ~/cvbench/data:/home/cvbench/data \
  -v ~/cvbench/workspace:/home/cvbench/workspace \
  -v ~/cvbench/experiments:/home/cvbench/experiments \
  --restart unless-stopped \
  mmgalushka/cvbench:latest
```

Or with Docker Compose:

```yaml
services:
  cvbench:
    image: mmgalushka/cvbench:latest
    container_name: cvbench
    ports:
      - "0.0.0.0:8000:8000"
      - "0.0.0.0:8888:8888"
    volumes:
      - ~/cvbench/data:/home/cvbench/data
      - ~/cvbench/workspace:/home/cvbench/workspace
      - ~/cvbench/experiments:/home/cvbench/experiments
    restart: unless-stopped
```

Then open the WebUI at `http://<server-ip>:8000` and JupyterLab at
`http://<server-ip>:8888`. Add `--gpus all` (docker run) or the GPU
`deploy.resources` block (Compose) for GPU acceleration.

**Full documentation →** https://mmgalushka.github.io/cvbench

Covers installation and volumes, generating and preparing datasets,
training (optimizer/loss/LR scheduling/two-phase fine-tuning), evaluation and
prediction, the WebUI, and deployment/export (TFLite, ONNX, Hailo HEF,
Jetson) — plus the full CLI reference.

---

## Quickstart

<!-- BEGIN QUICKSTART -->
```
1  commands                           # show this screen again any time
2  tm new <name>                      # start a tmux session so training survives closing your terminal
3  data generate                      # make a 4-class synthetic dataset in data/synthetic/
4  train data/synthetic --epochs 5    # train a model — prints the run name when it finishes
5  runs list                          # see every run, newest first
6  evaluate <run-name>                # score that run on the held-out test split
7  serve --host 0.0.0.0 --port 8000   # browse it all in the WebUI → http://<server-ip>:8000
```
<!-- END QUICKSTART -->

## CLI reference

<!-- BEGIN CLI REFERENCE -->
```
train           Train a model on DATA_DIR.
evaluate        Evaluate a trained model on the held-out test split.
predict         Run inference on INPUT using a trained EXPERIMENT.
serve           Start the CVBench WebUI server.

   Generate, inspect and reshape datasets.
   data clean      Copy a dataset, dropping OS/editor junk files.
   data dedup      Copy a dataset, dropping exact-duplicate images.
   data explore    Report per-class brightness and class balance.
   data flatten    Pool an already-split dataset back into one flat folder.
   data generate   Generate a synthetic geometric shapes dataset for pipeline testing.
   data hashify    Copy a dataset, renaming images to content hashes.
   data list       List datasets (default: data/).
   data split      Split a flat dataset into train/val/test, stratified by class.
   data upsample   Grow a class folder to TARGET images via augmentation.

   Discover, generate, and manage augmentation configurations.
   aug delete      Delete a saved augmentation config.
   aug edit        Open a saved augmentation config in your editor.
   aug generate    Interactively build and save a new augmentation config.
   aug list        List saved augmentation configs.
   aug show        Print a saved augmentation config.
   aug transforms  List every available transform with its default parameters.

   Manage and inspect experiment runs.
   runs best       Show the single best run by a metric.
   runs compare    Compare two runs side by side.
   runs delete     Delete a run, or just one of its exports.
   runs export     Export a run to TFLite / ONNX / Hailo, or print Jetson steps.
   runs list       List experiment runs (default: experiments/).
   runs rename     Rename a run directory and update its config.
   runs show       Show full details for a single run.
```

Every command has worked examples in its `--help`. In the container, run `commands` for the full picture (CLI plus the tmux session helpers).
<!-- END CLI REFERENCE -->

This block is generated from the CLI — run `./helper.sh docs` after changing a
command to refresh it.

---

See also: [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).
