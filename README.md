# CVBench

GPU-enabled computer vision training sandbox. Keras + TensorFlow + JupyterLab in one container.

## Prerequisites

- Docker 24+
- **GPU (optional):** [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) — required only if you want GPU acceleration. The container runs on CPU without it.

---

## Quick start

### Prepare a workspace

Create a directory to hold your data, workspace files, and outputs. `~/cvbench` is a convenient default:

```bash
mkdir -p ~/cvbench/{data,workspace,experiments}
cd ~/cvbench
```

---

### Option A — plain `docker run`

**With GPU:**

```bash
docker run -d \
  --name cvbench \
  --gpus all \
  -p 0.0.0.0:8888:8888 \
  -p 0.0.0.0:6006:6006 \
  -v ~/cvbench/data:/home/cvbench/data \
  -v ~/cvbench/workspace:/home/cvbench/workspace \
  -v ~/cvbench/experiments:/home/cvbench/experiments \
  --restart unless-stopped \
  mmgalushka/cvbench:latest
```

**CPU only** (drop `--gpus all`):

```bash
docker run -d \
  --name cvbench \
  -p 0.0.0.0:8888:8888 \
  -p 0.0.0.0:6006:6006 \
  -v ~/cvbench/data:/home/cvbench/data \
  -v ~/cvbench/workspace:/home/cvbench/workspace \
  -v ~/cvbench/experiments:/home/cvbench/experiments \
  --restart unless-stopped \
  mmgalushka/cvbench:latest
```

---

### Option B — Docker Compose

Save the appropriate file as `~/cvbench/docker-compose.yml` and run `docker compose up -d`.

**With GPU** (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

```yaml
services:
  cvbench:
    image: mmgalushka/cvbench:latest   # pin a release: mmgalushka/cvbench:0.2.0
    container_name: cvbench
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - CVBENCH_URL=http://<server-ip>:8000
    ports:
      - "0.0.0.0:8000:8000"
      - "0.0.0.0:8888:8888"
      - "0.0.0.0:6006:6006"
    volumes:
      - ~/cvbench/data:/home/cvbench/data
      - ~/cvbench/workspace:/home/cvbench/workspace
      - ~/cvbench/experiments:/home/cvbench/experiments
    restart: unless-stopped
```

**CPU only** (remove the GPU lines):

```yaml
services:
  cvbench:
    image: mmgalushka/cvbench:latest
    container_name: cvbench
    environment:
      - CVBENCH_URL=http://<server-ip>:8000
    ports:
      - "0.0.0.0:8000:8000"
      - "0.0.0.0:8888:8888"
      - "0.0.0.0:6006:6006"
    volumes:
      - ~/cvbench/data:/home/cvbench/data
      - ~/cvbench/workspace:/home/cvbench/workspace
      - ~/cvbench/experiments:/home/cvbench/experiments
    restart: unless-stopped
```

Replace `<server-ip>` with the actual IP or hostname of your Docker host.

After starting:

```bash
# CVBench WebUI → http://<server-ip>:8000  (starts automatically; set URL via CVBENCH_URL)
# JupyterLab    → http://<server-ip>:8888
# TensorBoard   → http://<server-ip>:6006
```

---

### Generate a synthetic dataset (smoke-test)

```bash
docker exec cvbench data generate --train 200 --val 50 --test 50
```

This writes to `data/synthetic/` by default. Point training at it immediately:

```bash
docker exec -it cvbench bash
train data/synthetic --epochs 5 --backbone efficientnet_b0
```

#### Classification dataset

```bash
data generate data/synthetic --train 200 --val 50 --test 50
```

```
data/synthetic/
├── train/
│   ├── circle/0000.jpg
│   ├── square/0000.jpg
│   ├── triangle/0000.jpg
│   └── star/0000.jpg
├── val/     (same 4 class folders)
└── test/    (same 4 class folders)
```

- One shape per image; the class is the folder name — no label files.
- `--train/--val/--test` count images **per class**.

#### YOLO txt dataset

```bash
data generate data/synthetic_yolo --format yolo \
    --train 200 --val 50 --test 50 --image-size 160 --max-objects 4
```

```
data/synthetic_yolo/
├── data.yaml                # class names + split paths
├── images/
│   ├── train/0000.jpg
│   ├── val/0000.jpg
│   └── test/0000.jpg
└── labels/
    ├── train/0000.txt       # one "class_id xc yc w h" line per shape, normalized
    ├── val/0000.txt
    └── test/0000.txt
```

- 1–`--max-objects` shapes per image (default 3), each with a bounding box.
- `--train/--val/--test` count images **per split**, since one image can hold
  several classes.
- Every image has a same-named `.txt` next to it in `labels/`.

The WebUI **Datasets** page reads both: it labels each dataset's format and, for
YOLO, draws the bounding boxes over every thumbnail (*Show boxes* toggles the
overlay, the class filter keeps only images containing a given class).

---

## Training

From a JupyterLab terminal or SSH session:

```bash
docker exec -it cvbench bash
tm -n train
train data --epochs 20 --backbone efficientnet_b0
# Ctrl+B D to detach — training continues after you close the terminal
```

### TensorBoard

```bash
docker exec cvbench tensorboard --logdir /home/cvbench/experiments --host 0.0.0.0 --port 6006
# → http://<server-ip>:6006
```

---

## Quickstart

Five commands, top to bottom, and you have a trained model. Run these inside
the container (`docker exec -it cvbench bash`):

<!-- BEGIN QUICKSTART -->
```
1  commands                           # show this screen again any time
2  tm -n <name>                       # start a tmux session so training survives closing your terminal
3  data generate                      # make a 4-class synthetic dataset in data/synthetic/
4  train data/synthetic --epochs 5    # train a model — prints the run name when it finishes
5  runs list                          # see every run, newest first
6  evaluate <run-name>                # score that run on the held-out test split
7  serve --host 0.0.0.0 --port 8000   # browse it all in the WebUI → http://<server-ip>:8000
```
<!-- END QUICKSTART -->

---

## CLI reference

<!-- BEGIN CLI REFERENCE -->
```
train                Train a model on DATA_DIR.
evaluate             Evaluate a trained model on the held-out test split.
predict              Run inference on INPUT using a trained EXPERIMENT.
serve                Start the CVBench WebUI server.

data                 Generate, inspect and reshape datasets.
data aug             Discover, generate, and manage augmentation configurations.
data aug delete      Delete a saved augmentation config.
data aug generate    Interactively build and save a new augmentation config.
data aug list        List saved augmentation configs.
data aug show        Print a saved augmentation config.
data aug transforms  List every available transform with its default parameters.
data clean           Copy a dataset, dropping OS/editor junk files.
data dedup           Copy a dataset, dropping exact-duplicate images.
data explore         Report per-class brightness and class balance.
data flatten         Pool an already-split dataset back into one flat folder.
data generate        Generate a synthetic geometric shapes dataset for pipeline testing.
data hashify         Copy a dataset, renaming images to content hashes.
data split           Split a flat dataset into train/val/test, stratified by class.
data upsample        Grow a class folder to TARGET images via augmentation.

runs                 Manage and inspect experiment runs.
runs best            Show the single best run by a metric.
runs compare         Compare two runs side by side.
runs delete          Delete a run, or just one of its exports.
runs export          Export a run to TFLite / ONNX / Hailo, or print Jetson steps.
runs list            List experiment runs (default: experiments/).
runs rename          Rename a run directory and update its config.
```

Every command has worked examples in its `--help`. In the container, run `commands` for the full picture (CLI plus the tmux session helpers).
<!-- END CLI REFERENCE -->

This block is generated from the CLI — run `./helper.sh docs` after changing a
command to refresh it.

---

## Volume mounts

| Host path                    | Container path                   | Notes                             |
|------------------------------|----------------------------------|-----------------------------------|
| `~/cvbench/data`             | `/home/cvbench/data`             | Image datasets (real + synthetic) |
| `~/cvbench/workspace`        | `/home/cvbench/workspace`        | Augmentation configs, user notebooks, and other working files |
| `~/cvbench/experiments`      | `/home/cvbench/experiments`      | Experiment directories            |

---

## Optimizer

By default training uses Adam. Use `--optimizer` to switch to SGD or to add weight decay (L2 regularization).

```bash
# Adam with weight decay
train data/ --optimizer adam:weight_decay=1e-4

# SGD with momentum and weight decay
train data/ --optimizer sgd:weight_decay=1e-4,momentum=0.9
```

| Option | Default | Description |
|---|---|---|
| `--optimizer adam` | ✓ | Adam optimizer |
| `--optimizer sgd` | — | SGD optimizer |
| `weight_decay=F` | `0.0` | L2 regularization penalty |
| `momentum=F` | `0.9` | Momentum (SGD only) |

The optimizer config is saved to `config.yaml` and applied automatically when resuming a run.

---

## Learning rate scheduling

By default the learning rate is fixed for the entire training run. Use `--lr-scheduler` to enable **ReduceLROnPlateau** — the LR is multiplied by `factor` whenever `val_loss` fails to improve for `patience` consecutive epochs.

```bash
# Reduce LR by 0.5x after 5 flat epochs (default factor and floor)
train data/ --lr 1e-3 --lr-scheduler patience=5

# Aggressive decay: cut to 20% after 3 flat epochs, floor at 1e-6
train data/ --lr 1e-3 --lr-scheduler patience=3,factor=0.2,min=1e-6
```

| Parameter | Default | Description |
|---|---|---|
| `patience=N` | required | Epochs with no `val_loss` improvement before reducing LR |
| `factor=F` | `0.5` | Multiplicative reduction factor |
| `min=F` | `1e-7` | Minimum LR floor |

The scheduler settings are saved to `config.yaml` and applied automatically when resuming a run.

---

## Two-phase training (freeze → fine-tune)

A common transfer-learning workflow is to first train with the backbone frozen, then unfreeze some layers and fine-tune at a lower learning rate.

**Phase 1 — train classifier head only (backbone frozen):**

```bash
train data/ --epochs 30 --output experiments/phase1
```

**Phase 2 — unfreeze top layers and fine-tune:**

```bash
train data/ \
  --from experiments/phase1 \
  --resume experiments/phase1/best.keras \
  --fine-tune-from-layer 100 \
  --lr 1e-5 \
  --epochs 50
```

`--from` loads the phase-1 config (backbone, input size, augmentation, etc.) and its recorded epoch count. `--resume` loads the saved weights. Training then continues from epoch 30 through epoch 50, adding 20 fine-tuning epochs — the training log is **appended**, so the full history (both phases) is preserved in `training_log.csv`.

> **Important:** `--epochs N` means *end at epoch N*, not *run N more epochs*. If phase 1 ran 30 epochs and you want 20 more, set `--epochs 50`.

**Resuming after an interrupt:**

If training is interrupted mid-run, CVBench saves an `interrupt_epochNNN.keras` checkpoint and prints the exact resume command:

```bash
train data/ --from experiments/phase1 --resume experiments/phase1/interrupt_epoch023.keras --epochs 30
```

| Option | Description |
|---|---|
| `--from <exp_dir>` | Load backbone, hyperparameters, and epoch count from a previous experiment |
| `--resume <checkpoint>` | Load weights from a `.keras` checkpoint and continue training from the recorded epoch |
| `--fine-tune-from-layer N` | Unfreeze backbone layers from index N onward (`-1` = unfreeze all) |

---

## Augmentation

Augmentation configs are named, saved artifacts managed under
`workspace/augmentations/` — the same way `runs` manages `experiments/`.
Once saved, a config is used by name with `train --augmentation` (applied
live during training) or `data upsample --augmentation` (materialized to
disk).

**Discover what's available:**

```bash
data aug transforms                    # every building-block transform + default params
data aug list                          # every config you've saved
```

**Generate a config** — `data aug generate` always walks a short wizard: seed
from a preset (`light` / `standard` / `heavy`) or start blank, keep/drop/
customize each transform, optionally add more from the full catalogue, then
save it under a name:

```bash
$ data aug generate --preset standard
 keras_flip  prob=1.0  mode: "horizontal"
  Keep 'keras_flip'? [Y/n]:
  Customize its parameters? [y/N]:
 ...
Add another transform from the catalogue? [y/N]: n
Save as [standard]: my_config
  ✓ Saved → workspace/augmentations/my_config.yaml  (5 transform(s))
  Usage:  train data/ --augmentation my_config
```

```bash
data aug show my_config                # print its YAML
train data/ --augmentation my_config --epochs 30
```

**`reference` preset** bypasses the wizard and saves a commented-out file
showing every available transform (including range-sampling and `one_of`
syntax) — open it in an editor and uncomment what you want:

```bash
data aug generate --preset reference --name aug_ref
```

### Upsampling a class folder

Use `data upsample` to materialise an augmented copy of a single class folder on disk. This is useful for correcting class imbalance before training — apply it only to the classes that need more samples (e.g. skip the `noise` class if it is already well-represented).

```bash
# Upsample the 'dog' class from however many originals it has to 1500 images.
# The destination folder must be empty or non-existent.
data upsample data/my_data/train/dog data/my_data_aug/train/dog \
  --augmentation my_config \
  --target 1500
```

**What it does:**

1. Copies every original image to `dst_dir` with a fresh 16-char random hex filename.
2. Randomly picks source images and augments them until `--target` is reached.
3. Uses MD5 hashing to detect exact duplicates; retries up to 10 times per sample before skipping.
4. If the source already has ≥ `--target` images, the command exits with a hint to use `data downsample` instead (not yet implemented).

| Option | Required | Description |
|---|---|---|
| `--augmentation NAME\|FILE` | ✓ | A saved `data aug` config name, or a path to an augmentation YAML file (same format as `--augmentation` in `train`) |
| `--target N` | ✓ | Total number of images the destination folder should contain |

### Cleaning a dataset

Use `data clean` to copy a dataset (classification or YOLO layout) while dropping OS/editor junk: `.DS_Store`, `Thumbs.db`, `__MACOSX/`, `.Spotlight-V100`, AppleDouble shadow files (`._*`), and editor swap/temp files. Directories left empty by junk removal are simply not created at the destination. The source is never modified.

```bash
data clean data/my_data data/my_data_clean --dry-run   # preview
data clean data/my_data data/my_data_clean             # write the cleaned copy
```

| Option | Required | Description |
|---|---|---|
| `--dry-run` |  | Print what would be removed without writing `DST` |

### Hashifying a dataset

Use `data hashify` to copy a dataset (classification or YOLO layout) while renaming every image to a content-hash filename (e.g. `0ca9c69d9741cb49.png`), instead of its original basename. This is deterministic and idempotent — the same image always gets the same name, in any dataset — which makes it easy to spot the same source image reappearing across collections. `hashify` never deletes anything: two images that land on the same destination name (byte-identical images sharing a directory) both survive, the second with a numeric suffix. Use `data dedup` to remove genuine duplicates.

```bash
data hashify data/my_data data/my_data_hashed
```

| Option | Required | Description |
|---|---|---|
| `--dry-run` |  | Print what would be renamed without writing `DST` |

### Deduplicating a dataset

Use `data dedup` to copy a dataset (classification or YOLO layout) while dropping exact-duplicate images. Duplicates are grouped by full image-content hash; within each group only the lexicographically-first path is kept. For YOLO, dropping an image also drops its paired label file. `--across-splits` additionally flags duplicate groups whose members span more than one split (train/val/test) — the highest-value check, since that's data leakage between splits.

```bash
data dedup data/my_data data/my_data_deduped --across-splits
```

| Option | Required | Description |
|---|---|---|
| `--across-splits` |  | Warn when a duplicate group spans more than one split |
| `--dry-run` |  | Print what would be removed without writing `DST` |

### Splitting a dataset

Use `data split` to copy a dataset (classification or YOLO layout) into train/val/test, stratified by class. `SRC` must be a flat pool (classification: `<class>/*`; YOLO: `images/*` + `labels/*`) — an already-split `SRC` is rejected; run `data flatten` first, then re-split the result. YOLO images can carry boxes of more than one class, so the stratification key is each image's *primary* (most frequent, ties broken by lowest id) box class; images with no boxes are split the same proportional, seeded way as every other group.

```bash
data split data/my_data data/my_data_split --train 0.8 --val 0.1 --test 0.1 --seed 42
```

| Option | Required | Description |
|---|---|---|
| `--train FLOAT` |  | Fraction assigned to train (default: 0.8) |
| `--val FLOAT` |  | Fraction assigned to val (default: 0.1) |
| `--test FLOAT` |  | Fraction assigned to test (default: 0.1) |
| `--seed N` |  | Random seed for the stratified shuffle (default: 42) |
| `--dry-run` |  | Print the planned split without writing `DST` |

### Flattening a dataset

Use `data flatten` to copy an already-split dataset (classification or YOLO layout) back into one flat pool — the exact inverse of `data split`. Every image from every split (`train`/`val`/`test`) is copied into a single flat destination (classification: `<class>/*`; YOLO: `images/*` + `labels/*`), with no split structure left. A `SRC` that isn't already split is rejected — there's nothing to flatten. This is the way to re-split a dataset with different ratios or a different seed: flatten it, then split the flattened result.

```bash
data flatten data/my_data_split data/my_data_flat
data split data/my_data_flat data/my_data_resplit --train 0.7 --val 0.15 --test 0.15
```

| Option | Required | Description |
|---|---|---|
| `--dry-run` |  | Print the flatten plan without writing `DST` |

---

## Loss function

By default training uses categorical cross-entropy. Use `--loss` to switch to focal loss or to enable label smoothing, which are particularly useful when you have false positive problems.

**Focal loss** shifts gradient weight toward hard, misclassified examples and away from easy ones — forcing the model to focus on the ambiguous signal/noise boundary:

```bash
# Focal loss with default gamma (2.0)
train data/ --loss focal

# Tune the focusing parameter
train data/ --loss focal:gamma=1.5
```

**Label smoothing** prevents the model from becoming overconfident, making threshold-based rejection more reliable:

```bash
train data/ --loss crossentropy:label_smoothing=0.1
```

**Both combined** — focal loss with smoothing:

```bash
train data/ --loss focal:gamma=2.0,label_smoothing=0.1
```

| Option | Default | Description |
|---|---|---|
| `--loss crossentropy` | ✓ | Standard categorical cross-entropy |
| `--loss focal` | — | Focal loss (gamma=2.0) — focuses on hard examples |
| `gamma=F` | `2.0` | Focusing parameter; higher = stronger focus on hard examples |
| `label_smoothing=F` | `0.0` | Smooths targets; applies to both loss types |

The loss config is saved to `config.yaml` and applied automatically when resuming a run.

