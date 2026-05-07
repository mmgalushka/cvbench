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
    ports:
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
    ports:
      - "0.0.0.0:8888:8888"
      - "0.0.0.0:6006:6006"
    volumes:
      - ~/cvbench/data:/home/cvbench/data
      - ~/cvbench/workspace:/home/cvbench/workspace
      - ~/cvbench/experiments:/home/cvbench/experiments
    restart: unless-stopped
```

After starting:

```bash
# JupyterLab → http://<server-ip>:8888
# TensorBoard → http://<server-ip>:6006
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

## CLI reference

```
train         <data_dir> [--epochs N] [--backbone NAME] [--lr FLOAT] [--batch-size N]
                         [--optimizer adam|sgd[:weight_decay=F][,momentum=F]]
                         [--lr-scheduler patience=N[,factor=F][,min=F]]
                         [--loss crossentropy|focal[:gamma=F][,label_smoothing=F]]
                         [--fine-tune-from-layer N] [--augmentation FILE]
                         [--val-split FLOAT] [--resume CHECKPOINT] [--output DIR]
evaluate      <experiment>  [--output-dir PATH]
predict       <experiment> <image-or-folder> [--format keras|onnx|tflite|plan|all]
runs          list       [dir] [--sort val_accuracy|date|backbone]
runs          compare    <experiment_a> <experiment_b>
runs          best       [dir] [--metric val_accuracy|val_loss|test_accuracy]
runs          rename     <experiment> <new-name>
runs          export     <experiment> --format tflite|onnx|plan|hailo [--quantize none|float16|int8] [--output DIR] [--calib-samples-per-class N]
data          generate   [out_dir]  [--train N] [--val N] [--test N] [--image-size N]
data          explore    <data_dir> [--split train|val|test] [--threshold N]
data          upsample   <src_dir> <dst_dir> --augmentation <file> --target <N>
augmentations list
augmentations example    [light|standard|heavy|reference] [--output FILE]
```

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

Augmentation is configured via a standalone YAML file and passed to `train` with `--augmentation`.

**Discover what's available:**

```bash
augmentations list                          # all transforms + default params
augmentations example                       # list presets (light / standard / heavy / reference)
```

**Generate a starting config:**

```bash
augmentations example standard --output workspace/my_aug.yaml
# edit workspace/my_aug.yaml as needed
train data/ --augmentation workspace/my_aug.yaml --epochs 30
```

**`reference` preset** generates a commented-out file showing every available transform — open it in an editor and uncomment what you want:

```bash
augmentations example reference --output workspace/aug_ref.yaml
```

### Upsampling a class folder

Use `data upsample` to materialise an augmented copy of a single class folder on disk. This is useful for correcting class imbalance before training — apply it only to the classes that need more samples (e.g. skip the `noise` class if it is already well-represented).

```bash
# Upsample the 'dog' class from however many originals it has to 1500 images.
# The destination folder must be empty or non-existent.
data upsample data/my_data/train/dog data/my_data_aug/train/dog \
  --augmentation workspace/my_aug.yaml \
  --target 1500
```

**What it does:**

1. Copies every original image to `dst_dir` with a fresh 16-char random hex filename.
2. Randomly picks source images and augments them until `--target` is reached.
3. Uses MD5 hashing to detect exact duplicates; retries up to 10 times per sample before skipping.
4. If the source already has ≥ `--target` images, the command exits with a hint to use `data downsample` instead (not yet implemented).

| Option | Required | Description |
|---|---|---|
| `--augmentation FILE` | ✓ | Augmentation YAML spec (same format as `--augmentation` in `train`) |
| `--target N` | ✓ | Total number of images the destination folder should contain |

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

