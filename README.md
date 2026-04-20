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
                         [--lr-patience N] [--lr-factor F] [--lr-min F]
                         [--fine-tune-from-layer N] [--augmentation FILE]
                         [--use-lcn] [--lcn-kernel-size N] [--lcn-epsilon F]
                         [--resume CHECKPOINT] [--output DIR]
evaluate      <experiment>  [--output-dir PATH]
predict       --checkpoint <path> --input <image-or-folder>
runs          list       [dir] [--sort val_accuracy|date|backbone]
runs          compare    <experiment_a> <experiment_b>
runs          best       [dir] [--metric val_accuracy|val_loss|test_accuracy]
data          generate   [out_dir]  [--train N] [--val N] [--test N] [--image-size N]
data          explore    <data_dir> [--split train|val|test] [--threshold N]
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

## Learning rate scheduling

By default the learning rate is fixed for the entire training run. Use `--lr-patience` to enable **ReduceLROnPlateau** — the LR is multiplied by `--lr-factor` whenever `val_loss` fails to improve for N consecutive epochs.

```bash
# Reduce LR by 0.5x after 5 flat epochs (default factor and floor)
train data/ --lr 1e-3 --lr-patience 5

# Aggressive decay: cut to 20% after 3 flat epochs, floor at 1e-6
train data/ --lr 1e-3 --lr-patience 3 --lr-factor 0.2 --lr-min 1e-6
```

| Option | Default | Description |
|---|---|---|
| `--lr-patience N` | disabled | Epochs with no `val_loss` improvement before reducing LR |
| `--lr-factor F` | `0.5` | Multiplicative reduction factor |
| `--lr-min F` | `1e-7` | Minimum LR floor |

The scheduler settings are saved to `config.yaml` and applied automatically when resuming a run.

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

---

## Local Contrast Normalization (LCN)

Use `--use-lcn` when your images come from sensors with different brightness levels or sensitivities (e.g. time-frequency waterfall images from multiple sensors). LCN inserts a preprocessing layer inside the model that removes local mean and normalizes by local standard deviation, making the model respond to *pattern structure* rather than absolute pixel intensity.

The layer is saved as part of the model — no separate preprocessing is needed at inference.

```bash
# Enable LCN with defaults (kernel=32px, epsilon=1e-3)
train data/ --use-lcn --epochs 30

# Tune neighbourhood size and stability constant
train data/ --use-lcn --lcn-kernel-size 48 --lcn-epsilon 0.01 --epochs 30
```

| Option | Default | Description |
|---|---|---|
| `--use-lcn` | off | Enable Local Contrast Normalization before the backbone |
| `--lcn-kernel-size N` | `32` | Gaussian neighbourhood size in pixels |
| `--lcn-epsilon F` | `1e-3` | Stability constant — increase if flat/noisy regions produce artefacts |

**Kernel size guidance:** for 224×224 images, `32` covers ~14% of the image width — appropriate for fine-grained patterns. Use `48`–`64` for coarser structures or if weak signals appear on a noisy background.
