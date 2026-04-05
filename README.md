# CVBench

GPU-enabled computer vision training sandbox. Keras + TensorFlow + JupyterLab in one container.

## Prerequisites

- Docker 24+
- **GPU (optional):** [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) — required only if you want GPU acceleration. The container runs on CPU without it.

---

## Quick start

### Prepare a workspace

Create a directory to hold your data, configs, and outputs. `~/cvbench` is a convenient default:

```bash
mkdir -p ~/cvbench/{data,configs,experiments}
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
  -v ~/cvbench/configs:/home/cvbench/configs \
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
  -v ~/cvbench/configs:/home/cvbench/configs \
  -v ~/cvbench/experiments:/home/cvbench/experiments \
  --restart unless-stopped \
  mmgalushka/cvbench:latest
```

---

### Option B — Docker Compose

Save the appropriate file as `~/cvbench/docker-compose.yml` and run `docker compose up -d`.

**With GPU** (`runtime: nvidia` requires NVIDIA Container Toolkit):

```yaml
services:
  cvbench:
    image: mmgalushka/cvbench:latest   # pin a release: mmgalushka/cvbench:0.2.0
    container_name: cvbench
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "0.0.0.0:8888:8888"
      - "0.0.0.0:6006:6006"
    volumes:
      - ~/cvbench/data:/home/cvbench/data
      - ~/cvbench/configs:/home/cvbench/configs
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
      - ~/cvbench/configs:/home/cvbench/configs
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
docker exec cvbench generate --train 200 --val 50 --test 50
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
                         [--augmentation FILE] [--resume CHECKPOINT] [--output DIR]
evaluate      <run_dir>  [--split val|test] [--output-dir PATH]
predict       --checkpoint <path> --input <image-or-folder>
runs          list       [dir] [--sort val_accuracy|date|backbone]
runs          compare    <run_a_dir> <run_b_dir>
runs          best       [dir] [--metric val_accuracy|val_loss|test_accuracy]
generate      [out_dir]  [--train N] [--val N] [--test N] [--image-size N]
augmentations list
augmentations example    [light|standard|heavy|reference] [--output FILE]
```

---

## Volume mounts

| Host path                    | Container path                   | Notes                             |
|------------------------------|----------------------------------|-----------------------------------|
| `~/cvbench/data`             | `/home/cvbench/data`             | Image datasets (real + synthetic) |
| `~/cvbench/configs`          | `/home/cvbench/configs`          | YAML configs and augmentation files |
| `~/cvbench/experiments`      | `/home/cvbench/experiments`      | Experiment directories            |

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
augmentations example standard --output configs/my_aug.yaml
# edit configs/my_aug.yaml as needed
train data/ --augmentation configs/my_aug.yaml --epochs 30
```

**`reference` preset** generates a commented-out file showing every available transform — open it in an editor and uncomment what you want:

```bash
augmentations example reference --output configs/aug_ref.yaml
```
