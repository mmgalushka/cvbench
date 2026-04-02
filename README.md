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
mkdir -p ~/cvbench/{data/train,data/val,data/test,work_dirs,augmentations,configs,experiments}
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
  -v ~/cvbench/data:/workspace/data:ro \
  -v ~/cvbench/work_dirs:/workspace/work_dirs \
  -v ~/cvbench/augmentations:/workspace/augmentations \
  -v ~/cvbench/configs:/workspace/configs \
  -v ~/cvbench/experiments:/workspace/experiments \
  --restart unless-stopped \
  mmgalushka/cvbench:latest
```

**CPU only** (drop `--gpus all`):

```bash
docker run -d \
  --name cvbench \
  -p 0.0.0.0:8888:8888 \
  -p 0.0.0.0:6006:6006 \
  -v ~/cvbench/data:/workspace/data:ro \
  -v ~/cvbench/work_dirs:/workspace/work_dirs \
  -v ~/cvbench/augmentations:/workspace/augmentations \
  -v ~/cvbench/configs:/workspace/configs \
  -v ~/cvbench/experiments:/workspace/experiments \
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
      - ~/cvbench/data:/workspace/data:ro
      - ~/cvbench/work_dirs:/workspace/work_dirs
      - ~/cvbench/augmentations:/workspace/augmentations
      - ~/cvbench/configs:/workspace/configs
      - ~/cvbench/experiments:/workspace/experiments
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
      - ~/cvbench/data:/workspace/data:ro
      - ~/cvbench/work_dirs:/workspace/work_dirs
      - ~/cvbench/augmentations:/workspace/augmentations
      - ~/cvbench/configs:/workspace/configs
      - ~/cvbench/experiments:/workspace/experiments
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
docker exec cvbench generate /workspace/data --train 200 --val 50 --test 50
```

---

## Training

From a JupyterLab terminal or SSH session:

```bash
docker exec -it cvbench bash
tmux new -s train
train /workspace/data --epochs 20 --backbone efficientnet_b0
# Ctrl+B D to detach — training continues after you close the terminal
```

### TensorBoard

```bash
docker exec cvbench tensorboard --logdir /workspace/work_dirs --host 0.0.0.0 --port 6006
# → http://<server-ip>:6006
```

---

## CLI reference

```
train     <data_dir> [--epochs N] [--backbone NAME] [--lr FLOAT] [--batch-size N]
                     [--aug-preset NAME] [--resume CHECKPOINT] [--output DIR]
evaluate  <run_dir>  [--split val|test] [--output-dir PATH]
predict   --checkpoint <path> --input <image-or-folder>
runs      list       [dir] [--sort val_accuracy|date|backbone]
runs      compare    <run_a_dir> <run_b_dir>
runs      best       [dir] [--metric val_accuracy|val_loss|test_accuracy]
generate  <out_dir>  [--train N] [--val N] [--test N] [--image-size N]
```

---

## Volume mounts

| Host path                    | Container path              | Notes                              |
|------------------------------|-----------------------------|------------------------------------|
| `~/cvbench/data`             | `/workspace/data` (ro)      | Image dataset — read-only          |
| `~/cvbench/work_dirs`        | `/workspace/work_dirs`      | Checkpoints, logs, eval reports    |
| `~/cvbench/augmentations`    | `/workspace/augmentations`  | Custom augmentation Python files   |
| `~/cvbench/configs`          | `/workspace/configs`        | YAML experiment configs            |
| `~/cvbench/experiments`      | `/workspace/experiments`    | Experiment directories             |

---

## Custom augmentations

Place a Python file in `~/cvbench/augmentations/` and reference it in your config:

```python
# augmentations/my_aug.py
from core.registry import register_aug
import keras

@register_aug("my_aug")
def build(params: dict) -> keras.Sequential:
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(params.get("rotation", 0.1)),
    ])
```

```bash
train /workspace/data --aug-preset my_aug
```
