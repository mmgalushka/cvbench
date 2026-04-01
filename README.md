# CVBench

GPU-enabled computer vision training sandbox. Keras + TensorFlow + JupyterLab in one container.

## Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Verify GPU access: `docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi`

---

## Quick start (Docker Hub)

Create a `docker-compose.yml` with the following content, then run `docker compose up -d`.

```yaml
services:
  cvbench:
    image: mmgalushka/cvbench:latest
    container_name: cvbench
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "0.0.0.0:8888:8888"
      - "0.0.0.0:6006:6006"
    volumes:
      - ./data:/workspace/data:ro
      - ./work_dirs:/workspace/work_dirs
      - ./augmentations:/workspace/augmentations
      - ./configs:/workspace/configs
      - ./experiments:/workspace/experiments
    restart: unless-stopped
```

```bash
# Create workspace directories
mkdir -p data/train data/val data/test work_dirs augmentations configs experiments

# Start the container
docker compose up -d

# JupyterLab → http://<server-ip>:8888
```

### Generate a synthetic dataset (smoke-test)

```bash
docker compose exec cvbench generate /workspace/data --train 200 --val 50 --test 50
```

---

## Training

From a JupyterLab terminal or SSH session:

```bash
docker compose exec cvbench bash
tmux new -s train
train /workspace/data --epochs 20 --backbone efficientnet_b0
# Ctrl+B D to detach — training continues after you close the terminal
```

### TensorBoard

```bash
docker compose exec cvbench tensorboard --logdir /workspace/work_dirs --host 0.0.0.0 --port 6006
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

| Host path       | Container path              | Notes                              |
|-----------------|-----------------------------|------------------------------------|
| `./data`        | `/workspace/data` (ro)      | Image dataset — read-only          |
| `./work_dirs`   | `/workspace/work_dirs`      | Checkpoints, logs, eval reports    |
| `./augmentations` | `/workspace/augmentations` | Custom augmentation Python files  |
| `./configs`     | `/workspace/configs`        | YAML experiment configs            |
| `./experiments` | `/workspace/experiments`    | Experiment directories             |

---

## Custom augmentations

Place a Python file in `./augmentations/` and reference it in your config:

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

---

## Development

```yaml
# docker-compose.yml (local dev — builds from source)
services:
  cvbench:
    build:
      context: .
      target: dev
    image: cvbench
    container_name: cvbench
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "0.0.0.0:8888:8888"
      - "0.0.0.0:6006:6006"
    volumes:
      - ./data:/workspace/data:ro
      - ./configs:/workspace/configs
      - ./augmentations:/workspace/augmentations
      - ./work_dirs:/workspace/work_dirs
    restart: unless-stopped
```

```bash
./helper.sh init        # create .venv and install dependencies
./helper.sh test        # run test suite
./helper.sh test -m "not tf"   # skip TensorFlow-dependent tests
```
