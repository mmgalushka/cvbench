# Installation & Volumes

## Prerequisites

- Docker 24+
- **GPU (optional):** [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) — required only if you want GPU acceleration. The container runs on CPU without it.

<!-- IMAGE PLACEHOLDER: architecture diagram — see design spec for generation prompt. Suggested alt text: "CVBench container architecture: CLI, WebUI, and JupyterLab modules with mounted data/workspace/experiments volumes" -->

## Prepare a workspace

Create a directory to hold your data, workspace files, and outputs. `~/cvbench` is a convenient default:

```bash
mkdir -p ~/cvbench/{data,workspace,experiments}
cd ~/cvbench
```

---

## Option A — plain `docker run`

**With GPU:**

```bash
docker run -d \
  --name cvbench \
  --gpus all \
  -p 0.0.0.0:8888:8888 \
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
  -v ~/cvbench/data:/home/cvbench/data \
  -v ~/cvbench/workspace:/home/cvbench/workspace \
  -v ~/cvbench/experiments:/home/cvbench/experiments \
  --restart unless-stopped \
  mmgalushka/cvbench:latest
```

---

## Option B — Docker Compose

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
```

---

## Volume mounts

| Host path                    | Container path                   | Notes                             |
|------------------------------|----------------------------------|-----------------------------------|
| `~/cvbench/data`             | `/home/cvbench/data`             | Image datasets (real + synthetic) |
| `~/cvbench/workspace`        | `/home/cvbench/workspace`        | Augmentation configs, user notebooks, and other working files |
| `~/cvbench/experiments`      | `/home/cvbench/experiments`      | Experiment directories            |

!!! tip
    All three directories are just bind mounts — nothing on the host is
    special. Point `~/cvbench` at any location with enough disk space (an
    external drive, a NAS mount) and the container works the same way.

Next: [Your First Model](first-model.md) walks through generating data,
training, evaluating, and serving a run end to end.
