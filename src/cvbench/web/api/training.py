"""Training API — start and monitor training runs.

Planned endpoints
-----------------
    POST /api/train                   start a training run (accepts same params as CLI `train`)
    GET  /api/train/{name}/status     poll run status + latest metrics from config.yaml

Each handler should call:
    from cvbench.services.training import run_training

Progress streaming design
-------------------------
Training is long-running, so POST /api/train must start work in a background
thread and return immediately with {"run": "my_run_name"}.

Live progress is delivered via Server-Sent Events (SSE):
    GET /api/train/{name}/stream  →  text/event-stream

The SSE endpoint pushes one event per epoch:
    data: {"epoch": 3, "loss": 0.42, "val_accuracy": 0.87, "done": false}
    data: {"epoch": 10, "loss": 0.21, "val_accuracy": 0.94, "done": true}

The browser subscribes with:
    const es = new EventSource("/api/train/my_run/stream");
    es.onmessage = e => updateProgressUI(JSON.parse(e.data));

This is wired through services.training.run_training(on_epoch_end=...).
See the on_epoch_end note in cvbench/services/training.py for implementation details.

TODO: implement (tracked in a follow-up GitHub issue)
"""

from fastapi import APIRouter

router = APIRouter()


# @router.post("/train")
# def start_training(params: TrainingParams): ...

# @router.get("/train/{name}/status")
# def training_status(name: str): ...
