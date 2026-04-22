"""Explain API — Grad-CAM heatmap for a single image."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cvbench.core.runs import resolve_run_dir
from cvbench.core.config import load_config

router = APIRouter()


class GradCamRequest(BaseModel):
    image_path: str
    class_index: int


@router.post("/runs/{name}/explain/gradcam")
def gradcam(name: str, body: GradCamRequest):
    try:
        run_dir = resolve_run_dir(name)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Run '{name}' not found")

    cfg = load_config(run_dir)

    # Resolve and validate image path stays inside test_dir
    test_dir = Path(cfg.data.test_dir).resolve()
    img_path = (test_dir / body.image_path).resolve()
    try:
        img_path.relative_to(test_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {body.image_path}")

    # Find the best checkpoint for this run
    checkpoint = _find_checkpoint(Path(run_dir))
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="No model checkpoint found for this run")

    try:
        from cvbench.services.gradcam import compute_gradcam
        heatmap_b64 = compute_gradcam(
            checkpoint=str(checkpoint),
            image_path=str(img_path),
            class_index=body.class_index,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse({"heatmap_b64": heatmap_b64})


def _find_checkpoint(run_dir: Path) -> Path | None:
    for pattern in ("best_model.keras", "*.keras", "best_model.h5", "*.h5"):
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return matches[0]
    return None
