"""Prediction API — single-image inference, augmentation, and XAI."""

from __future__ import annotations

import json

from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from cvbench.augmentations.registry import get_schema
from cvbench.core.runs import resolve_run_dir

router = APIRouter()


@router.get("/augmentations")
def list_augmentations():
    """Return UI parameter schemas for all registered augmentations."""
    return get_schema()


@router.post("/predict/single")
async def predict_single(
    file: UploadFile = File(...),
    run: str = Form(...),
):
    """Run inference on an uploaded image using the checkpoint from the given run."""
    image_bytes = await file.read()
    try:
        from cvbench.services.prediction import predict_image
        result = predict_image(run, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(result)


@router.post("/predict/augmented")
async def predict_augmented(
    file: UploadFile = File(...),
    run: str = Form(...),
    augmentations: str = Form(...),
):
    """Apply augmentations to an uploaded image then run inference.

    `augmentations` is a JSON string: [{name: str, params: {…}}, …]
    """
    image_bytes = await file.read()
    try:
        aug_list = json.loads(augmentations)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="augmentations must be valid JSON")

    try:
        from cvbench.services.prediction import predict_augmented as _predict_augmented
        result = _predict_augmented(run, image_bytes, aug_list)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(result)


@router.post("/predict/xai")
async def predict_xai(
    file: UploadFile = File(...),
    run: str = Form(...),
    class_index: int = Form(...),
):
    """Compute a Grad-CAM heatmap for an uploaded image."""
    image_bytes = await file.read()

    try:
        run_dir = resolve_run_dir(run)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Run '{run}' not found")

    from pathlib import Path
    from cvbench.web.api.explain import _find_checkpoint

    checkpoint = _find_checkpoint(Path(run_dir))
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="No model checkpoint found for this run")

    try:
        from cvbench.services.gradcam import compute_gradcam_from_bytes
        heatmap_b64 = compute_gradcam_from_bytes(str(checkpoint), image_bytes, class_index)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse({"heatmap_b64": heatmap_b64})
