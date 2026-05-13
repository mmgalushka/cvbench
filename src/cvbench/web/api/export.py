"""Exports API — list, trigger, download, and delete model exports."""

import asyncio
import io
import json
import os
import shutil
import tarfile
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from cvbench.core.runs import resolve_run_dir

router = APIRouter()

_VALID_FORMATS = {"tflite", "onnx", "hailo", "plan"}
_VALID_QUANTIZE = {"none", "float16", "int8"}
_MODEL_FILENAMES = ("model.tflite", "model_float16.tflite", "model_int8.tflite", "model.onnx")
_HAILO_PACKAGE_FILES = ("model.tflite", "calib_set.npy", "model.alls", "export_info.json")


def _scan_exports(run_dir: Path) -> list[dict]:
    export_base = run_dir / "export"
    if not export_base.exists():
        return []
    results = []
    for subfolder in sorted(export_base.iterdir()):
        if not subfolder.is_dir():
            continue
        info_path = subfolder / "export_info.json"
        if not info_path.exists():
            continue
        info = json.loads(info_path.read_text())
        model_file = None
        size_mb = None
        for fname in _MODEL_FILENAMES:
            candidate = subfolder / fname
            if candidate.exists():
                model_file = fname
                size_mb = round(candidate.stat().st_size / (1024 * 1024), 2)
                break
        results.append({
            "subfolder": subfolder.name,
            "model_file": model_file,
            "size_mb": size_mb,
            **info,
        })
    return results


@router.get("/runs/{name}/exports")
def list_exports(name: str):
    try:
        run_dir = Path(resolve_run_dir(name))
    except Exception:
        raise HTTPException(status_code=404, detail=f"Run '{name}' not found")
    return _scan_exports(run_dir)


class ExportRequest(BaseModel):
    format: str
    quantize: str = "none"
    calib_total: int = 1024
    calib_strategy: str = "stratified"


@router.post("/runs/{name}/exports")
async def create_export(name: str, req: ExportRequest):
    if req.format not in _VALID_FORMATS:
        raise HTTPException(status_code=400, detail=f"Invalid format. Choose from: {', '.join(sorted(_VALID_FORMATS))}")
    if req.format != "hailo" and req.quantize not in _VALID_QUANTIZE:
        raise HTTPException(status_code=400, detail=f"Invalid quantize. Choose from: {', '.join(sorted(_VALID_QUANTIZE))}")
    if req.calib_total < 1:
        raise HTTPException(status_code=400, detail="calib_total must be at least 1")
    if req.calib_strategy not in {"stratified", "proportional", "equal", "diverse"}:
        raise HTTPException(status_code=400, detail="calib_strategy must be 'stratified', 'proportional', 'equal', or 'diverse'")

    try:
        run_dir = Path(resolve_run_dir(name))
    except Exception:
        raise HTTPException(status_code=404, detail=f"Run '{name}' not found")

    if not (run_dir / "best.keras").exists():
        raise HTTPException(status_code=400, detail="No trained checkpoint found (best.keras missing)")

    from cvbench.services.export import run_export
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            lambda: run_export(
                name, req.format, req.quantize,
                calib_total=req.calib_total,
                calib_strategy=req.calib_strategy,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _scan_exports(run_dir)


def _stream_tar_gz(files: list[tuple[Path, str]], chunk_size: int = 65536):
    """Yield tar.gz chunks while the archive is being built (no full-buffer wait)."""
    read_fd, write_fd = os.pipe()

    def _build():
        with open(write_fd, "wb") as wf:
            with tarfile.open(fileobj=wf, mode="w:gz") as tar:
                for path, arcname in files:
                    tar.add(path, arcname=arcname)

    t = threading.Thread(target=_build, daemon=True)
    t.start()

    with open(read_fd, "rb") as rf:
        while chunk := rf.read(chunk_size):
            yield chunk

    t.join()


@router.get("/runs/{name}/exports/{subfolder}/download/{filename}")
def download_export(name: str, subfolder: str, filename: str):
    try:
        run_dir = Path(resolve_run_dir(name))
    except Exception:
        raise HTTPException(status_code=404, detail=f"Run '{name}' not found")

    export_base = (run_dir / "export").resolve()
    export_dir = (export_base / subfolder).resolve()
    try:
        export_dir.relative_to(export_base)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not export_dir.is_dir():
        raise HTTPException(status_code=404, detail="Export not found")

    archive_name = f"{name}_{subfolder}.tar.gz"

    if subfolder == "hailo":
        files_to_pack = [
            (export_dir / fname, fname)
            for fname in _HAILO_PACKAGE_FILES
            if (export_dir / fname).exists()
        ]
        if not files_to_pack:
            raise HTTPException(status_code=404, detail="Hailo package files not found")
    else:
        model_file = next(
            (export_dir / f for f in _MODEL_FILENAMES if (export_dir / f).exists()),
            None,
        )
        if model_file is None:
            raise HTTPException(status_code=404, detail="Model file not found in export")
        files_to_pack = [(model_file, model_file.name)]
        info_path = export_dir / "export_info.json"
        if info_path.exists():
            files_to_pack.append((info_path, "export_info.json"))

    return StreamingResponse(
        _stream_tar_gz(files_to_pack),
        media_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{archive_name}"'},
    )


@router.delete("/runs/{name}/exports/{subfolder}")
def delete_export(name: str, subfolder: str):
    try:
        run_dir = Path(resolve_run_dir(name))
    except Exception:
        raise HTTPException(status_code=404, detail=f"Run '{name}' not found")

    export_base = (run_dir / "export").resolve()
    export_dir = (export_base / subfolder).resolve()
    try:
        export_dir.relative_to(export_base)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not export_dir.is_dir():
        raise HTTPException(status_code=404, detail="Export not found")

    shutil.rmtree(export_dir)
    return _scan_exports(run_dir)
