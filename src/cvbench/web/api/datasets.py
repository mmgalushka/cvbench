"""Datasets API — browse and manage dataset image files."""
import base64
import shutil
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from cvbench.core.runs import EXPERIMENTS_DIR, scan_experiments, resolve_run_dir
from cvbench.core.config import load_config

router = APIRouter()

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
PAGE_SIZE_DEFAULT = 60
PAGE_SIZE_MAX = 200


def _encode_dir(path: Path) -> str:
    padded = base64.urlsafe_b64encode(str(path).encode()).decode()
    return padded.rstrip('=')


def _decode_dir(dir_id: str) -> Path:
    pad = (4 - len(dir_id) % 4) % 4
    try:
        raw = base64.urlsafe_b64decode(dir_id + '=' * pad).decode()
        return Path(raw).resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid directory ID")


def _assert_safe(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")


@router.get("/datasets")
def list_datasets():
    seen: dict[str, dict] = {}

    for run in scan_experiments(EXPERIMENTS_DIR):
        try:
            run_dir = resolve_run_dir(run['name'])
            cfg = load_config(run_dir)
        except Exception:
            continue

        data_dir = Path(cfg.data.data_dir).resolve()
        if not data_dir.is_dir() or str(data_dir) in seen:
            continue

        splits: dict[str, dict] = {}
        for split_name, raw_path in [
            ('train', cfg.data.train_dir),
            ('val',   cfg.data.val_dir),
            ('test',  cfg.data.test_dir),
        ]:
            if not raw_path:
                continue
            sp = Path(raw_path).resolve()
            if sp.is_dir():
                splits[split_name] = {'id': _encode_dir(sp)}

        seen[str(data_dir)] = {
            'id':          _encode_dir(data_dir),
            'name':        data_dir.name,
            'path':        str(data_dir),
            'num_classes': len(cfg.data.classes),
            'classes':     cfg.data.classes,
            'splits':      splits,
        }

    return list(seen.values())


@router.get("/datasets/{dir_id}/images")
def list_images(
    dir_id: str,
    cls: Optional[str] = Query(None, alias='class'),
    page: int = Query(1, ge=1),
    page_size: int = Query(PAGE_SIZE_DEFAULT, ge=1, le=PAGE_SIZE_MAX),
):
    root = _decode_dir(dir_id)
    if not root.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")

    if cls:
        search_dir = (root / cls).resolve()
        _assert_safe(search_dir, root)
        if not search_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Class '{cls}' not found")
    else:
        search_dir = root

    all_images = sorted(
        f for f in search_dir.rglob('*')
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )

    classes = sorted(p.name for p in root.iterdir() if p.is_dir())

    total = len(all_images)
    start = (page - 1) * page_size
    items = []
    for img_path in all_images[start: start + page_size]:
        rel = img_path.relative_to(root)
        parts = rel.parts
        img_cls = parts[0] if len(parts) > 1 else None
        items.append({
            'path':     str(rel),
            'class':    img_cls,
            'filename': img_path.name,
        })

    return {
        'items':     items,
        'classes':   classes,
        'total':     total,
        'page':      page,
        'page_size': page_size,
        'pages':     max(1, (total + page_size - 1) // page_size),
    }


@router.get("/datasets/{dir_id}/file/{path:path}")
def serve_image(dir_id: str, path: str):
    root = _decode_dir(dir_id)
    if not root.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")
    img_path = (root / path).resolve()
    _assert_safe(img_path, root)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(img_path))


@router.post("/datasets/{dir_id}/images")
async def upload_images(
    dir_id: str,
    files: list[UploadFile] = File(...),
    cls: Optional[str] = Query(None, alias='class'),
):
    root = _decode_dir(dir_id)
    if not root.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")

    dest_dir = root
    if cls:
        dest_dir = (root / cls).resolve()
        _assert_safe(dest_dir, root)
        dest_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for upload in files:
        if not upload.filename:
            continue
        fname = Path(upload.filename).name
        if Path(fname).suffix.lower() not in IMAGE_EXTS:
            continue
        dest = dest_dir / fname
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            i = 1
            while dest.exists():
                dest = dest_dir / f"{stem}_{i}{suffix}"
                i += 1
        with dest.open('wb') as f:
            shutil.copyfileobj(upload.file, f)
        saved.append(str(dest.relative_to(root)))

    return {'uploaded': len(saved), 'files': saved}


@router.delete("/datasets/{dir_id}/images/{path:path}")
def delete_image(dir_id: str, path: str):
    root = _decode_dir(dir_id)
    if not root.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")
    img_path = (root / path).resolve()
    _assert_safe(img_path, root)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    if not img_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")
    img_path.unlink()
    return {'deleted': path}
