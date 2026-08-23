"""Datasets API — browse and manage dataset image files."""
import base64
import secrets
import shutil
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from cvbench.core.runs import EXPERIMENTS_DIR, scan_experiments, resolve_run_dir
from cvbench.core.config import load_config
from cvbench.datasets import layout as dataset_layout

router = APIRouter()

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


DATA_DIR = Path("data")

SPLIT_NAMES = ('train', 'val', 'test')


# ── YOLO datasets ───────────────────────────────────────────────────────────
#
# Layout rules and parsing live in cvbench.datasets.layout; this module only
# shapes the HTTP responses on top of them.


def _yolo_item(img_path: Path, root: Path, label_dir: Path, names: list[str]) -> dict:
    raw = dataset_layout.read_yolo_boxes(label_dir / f"{img_path.stem}.txt")
    boxes = [
        {
            'class_id': cls_id,
            'class':    names[cls_id] if 0 <= cls_id < len(names) else str(cls_id),
            'x': x, 'y': y, 'w': w, 'h': h,
        }
        for cls_id, (x, y, w, h) in raw
    ]
    return {
        'path':     str(img_path.relative_to(root)),
        'class':    None,
        'filename': img_path.name,
        'boxes':    boxes,
    }


def _yolo_dataset_entry(data_dir: Path) -> dict:
    splits: dict[str, dict] = {}
    for split_name in SPLIT_NAMES:
        sp = (data_dir / 'images' / split_name).resolve()
        if sp.is_dir():
            splits[split_name] = {'id': _encode_dir(sp)}
    classes = dataset_layout.yolo_class_names(data_dir)
    return {
        'id':          _encode_dir(data_dir),
        'name':        data_dir.name,
        'path':        str(data_dir),
        'format':      'yolo',
        'num_classes': len(classes),
        'classes':     classes,
        'splits':      splits,
    }


def _dataset_entry(data_dir: Path, classes: list[str] | None = None) -> dict:
    if dataset_layout.is_yolo_dataset(data_dir):
        return _yolo_dataset_entry(data_dir)

    splits: dict[str, dict] = {}
    for split_name in SPLIT_NAMES:
        sp = (data_dir / split_name).resolve()
        if sp.is_dir():
            splits[split_name] = {'id': _encode_dir(sp)}
    if classes is None:
        classes = sorted(
            p.name for p in data_dir.iterdir()
            if p.is_dir() and p.name not in SPLIT_NAMES
        )
    return {
        'id':          _encode_dir(data_dir),
        'name':        data_dir.name,
        'path':        str(data_dir),
        'format':      'classification',
        'num_classes': len(classes),
        'classes':     classes,
        'splits':      splits,
    }


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

        if dataset_layout.is_yolo_dataset(data_dir):
            seen[str(data_dir)] = _yolo_dataset_entry(data_dir)
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
            'format':      'classification',
            'num_classes': len(cfg.data.classes),
            'classes':     cfg.data.classes,
            'splits':      splits,
        }

    # Also include any dataset folders in data/ not referenced by any run
    if DATA_DIR.is_dir():
        for candidate in sorted(DATA_DIR.iterdir()):
            if not candidate.is_dir():
                continue
            resolved = candidate.resolve()
            if str(resolved) in seen:
                continue
            seen[str(resolved)] = _dataset_entry(resolved)

    return list(seen.values())


def _list_yolo_images(root: Path, ds_root: Path, cls: Optional[str],
                      page: int, page_size: int) -> dict:
    """Paginate a YOLO split directory, attaching annotations to every item."""
    label_dir = dataset_layout.yolo_label_dir(root, ds_root)
    names = dataset_layout.yolo_class_names(ds_root)

    all_images = dataset_layout.list_images(root)

    if cls:
        if cls not in names:
            raise HTTPException(status_code=404, detail=f"Class '{cls}' not found")
        # Filtering needs every label file parsed, so build the items up front.
        items = [_yolo_item(p, root, label_dir, names) for p in all_images]
        items = [it for it in items if any(b['class'] == cls for b in it['boxes'])]
        total = len(items)
        start = (page - 1) * page_size
        page_items = items[start: start + page_size]
    else:
        total = len(all_images)
        start = (page - 1) * page_size
        page_items = [
            _yolo_item(p, root, label_dir, names)
            for p in all_images[start: start + page_size]
        ]

    return {
        'items':     page_items,
        'classes':   names,
        'format':    'yolo',
        'total':     total,
        'page':      page,
        'page_size': page_size,
        'pages':     max(1, (total + page_size - 1) // page_size),
    }


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

    ds_root = dataset_layout.yolo_root(root)
    if ds_root is not None:
        return _list_yolo_images(root, ds_root, cls, page, page_size)

    if cls:
        search_dir = (root / cls).resolve()
        _assert_safe(search_dir, root)
        if not search_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Class '{cls}' not found")
    else:
        search_dir = root

    all_images = dataset_layout.list_images(search_dir)

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
        'format':    'classification',
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

    used_tokens: set[str] = set()

    def _fresh_token(suffix: str) -> Path:
        for _ in range(10_000):
            token = secrets.token_hex(8)  # 16 hex chars
            if token not in used_tokens:
                dest = dest_dir / f"{token}{suffix}"
                if not dest.exists():
                    used_tokens.add(token)
                    return dest
        raise RuntimeError("Could not generate a unique token after 10 000 attempts.")

    saved = []
    for upload in files:
        if not upload.filename:
            continue
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in dataset_layout.IMAGE_EXTS:
            continue
        dest = _fresh_token(suffix)
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

    # For YOLO splits the annotation file is worthless without its image.
    ds_root = dataset_layout.yolo_root(root)
    if ds_root is not None:
        label_path = dataset_layout.yolo_label_dir(root, ds_root) / f"{img_path.stem}.txt"
        if label_path.is_file():
            label_path.unlink()

    return {'deleted': path}
