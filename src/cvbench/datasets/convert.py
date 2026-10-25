"""``data convert`` — turn a Pascal VOC or COCO detection dataset into CVBench's YOLO layout.

YOLO stays the single internal format. Other formats are recognised (``detect_format``) but
never trained on directly: the conversion is an explicit step, so the choice of classes and
splits is made by whoever has looked at the data.

* Pascal VOC — ``Annotations/*.xml`` + ``JPEGImages/*`` (+ ``ImageSets/Main/<split>.txt``).
* COCO — instances JSON files (``annotations/instances_<split>.json``, a root ``*.json``, or
  Roboflow's ``<split>/_annotations.coco.json``); the images are found next to the JSON.

When the source ships its own split lists they are used as they are; otherwise (or with
``resplit``) the images are split with the same seeded, class-stratified shuffle as
``data split``. TensorFlow-free.
"""
from __future__ import annotations

import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cvbench.core.errors import DatasetError
from cvbench.datasets import layout
from cvbench.datasets.split import stratified_split

FORMAT_VOC = "voc"
FORMAT_COCO = "coco"
FORMAT_LABELS = {FORMAT_VOC: "Pascal VOC", FORMAT_COCO: "COCO"}

_VOC_ANNOTATIONS = "Annotations"
_VOC_IMAGES = "JPEGImages"
_VOC_SETS = Path("ImageSets") / "Main"
# Source split names (as spelled in file names and folders) -> CVBench split names.
_COCO_PARSE_LIMIT = 50 * 1024 * 1024
_SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}


@dataclass
class Box:
    """One box in pixels: top-left ``(x1, y1)``, bottom-right ``(x2, y2)``."""

    cls: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Sample:
    image: Path
    width: float
    height: float
    boxes: list[Box]
    split: str | None = None  # the source's own split, when it has one


@dataclass
class Source:
    """Everything read from a VOC or COCO dataset, before classes and splits are chosen."""

    format: str
    samples: list[Sample] = field(default_factory=list)
    dropped: Counter[str] = field(default_factory=Counter)  # reason -> count
    kept_notes: Counter[str] = field(default_factory=Counter)  # things kept the user may expect gone


@dataclass
class ConvertAction:
    src_image: Path
    dst_image: Path  # relative to dst
    dst_label: Path  # relative to dst
    lines: list[str]


@dataclass
class ConvertPlan:
    source_format: str
    class_names: list[str]
    actions: list[ConvertAction] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)  # split -> class -> boxes
    images: dict[str, int] = field(default_factory=dict)  # split -> images
    dropped: Counter[str] = field(default_factory=Counter)
    kept_notes: Counter[str] = field(default_factory=Counter)
    used_source_splits: bool = False

    @property
    def splits_written(self) -> list[str]:
        return [s for s in layout.SPLIT_NAMES if self.images.get(s)]


# ---- format detection ---------------------------------------------------------------------


def _split_of(name: str) -> str | None:
    """CVBench split named by the first token of NAME ("instances_val2017" -> "val")."""
    low = name.lower()
    for key, split in _SPLIT_ALIASES.items():
        if key in low:
            return split
    return None


def _is_coco_json(path: Path) -> bool:
    """Whether PATH is a COCO detection file.

    A small file is parsed and its top-level keys checked. A big one (``instances_train2017.json``
    is hundreds of MB) is only recognised by an ``"images"`` key in its first bytes; ``read_coco``
    then reports a real problem if it turns out not to be COCO.
    """
    try:
        if path.stat().st_size > _COCO_PARSE_LIMIT:
            with open(path) as f:
                return '"images"' in f.read(4096)
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (
        isinstance(data, dict)
        and isinstance(data.get("images"), list)
        and isinstance(data.get("annotations"), list)
        and isinstance(data.get("categories"), list)
    )


def coco_files(root: Path) -> list[Path]:
    """COCO annotation files found under ROOT (sorted), at the places exporters put them."""
    if not root.is_dir():
        return []
    candidates = [*root.glob("*.json"), *(root / "annotations").glob("*.json"),
                  *root.glob("*/_annotations*.json")]
    return sorted(p for p in dict.fromkeys(candidates) if p.is_file() and _is_coco_json(p))


def is_voc_dataset(root: Path) -> bool:
    return (root / _VOC_ANNOTATIONS).is_dir() and (root / _VOC_IMAGES).is_dir()


def detect_format(root: Path) -> str | None:
    """``"voc"`` or ``"coco"`` when ROOT looks like one of them, ``None`` otherwise."""
    if layout.is_yolo_dataset(root):
        return None
    if is_voc_dataset(root):
        return FORMAT_VOC
    if coco_files(root):
        return FORMAT_COCO
    return None


# ---- readers ------------------------------------------------------------------------------


def _image_size(path: Path) -> tuple[float, float]:
    from PIL import Image

    with Image.open(path) as im:
        return float(im.width), float(im.height)


def _voc_split_lists(root: Path) -> dict[str, set[str]]:
    """``{cvbench split: image stems}`` from ``ImageSets/Main/{train,val,test}.txt`` (those present)."""
    lists: dict[str, set[str]] = {}
    for name, split in (("train", "train"), ("val", "val"), ("test", "test")):
        path = root / _VOC_SETS / f"{name}.txt"
        if path.is_file():
            # A line is "<stem>" or, in the per-class lists, "<stem> <1|0|-1>".
            lists[split] = {ln.split()[0] for ln in path.read_text().splitlines() if ln.strip()}
    return lists


def _voc_image(root: Path, xml_root: ET.Element, stem: str) -> Path | None:
    named = xml_root.findtext("filename")
    for name in (named, *(f"{stem}{ext}" for ext in (".jpg", ".jpeg", ".png"))):
        if name and (root / _VOC_IMAGES / name).is_file():
            return root / _VOC_IMAGES / name
    return None


def read_voc(root: Path) -> Source:
    src = Source(FORMAT_VOC)
    lists = _voc_split_lists(root)
    for xml_path in sorted((root / _VOC_ANNOTATIONS).glob("*.xml")):
        try:
            tree = ET.parse(xml_path).getroot()
        except ET.ParseError:
            src.dropped["unreadable annotation file"] += 1
            continue
        image = _voc_image(root, tree, xml_path.stem)
        if image is None:
            src.dropped["annotation without an image"] += 1
            continue
        try:
            width = float(tree.findtext("size/width") or 0)
            height = float(tree.findtext("size/height") or 0)
        except ValueError:
            width = height = 0.0
        if width <= 0 or height <= 0:
            width, height = _image_size(image)

        boxes = []
        for obj in tree.findall("object"):
            bnd = obj.find("bndbox")
            name = (obj.findtext("name") or "").strip()
            if bnd is None or not name:
                src.dropped["object without a box or name"] += 1
                continue
            if (obj.findtext("difficult") or "0").strip() == "1":
                src.dropped["objects marked difficult"] += 1
                continue
            if (obj.findtext("truncated") or "0").strip() == "1":
                src.kept_notes["objects marked truncated (kept)"] += 1
            try:
                x1, y1, x2, y2 = (float(bnd.findtext(k) or "nan") for k in ("xmin", "ymin", "xmax", "ymax"))
            except ValueError:
                src.dropped["object with a malformed box"] += 1
                continue
            boxes.append(Box(name, x1, y1, x2, y2))

        split = next((s for s, stems in lists.items() if xml_path.stem in stems), None)
        src.samples.append(Sample(image, width, height, boxes, split))
    return src


def _coco_image_dirs(root: Path, json_path: Path, split: str | None) -> list[Path]:
    """Folders that may hold a COCO file's images, most specific first."""
    dirs = [json_path.parent, json_path.parent.parent, root, root / "images"]
    if split:  # COCO proper: train2017/, val2017/ next to annotations/
        for parent in (root, root / "images"):
            if parent.is_dir():
                dirs += [d for d in sorted(parent.iterdir()) if d.is_dir() and _split_of(d.name) == split]
    return list(dict.fromkeys(dirs))


def read_coco(root: Path) -> Source:
    src = Source(FORMAT_COCO)
    files = coco_files(root)
    has_splits = len(files) > 1 and all(_split_of(p.stem if p.name.startswith("instances") else p.parent.name)
                                        for p in files)
    for json_path in files:
        data = json.loads(json_path.read_text())
        if not all(isinstance(data.get(k), list) for k in ("images", "annotations", "categories")):
            raise DatasetError(f"'{json_path}' is not a COCO file (needs images, annotations and categories).")
        label = json_path.stem if json_path.name.startswith("instances") else json_path.parent.name
        split = _split_of(label) if has_splits else None
        names = {c["id"]: str(c["name"]) for c in data["categories"]}
        by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in data["annotations"]:
            by_image[ann["image_id"]].append(ann)

        search = _coco_image_dirs(root, json_path, split)
        for im in data["images"]:
            rel = Path(im["file_name"])
            image = next((d / rel for d in search if (d / rel).is_file()), None)
            if image is None:
                src.dropped["image listed but not found on disk"] += 1
                continue
            width, height = float(im.get("width") or 0), float(im.get("height") or 0)
            if width <= 0 or height <= 0:
                width, height = _image_size(image)
            boxes = []
            for ann in by_image.get(im["id"], []):
                if ann.get("iscrowd"):
                    src.dropped["crowd regions (iscrowd)"] += 1
                    continue
                name = names.get(ann["category_id"])
                bbox = ann.get("bbox") or []
                if name is None or len(bbox) != 4:
                    src.dropped["object without a class or box"] += 1
                    continue
                x, y, w, h = (float(v) for v in bbox)
                boxes.append(Box(name, x, y, x + w, y + h))
            src.samples.append(Sample(image, width, height, boxes, split))
    return src


def read_source(root: Path) -> Source:
    fmt = detect_format(root)
    if fmt == FORMAT_VOC:
        return read_voc(root)
    if fmt == FORMAT_COCO:
        return read_coco(root)
    raise DatasetError(
        f"'{root}' is not a Pascal VOC or COCO dataset. VOC needs {_VOC_ANNOTATIONS}/ and {_VOC_IMAGES}/; "
        "COCO needs an instances JSON (annotations/instances_*.json, a *.json in the root, or "
        "<split>/_annotations.coco.json)."
    )


# ---- planning -----------------------------------------------------------------------------


def _yolo_line(box: Box, class_id: int, width: float, height: float) -> str | None:
    x1, x2 = sorted((min(max(box.x1, 0.0), width), min(max(box.x2, 0.0), width)))
    y1, y2 = sorted((min(max(box.y1, 0.0), height), min(max(box.y2, 0.0), height)))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return (f"{class_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")


def _primary_class(lines: list[str]) -> str:
    """Most frequent class id among LINES, ties broken by the lowest id (as ``data split`` does)."""
    counts = Counter(int(ln.split()[0]) for ln in lines)
    top = max(counts.values())
    return str(min(c for c, n in counts.items() if n == top))


def plan_convert(
    root: Path,
    *,
    classes: list[str] | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    limit: int | None = None,
    resplit: bool = False,
) -> ConvertPlan:
    """Read ROOT and decide what ``apply_plan`` will write. Read-only.

    CLASSES picks (and orders) the classes to keep; by default every class that has a box,
    sorted by name. LIMIT keeps a seeded random sample of that many images. The source's
    own split lists are used unless RESPLIT.
    """
    src = read_source(root)
    available = sorted({b.cls for s in src.samples for b in s.boxes})
    if classes:
        missing = [c for c in classes if c not in available]
        if missing:
            raise DatasetError(
                f"Class(es) not found in '{root}': {', '.join(missing)}. Available: {', '.join(available)}."
            )
        class_names = list(dict.fromkeys(classes))
    else:
        class_names = available
    ids = {name: i for i, name in enumerate(class_names)}

    plan = ConvertPlan(src.format, class_names, dropped=Counter(src.dropped), kept_notes=Counter(src.kept_notes))
    kept: list[tuple[Sample, list[str]]] = []
    for sample in src.samples:
        lines = []
        for box in sample.boxes:
            if box.cls not in ids:
                plan.dropped["objects of classes not selected"] += 1
                continue
            line = _yolo_line(box, ids[box.cls], sample.width, sample.height)
            if line is None:
                plan.dropped["boxes with no area"] += 1
                continue
            lines.append(line)
        if lines:
            kept.append((sample, lines))
        else:
            plan.dropped["images left with no boxes"] += 1

    if limit is not None and limit < len(kept):
        plan.dropped["images beyond --limit"] += len(kept) - limit
        kept = random.Random(seed).sample(kept, limit)

    plan.used_source_splits = not resplit and any(s.split for s, _ in kept)
    assigned: dict[str, list[tuple[Sample, list[str]]]]
    if plan.used_source_splits:
        assigned = defaultdict(list)
        for sample, lines in kept:
            if sample.split is None:
                plan.dropped["images in none of the source's split lists"] += 1
            else:
                assigned[sample.split].append((sample, lines))
    else:
        assigned = stratified_split(kept, key_fn=lambda t: _primary_class(t[1]), ratios=ratios, seed=seed)

    for split in layout.SPLIT_NAMES:
        used: set[str] = set()
        for sample, lines in assigned.get(split, []):
            name = layout.dedupe_filename(sample.image.name, used)
            plan.actions.append(ConvertAction(
                sample.image,
                Path(layout.IMAGES_DIRNAME) / split / name,
                Path(layout.LABELS_DIRNAME) / split / f"{Path(name).stem}.txt",
                lines,
            ))
            plan.images[split] = plan.images.get(split, 0) + 1
            per_class = plan.counts.setdefault(split, {})
            for ln in lines:
                cls = class_names[int(ln.split()[0])]
                per_class[cls] = per_class.get(cls, 0) + 1
    if not plan.actions:
        raise DatasetError(f"Nothing to convert: no images with boxes of the selected classes in '{root}'.")
    return plan


def apply_plan(plan: ConvertPlan, dst: Path, progress: Callable[[int], None] | None = None) -> None:
    """Write PLAN at DST: copy each image, write its label file, then ``data.yaml``."""
    for action in plan.actions:
        image, label = dst / action.dst_image, dst / action.dst_label
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(action.src_image, image)
        label.write_text("\n".join(action.lines) + "\n")
        if progress:
            progress(1)
    layout.write_data_yaml(dst, plan.splits_written, plan.class_names)
