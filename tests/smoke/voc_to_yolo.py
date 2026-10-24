"""Convert a Pascal-VOC-style XML dataset into a CVBench YOLO dataset subset.

Usage: voc_to_yolo.py VOC_ROOT DST [--classes a,b,c] [--train N] [--val N] [--test N]

VOC_ROOT holds JPEGImages/ and Annotations/ (VOC2007 layout).
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def parse(xml_path, class_ids):
    root = ET.parse(xml_path).getroot()
    w = float(root.findtext("size/width"))
    h = float(root.findtext("size/height"))
    lines = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in class_ids:
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        lines.append(
            f"{class_ids[name]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
        )
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voc_root", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--classes", default="cat,dog,car,person")
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--val", type=int, default=60)
    ap.add_argument("--test", type=int, default=60)
    args = ap.parse_args()

    classes = args.classes.split(",")
    class_ids = {c: i for i, c in enumerate(classes)}
    items = []
    for xml in sorted((args.voc_root / "Annotations").glob("*.xml")):
        lines = parse(xml, class_ids)
        if lines:
            items.append((xml.stem, lines))
    random.Random(0).shuffle(items)

    if args.dst.exists():
        shutil.rmtree(args.dst)
    start = 0
    for split, n in (("train", args.train), ("val", args.val), ("test", args.test)):
        (args.dst / "images" / split).mkdir(parents=True)
        (args.dst / "labels" / split).mkdir(parents=True)
        for stem, lines in items[start:start + n]:
            shutil.copy(args.voc_root / "JPEGImages" / f"{stem}.jpg", args.dst / "images" / split)
            (args.dst / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        start += n

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    (args.dst / "data.yaml").write_text(
        f"path: {args.dst}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n"
    )
    print(f"wrote {args.dst} ({start} images, {len(classes)} classes)")


if __name__ == "__main__":
    main()
