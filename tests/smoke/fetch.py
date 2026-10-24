"""Download real public datasets and cut them into small CVBench datasets for the smoke test.

Downloads and prepared subsets are cached under ``$CVBENCH_SMOKE_CACHE`` (default
``~/.cache/cvbench-smoke``), so only the first run pays for the network. Subsets are
seeded, so every run trains on the same images.
"""
from __future__ import annotations

import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).parent
SEED = 0


def cache_root() -> Path:
    root = Path(os.environ.get("CVBENCH_SMOKE_CACHE", "~/.cache/cvbench-smoke")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_manifest() -> list[dict]:
    return yaml.safe_load((HERE / "datasets.yaml").read_text())


def download(url: str, dest: Path) -> Path:
    """Fetch URL to DEST unless it is already there (written via a .part file, so a killed run leaves no stub)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
        shutil.copyfileobj(resp, out)
    part.rename(dest)
    return dest


def _sample_into(files: list[Path], dst: Path, n: int, rng: random.Random) -> list[Path]:
    """Copy N random files into DST; return the files that were not used."""
    files = sorted(files)
    rng.shuffle(files)
    dst.mkdir(parents=True, exist_ok=True)
    for f in files[:n]:
        shutil.copy(f, dst / f.name)
    return files[n:]


def _roboflow_zips(entry: dict, raw: Path, out: Path) -> None:
    rng = random.Random(SEED)
    for src_split, split in (("train", "train"), ("valid", "val"), ("test", "test")):
        z = download(f"{entry['base_url']}/{src_split}.zip", raw / f"{src_split}.zip")
        extracted = raw / src_split
        if not extracted.exists():
            zipfile.ZipFile(z).extractall(extracted)
        classes = sorted(p for p in extracted.iterdir() if p.is_dir())
        for cls in classes:
            _sample_into(list(cls.glob("*.jpg")), out / split / cls.name, entry[split] // len(classes), rng)


def _hf_parquet(entry: dict, raw: Path, out: Path) -> None:
    parquet = download(entry["url"], raw / "data.parquet")
    script = (
        "import sys, random, pyarrow.parquet as pq\n"
        "from pathlib import Path\n"
        "src, dst, names = sys.argv[1], Path(sys.argv[2]), sys.argv[3].split(',')\n"
        "counts = [int(c) for c in sys.argv[4].split(',')]\n"
        "rows = pq.read_table(src).to_pylist()\n"
        "random.Random(0).shuffle(rows)\n"
        "start = 0\n"
        "for split, n in zip(('train', 'val', 'test'), counts):\n"
        "    for i, r in enumerate(rows[start:start + n]):\n"
        "        d = dst / split / names[r['label']]\n"
        "        d.mkdir(parents=True, exist_ok=True)\n"
        "        (d / f'{start + i}.jpg').write_bytes(r['image']['bytes'])\n"
        "    start += n\n"
    )
    counts = ",".join(str(entry[s]) for s in ("train", "val", "test"))
    # pyarrow needs NumPy 2, which the project pins below: run it in a throwaway environment.
    subprocess.run(
        ["uv", "run", "--isolated", "--no-project", "-q", "--with", "pyarrow", "--with", "numpy>=2",
         "python", "-c", script, str(parquet), str(out), ",".join(entry["classes"]), counts],
        check=True,
    )


def _pets(entry: dict, raw: Path, out: Path) -> None:
    archive = download(entry["url"], raw / "images.tar.gz")
    images = raw / "images"
    if not images.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(raw, filter="data")
    by_breed: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(images.glob("*.jpg")):
        m = re.match(r"(.+)_\d+$", f.stem)
        if m:
            by_breed[m.group(1)].append(f)
    rng = random.Random(SEED)
    for breed, files in by_breed.items():
        for split, n in entry["per_class"].items():
            files = _sample_into(files, out / split / breed, n, rng)


def _voc(entry: dict, raw: Path, out: Path) -> None:
    archive = download(entry["url"], raw / "voc.tar")
    root = raw / "VOCdevkit" / "VOC2007"
    if not root.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(raw, filter="data")
    subprocess.run(
        [sys.executable, str(HERE / "voc_to_yolo.py"), str(root), str(out),
         "--classes", ",".join(entry["classes"]),
         "--train", str(entry["train"]), "--val", str(entry["val"]), "--test", str(entry["test"])],
        check=True,
    )


FETCHERS = {"roboflow_zips": _roboflow_zips, "hf_parquet": _hf_parquet, "pets": _pets, "voc": _voc}


def prepare(entry: dict) -> Path:
    """The small, ready-to-train dataset for ENTRY, building it on first use."""
    out = cache_root() / "prepared" / entry["name"]
    if (out / ".ready").exists():
        return out
    shutil.rmtree(out, ignore_errors=True)
    FETCHERS[entry["kind"]](entry, cache_root() / "raw" / entry["name"], out)
    (out / ".ready").touch()
    return out
