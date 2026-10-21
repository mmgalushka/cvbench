"""Every web route through a real ``TestClient`` — status code and response shape.

The route functions are exercised through the ASGI app, so routing, request parsing,
the domain-error handler and the static mount are all part of what is under test.
"""
import base64
import io
import json
import tarfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="requires the 'web' extra")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from cvbench.core.config import load_config, save_config  # noqa: E402
from cvbench.datasets.layout import list_images  # noqa: E402
from cvbench.web.api.datasets import _encode_dir  # noqa: E402
from cvbench.web.app import create_app  # noqa: E402


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture
def client_500():
    """Client that returns unexpected server errors as 500 responses instead of re-raising them."""
    return TestClient(create_app(), raise_server_exceptions=False)


def _png_bytes(size: int = 32) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (120, 60, 200)).save(buf, format="PNG")
    return buf.getvalue()


def _first_test_image(run_dir) -> str:
    test_dir = load_config(str(run_dir)).data.test_dir
    return str(list_images(Path(test_dir))[0].relative_to(test_dir))


# ---------------------------------------------------------------------------
# App: docs, static mount, domain errors
# ---------------------------------------------------------------------------

def test_app_serves_the_frontend_at_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_app_serves_openapi_docs(client):
    assert client.get("/api/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/api/runs" in schema["paths"]


def test_unknown_api_route_is_404(client):
    assert client.get("/api/definitely-not-a-route").status_code == 404


def test_domain_errors_map_to_http_status(client, exps, monkeypatch):
    from cvbench.core.errors import CVBenchError, WrongKindError

    def boom(_name, *_a, **_k):
        raise WrongKindError("'x' is a sweep, not a run")

    monkeypatch.setattr("cvbench.web.api.runs.resolve_run_dir", boom)
    resp = client.get("/api/runs/x")
    assert resp.status_code == 400
    assert resp.json() == {"detail": "'x' is a sweep, not a run"}
    assert issubclass(WrongKindError, CVBenchError)


def test_conflict_error_maps_to_409(client, monkeypatch):
    from cvbench.core.errors import ConflictError

    def boom(_name, *_a, **_k):
        raise ConflictError("A run named 'x' already exists.")

    monkeypatch.setattr("cvbench.web.api.runs.resolve_run_dir", boom)
    resp = client.get("/api/runs/x")
    assert resp.status_code == 409
    assert resp.json() == {"detail": "A run named 'x' already exists."}


def test_unexpected_exception_maps_to_500_json_and_is_logged(client_500, monkeypatch, caplog):

    def boom(_name, *_a, **_k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("cvbench.web.api.runs.resolve_run_dir", boom)
    with caplog.at_level("ERROR", logger="cvbench.web"):
        resp = client_500.get("/api/runs/x")
    assert resp.status_code == 500
    assert resp.json() == {"detail": "kaboom"}
    assert any(r.name == "cvbench.web" and r.exc_info for r in caplog.records)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def test_get_present_returns_labels(client):
    resp = client.get("/api/present")
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict) and resp.json()


def test_list_runs_empty_and_populated(client, exps, write_exp):
    assert client.get("/api/runs").json() == []
    write_exp(exps, "run_a", val_accuracy=0.9)
    (entry,) = client.get("/api/runs").json()
    assert entry["name"] == "run_a"
    assert "display" in entry


def test_list_sweeps_and_get_sweep(client, exps, write_exp, make_sweep):
    sweep = make_sweep(exps)
    write_exp(sweep, "sw_001", val_loss=0.5)
    write_exp(sweep, "sw_002", val_loss=0.2)

    (listed,) = client.get("/api/sweeps").json()
    assert listed["name"] == "sw"

    resp = client.get("/api/sweeps/sw")
    assert resp.status_code == 200
    body = resp.json()
    assert body["metric"] == "val_loss"
    assert [t["is_best"] for t in body["trials"]] == [False, True]
    assert set(body["trials"][0]) >= {"index", "dir", "params", "status", "value", "display"}


def test_get_sweep_errors(client, exps, write_exp):
    assert client.get("/api/sweeps/nope").status_code == 404
    write_exp(exps, "manual")
    assert client.get("/api/sweeps/manual").status_code == 404


def test_get_sweep_with_a_broken_manifest_is_400(client, exps, make_sweep):
    sweep = make_sweep(exps)
    (sweep / "sweep.yaml").write_text("not: [a, valid manifest")
    resp = client.get("/api/sweeps/sw")
    assert resp.status_code == 400
    assert "Invalid" in resp.json()["detail"]


def test_get_run_detail_shape(client, cls_run):
    (cls_run / "training_log.csv").write_text("epoch,loss,val_loss\n0,1.0,0.9\n1,0.8,0.7\n")
    (cls_run / "eval_report.json").write_text(json.dumps({
        "task": "classification", "split": "test", "n_images": 2,
        "overall": {"metric": "accuracy", "value": 0.75, "label": "Accuracy"},
        "per_class": {}, "samples": [],
    }))

    resp = client.get("/api/runs/cls_run")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "cls_run"
    assert body["task"] == "classification"
    assert body["status"] == "done"
    assert body["sweep"] is None
    assert body["primary_value"] == 0.75
    assert [row["epoch"] for row in body["training_log"]] == [0.0, 1.0]
    assert body["eval_report"]["overall"]["value"] == 0.75
    assert body["task_meta"]["name"] == "classification"
    assert set(body["config"]) == {"task", "data", "model", "augmentation", "training", "run"}
    assert body["config"]["training"]["optimizer"]["type"] == "adam"  # the whole config, not a hand-picked subset
    assert set(body["display"]) and body["eval_display"] is not None


def test_get_run_without_artifacts(client, cls_run):
    body = client.get("/api/runs/cls_run").json()
    assert body["training_log"] == [] and body["eval_report"] is None
    assert body["eval_display"] is None and body["advisories"] == []


def test_get_run_unknown_is_404_with_candidates(client, exps, write_exp):
    write_exp(exps, "resnet_run")
    resp = client.get("/api/runs/resnet_rum")
    assert resp.status_code == 404
    assert resp.json()["candidates"] == ["resnet_run"]


def test_get_run_image_serves_a_test_image(client, cls_run):
    rel = _first_test_image(cls_run)
    resp = client.get(f"/api/runs/cls_run/images/{rel}")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")


def test_get_run_image_rejects_traversal_and_missing(client, cls_run):
    assert client.get("/api/runs/cls_run/images/%2e%2e/%2e%2e/config.yaml").status_code == 403
    assert client.get("/api/runs/cls_run/images/nope/missing.png").status_code == 404


def test_rename_run(client, exps, write_exp):
    write_exp(exps, "old_name")
    resp = client.patch("/api/runs/old_name", json={"new_name": "new_name"})
    assert resp.status_code == 200
    assert resp.json() == {"name": "new_name"}
    assert not (exps / "old_name").exists()
    assert load_config(str(exps / "new_name")).run.name == "new_name"


def test_rename_run_rejections(client, exps, write_exp, make_sweep):
    write_exp(exps, "taken")
    write_exp(exps, "mine")
    live = write_exp(exps, "live")
    cfg = load_config(str(live))
    cfg.run.status, cfg.run.pid = "running", __import__("os").getpid()
    save_config(cfg, str(live))
    make_sweep(exps)

    assert client.patch("/api/runs/mine", json={"new_name": "taken"}).status_code == 409
    assert client.patch("/api/runs/mine", json={"new_name": "bad/name"}).status_code == 400
    assert client.patch("/api/runs/live", json={"new_name": "other"}).status_code == 409
    assert client.patch("/api/runs/sw", json={"new_name": "other"}).status_code == 400
    assert client.patch("/api/runs/ghost", json={"new_name": "other"}).status_code == 404
    assert client.patch("/api/runs/mine", json={}).status_code == 422  # body validation


def test_delete_run(client, exps, write_exp):
    run = write_exp(exps, "doomed")
    resp = client.delete("/api/runs/doomed")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": "doomed"}
    assert not run.exists()
    assert client.delete("/api/runs/doomed").status_code == 404


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@pytest.fixture
def data_project(exps, tiny_cls_root, tiny_yolo_root):
    """cwd with both tiny datasets under ``data/`` (the folder the datasets route scans)."""
    import shutil

    data = exps.parent / "data"
    data.mkdir()
    shutil.move(str(tiny_cls_root), data / "cls")
    shutil.move(str(tiny_yolo_root), data / "yolo")
    return data


def test_list_datasets(client, data_project):
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    by_format = {d["format"]: d for d in resp.json()}
    assert set(by_format) == {"classification", "yolo"}
    cls = by_format["classification"]
    assert set(cls["splits"]) == {"train", "val", "test"}
    assert cls["splits"]["train"]["id"] and "display" in cls


def test_list_datasets_puts_the_dataset_of_a_run_first(client, cls_run, tiny_cls_root, tiny_yolo_root):
    import shutil

    (cls_run.parent.parent / "data").mkdir()
    shutil.move(str(tiny_yolo_root), cls_run.parent.parent / "data" / "yolo")

    listed = client.get("/api/datasets").json()
    assert [d["format"] for d in listed] == ["classification", "yolo"]
    assert listed[0]["id"] == _encode_dir(tiny_cls_root)


def test_list_datasets_without_a_data_folder(client, exps):
    assert client.get("/api/datasets").json() == []


def _split_id(client, fmt: str, split: str = "train") -> str:
    entry = next(d for d in client.get("/api/datasets").json() if d["format"] == fmt)
    return entry["splits"][split]["id"]


def test_list_images_classification_paging_and_filter(client, data_project):
    split = _split_id(client, "classification")
    body = client.get(f"/api/datasets/{split}/images", params={"page_size": 5}).json()
    assert body["format"] == "classification"
    assert body["page"] == 1 and len(body["items"]) == 5 and body["pages"] >= 2
    assert set(body["items"][0]) == {"path", "class", "filename"}

    cls = body["classes"][0]
    filtered = client.get(f"/api/datasets/{split}/images", params={"class": cls}).json()
    assert filtered["total"] == 4 and {i["class"] for i in filtered["items"]} == {cls}


def test_list_images_errors(client, data_project):
    split = _split_id(client, "classification")
    assert client.get(f"/api/datasets/{split}/images", params={"class": "ghost"}).status_code == 404
    assert client.get(f"/api/datasets/{split}/images", params={"class": "../.."}).status_code in (403, 404)
    assert client.get(f"/api/datasets/{split}/images", params={"page": 0}).status_code == 422
    assert client.get(f"/api/datasets/{split}/images", params={"page_size": 9999}).status_code == 422
    not_utf8 = base64.urlsafe_b64encode(b"\xff\xfe").decode().rstrip("=")
    assert client.get(f"/api/datasets/{not_utf8}/images").status_code == 400
    missing = _encode_dir(data_project / "nowhere")
    assert client.get(f"/api/datasets/{missing}/images").status_code == 404


def test_list_images_yolo_has_boxes_and_class_filter(client, data_project):
    split = _split_id(client, "yolo")
    body = client.get(f"/api/datasets/{split}/images").json()
    assert body["format"] == "yolo"
    assert set(body["items"][0]) == {"path", "class", "filename", "boxes"}
    assert any(item["boxes"] for item in body["items"])

    cls = body["classes"][0]
    filtered = client.get(f"/api/datasets/{split}/images", params={"class": cls}).json()
    assert all(any(b["class"] == cls for b in i["boxes"]) for i in filtered["items"])
    assert client.get(f"/api/datasets/{split}/images", params={"class": "ghost"}).status_code == 404


def test_serve_dataset_file(client, data_project):
    split = _split_id(client, "classification")
    item = client.get(f"/api/datasets/{split}/images").json()["items"][0]
    resp = client.get(f"/api/datasets/{split}/file/{item['path']}")
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("image/")
    assert client.get(f"/api/datasets/{split}/file/missing.png").status_code == 404
    assert client.get(f"/api/datasets/{split}/file/%2e%2e/%2e%2e/pyproject.toml").status_code == 403


def test_upload_images(client, data_project):
    split = _split_id(client, "classification")
    before = client.get(f"/api/datasets/{split}/images").json()["total"]
    files = [
        ("files", ("a.png", _png_bytes(), "image/png")),
        ("files", ("notes.txt", b"not an image", "text/plain")),
    ]
    resp = client.post(f"/api/datasets/{split}/images", params={"class": "fresh"}, files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["uploaded"] == 1 and body["files"][0].startswith("fresh/")
    assert client.get(f"/api/datasets/{split}/images").json()["total"] == before + 1


def test_upload_images_rejections(client, data_project):
    split = _split_id(client, "classification")
    one = [("files", ("a.png", _png_bytes(), "image/png"))]
    assert client.post(f"/api/datasets/{split}/images", params={"class": "../.."}, files=one).status_code == 403
    assert client.post(f"/api/datasets/{split}/images").status_code == 422
    missing = _encode_dir(data_project / "nowhere")
    assert client.post(f"/api/datasets/{missing}/images", files=one).status_code == 404


def test_delete_image_removes_the_label_file_of_a_yolo_split(client, data_project):
    split = _split_id(client, "yolo")
    item = client.get(f"/api/datasets/{split}/images").json()["items"][0]
    label = data_project / "yolo" / "labels" / "train" / (item["filename"].rsplit(".", 1)[0] + ".txt")
    assert label.is_file()

    resp = client.delete(f"/api/datasets/{split}/images/{item['path']}")
    assert resp.status_code == 200 and resp.json() == {"deleted": item["path"]}
    assert not label.exists()
    assert client.delete(f"/api/datasets/{split}/images/{item['path']}").status_code == 404


def test_delete_image_rejections(client, data_project):
    split = _split_id(client, "classification")
    cls = client.get(f"/api/datasets/{split}/images").json()["classes"][0]
    assert client.delete(f"/api/datasets/{split}/images/{cls}").status_code == 400  # a directory
    assert client.delete(f"/api/datasets/{split}/images/%2e%2e/%2e%2e/pyproject.toml").status_code == 403


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def _stub_export(exp_dir, subfolder="tflite", model="model.tflite", extra=()):
    folder = exp_dir / "export" / subfolder
    folder.mkdir(parents=True)
    (folder / model).write_bytes(b"\0" * 2048)
    for name in extra:
        (folder / name).write_bytes(b"x")
    (folder / "export_info.json").write_text(json.dumps({"format": subfolder, "quantize": "none"}))
    return folder


def test_list_exports(client, cls_run):
    assert client.get("/api/runs/cls_run/exports").json() == []
    _stub_export(cls_run)
    (entry,) = client.get("/api/runs/cls_run/exports").json()
    assert entry["subfolder"] == "tflite" and entry["model_file"] == "model.tflite"
    assert entry["size_mb"] == pytest.approx(0.0, abs=0.01) and entry["format"] == "tflite"


def test_list_exports_skips_folders_without_info(client, cls_run):
    (cls_run / "export" / "junk").mkdir(parents=True)
    assert client.get("/api/runs/cls_run/exports").json() == []


@pytest.mark.parametrize("payload, fragment", [
    ({"format": "nope"}, "Invalid format"),
    ({"format": "tflite", "quantize": "int4"}, "Invalid quantize"),
    ({"format": "tflite", "calib_total": 0}, "calib_total"),
    ({"format": "tflite", "calib_strategy": "magic"}, "calib_strategy"),
])
def test_create_export_validation(client, cls_run, payload, fragment):
    resp = client.post("/api/runs/cls_run/exports", json=payload)
    assert resp.status_code == 400 and fragment in resp.json()["detail"]


def test_create_export_requires_a_checkpoint(client, cls_run):
    resp = client.post("/api/runs/cls_run/exports", json={"format": "tflite"})
    assert resp.status_code == 400 and "best.keras" in resp.json()["detail"]


def test_create_export_unknown_run_is_404(client, exps):
    assert client.post("/api/runs/ghost/exports", json={"format": "tflite"}).status_code == 404


def test_create_export_runs_the_service_and_lists_the_result(client, cls_run, monkeypatch):
    (cls_run / "best.keras").write_bytes(b"checkpoint")
    calls = []

    def fake_export(run, fmt, quantize, **kwargs):
        calls.append((run, fmt, quantize, kwargs))
        _stub_export(cls_run, subfolder=fmt)

    monkeypatch.setattr("cvbench.services.export.run_export", fake_export)
    resp = client.post("/api/runs/cls_run/exports", json={"format": "tflite", "quantize": "float16"})
    assert resp.status_code == 200
    assert [e["subfolder"] for e in resp.json()] == ["tflite"]
    assert calls == [("cls_run", "tflite", "float16", {"calib_total": 1024, "calib_strategy": "stratified"})]


def test_create_export_service_failure_is_500(client_500, cls_run, monkeypatch):
    (cls_run / "best.keras").write_bytes(b"checkpoint")

    def failing(*_a, **_k):
        raise RuntimeError("converter exploded")

    monkeypatch.setattr("cvbench.services.export.run_export", failing)
    resp = client_500.post("/api/runs/cls_run/exports", json={"format": "onnx"})
    assert resp.status_code == 500 and resp.json()["detail"] == "converter exploded"


def _tar_members(content: bytes) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
        return sorted(tar.getnames())


def test_download_export_archive(client, cls_run):
    _stub_export(cls_run)
    resp = client.get("/api/runs/cls_run/exports/tflite/download/model.tflite")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/gzip"
    assert "cls_run_tflite.tar.gz" in resp.headers["content-disposition"]
    assert _tar_members(resp.content) == ["export_info.json", "model.tflite"]


def test_download_hailo_package(client, cls_run):
    _stub_export(cls_run, subfolder="hailo", extra=("calib_set.npy", "model.alls"))
    resp = client.get("/api/runs/cls_run/exports/hailo/download/package")
    assert resp.status_code == 200
    assert _tar_members(resp.content) == ["calib_set.npy", "export_info.json", "model.alls", "model.tflite"]


def test_download_export_errors(client, cls_run):
    assert client.get("/api/runs/cls_run/exports/ghost/download/x").status_code == 404
    assert client.get("/api/runs/cls_run/exports/%2e%2e/download/x").status_code in (403, 404)
    (cls_run / "export" / "tflite").mkdir(parents=True)
    assert client.get("/api/runs/cls_run/exports/tflite/download/x").status_code == 404  # no model file
    (cls_run / "export" / "hailo").mkdir(parents=True)
    assert client.get("/api/runs/cls_run/exports/hailo/download/x").status_code == 404  # no package


def test_delete_export(client, cls_run):
    _stub_export(cls_run, "tflite")
    _stub_export(cls_run, "onnx", model="model.onnx")
    resp = client.delete("/api/runs/cls_run/exports/tflite")
    assert resp.status_code == 200
    assert [e["subfolder"] for e in resp.json()] == ["onnx"]
    assert client.delete("/api/runs/cls_run/exports/tflite").status_code == 404
    assert client.delete("/api/runs/cls_run/exports/%2e%2e").status_code in (403, 404)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_list_augmentations_schema(client):
    resp = client.get("/api/augmentations")
    assert resp.status_code == 200
    schema = resp.json()
    assert any(a["name"] == "aug_blur" for a in schema)
    assert all(set(a) == {"name", "label", "params"} for a in schema)


def test_predict_single_unknown_run_is_404(client, exps):
    resp = client.post("/api/predict/single", data={"run": "ghost"}, files={"file": ("a.png", _png_bytes())})
    assert resp.status_code == 404


def test_predict_single_without_checkpoint_is_404(client, cls_run):
    resp = client.post("/api/predict/single", data={"run": "cls_run"}, files={"file": ("a.png", _png_bytes())})
    assert resp.status_code == 404 and "No keras checkpoint" in resp.json()["detail"]


def test_predict_single_requires_the_form_fields(client, cls_run):
    assert client.post("/api/predict/single", data={"run": "cls_run"}).status_code == 422
    assert client.post("/api/predict/single", files={"file": ("a.png", _png_bytes())}).status_code == 422


@pytest.mark.tf
def test_predict_single_returns_classification_result(client, cls_run_with_model):
    resp = client.post(
        "/api/predict/single", data={"run": "cls_run", "top_k": 2},
        files={"file": ("a.png", _png_bytes())},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["task"] == "classification"
    classes = load_config(str(cls_run_with_model)).data.classes
    assert body["class_name"] == classes[body["class_index"]]
    assert len(body["top_k"]) == 2 and len(body["probs"]) == len(classes)


@pytest.mark.tf
def test_predict_single_rejects_an_option_the_task_ignores(client, cls_run_with_model):
    resp = client.post(
        "/api/predict/single", data={"run": "cls_run", "conf": 0.3},
        files={"file": ("a.png", _png_bytes())},
    )
    assert resp.status_code == 400 and "--conf only applies to detection" in resp.json()["detail"]


def test_predict_augmented_rejects_bad_json(client, cls_run):
    resp = client.post(
        "/api/predict/augmented", data={"run": "cls_run", "augmentations": "{not json"},
        files={"file": ("a.png", _png_bytes())},
    )
    assert resp.status_code == 400 and "valid JSON" in resp.json()["detail"]


@pytest.mark.tf
def test_predict_augmented_returns_the_augmented_image(client, cls_run_with_model):
    augs = json.dumps([{"name": "aug_blur", "params": {"radius": 1.0}}])
    resp = client.post(
        "/api/predict/augmented", data={"run": "cls_run", "augmentations": augs},
        files={"file": ("a.png", _png_bytes())},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["task"] == "classification"
    assert Image.open(io.BytesIO(base64.b64decode(body["augmented_image_b64"]))).size == (32, 32)


@pytest.mark.tf
def test_predict_augmented_unknown_augmentation_is_400(client, cls_run_with_model):
    augs = json.dumps([{"name": "aug_does_not_exist"}])
    resp = client.post(
        "/api/predict/augmented", data={"run": "cls_run", "augmentations": augs},
        files={"file": ("a.png", _png_bytes())},
    )
    assert resp.status_code == 400 and "Unknown augmentation" in resp.json()["detail"]


def test_predict_xai_errors(client, exps, cls_run):
    files = {"file": ("a.png", _png_bytes())}
    assert client.post("/api/predict/xai", data={"run": "ghost", "class_index": 0}, files=files).status_code == 404
    assert client.post("/api/predict/xai", data={"run": "cls_run", "class_index": 0}, files=files).status_code == 404
    assert client.post("/api/predict/xai", data={"run": "cls_run"}, files=files).status_code == 422


@pytest.mark.tf
def test_predict_xai_returns_a_heatmap(client, cls_run_with_model):
    resp = client.post(
        "/api/predict/xai", data={"run": "cls_run", "class_index": 0},
        files={"file": ("a.png", _png_bytes())},
    )
    assert resp.status_code == 200
    Image.open(io.BytesIO(base64.b64decode(resp.json()["heatmap_b64"]))).verify()


@pytest.mark.tf
def test_predict_xai_corrupt_image_is_500(client_500, cls_run_with_model):
    resp = client_500.post(
        "/api/predict/xai", data={"run": "cls_run", "class_index": 0},
        files={"file": ("a.png", b"not an image")},
    )
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------

def _gradcam(client, run="cls_run", image_path="x.png", class_index=0):
    return client.post(
        f"/api/runs/{run}/explain/gradcam", json={"image_path": image_path, "class_index": class_index}
    )


def test_gradcam_guards(client, cls_run):
    rel = _first_test_image(cls_run)
    assert _gradcam(client, "ghost").status_code == 404
    assert _gradcam(client, image_path="../../config.yaml").status_code == 403
    assert _gradcam(client, image_path="missing.png").status_code == 404
    assert _gradcam(client, image_path=rel).status_code == 404  # image fine, no checkpoint
    assert client.post("/api/runs/cls_run/explain/gradcam", json={}).status_code == 422


@pytest.mark.tf
def test_gradcam_returns_a_heatmap(client, cls_run_with_model):
    resp = _gradcam(client, image_path=_first_test_image(cls_run_with_model), class_index=1)
    assert resp.status_code == 200
    Image.open(io.BytesIO(base64.b64decode(resp.json()["heatmap_b64"]))).verify()


@pytest.mark.tf
def test_gradcam_bad_class_index_is_500(client_500, cls_run_with_model):
    resp = _gradcam(client_500, image_path=_first_test_image(cls_run_with_model), class_index=999)
    assert resp.status_code == 500
