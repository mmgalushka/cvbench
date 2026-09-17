# Jetson Deployment

NVIDIA Jetson devices run models as compiled TensorRT `.plan` engines. A
`.plan` file is compiled for a specific GPU architecture, so it **must be
built on the Jetson itself** — CVBench prepares the ONNX model and prints the
exact steps.

```bash
runs export my_run --format plan
```

This exports (or reuses) `model.onnx` under
`experiments/my_run/export/plan/`, writes `export_info.json`, and prints:

**Step 1 — copy the ONNX model to your Jetson:**

```bash
scp experiments/my_run/export/plan/model.onnx \
    user@jetson:/home/user/model.onnx
```

**Step 2 — on the Jetson, convert to a TensorRT engine plan:**

```bash
trtexec --onnx=model.onnx --saveEngine=model.plan --noTF32
```

**Step 3 — run inference** using the TensorRT Python API or DeepStream.

## Ready-made inference script

`predict --format plan` doesn't run inference (there's no Jetson GPU in the
container) — it prints a ready-to-use inference script for the device:

```bash
predict --format plan
```

Save the printed script as `infer.py` on the Jetson, make it executable, and
run it against a single image or a folder:

```bash
chmod +x infer.py
./infer.py model.plan image.jpg
./infer.py model.plan images/
```

The script loads the TensorRT engine, resizes each input image to the
training input size, runs `execute_v2`, and prints the predicted class index
and confidence for each image — a minimal starting point you can adapt for
your own deployment.

Next: back to [Export Formats (TFLite/ONNX)](export.md) or
[Hailo HEF Export](hailo.md) for other deployment targets.
