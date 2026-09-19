# Jetson Deployment

[NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) devices run
models as compiled [TensorRT](https://developer.nvidia.com/tensorrt) `.plan`
engines. A
`.plan` file is compiled for a specific GPU architecture, so it **must be
built on the Jetson itself** — CVBench prepares the ONNX model and prints the
exact steps.

!!! info "Why build on the device?"
    TensorRT optimizes the network for the specific GPU and TensorRT version
    it is built on, so an engine compiled on your desktop GPU will not load
    on a Jetson. Always run `trtexec` on the target device.

```bash
runs export my_run --format plan
```

This exports (or reuses) an [ONNX](https://onnx.ai/) `model.onnx` under
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

**Step 3 — run inference** using the
[TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/latest/)
or [DeepStream](https://developer.nvidia.com/deepstream-sdk).

!!! tip
    `trtexec` is bundled with TensorRT (JetPack installs it, usually under
    `/usr/src/tensorrt/bin/`). `--noTF32` keeps full float32 precision so
    results match the exported model; see the TensorRT docs for `--fp16` and
    `--int8` if you need more speed.

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

## Further reading

- [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) — modules,
  JetPack SDK, and setup guides
- [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/)
  — `trtexec`, the Python API, and precision options
- [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) — video
  analytics pipelines on Jetson
- [Technology References](../references.md) — every external tool CVBench uses

Next: back to [Export Formats (TFLite/ONNX)](export.md) or
[Hailo HEF Export](hailo.md) for other deployment targets.
