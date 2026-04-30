## v0.25.1 (2026-04-30)

### Fix

- add missing FileResponse import for run image endpoint
- generate random seed for validation split when --seed is not provided

## v0.25.0 (2026-04-29)

### Feat

- add --seed flag for reproducible training
- add rename experiment (CLI, API, web UI)
- delete run/export CLI command; TensorRT plan package export; web UI delete action and dynamic format dropdown

### Fix

- include filename in export download URL for curl/wget compatibility

### Refactor

- remove --normalization external option

## v0.24.1 (2026-04-29)

### Fix

- remove calib images-per-class selector from web UI

## v0.24.0 (2026-04-29)

### Feat

- stratified proportional calibration set (fixed 1024 images, shuffled)

## v0.23.1 (2026-04-29)

### Fix

- apply external normalization after augmentation, not before
- pin TensorFlow base image to 2.16.2-gpu for reproducibility

## v0.23.0 (2026-04-28)

### Feat

- add --normalization flag (internal/external) for pipeline compatibility

## v0.22.3 (2026-04-28)

### Refactor

- standardize on RGB, replace opencv with Pillow, add --calib-images-per-class

## v0.22.2 (2026-04-28)

### Refactor

- remove LCN — not exportable to Hailo HEF

### Perf

- cache pip deps layer to speed up code-only Docker builds

## v0.22.1 (2026-04-27)

### Fix

- strip Rescaling and fix avgpool quantization for Hailo export

## v0.22.0 (2026-04-27)

### Feat

- switch image pipeline to BGR (OpenCV) for packing compatibility

## v0.21.0 (2026-04-27)

### Feat

- add curl/wget copy buttons to exports table

### Fix

- mixup clamps lambda so signal always dominates pixel blend over background
- CLI copy button works over HTTP, long commands wrap in UI
- hailo calibration uses train set, stratified per-class sampling, 1024-image default

## v0.20.0 (2026-04-26)

### Feat

- add Hailo HEF export format with calibration dataset preparation

## v0.19.0 (2026-04-26)

### Feat

- add TensorRT Plan (Jetson) option to Export tab with step-by-step instructions and CLI copy bars
- show CLI command used for training in run detail Training tab
- add Export tab to run detail with list, generate, and tar.gz download

### Fix

- display class_weight dict properly in web UI instead of [object Object]
- show test accuracy in experiments table for evaluated runs
- handle one_of augmentation transforms in run detail API

### Refactor

- split monolithic style.css and app.js into per-page CSS/JS modules

## v0.18.0 (2026-04-25)

### Feat

- move mixup config to augmentation YAML and add one_of transform groups
- add runs predict command with multi-format inference support
- add web predict page with image upload and GradCAM visualization

### Fix

- replace --fp16 with --noTF32 in TensorRT plan build instructions

### Refactor

- promote runs predict to top-level predict command
- rename Predict page to Inference in web UI

## v0.17.0 (2026-04-23)

### Feat

- add mixup augmentation with background-class blending

## v0.16.0 (2026-04-23)

### Feat

- add --loss flag for focal loss and label smoothing support

## v0.15.0 (2026-04-22)

### Feat

- add runs export command for TFLite, ONNX, and Jetson TensorRT plan

## v0.14.1 (2026-04-22)

### Fix

- use val_loss as primary metric for best checkpoint selection and experiment ranking

## v0.14.0 (2026-04-22)

### Feat

- auto-install tensorflow-metal and show Metal GPU message on Apple Silicon
- add Grad-CAM explanation to WebUI gallery

### Fix

- use stored epochs_run as initial_epoch when resuming training

## v0.13.0 (2026-04-21)

### Feat

- add --val-split flag to split train set when no val/ directory exists

## v0.12.0 (2026-04-20)

### Feat

- add --use-lcn flag for brightness-invariant training via Local Contrast Normalization

## v0.11.0 (2026-04-19)

### Feat

- add data group with generate and explore subcommands for class and brightness analysis

## v0.10.2 (2026-04-18)

### Fix

- expose --fine-tune-from-layer CLI option for backbone fine-tuning control

## v0.10.1 (2026-04-18)

### Fix

- include web/static files in package data so non-editable installs serve the WebUI

## v0.10.0 (2026-04-18)

### Feat

- implement WebUI experiments explorer with interactive confusion matrix
- lay WebUI foundation — services layer, REST API skeleton, serve command
- add ReduceLROnPlateau support via --lr-patience/--lr-factor/--lr-min

## v0.9.0 (2026-04-16)

### Feat

- polish CLI output — full-width rules, color hierarchy, table formatting

### Fix

- opt into Node.js 24 for GitHub Actions and clean up GPU status messages

## v0.8.0 (2026-04-11)

### Feat

- add samples and raw confusion matrix to eval_report.json, drop confusion_matrix.png
- rename CLI args to EXPERIMENT for clarity in evaluate and runs compare
- accept bare run name in evaluate and runs compare, resolving under experiments/
- add terminal confusion matrix with auto staircase layout for wide matrices
- render confusion matrix in terminal after evaluate
- add progress bar, GPU/CPU indicator, and suppress TF logs in evaluate and train
- add progress bar, GPU indicator, and suppress TF logs in evaluate
- make release manual-only, remove docker PR build, add GitHub Release creation

### Fix

- pin all dependencies to exact versions, resolve keras/keras-hub incompatibility

### Refactor

- adopt src/ layout and cvbench package namespace
- centralise EXPERIMENTS_DIR and resolve_run_dir in core/runs, sync README and helper.sh

## v0.7.0 (2026-04-06)

### Feat

- replace configs mount with workspace directory

## v0.6.1 (2026-04-05)

### Fix

- apply keras augmentation layers in native tf.data.map to avoid numpy_function shape errors

## v0.6.0 (2026-04-05)

### Feat

- add GPU status line to training header and suppress TF CUDA warnings

### Fix

- use tensorflow[and-cuda] to include CUDA libraries on Linux GPU installs

## v0.5.3 (2026-04-05)

### Fix

- add sentencepiece as explicit dependency to resolve keras_hub import error

## v0.5.2 (2026-04-05)

### Fix

- add tokenizers as explicit dependency to resolve keras_hub import error

## v0.5.1 (2026-04-05)

### Fix

- use deploy.resources for GPU in docker-compose example (#3)

## v0.5.0 (2026-04-05)

### Feat

- class imbalance detection, report, and class_weight support
- auto val-split fallback, clean dataset output, suppress TF noise
- add range sampling for aug_* params; fix lines.py width/brightness naming
- add --augmentation flag, augmentations CLI, remove placement config
- add augmentation visualization notebook

### Fix

- handle colour/grayscale images in aug pipeline; exclude configs from git
- remove aug_brightness/contrast/gaussian_noise from visualizer notebook (covered by Keras)

### Refactor

- evaluate always uses test_ds; drop --split option
- remove aug_exposure augmentation
- type-based range sampling, fix deterministic seed, add both-side fade
- replace preset/registry augmentation with unified declarative pipeline
- extract notebook augmentations into per-category package modules

## v0.4.1 (2026-04-02)

### Fix

- suppress TF banner and fix terminal colour readability

## v0.4.0 (2026-04-02)

### Feat

- improve container UX — non-root user, clean home, tmux aliases, writable data

## v0.3.0 (2026-04-02)

### Feat

- make generate output directory a positional argument

## v0.2.2 (2026-04-02)

### Fix

- correct generate CLI syntax in README (--output flag, not positional arg)

## v0.2.1 (2026-04-01)

### Fix

- allow README.md in Docker build context

## v0.2.0 (2026-04-01)

### Feat

- add Docker Hub publishing pipeline
