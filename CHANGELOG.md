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
