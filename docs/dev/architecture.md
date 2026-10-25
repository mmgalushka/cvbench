# Architecture

This page is for people changing CVBench or adding a new model family (a segmentation net, a VAE, a GAN, ...) next to classification and detection. It describes the layers, the rules every layer follows, and step-by-step guides for the three extension points: tasks, backbones and exporters.

## Layers

```text
  cli/            web/                 front ends: parse input, show output
     \             /
      services/                        orchestration: train, evaluate, predict, export, sweep
          |
   tasks.py  classification/  detection/  ...     one Task per model family
          |
        core/                          Task interface, config, events, registries, trainer
```

Dependencies point down only. `datasets/` and `augmentations/` are leaf helpers used by the layers above them.

| Layer | Owns | Must not |
|---|---|---|
| `cli/`, `web/` | Argument parsing, HTTP, rendering, translating errors for the user | Hold training logic, compare against a task name, import TensorFlow at module scope |
| `services/` | Orchestration: build config, ask the `Task` what to do, call the trainer, write results | Print (they emit events), branch on a task name |
| `tasks.py` and task packages | Everything that differs between model families | Be imported by `core/` |
| `core/` | The `Task` contract, config, events, errors, registries (backbones, layouts), trainer and fit loop, checkpoints, the input pipeline | Name a task package |

## Rules

### Errors

User-facing problems raise a `CVBenchError` subclass from `cvbench.core.errors` (`NotFoundError`, `ConfigError`, `DatasetError`, `ConflictError`, `ExportError`, ...). Built-in exceptions are for programmer errors. Never turn a `CVBenchError` into a click or HTTP error yourself: the CLI does that in `cli/_help.py` and the WebUI in `web/app.py`. The table of classes is in [CONTRIBUTING.md](https://github.com/mmgalushka/cvbench/blob/main/CONTRIBUTING.md#errors).

### Events and listeners

Code below the front ends never prints. Progress is reported as small frozen dataclasses from `cvbench.core.events` (`RunStarted`, `EpochEnded`, `RunFinished`, `Notice`, ...) handed to an optional `listener` callable, and `emit(listener, event)` delivers them. The CLI renders them in `cli/_listeners.py`; the WebUI queues them. A new task reports epochs through `FitContext.end_epoch`, which emits `EpochEnded` for it.

### The TF-free import boundary

Asking a task about itself must not load TensorFlow: `get_task(name)`, `meta()`, `metrics_schema()`, `artifacts()`, `export_targets()`, `label_spec`, layout sniffing and report rendering all work without it, which keeps `--help`, `runs list` and the WebUI listings fast. So:

- `core/task.py`, `core/fit.py`, `core/layout.py`, `core/config.py`, `core/backbone.py` and `datasets/` import no TensorFlow or Keras at module scope.
- In a task package import `keras` / `tensorflow` inside the methods that build or run models (`build_model`, `build_datasets`, `fit`, `evaluate`).
- `tests/test_import_boundaries.py` and `tests/test_tasks_contract.py` fail when this is broken.

Also: only a task package and `tasks.py` may compare against a task name; everything else asks the `Task`.

## The extension hooks

A `Task` subclass answers every question that differs between model families. Each hook has a default that reproduces classification behaviour, so a task overrides only what differs.

| Hook | Default | Override when |
|---|---|---|
| `fit(ctx)` | `KerasFitLoop`: `model.fit` on the primary model with CSV log, checkpoint, LR scheduler, epoch events and graceful Ctrl-C | The model is not trained by a single `model.fit` (GAN alternation, VAE with extra terms) |
| `artifacts()` | `("model",)`: one `best.keras` | A run produces several models (encoder + decoder, generator + discriminator) |
| `build_models(cfg, primary)` | `{primary_artifact: primary}` | You declare more than one artifact |
| `load_models(checkpoint)` | loads every artifact via `load_model` | You need custom loading beyond `custom_objects()` |
| `prepare_resume(models, cfg, listener)` | re-applies `fine_tune_from_layer` and syncs the learning rate on the primary model | Your models have no backbone or your loop owns its optimizer |
| `label_spec` | `LabelSpec("class")` | Your samples are `boxes`, `masks` or unlabeled (`none`) |
| `layout` | `None` (override `matches_layout`, `layout_errors`, `resolve_layout` instead) | Your dataset shape deserves a reusable `LayoutAdapter` |
| `layout_priority` | `0` | Your layout can be mistaken for another one (higher is sniffed first; classification is the catch-all at 0, detection is 10) |
| `export_targets()` | `None`: every registered exporter | Some exporters make no sense for your models |
| `calibration_groups(cfg)` | `None`: training images grouped by class folder | Your calibration inputs are not class-folder images |
| `config_section` | `None` | Your task has its own settings (read as `cfg.<name>`, settable with `--set name.key=value`) |

How the pieces fit together:

- **Artifacts.** The first artifact is primary and lives at the checkpoint path itself (`best.keras`); the others sit next to it as `best.<name>.keras`, `epoch_003.<name>.keras`, `interrupt_epoch003.<name>.keras`. `Task.artifact_path` is the single place that maps names to files. `ArtifactCheckpointer` (best-only and every-N strategies, pruning) and the interrupt checkpoint save and prune all artifacts together; resume reloads all of them and reports a missing one as a `NotFoundError`.
- **Custom fit loop.** `core/trainer.py` builds a `FitContext` (config, models by artifact name, datasets, steps per epoch, initial epoch, a `stop_requested()` callable, the checkpointer) and calls `task.fit(ctx)`, which returns a `FitResult(history, epochs)`. A custom loop calls `ctx.end_epoch(epoch, logs)` once per epoch (CSV row, `EpochEnded`, checkpoint decision) and checks `ctx.stop_requested()` to honour Ctrl-C.
- **Layout adapters.** A `LayoutAdapter` (`core/layout.py`) has `sniff(root)`, `errors(root)` and `resolve(cfg, task)`. Register it with `register_layout` and set `Task.layout = "<adapter name>"`; `matches_layout`, `layout_errors` and `resolve_layout` then delegate to it. `tasks.detect_task_name` tries tasks by `layout_priority`. The built-in adapters are `class_folders` and `yolo` in `datasets/adapters.py`, plus `voc` and `coco`, which are recognised but not trainable: their `errors` point at `data convert` (`datasets/convert.py`) and `resolve` raises.
- **Label kinds and joint augmentation.** `LabelSpec.kind` is `class`, `boxes`, `masks` or `none`. For `masks` pass `joint_targets=True` to `build_image_dataset` with one mask file path per sample: geometric `keras_*` transforms (flip, rotation, zoom, translation, crop) are applied to the image and the mask together with nearest-neighbour interpolation, photometric ones and `aug_*` touch the image only. A `boxes` task still drops the geometric transforms in `filter_transforms`; box-aware geometric augmentation is not implemented.
- **Export.** `export_targets()` names exporters from `services.export.EXPORTERS`; `run_export` refuses a format the task does not list, and a name that is not registered is a `ConfigError`. `calibration_groups` extends `services/export/calibration.py`.

### Not yet exercised by a real model

These hooks are covered by a test-only autoencoder task (`tests/autoencoder_task.py`) and unit tests, but no shipped model uses them. Expect rough edges when the first real family does:

- Multi-artifact runs and a custom `fit`. Evaluation, prediction, Grad-CAM, sweeps and export all load only the **primary** artifact (`best.keras`). A task that needs the other models in `evaluate` or `predict` must call `load_models` itself.
- `LabelSpec` `masks` and `none`, `joint_targets` augmentation, and `LayoutAdapter` registration by anything other than the two built-in layouts. Nothing yet writes mask files, previews them in the WebUI or computes mask metrics.
- `export_targets()` and `calibration_groups()` overrides. The exporters themselves still read `best.keras` and assume a single image input (`ConvertExporter`, `build_export_info`).
- The CLI `--format` choice of `runs export` (`cli/runs.py`) and the WebUI's `_VALID_FORMATS` (`web/api/export.py`) are literal lists, not read from `EXPORTERS`; a new exporter name must be added there too.
- Classification-shaped fields remain in `RunConfig` / `DataConfig` (`classes`, `num_classes`) and in the WebUI's confusion-matrix and per-class views; an unlabeled task leaves them empty.

## Guide: add a task

The steps below are exactly what `tests/autoencoder_task.py` does; read it next to this guide.

1. **Create the package** `my_pkg/mytask.py` (or `src/cvbench/mytask/__init__.py` for a built-in) with a `Task` subclass. Set the class attributes: `name`, `short_name` (run-name prefix), `default_backbone` (a registered backbone name, even if unused), `primary_metric = MetricSpec(...)`, `sweep_metric`, `selection_metric_names`.
2. **Declare the data.** Set `label_spec`. Write a `LayoutAdapter` (or override `matches_layout`, `layout_errors` and `resolve_layout`), register it with `register_layout`, and set `layout` and `layout_priority`. `resolve` is the only code allowed to write `cfg.data.{train,val,test}_dir`, `cfg.data.classes` and `cfg.model.num_classes`; for an unlabeled task leave the classes empty. Implement `count_images`, `build_datasets` and `build_eval_dataset`, using `core.input_pipeline.build_image_dataset`.
3. **Build the model.** `build_model(cfg)` returns the primary model. For several artifacts also override `artifacts()` and `build_models(cfg, primary)`. Import Keras inside the method.
4. **Train.** Keep the default loop if `model.fit` fits your model; otherwise override `fit(ctx)` and report each epoch with `ctx.end_epoch`. Override `prepare_resume` when the default (backbone freeze flags, optimizer learning rate) does not apply. Implement `headline_metrics(final_metrics)`.
5. **Evaluate and present.** Implement `evaluate` (write `eval_report.json`, return a dict with `task` and `overall.metric/value`), `predict`, `predict_one`, and a `TaskView` with `render_predictions`, `eval_summary`, `per_class_table` and `render_report`.
6. **Optional hooks.** `config_section = ("name", SettingsDataclass)` for task settings, `export_targets()`, `calibration_groups()`.
7. **Register it.** For a plugin add an entry point; for a built-in add it to `_TASKS` in `cvbench/tasks.py`:

   ```toml
   [project.entry-points."cvbench.tasks"]
   mytask = "my_pkg.mytask:MyTask"
   ```

   A built-in name always wins over a plugin with the same name, with a warning.
8. **Prove it with the conformance suite.** Add a tiny dataset fixture for the task to `_TASK_DATASET_FIXTURES` in `tests/conftest.py` and the task to `tests/test_tasks_contract.py`: it checks every abstract method, metadata, JSON-able `meta()`, the TF-free import boundary and one pass through layout, datasets, model and evaluation. Then write a train, resume and interrupt test through `run_training` like `tests/test_task_hooks.py` does. Nothing in `services/`, `cli/` or `web/` should need an edit; if it does, that is a gap to fix in the hook, not to patch around.

## Guide: add a backbone

1. Pick the form. A keras-hub preset needs `hub_class` and `preset`; anything else needs a `factory` string `"pkg.module:fn"` whose `fn(cfg, load_weights)` returns a `keras.Model` with a `.pyramid_outputs` dict. Set `quant_friendly=True` for pure-ReLU graphs that quantize to INT8.
2. Register it, either in code before the `train` CLI module is imported (its `--backbone` choices are read once):

   ```python
   from cvbench.core.backbone import BackboneSpec, register_backbone

   register_backbone(BackboneSpec("mobilenet_v3", factory="my_pkg.backbones:build_mobilenet", quant_friendly=True))
   ```

   or, for a package, through an entry point whose value is a `BackboneSpec` or a callable returning a list of them:

   ```toml
   [project.entry-points."cvbench.backbones"]
   my_backbone = "my_pkg.backbones:MY_BACKBONE"
   ```

   Entry points are registered on the first `backbone_names()` call. A duplicate name is a `ConfigError`.
3. Keep the module importing the spec TF-free: the factory string is resolved only when a model is built.
4. Check with `tests/test_backbone_registry.py`-style tests: the name appears in `backbone_names()` and `--backbone <name>` builds.

## Guide: add an exporter

1. Subclass `Exporter` (or `ConvertExporter` for "load the model, convert it to one file") in `services/export/`. Set `name`; implement `export(run_dir, cfg, export_dir, opts)` returning extra `export_info.json` fields, and `convert` + `model_filename` for a `ConvertExporter`. Override `validate` to reject bad options with a `ConfigError` before any work starts, and `print_instructions` for follow-up steps.
2. Register an instance in `EXPORTERS` in `services/export/__init__.py`. Exporters have no entry point group; they are added to that dict.
3. Add the name to the literal lists that still exist: the `--format` choice in `cli/runs.py` and `_VALID_FORMATS` in `web/api/export.py`.
4. Tasks reach it by default (`export_targets()` returns `None`). A task that cannot use it lists the names it does support, and `run_export` then refuses the new one for that task.
5. Import TensorFlow and converters inside the functions, and raise `ExportError` when an optional runtime is missing.
