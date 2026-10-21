"""Terminal output for the progress events of training, evaluation and sweeps.

The trainer and the services print nothing; they emit ``cvbench.core.events``
events. ``ConsoleListener`` is the CLI's rendering of them, so a terminal run
reads the same as it always did.
"""
from __future__ import annotations

import platform
from typing import Any

from cvbench.core import console
from cvbench.core.config import CVBenchConfig
from cvbench.core.events import (
    DeviceDetected,
    EvalProgress,
    EvalStarted,
    Event,
    InterruptRequested,
    Notice,
    RunFinished,
    RunStarted,
    SeedSet,
    SweepEvalTrialStarted,
)

# Below this many validation images, val metrics are too noisy to trust.
_SMALL_VAL_IMAGES = 30


def fine_tune_label(fine_tune_from_layer: int) -> str:
    if fine_tune_from_layer == 0:
        return "frozen backbone (only the head trains)"
    if fine_tune_from_layer == -1:
        return "all layers unfrozen"
    return f"unfrozen from layer {fine_tune_from_layer}"


def print_header(exp_dir: str, cfg: CVBenchConfig, val_images: int | None = None) -> None:
    print(console.rule())
    print(f" {console.bold('CVBench — train')}")
    print(console.rule())
    print(f" Task      : {cfg.task}")
    print(f" Data      : {console.dim(cfg.data.data_dir)}")
    print(f" Backbone  : {cfg.model.backbone}  (weights: {cfg.model.weights})")
    print(f" Fine-tune : {fine_tune_label(cfg.model.fine_tune_from_layer)}")
    print(f" Input size: {cfg.model.input_size}px   Batch size: {cfg.data.batch_size}")
    print(f" Epochs    : {cfg.training.epochs}")
    seed = cfg.training.seed
    print(f" Seed      : {seed if seed is not None else 'none (not reproducible)'}")
    opt = cfg.training.optimizer
    opt_label = opt.type.upper()
    if opt.weight_decay > 0:
        opt_label += f"  weight_decay={opt.weight_decay}"
    if opt.type == "sgd":
        opt_label += f"  momentum={opt.momentum}"
    print(f" Optimizer : {opt_label}")
    lrs = cfg.training.lr_scheduler
    if lrs.patience > 0:
        print(
            f" LR        : {cfg.training.learning_rate}  →  reduce by {lrs.factor}x after"
            f" {lrs.patience} flat epoch(s) (min {lrs.min_lr})"
        )
    else:
        print(f" LR        : {cfg.training.learning_rate}")
    n_transforms = len(cfg.augmentation.transforms)
    print(f" Aug       : {n_transforms} transform(s)")
    print(f" Output    : {console.dim(exp_dir)}")
    print(console.rule())
    if val_images is not None and val_images < _SMALL_VAL_IMAGES:
        console.warning(
            f"Validation set has only {val_images} image(s); val metrics will be noisy."
        )
    print()  # breathing room before epoch output


def print_device_banner(purpose: str, gpus: tuple[str, ...]) -> None:
    """Print a GPU/CPU banner. PURPOSE is the present participle, e.g. "training"."""
    if gpus:
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            print(console.dim(f"🟢 Apple Silicon GPU (Metal) detected — {purpose} on {len(gpus)} device(s)"))
        else:
            print(console.dim(f"🟢 GPU detected: {len(gpus)} device(s) — {', '.join(gpus)}"))
    else:
        console.warning(f"GPU not available, {purpose} on CPU")


def print_run_summary(event: RunFinished) -> None:
    interrupted = event.status == "interrupted"
    if event.interrupt_checkpoint:
        print(f"\n {console.green('Checkpoint saved')} → {console.dim(event.interrupt_checkpoint)}")
        print(
            f" To resume: train {event.data_dir}"
            f" --from {event.exp_dir}"
            f" --resume {event.interrupt_checkpoint}\n"
        )
    status_label = "interrupted" if interrupted else "complete"
    status_color = console.yellow if interrupted else console.green
    print()  # breathing room after last epoch line
    print(console.rule())
    print(f" {console.bold(f'Training {status_color(status_label)}')}")
    if event.final_metrics:
        max_k = max(len(k) for k in event.final_metrics)
        for k, v in event.final_metrics.items():
            print(f"   {console.bold(f'{k:<{max_k}}')} : {v:.4f}")
    if event.best_events:
        metric_name = event.best_events[0]["metric"]
        print(f"\n {console.bold('Best weights history')}  ({console.dim(metric_name)})")
        for e in event.best_events:
            val_str = f"{e['value']:.4f}" if e["value"] is not None else "—"
            epoch_str = f"{e['epoch']:>4}"
            print(f"   epoch {console.bold(epoch_str)}  →  {val_str}")
    for advisory in event.advisories:
        print()
        console.warning(advisory)
    print(f"\n Run name      : {console.bold(event.run_name)}")
    print(f" Run directory → {console.dim(event.exp_dir)}")
    if not interrupted:
        print(f" Next          : {console.green(f'evaluate {event.run_name}')}")
    print(console.rule())


class ConsoleListener:
    """Render events to the terminal. Subclasses extend ``__call__`` for further event types."""

    def __init__(self) -> None:
        self._bar: Any = None

    def __call__(self, event: Event) -> None:
        if isinstance(event, Notice):
            {"error": console.error, "info": console.info}.get(event.level, console.warning)(event.message)
        elif isinstance(event, DeviceDetected):
            print_device_banner(event.purpose, event.gpus)
        elif isinstance(event, SeedSet):
            print(console.dim(f" Seed: {event.seed} (reproducible run)"))
        elif isinstance(event, RunStarted):
            print_header(event.exp_dir, event.cfg, event.val_images)
            if event.resume_checkpoint:
                print(f" Resuming from: {event.resume_checkpoint}")
        elif isinstance(event, InterruptRequested):
            # \r\033[K is cursor control (clear the in-progress progress-bar line),
            # not color — left as a deliberate exception to the "route color
            # through console" rule. The warning itself still goes through
            # console.warning() so NO_COLOR/non-TTY gating still applies.
            print("\r\033[K")
            console.warning("Interrupt received — finishing current batch then saving...\n")
        elif isinstance(event, RunFinished):
            print_run_summary(event)
        elif isinstance(event, EvalStarted):
            print(console.dim(f" Found {event.num_files} files for evaluation ({event.num_classes} classes)."))
        elif isinstance(event, EvalProgress):
            self._eval_progress(event)
        elif isinstance(event, SweepEvalTrialStarted):
            print()
            print(f" {console.bold(f'[{event.index}/{event.total}] {event.trial}')}")
        # EpochEnded: Keras' own progress bar already shows each epoch in the terminal.

    def _eval_progress(self, event: EvalProgress) -> None:
        import tqdm

        if self._bar is None:
            self._bar = tqdm.tqdm(
                total=event.total, desc=" Evaluating", unit="batch", disable=not console.is_tty(),
            )
        else:
            self._bar.update(event.done - self._bar.n)
        if event.finished:
            self._bar.close()
            self._bar = None

