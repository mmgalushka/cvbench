# Two-Phase Training (freeze → fine-tune)

A common [transfer-learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
workflow is to first train with the backbone
frozen, then unfreeze some layers and fine-tune at a lower learning rate.

```text
Phase 1 (epochs 0–30)              Phase 2 (epochs 30–50)
┌──────────────────────────┐       ┌──────────────────────────┐
│ backbone: frozen         │  ──▶  │ backbone: unfrozen from   │
│ head: training           │       │ layer 100, lr=1e-5        │
│ lr: default              │       │ head: continues training  │
└──────────────────────────┘       └──────────────────────────┘
        training_log.csv  ─────────────▶  (appended, both phases)
```

**Phase 1 — train classifier head only (backbone frozen):**

```bash
train data/ --epochs 30 --output experiments/phase1
```

**Phase 2 — unfreeze top layers and fine-tune:**

```bash
train data/ \
  --from experiments/phase1 \
  --resume experiments/phase1/best.keras \
  --fine-tune-from-layer 100 \
  --lr 1e-5 \
  --epochs 50
```

`--from` loads the phase-1 config (backbone, input size, augmentation, etc.)
and its recorded epoch count. `--resume` loads the saved weights. Training
then continues from epoch 30 through epoch 50, adding 20 fine-tuning epochs —
the training log is **appended**, so the full history (both phases) is
preserved in `training_log.csv`.

!!! warning "`--epochs N` means *end at* epoch N"
    Not *run N more epochs*. If phase 1 ran 30 epochs and you want 20 more,
    set `--epochs 50`.

## Resuming after an interrupt

If training is interrupted mid-run, CVBench saves an `interrupt_epochNNN.keras`
[`.keras`](https://keras.io/api/models/model_saving_apis/) checkpoint and prints the exact resume command:

```bash
train data/ --from experiments/phase1 --resume experiments/phase1/interrupt_epoch023.keras --epochs 30
```

| Option | Description |
|---|---|
| `--from <exp_dir>` | Load backbone, hyperparameters, and epoch count from a previous experiment |
| `--resume <checkpoint>` | Load weights from a `.keras` checkpoint and continue training from the recorded epoch |
| `--fine-tune-from-layer N` | Unfreeze backbone layers from index N onward (`-1` = unfreeze all) |

!!! tip "Why a lower learning rate in phase 2?"
    Pretrained backbone weights are already good. Large updates can wipe out
    what they learned, so fine-tune with a learning rate roughly 10–100×
    smaller than the one used for the head.

## Further reading

- [Transfer learning and fine-tuning](https://www.tensorflow.org/tutorials/images/transfer_learning)
  (TensorFlow guide)
- [Keras model saving](https://keras.io/api/models/model_saving_apis/)
- [Technology References](../references.md)

Next: [Evaluating a Run](../evaluation/evaluate.md).
