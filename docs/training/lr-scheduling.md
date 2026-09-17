# Learning Rate Scheduling

By default the learning rate is fixed for the entire training run. Use
`--lr-scheduler` to enable **ReduceLROnPlateau** — the LR is multiplied by
`factor` whenever `val_loss` fails to improve for `patience` consecutive
epochs.

```bash
# Reduce LR by 0.5x after 5 flat epochs (default factor and floor)
train data/ --lr 1e-3 --lr-scheduler patience=5

# Aggressive decay: cut to 20% after 3 flat epochs, floor at 1e-6
train data/ --lr 1e-3 --lr-scheduler patience=3,factor=0.2,min=1e-6
```

| Parameter | Default | Description |
|---|---|---|
| `patience=N` | required | Epochs with no `val_loss` improvement before reducing LR |
| `factor=F` | `0.5` | Multiplicative reduction factor |
| `min=F` | `1e-7` | Minimum LR floor |

!!! note
    The scheduler settings are saved to `config.yaml` and applied
    automatically when resuming a run.

Next: [Two-Phase Training](two-phase.md).
