"""Augmentation registry — builds UI parameter schemas from function signatures.

Design rationale
----------------
`inspect` extracts parameter names, type annotations, and default values directly
from each aug_* function.  The registry stores *only* what introspection cannot
provide: numeric ranges (min/max/step) and string option lists (choices).

Computation modules (blur.py, noise.py, …) remain pure — no UI concepts bleed in.
To add a new augmentation to the UI: write the function, add one entry to _RANGES.

Parameters always excluded from the UI schema:
    img   – implicit array input
    seed  – internal randomness; the UI never exposes a seed (each Run is a fresh draw)
"""

from __future__ import annotations

import inspect
from typing import Any

# ── Range / option metadata ───────────────────────────────────────────────────
# Keys are aug_* function names.  Values are per-parameter dicts containing
# only the fields that cannot be read from the function signature:
#   min, max, step  – for numeric (int / float) params
#   choices         – for str params with a fixed option set
#   label           – human-readable augmentation name shown in the UI
#   default         – fallback only when the function signature has no default value

_RANGES: dict[str, dict] = {
    "aug_blur": {
        "label": "Gaussian Blur",
        "params": {
            "radius": {"min": 0.5, "max": 15.0, "step": 0.5, "default": 3.0}
        },
    },
    "aug_salt_pepper": {
        "label": "Salt & Pepper Noise",
        "params": {
            "density": {"min": 0.0, "max": 0.5, "step": 0.01, "default": 0.05}
        },
    },
    "aug_gamma": {
        "label": "Gamma Correction",
        "params": {
            "gamma": {"min": 0.1, "max": 3.0, "step": 0.1, "default": 1.0}
        },
    },
    "aug_fog": {
        "label": "Fog / Haze",
        "params": {
            "strength": {"min": 0.0, "max": 1.0, "step": 0.05, "default": 0.3}
        },
    },
    "aug_lines_h": {
        "label": "Horizontal Lines",
        "params": {
            "n_lines": {"min": 1, "max": 20, "step": 1},
            "width": {"min": 1, "max": 10, "step": 1},
            "brightness": {"min": 0, "max": 255, "step": 5},
        },
    },
    "aug_lines_v": {
        "label": "Vertical Lines",
        "params": {
            "n_lines": {"min": 1, "max": 20, "step": 1},
            "width": {"min": 1, "max": 10, "step": 1},
            "brightness": {"min": 0, "max": 255, "step": 5},
        },
    },
    "aug_fade_horizontal": {
        "label": "Horizontal Fade",
        "params": {
            "fade_to": {"min": 0, "max": 255, "step": 5},
            "side": {"choices": ["left", "right", "both"]},
            "strength": {"min": 0.0, "max": 1.0, "step": 0.05},
        },
    },
    "aug_fade_vertical": {
        "label": "Vertical Fade",
        "params": {
            "fade_to": {"min": 0, "max": 255, "step": 5},
            "side": {"choices": ["top", "bottom", "both"]},
            "strength": {"min": 0.0, "max": 1.0, "step": 0.05},
        },
    },
    "aug_brighten_edges_h": {
        "label": "Brighten Edges H",
        "params": {
            "fade_to": {"min": 0, "max": 255, "step": 5},
            "strength": {"min": 0.0, "max": 1.0, "step": 0.05},
            "edge_fraction": {"min": 0.05, "max": 0.5, "step": 0.05},
        },
    },
    "aug_brighten_edges_v": {
        "label": "Brighten Edges V",
        "params": {
            "fade_to": {"min": 0, "max": 255, "step": 5},
            "strength": {"min": 0.0, "max": 1.0, "step": 0.05},
            "edge_fraction": {"min": 0.05, "max": 0.5, "step": 0.05},
        },
    },
    "aug_random_profile_h": {
        "label": "Random Profile H",
        "params": {
            "n_changes": {"min": 1, "max": 10, "step": 1},
            "max_delta": {"min": 5, "max": 120, "step": 5},
        },
    },
    "aug_random_profile_v": {
        "label": "Random Profile V",
        "params": {
            "n_changes": {"min": 1, "max": 10, "step": 1},
            "max_delta": {"min": 5, "max": 120, "step": 5},
        },
    },
    "aug_mask_h": {
        "label": "Horizontal Band Mask",
        "params": {
            "n_masks": {"min": 1, "max": 5, "step": 1},
            "max_width": {"min": 1, "max": 100, "step": 1},
            "fill_value": {"min": 0, "max": 255, "step": 5},
        },
    },
    "aug_mask_v": {
        "label": "Vertical Band Mask",
        "params": {
            "n_masks": {"min": 1, "max": 5, "step": 1},
            "max_width": {"min": 1, "max": 100, "step": 1},
            "fill_value": {"min": 0, "max": 255, "step": 5},
        },
    },
    "aug_rf_transmission": {
        "label": "RF Transmission",
        "params": {
            "bandwidth": {"min": 0.02, "max": 0.25, "step": 0.01},
            "brightness_delta": {"min": 0, "max": 80, "step": 5},
            "rectangular": {"choices": [True, False]},
            "edge_rolloff": {"min": 0.0, "max": 0.01, "step": 0.001},
            "ripple": {"min": 0.0, "max": 0.2, "step": 0.01},
            "drift_speed": {"min": 0.0, "max": 0.08, "step": 0.005},
            "noise_floor": {"min": 0.0, "max": 0.25, "step": 0.01},
        },
    },
}

_SKIP = {"img", "seed"}


def get_schema() -> list[dict]:
    """Return UI-ready parameter schemas for all registered augmentations.

    Merges live function introspection (names, types, defaults) with the range
    metadata in _RANGES.  The result is a JSON-serialisable list consumed by
    GET /api/augmentations.
    """
    import cvbench.augmentations as aug_mod

    result = []
    for func_name, meta in _RANGES.items():
        fn = getattr(aug_mod, func_name, None)
        if fn is None:
            continue

        sig = inspect.signature(fn)
        params: list[dict[str, Any]] = []

        for pname, p in sig.parameters.items():
            if pname in _SKIP:
                continue

            range_meta = meta["params"].get(pname, {})
            has_default = p.default is not inspect.Parameter.empty
            default = p.default if has_default else range_meta.get("default")

            if "choices" in range_meta:
                ptype = "choice"
            elif p.annotation is int or (
                has_default and isinstance(p.default, int)
            ):
                ptype = "int"
            else:
                ptype = "float"

            entry: dict[str, Any] = {
                "name": pname,
                "type": ptype,
                "default": default,
            }
            for k, v in range_meta.items():
                if k != "default":  # default already resolved above
                    entry[k] = v
            params.append(entry)

        result.append(
            {"name": func_name, "label": meta["label"], "params": params}
        )

    return result
