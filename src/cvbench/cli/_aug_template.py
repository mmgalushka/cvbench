"""Content for `data aug`: built-in presets, the commented reference sheet, and
the YAML renderer. Plain data only, so nothing here pulls in TF/Keras."""
import copy
from typing import Any

import yaml

from cvbench.augmentations.registry import _RANGES

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

_PRESETS: dict[str, dict[str, Any]] = {
    "light": {
        "description": "Horizontal flip + tiny rotation. Safe for any dataset.",
        "transforms": [
            {"name": "keras_flip",     "prob": 1.0, "mode": "horizontal"},
            {"name": "keras_rotation", "prob": 0.5, "factor": 0.05},
        ],
    },
    "standard": {
        "description": "Flip + rotation + brightness/contrast + blur. Good general starting point.",
        "transforms": [
            {"name": "keras_flip",       "prob": 1.0, "mode": "horizontal"},
            {"name": "keras_rotation",   "prob": 0.7, "factor": 0.1},
            {"name": "keras_brightness", "prob": 0.5, "factor": 0.2},
            {"name": "keras_contrast",   "prob": 0.5, "factor": 0.2},
            {"name": "aug_blur",         "prob": 0.3, "radius": 1.0},
        ],
    },
    "heavy": {
        "description": "Everything in standard + zoom + fog + noise. Aggressive regularisation.",
        "transforms": [
            {"name": "keras_flip",       "prob": 1.0, "mode": "horizontal_and_vertical"},
            {"name": "keras_rotation",   "prob": 0.8, "factor": 0.15},
            {"name": "keras_zoom",       "prob": 0.5, "height_factor": 0.15},
            {"name": "keras_brightness", "prob": 0.6, "factor": 0.3},
            {"name": "keras_contrast",   "prob": 0.6, "factor": 0.3},
            {"name": "aug_blur",         "prob": 0.4, "radius": 1.5},
            {"name": "aug_fog",          "prob": 0.2, "strength": 0.15},
            {"name": "aug_salt_pepper",  "prob": 0.3, "density": 0.02},
        ],
    },
}

# ---------------------------------------------------------------------------
# Short, one-line descriptions for every catalogue transform — shown as a
# comment above each block in a generated config.
# Per-parameter notes (ranges/choices) come from augmentations/registry.py's
# _RANGES instead of being duplicated here.
# ---------------------------------------------------------------------------

_DESCRIPTIONS = {
    "keras_flip": "Randomly flip the image.",
    "keras_rotation": "Randomly rotate the image.",
    "keras_zoom": "Randomly zoom in or out.",
    "keras_translation": "Randomly shift the image horizontally/vertically.",
    "keras_crop": "Randomly crop to a fixed size.",
    "keras_brightness": "Randomly adjust brightness.",
    "keras_contrast": "Randomly adjust contrast.",
    "keras_noise": "Add Gaussian noise.",
    "aug_blur": "Gaussian blur.",
    "aug_brighten_edges": "Brighten or darken an edge (choose orientation).",
    "aug_brighten_edges_h": "Brighten or darken the left/right edges.",
    "aug_brighten_edges_v": "Brighten or darken the top/bottom edges.",
    "aug_chirp_artifacts": "Add curved, chirp-like line artifacts.",
    "aug_fade": "Fade an edge to a fixed value (choose orientation/side).",
    "aug_fade_horizontal": "Fade the left or right edge to a fixed value.",
    "aug_fade_vertical": "Fade the top or bottom edge to a fixed value.",
    "aug_fog": "Add a fog/haze effect.",
    "aug_gamma": "Gamma correction (brightens or darkens midtones).",
    "aug_interference": "Add a wave-like interference pattern.",
    "aug_lines": "Draw random lines (choose orientation).",
    "aug_lines_h": "Draw random horizontal lines.",
    "aug_lines_v": "Draw random vertical lines.",
    "aug_mask": "Blank out a random band (choose orientation).",
    "aug_mask_h": "Blank out a random horizontal band.",
    "aug_mask_v": "Blank out a random vertical band.",
    "aug_net": "Overlay a net-like grid of lines/stripes.",
    "aug_random_profile_h": "Apply a smooth random brightness profile along rows.",
    "aug_random_profile_v": "Apply a smooth random brightness profile along columns.",
    "aug_rf_transmission": "Simulate RF/radio transmission-style banding noise.",
    "aug_salt_pepper": "Add salt-and-pepper (random black/white pixel) noise.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _preset_transforms(name: str) -> list[dict]:
    """Deep-copy of a built-in preset's transform list, key order name/prob/...params preserved."""
    out = []
    for t in _PRESETS[name]["transforms"]:
        entry = {"name": t["name"], "prob": t["prob"]}
        entry.update({k: v for k, v in t.items() if k not in ("name", "prob")})
        out.append(entry)
    return copy.deepcopy(out)



def _reference_yaml() -> str:
    lines = [
        "# Augmentation reference — uncomment the transforms you want to use.",
        "# Pass this file to training with:  train data/ --augmentation <this_file>",
        "#",
        "# Range sampling: any numeric parameter can be written as a 2-element list,",
        "# e.g.  radius: [0.5, 2.0]  — a fresh value is sampled per image.",
        "#",
        "# Mutually exclusive groups: wrap candidates in a one_of block so that at",
        "# most one fires per image. Use weight to control relative frequency.",
        "#",
        "# - one_of:",
        "#     prob: 0.5          # probability the group fires at all",
        "#     candidates:",
        "#       - name: aug_fade_horizontal",
        "#         weight: 2      # picked 2x as often as weight-1 candidates",
        "#         side: both",
        "#         strength: [0.3, 0.8]",
        "#       - name: aug_fog",
        "#         weight: 1",
        "#         strength: [0.1, 0.3]",
        "#",
        "transforms:",
        "",
        "  # ── Keras built-in layers ─────────────────────────────────────────────",
        "",
        "  # Randomly flip images.",
        "  # - name: keras_flip",
        '  #   prob: 1.0',
        '  #   mode: horizontal    # horizontal | vertical | horizontal_and_vertical',
        "",
        "  # Randomly rotate images (factor = max rotation as fraction of 2π).",
        "  # - name: keras_rotation",
        "  #   prob: 0.8",
        "  #   factor: 0.1",
        "",
        "  # Randomly zoom in/out.",
        "  # - name: keras_zoom",
        "  #   prob: 0.5",
        "  #   height_factor: 0.1",
        "",
        "  # Randomly translate (shift) images.",
        "  # - name: keras_translation",
        "  #   prob: 0.5",
        "  #   height_factor: 0.1",
        "  #   width_factor: 0.1",
        "",
        "  # Randomly crop images to a fixed size.",
        "  # - name: keras_crop",
        "  #   prob: 1.0",
        "  #   height: 196",
        "  #   width: 196",
        "",
        "  # Randomly adjust brightness.",
        "  # - name: keras_brightness",
        "  #   prob: 0.5",
        "  #   factor: 0.2",
        "",
        "  # Randomly adjust contrast.",
        "  # - name: keras_contrast",
        "  #   prob: 0.5",
        "  #   factor: 0.2",
        "",
        "  # Add Gaussian noise.",
        "  # - name: keras_noise",
        "  #   prob: 0.3",
        "  #   stddev: 0.05",
        "",
        "  # ── Custom functions ───────────────────────────────────────────────────",
        "",
        "  # Gaussian blur.",
        "  # - name: aug_blur",
        "  #   prob: 0.3",
        "  #   radius: [0.5, 2.0]",
        "",
        "  # Salt-and-pepper noise.",
        "  # - name: aug_salt_pepper",
        "  #   prob: 0.3",
        "  #   density: [0.01, 0.05]",
        "",
        "  # Gamma correction (< 1 brightens, > 1 darkens).",
        "  # - name: aug_gamma",
        "  #   prob: 0.4",
        "  #   gamma: [0.8, 1.4]",
        "",
        "  # Fog / haze effect.",
        "  # - name: aug_fog",
        "  #   prob: 0.2",
        "  #   strength: [0.05, 0.3]",
        "",
        "  # Fade edge(s) to grey.",
        "  # - name: aug_fade_horizontal",
        "  #   prob: 0.3",
        "  #   fade_to: [100, 180]",
        '  #   side: [left, right, both]    # randomly chosen per image',
        "  #   strength: [0.5, 1.0]",
        "",
        "  # Fade top or bottom edge to grey.",
        "  # - name: aug_fade_vertical",
        "  #   prob: 0.3",
        "  #   fade_to: [100, 180]",
        '  #   side: [top, bottom, both]    # randomly chosen per image',
        "  #   strength: [0.5, 1.0]",
        "",
        "  # Brighten/darken left and right edges (Gaussian falloff).",
        "  # - name: aug_brighten_edges_h",
        "  #   prob: 0.3",
        "  #   fade_to: [200, 255]",
        "  #   strength: [0.5, 1.0]",
        "  #   edge_fraction: [0.1, 0.25]",
        "",
        "  # Brighten/darken top and bottom edges (Gaussian falloff).",
        "  # - name: aug_brighten_edges_v",
        "  #   prob: 0.3",
        "  #   fade_to: [200, 255]",
        "  #   strength: [0.5, 1.0]",
        "  #   edge_fraction: [0.1, 0.25]",
        "",
        "  # Random smooth brightness profile along the horizontal axis.",
        "  # - name: aug_random_profile_h",
        "  #   prob: 0.4",
        "  #   n_changes: [3, 8]",
        "  #   max_delta: [20.0, 80.0]",
        "",
        "  # Random smooth brightness profile along the vertical axis.",
        "  # - name: aug_random_profile_v",
        "  #   prob: 0.4",
        "  #   n_changes: [3, 8]",
        "  #   max_delta: [20.0, 80.0]",
        "",
        "  # Random horizontal lines.",
        "  # - name: aug_lines_h",
        "  #   prob: 0.3",
        "  #   n_lines: [2, 8]       # sampled once per image",
        "  #   width: [1, 3]         # sampled once per image",
        "  #   brightness: [0, 255]  # sampled once per image",
        "",
        "  # Random vertical lines.",
        "  # - name: aug_lines_v",
        "  #   prob: 0.3",
        "  #   n_lines: [2, 8]       # sampled once per image",
        "  #   width: [1, 3]         # sampled once per image",
        "  #   brightness: [0, 255]  # sampled once per image",
    ]
    return "\n".join(lines) + "\n"


def _param_note(t_name: str, pname: str) -> str | None:
    """A short 'how to change this' note for one transform's parameter, from _RANGES."""
    meta = _RANGES.get(t_name, {}).get("params", {}).get(pname)
    if not meta:
        return None
    if "choices" in meta:
        return " | ".join(str(c) for c in meta["choices"])
    if "min" in meta and "max" in meta:
        return f"range {meta['min']}–{meta['max']}"
    return None


def _yaml_scalar(v) -> str:
    """Render a single Python value the way it would appear as a YAML scalar."""
    return yaml.safe_dump(v, default_flow_style=True).split("\n", 1)[0]


def _render_config_yaml(transforms: list[dict]) -> str:
    """Render TRANSFORMS as YAML with a compact description + param notes per block.

    This is what `data aug --preset <name>` writes — meant to be opened in an
    editor afterward, so each block carries enough context to tweak it without
    looking anything up (`data aug --preset reference` lists every transform).
    """
    lines = ["transforms:"]
    for t in transforms:
        t_name = t["name"]
        rest = {k: v for k, v in t.items() if k not in ("name", "prob")}
        desc = _DESCRIPTIONS.get(t_name)
        if desc:
            lines.append(f"  # {desc}")
        lines.append(f"  - name: {t_name}")
        lines.append(f"    prob: {_yaml_scalar(t['prob'])}  # chance this fires per image")
        for pname, v in rest.items():
            note = _param_note(t_name, pname)
            trailer = f"  # {note}" if note else ""
            lines.append(f"    {pname}: {_yaml_scalar(v)}{trailer}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

