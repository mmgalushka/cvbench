import inspect
import sys

import click
import yaml


# ---------------------------------------------------------------------------
# Keras transform catalogue (name → default params)
# Defined here as plain data to avoid triggering TF/Keras import.
# ---------------------------------------------------------------------------

_KERAS_TRANSFORMS = [
    ("keras_flip",        {"mode": "horizontal"}),
    ("keras_rotation",    {"factor": 0.1}),
    ("keras_zoom",        {"height_factor": 0.1}),
    ("keras_translation", {"height_factor": 0.1, "width_factor": 0.1}),
    ("keras_crop",        {"height": 196, "width": 196}),
    ("keras_brightness",  {"factor": 0.2}),
    ("keras_contrast",    {"factor": 0.2}),
    ("keras_noise",       {"stddev": 0.05}),
]

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

_PRESETS = {
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
# Helpers
# ---------------------------------------------------------------------------

def _aug_function_defaults() -> list[tuple[str, dict]]:
    """Return (name, defaults_dict) for each aug_* function in augmentations package."""
    import augmentations as aug_mod
    result = []
    for name in sorted(n for n in dir(aug_mod) if n.startswith("aug_")):
        fn = getattr(aug_mod, name)
        sig = inspect.signature(fn)
        defaults = {}
        for pname, param in sig.parameters.items():
            if pname == "img":
                continue
            if param.default is inspect.Parameter.empty:
                defaults[pname] = "<required>"
            else:
                defaults[pname] = param.default
        result.append((name, defaults))
    return result


def _fmt_params(d: dict) -> str:
    parts = []
    for k, v in d.items():
        if isinstance(v, str):
            parts.append(f'{k}: "{v}"')
        else:
            parts.append(f"{k}: {v}")
    return ",  ".join(parts)


def _preset_to_yaml(name: str) -> str:
    if name == "reference":
        return _reference_yaml()
    preset = _PRESETS[name]
    transforms = preset["transforms"]
    # Build a list of dicts preserving key order: name, prob, then rest
    out_transforms = []
    for t in transforms:
        entry = {"name": t["name"], "prob": t["prob"]}
        entry.update({k: v for k, v in t.items() if k not in ("name", "prob")})
        out_transforms.append(entry)
    return yaml.dump({"transforms": out_transforms}, default_flow_style=False, sort_keys=False)


def _reference_yaml() -> str:
    lines = [
        "# Augmentation reference — uncomment the transforms you want to use.",
        "# Pass this file to training with:  train data/ --augmentation <this_file>",
        "#",
        "# Range sampling: any numeric parameter can be written as a 2-element list,",
        "# e.g.  radius: [0.5, 2.0]  — a fresh value is sampled per image.",
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def augmentations():
    """Discover and generate augmentation configurations."""


@augmentations.command("list")
def list_transforms():
    """List all available transforms with their default parameters."""
    w = 55
    print("━" * w)
    print(" Available transforms")
    print("━" * w)
    print(" Keras layers:")
    for name, defaults in _KERAS_TRANSFORMS:
        params_str = _fmt_params(defaults)
        print(f"   {name:<26}  {params_str}")
    print()
    print(" Custom functions:")
    for name, defaults in _aug_function_defaults():
        params_str = _fmt_params(defaults)
        print(f"   {name:<26}  {params_str}")
    print("━" * w)


@augmentations.command("example")
@click.argument("preset", required=False,
                type=click.Choice(["light", "standard", "heavy", "reference"]))
@click.option("--output", "output_file", default=None, type=click.Path(),
              help="Write to this file instead of stdout.")
def example(preset, output_file):
    """Generate a preset augmentation configuration.

    Available presets: light, standard, heavy, reference.
    Run without a preset name to see descriptions.
    """
    if preset is None:
        w = 55
        print("━" * w)
        print(" Augmentation presets")
        print("━" * w)
        for name, data in _PRESETS.items():
            print(f"   {name:<12}  {data['description']}")
        print(f"   {'reference':<12}  All transforms commented out — a lookup sheet.")
        print("━" * w)
        print(f" Usage: augmentations example <preset> [--output FILE]")
        return

    content = _preset_to_yaml(preset)

    if output_file:
        from pathlib import Path
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(content)
        print(f" Saved → {output_file}")
        print(f" Usage:  train data/ --augmentation {output_file}")
    else:
        sys.stdout.write(content)
