import copy
import inspect
from pathlib import Path

import click

from cvbench.cli import _help
from cvbench.core.augmentations_store import (
    AUGMENTATIONS_DIR,
    list_saved_augmentations,
    resolve_aug_file,
    save_augmentation_config,
)


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
    """Return (name, defaults_dict) for each aug_* function in the augmentations package."""
    import cvbench.augmentations as aug_mod
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


def _catalogue() -> list[tuple[str, dict]]:
    """Every available transform (keras layers + custom functions), name → defaults."""
    return list(_KERAS_TRANSFORMS) + _aug_function_defaults()


def _preset_transforms(name: str) -> list[dict]:
    """Deep-copy of a preset's transform list, key order name/prob/...params preserved."""
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


def _prompt_params(defaults: dict, current: dict | None = None) -> dict:
    """Prompt for each param in DEFAULTS, showing CURRENT (or the default) as the default.

    Type is inferred from the shown default value. Returns the collected params.
    """
    current = current or {}
    params = {}
    for pname, default in defaults.items():
        shown = current.get(pname, default)
        if shown == "<required>":
            value = click.prompt(f"    {pname} (required)", type=str)
            if value == "":
                raise click.ClickException(f"'{pname}' is required.")
            params[pname] = value
        elif shown is None:
            # e.g. seed=None — optional, blank keeps it unset.
            raw = click.prompt(f"    {pname} (optional)", default="", show_default=False)
            params[pname] = raw if raw != "" else None
        else:
            params[pname] = click.prompt(f"    {pname}", default=shown, type=type(shown))
    return params


def _print_catalogue():
    from cvbench.core import _fmt

    print(_fmt.rule(thick=True))
    print(f" {_fmt.bold('Available transforms')}")
    print(_fmt.rule(thick=True))
    print(" Keras layers:")
    for name, defaults in _KERAS_TRANSFORMS:
        print(f"   {name:<26}  {_fmt_params(defaults)}")
    print()
    print(" Custom functions:")
    for name, defaults in _aug_function_defaults():
        print(f"   {name:<26}  {_fmt_params(defaults)}")
    print(_fmt.rule(thick=True))


def _choose_from_catalogue() -> dict:
    """Print the numbered catalogue, prompt for a pick, and return a new transform dict."""
    catalogue = _catalogue()
    print()
    for i, (name, defaults) in enumerate(catalogue, 1):
        print(f"   {i:>2}) {name:<26}  {_fmt_params(defaults)}")
    print()
    choice = click.prompt("  Transform number or name")
    match = None
    if choice.isdigit() and 1 <= int(choice) <= len(catalogue):
        match = catalogue[int(choice) - 1]
    else:
        match = next((c for c in catalogue if c[0] == choice), None)
    if match is None:
        raise click.ClickException(f"Unknown transform: '{choice}'")

    name, defaults = match
    print(f"  Adding '{name}':")
    prob = click.prompt("    prob", default=1.0, type=float)
    params = _prompt_params(defaults)
    entry = {"name": name, "prob": prob}
    entry.update(params)
    return entry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@_help.group(
    examples=[
        ("data aug generate --preset standard", "generate + customize a config, save it under a name"),
        ("data aug list", "see every augmentation config you've saved"),
    ],
)
def augmentations():
    """Discover, generate, and manage augmentation configurations."""


@augmentations.command(
    "transforms",
    short_help="List every available transform with its default parameters.",
    examples=[("data aug transforms", "Print the full transform catalogue")],
    see_also=[("data aug generate", "build a config from these building blocks")],
)
def transforms_cmd():
    """List every available transform (keras layers + custom functions)."""
    _print_catalogue()


@augmentations.command(
    "list",
    short_help="List saved augmentation configs.",
    examples=[("data aug list", "See every config you've generated and saved")],
    see_also=[("data aug generate", "create a new saved config")],
)
def list_saved():
    """List augmentation configs saved under workspace/augmentations/."""
    from cvbench.core import _fmt

    entries = list_saved_augmentations()
    if not entries:
        print(f" No augmentation configs found in '{AUGMENTATIONS_DIR}'.")
        print(" Run 'data aug generate' to create one.")
        return

    max_name = max(len(e["name"]) for e in entries)
    tr = _fmt.rule(thick=True)
    print(tr)
    print(f" {'Name':<{max_name}}  {'Preset':<10}  {'Transforms':>10}  {'Modified':>10}")
    print(tr)
    for e in entries:
        print(
            f" {e['name']:<{max_name}}  {e['preset']:<10}  "
            f"{e['n_transforms']:>10}  {e['modified']:>10}"
        )
    print(tr)


@augmentations.command(
    "generate",
    short_help="Interactively build and save a new augmentation config.",
    examples=[
        ("data aug generate", "Start from a blank config and add transforms one at a time"),
        ("data aug generate --preset standard", "Start from the 'standard' preset and customize it"),
        ("data aug generate --preset reference --name ref", "Save the fully-commented reference sheet"),
    ],
    see_also=[("train data/ --augmentation <name>", "train with the config you saved")],
)
@click.option("--preset", type=click.Choice(["light", "standard", "heavy", "reference"]),
              default=None, help="Seed the wizard from this preset instead of starting blank.")
@click.option("--name", default=None, help="Save under this name (prompted if omitted).")
def generate(preset, name):
    """Interactively build an augmentation config, then save it under a name.

    With --preset, the wizard starts from that preset's transforms — keep,
    drop, or customize each one, then optionally add more from the full
    catalogue. Without --preset, it starts blank and only adds what you pick.
    """
    from cvbench.core import _fmt

    if preset == "reference":
        save_name = name or click.prompt("Save as", default="reference")
        path = _save_reference(save_name)
        print(f"  {_fmt.green('✓')} Saved → {path}")
        print(f"  Usage:  train data/ --augmentation {save_name}")
        return

    transforms = _preset_transforms(preset) if preset else []

    print(_fmt.rule(thick=True))
    print(f" {_fmt.bold('Augmentation wizard')}")
    print(_fmt.rule(thick=True))

    kept = []
    for t in transforms:
        t_name = t["name"]
        rest = {k: v for k, v in t.items() if k not in ("name", "prob")}
        print(f" {t_name}  prob={t['prob']}  {_fmt_params(rest)}")
        if not click.confirm(f"  Keep '{t_name}'?", default=True):
            continue
        if click.confirm("  Customize its parameters?", default=False):
            prob = click.prompt("    prob", default=t["prob"], type=float)
            params = _prompt_params(rest, current=rest)
            entry = {"name": t_name, "prob": prob}
            entry.update(params)
            kept.append(entry)
        else:
            kept.append(t)

    while click.confirm("Add another transform from the catalogue?", default=False):
        kept.append(_choose_from_catalogue())

    if not kept:
        raise click.ClickException("No transforms selected — nothing to save.")

    save_name = name or click.prompt("Save as", default=preset or "config")
    path = save_augmentation_config(save_name, kept, preset or "custom")
    print(f"  {_fmt.green('✓')} Saved → {path}  ({len(kept)} transform(s))")
    print(f"  Usage:  train data/ --augmentation {save_name}")


def _save_reference(name: str) -> Path:
    from datetime import date

    out_dir = Path(AUGMENTATIONS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.yaml"
    header = f"meta:\n  preset: reference\n  created: '{date.today().isoformat()}'\n"
    with open(path, "w") as f:
        f.write(header + _reference_yaml())
    return path


@augmentations.command(
    "show",
    short_help="Print a saved augmentation config.",
    examples=[("data aug show standard", "Print the saved 'standard' config's YAML")],
)
@click.argument("name")
def show(name):
    """Print the raw YAML of a saved augmentation config."""
    path = resolve_aug_file(name)
    print(Path(path).read_text(), end="")


@augmentations.command(
    "delete",
    short_help="Delete a saved augmentation config.",
    examples=[("data aug delete standard -y", "Remove the saved 'standard' config without confirming")],
)
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip the confirmation prompt.")
def delete(name, yes):
    """Delete a saved augmentation config."""
    from cvbench.core import _fmt

    path = Path(resolve_aug_file(name))

    if not yes:
        click.confirm(
            f"{_fmt.yellow('Warning:')} This will permanently delete '{path}'. Continue?",
            abort=True,
        )

    path.unlink()
    print(_fmt.green(f" Deleted '{path}'."))
