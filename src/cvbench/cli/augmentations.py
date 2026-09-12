import copy
import inspect
from pathlib import Path

import click
import questionary
import yaml

from cvbench.augmentations.registry import _RANGES
from cvbench.cli import _help
from cvbench.core.augmentations_store import (
    AUGMENTATIONS_DIR,
    list_saved_augmentations,
    resolve_aug_file,
    write_augmentation_text,
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
# Short, one-line descriptions for every catalogue transform — shown in the
# `generate` checklist and as a comment above each block in a saved config.
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

def _aug_function_defaults() -> list[tuple[str, dict]]:
    """Return (name, defaults_dict) for each aug_* function in the augmentations package.

    A parameter with no default in the function signature (e.g. aug_blur's
    radius) is filled in from registry.py's _RANGES fallback default instead
    of a placeholder, so every value here is directly usable.
    """
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
                fallback = _RANGES.get(name, {}).get("params", {}).get(pname, {}).get("default")
                defaults[pname] = fallback if fallback is not None else "<required>"
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


def _render_config_yaml(transforms: list[dict], preset_label: str) -> str:
    """Render TRANSFORMS as YAML with a compact description + param notes per block.

    This is what `aug generate` saves — meant to be opened in an editor
    afterward, so each block carries enough context to tweak it without
    looking anything up (`aug transforms` / future docs have the full detail).
    """
    from datetime import date

    lines = [
        "meta:",
        f"  preset: {preset_label}",
        f"  created: '{date.today().isoformat()}'",
        "",
        "transforms:",
    ]
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


def _print_catalogue():
    from cvbench.core import _fmt

    name_width = max(len(name) for name, _ in _catalogue())

    def _print_group(title: str, rows: list[tuple[str, dict]]):
        print(f" {_fmt.bold(f'{title}:')}")
        for name, defaults in rows:
            colored_name = _fmt.blue(f"{name:<{name_width}}")
            print(f"   {colored_name}  {_DESCRIPTIONS.get(name, '')}")
            print(f"   {'':<{name_width}}  {_fmt.dim(_fmt_params(defaults))}")
        print()

    print(_fmt.rule(thick=True))
    print(f" {_fmt.bold('Available transforms')}")
    print(_fmt.rule(thick=True))
    _print_group("Keras layers", _KERAS_TRANSFORMS)
    _print_group("Custom functions", _aug_function_defaults())
    print(_fmt.rule(thick=True))


def _checklist_prompt(catalogue: list[tuple[str, dict]], preselected: set) -> list[str] | None:
    """Show a [x]/[ ] checkbox list of every transform in CATALOGUE (arrow keys to
    move, space to toggle, enter to confirm), pre-checking names in PRESELECTED.

    Returns the selected names in catalogue order, or None if the user aborted
    (Ctrl-C / Esc). Split out so tests can monkeypatch this one function instead
    of driving a real terminal.
    """
    name_width = max(len(name) for name, _ in catalogue)
    choices = [
        questionary.Choice(
            title=f"{name:<{name_width}}  {_DESCRIPTIONS.get(name, '')}",
            value=name,
            checked=name in preselected,
        )
        for name, _defaults in catalogue
    ]
    selected = questionary.checkbox(
        "Select transforms to include (space to toggle, enter to confirm):",
        choices=choices,
    ).ask()
    if selected is None:
        return None
    order = {name: i for i, (name, _) in enumerate(catalogue)}
    return sorted(selected, key=order.get)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@_help.group(
    examples=[
        ("aug generate --preset standard", "check off transforms, save under a name"),
        ("aug list", "see every augmentation config you've saved"),
    ],
)
def augmentations():
    """Discover, generate, and manage augmentation configurations."""


@augmentations.command(
    "transforms",
    short_help="List every available transform with its default parameters.",
    examples=[("aug transforms", "Print the full transform catalogue")],
    see_also=[("aug generate", "build a config from these building blocks")],
)
def transforms_cmd():
    """List every available transform (keras layers + custom functions)."""
    _print_catalogue()


@augmentations.command(
    "list",
    short_help="List saved augmentation configs.",
    examples=[("aug list", "See every config you've generated and saved")],
    see_also=[("aug generate", "create a new saved config")],
)
def list_saved():
    """List augmentation configs saved under workspace/augmentations/."""
    from cvbench.core import _fmt

    entries = list_saved_augmentations()
    if not entries:
        print(f" No augmentation configs found in '{AUGMENTATIONS_DIR}'.")
        print(" Run 'aug generate' to create one.")
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
        ("aug generate", "Check off transforms from the full catalogue, nothing pre-selected"),
        ("aug generate --preset standard", "Same checklist, pre-checked with the 'standard' preset"),
        ("aug generate --preset reference --name ref", "Save the fully-commented reference sheet"),
    ],
    see_also=[("train data/ --augmentation <name>", "train with the config you saved")],
)
@click.option("--preset", type=click.Choice(["light", "standard", "heavy", "reference"]),
              default=None, help="Seed the wizard from this preset instead of starting blank.")
@click.option("--name", default=None, help="Save under this name (prompted if omitted).")
def generate(preset, name):
    """Interactively select transforms and save them as a named config.

    Shows a checklist of every available transform, each with a short
    description — space to toggle, enter to confirm — pre-checked for
    whichever ones --preset includes (nothing pre-checked without --preset).
    The saved file gets a compact comment above each transform explaining
    what it does and what its parameters mean, so it's ready to fine-tune by
    opening it in an editor afterward.
    """
    from datetime import date

    from cvbench.core import _fmt

    if preset == "reference":
        save_name = name or click.prompt("Save as", default="reference")
        header = f"meta:\n  preset: reference\n  created: '{date.today().isoformat()}'\n"
        path = write_augmentation_text(save_name, header + _reference_yaml())
        print(f"  {_fmt.green('✓')} Saved → {path}")
        print(f"  Usage:  train data/ --augmentation {save_name}")
        return

    preset_map = {t["name"]: t for t in _preset_transforms(preset)} if preset else {}
    catalogue = _catalogue()

    print(_fmt.rule(thick=True))
    print(f" {_fmt.bold('Augmentation wizard')}")
    print(_fmt.rule(thick=True))

    selected = _checklist_prompt(catalogue, set(preset_map))
    if selected is None:
        raise click.Abort()
    if not selected:
        raise click.ClickException("No transforms selected — nothing to save.")

    defaults_by_name = dict(catalogue)
    kept = [
        preset_map.get(t_name) or {"name": t_name, "prob": 1.0, **defaults_by_name[t_name]}
        for t_name in selected
    ]

    save_name = name or click.prompt("Save as", default=preset or "config")
    content = _render_config_yaml(kept, preset or "custom")
    path = write_augmentation_text(save_name, content)
    print(f"  {_fmt.green('✓')} Saved → {path}  ({len(kept)} transform(s))")
    print("  Open it in an editor to fine-tune any value.")
    print(f"  Usage:  train data/ --augmentation {save_name}")


@augmentations.command(
    "show",
    short_help="Print a saved augmentation config.",
    examples=[("aug show standard", "Print the saved 'standard' config's YAML")],
)
@click.argument("name")
def show(name):
    """Print the raw YAML of a saved augmentation config, syntax-highlighted."""
    from rich.console import Console
    from rich.syntax import Syntax

    path = resolve_aug_file(name)
    content = Path(path).read_text()
    console = Console()
    if console.is_terminal:
        console.print(Syntax(content, "yaml", theme="ansi_dark", background_color="default",
                              word_wrap=True))
    else:
        print(content, end="")


@augmentations.command(
    "delete",
    short_help="Delete a saved augmentation config.",
    examples=[("aug delete standard -y", "Remove the saved 'standard' config without confirming")],
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
