import importlib.util
import sys
from pathlib import Path

_REGISTRY: dict[str, callable] = {}


def register_aug(name: str):
    """Decorator that registers an augmentation builder function by name.

    Usage:
        @register_aug("my_aug")
        def build(params: dict) -> keras.Sequential:
            ...
    """
    def decorator(fn):
        _REGISTRY[name] = fn
        return fn
    return decorator


def resolve_aug(name: str, params: dict):
    """Return a built augmentation layer for the given registered name.

    Raises KeyError with a helpful message if name is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Augmentation '{name}' is not registered. "
            f"Available: {available}"
        )
    return _REGISTRY[name](params)


def load_aug_source(source_path: str):
    """Dynamically import a custom augmentation file to trigger registration."""
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Augmentation source not found: {source_path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
