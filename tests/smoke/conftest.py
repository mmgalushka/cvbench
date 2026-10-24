"""Smoke tests download real datasets and train: they only run when asked for with ``-m smoke``."""
import pytest


def pytest_collection_modifyitems(config, items):
    if "smoke" in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(reason="smoke tests download data and train; run with: pytest -m smoke")
    for item in items:
        if "smoke" in item.keywords:
            item.add_marker(skip)
