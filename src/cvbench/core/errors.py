"""Domain errors raised by `core/` (and the services above it).

`core/` knows nothing about its callers, so a lookup that fails raises one of
these instead of a Click or HTTP error. Each front end maps them in exactly one
place: the CLI in `cli/_help.py` (a usage error naming `param_hint`), the WebUI
in `web/app.py` (a JSON error response carrying the message and candidates).
"""
from __future__ import annotations


class CVBenchError(Exception):
    """Base class for expected, user-facing errors (as opposed to bugs)."""

    def __init__(self, message: str, *, param_hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.param_hint = param_hint


class NotFoundError(CVBenchError):
    """A named entry (run, dataset, config, ...) does not exist.

    `suggestion` is the closest existing name, if any is close enough.
    """

    def __init__(
        self,
        message: str,
        *,
        param_hint: str | None = None,
        suggestion: str | None = None,
        candidates: list[str] | None = None,
    ) -> None:
        super().__init__(message, param_hint=param_hint)
        self.suggestion = suggestion
        self.candidates = candidates or ([suggestion] if suggestion else [])


class AmbiguousNameError(CVBenchError):
    """A name prefix matches more than one entry; `candidates` lists them."""

    def __init__(self, message: str, *, candidates: list[str], param_hint: str | None = None) -> None:
        super().__init__(message, param_hint=param_hint)
        self.candidates = candidates


class WrongKindError(CVBenchError):
    """The entry exists but is the wrong kind (e.g. a sweep where a single run is expected)."""
