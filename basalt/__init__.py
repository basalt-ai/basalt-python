from typing import TYPE_CHECKING, Any

from ._version import __version__

# For static analysis / type checkers, expose Basalt symbol; at runtime we'll lazily import it.
if TYPE_CHECKING:
    from .basalt_facade import BasaltFacade as Basalt  # pragma: no cover
else:
    Basalt: Any = None

# Lazily import BasaltFacade to avoid importing runtime dependencies (like requests)
# during build-time metadata inspection.
__all__ = ["Basalt"]

def __getattr__(name: str):
    if name == "Basalt":
        from .basalt_facade import BasaltFacade as Basalt  # imported only when accessed
        return Basalt
    raise AttributeError(name)
