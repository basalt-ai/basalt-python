from .basalt_facade import BasaltFacade as Basalt
from ._version import __version__

# make only Basalt publicly accessible
__all__ = ["Basalt"]
