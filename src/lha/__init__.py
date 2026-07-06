"""lha — a verification-first long-horizon agent harness.

The core is the verification loop (context -> tool -> execute -> verify ->
repair -> checkpoint -> repeat). The live-context layer (code/paper/experiment
search) is infrastructure underneath it, reachable only through
``lha.live_context``.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # single-source the version from the installed package metadata (pyproject)
    __version__ = version("lha")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
