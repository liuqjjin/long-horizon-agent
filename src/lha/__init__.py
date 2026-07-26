"""LHA runs model-generated task steps behind executable checks.

The main loop reads context, executes a step, runs its checks, then either
repairs the result or saves a checkpoint and advances. Code and document search
are available only through ``lha.live_context``.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # single-source the version from the installed package metadata (pyproject)
    __version__ = version("lha")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
