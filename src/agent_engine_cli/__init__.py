"""Agent Engine CLI package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agent-engine-cli")
except PackageNotFoundError:
    __version__ = "unknown"
