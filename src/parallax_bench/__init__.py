"""parallax-bench — measuring language-induced retrieval displacement in RAG systems."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("parallax-bench")
except PackageNotFoundError:  # running from a checkout without installation
    __version__ = "0.0.0.dev0"
