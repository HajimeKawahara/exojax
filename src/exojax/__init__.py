# src/exojax/__init__.py

"""ExoJAX
Auto-differentiable line-by-line spectral modeling of exoplanets/brown dwarfs using JAX.
"""

from importlib.metadata import version as _version, PackageNotFoundError

__all__ = []

try:
    # Prefer version written by setuptools-scm into ExoJAX_version.py
    from .ExoJAX_version import __version__  # type: ignore
except ImportError:
    try:
        # Fallback: use installed package metadata
        __version__ = _version(__name__)
    except PackageNotFoundError:
        __version__ = "0.0.0"

__uri__ = "http://secondearths.sakura.ne.jp/exojax/"
__author__ = "Hajime Kawahara and collaborators"
__email__ = "divrot@gmail.com"
__license__ = "MIT"
__description__ = (
    "Auto-differentiable line-by-line spectral modeling of exoplanets/"
    "brown dwarfs using JAX."
)
