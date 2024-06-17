"""Top-level package for fvGP."""

__author__ = """Marcus Michael Noack"""
__email__ = 'MarcusNoack@lbl.gov'

try:
    from ._version import __version__
except (ImportError, ModuleNotFoundError) as ex:
    raise RuntimeError('Running fvgp from source code requires installation. If you would like an editable source '
                       'install, use "pip install -e ." to perform and editable installation.') from ex

from loguru import logger
import sys
from .gp import GP
from .fvgp import fvGP
from .gpMCMC import gpMCMC

__all__ = ['GP', 'fvGP', 'gpMCMC']

logger.disable('fvgp')
