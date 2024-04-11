"""Top-level package for fvGP."""

__author__ = """Marcus Michael Noack"""
__email__ = 'MarcusNoack@lbl.gov'

from . import _version
from loguru import logger
import sys
from .gp import GP
from .gp2 import GP as GP2
from .fvgp2 import fvGP as fvGP2
from .fvgp import fvGP
from .gpMCMC import gpMCMC

__all__ = ['GP','GP2', 'fvGP', 'fvGP2']
__version__ = _version.get_versions()['version']

logger.disable('fvgp')
