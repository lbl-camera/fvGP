"""Top-level package for fvGP."""

__author__ = """Marcus Michael Noack"""
__email__ = 'MarcusNoack@lbl.gov'

from . import _version
from loguru import logger
import sys
from .gp import GP
from .fvgp import fvGP

__all__ = ['GP', 'fvGP']
__version__ = _version.get_versions()['version']

logger.disable('fvgp')
