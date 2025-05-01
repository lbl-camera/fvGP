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
from .gp_mcmc import gpMCMC, ProposalDistribution



__all__ = ['GP', 'fvGP', 'gpMCMC', 'ProposalDistribution']

logger.disable('fvgp')
