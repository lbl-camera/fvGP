# fvGP

[![PyPI](https://img.shields.io/pypi/v/fvGP)](https://pypi.org/project/fvgp/)
[![Documentation Status](https://readthedocs.org/projects/fvgp/badge/?version=latest)](https://fvgp.readthedocs.io/en/latest/?badge=latest)
[![fvGP CI](https://github.com/lbl-camera/fvGP/actions/workflows/fvGP-CI.yml/badge.svg)](https://github.com/lbl-camera/fvGP/actions/workflows/fvGP-CI.yml)
[![Codecov](https://img.shields.io/codecov/c/github/lbl-camera/fvGP)](https://app.codecov.io/gh/lbl-camera/fvGP)
[![PyPI - License](https://img.shields.io/badge/license-GPL%20v3-lightgrey)](https://pypi.org/project/fvgp/)
[<img src="https://img.shields.io/badge/slack-@gpCAM-purple.svg?logo=slack">](https://gpCAM.slack.com/)
[![DOI](https://zenodo.org/badge/434769505.svg)](https://zenodo.org/badge/latestdoi/434769505)


Python package for highly flexible function-valued Gaussian processes (fvGP)

It is recommended to use this package via [gpCAM](https://gpcam.lbl.gov/).

Specialties: Extreme-Scale GPs, GPs Tailored for HPC training, Advanced Kernel Designs, Domain-Aware Stochastic Function Approximation

Coming soon: All those capabilities for stochastic manifold learning

fvGP holds the world record for exact large-scale Gaussian Processes!
## Credits

This code was developed with help from Ron Pandolfi (LBNL), Mark Risser (LBNL), Hengrui Luo (Rice U.), and Vardaan Tekriwal (UCB).

Additional nights benefiting fvGP came from across the community, in particular, Kevin Yager, Masafumi Fukuto, and their teams (Brookhaven National Lab)

We acknowledge support from several DOE ASCR, BER, and BES projects, including CAMERA (James Sethian), SPECTRA (Sherry Li), CASCADE (Bill Collins), as well as support directly from Lawrence Berkeley National Laboratory. 

This package uses the HGDL package of David Perryman and Marcus Noack, which
is based on the [HGDN](https://www.sciencedirect.com/science/article/pii/S037704271730225X) algorithm by Noack and Funke.
