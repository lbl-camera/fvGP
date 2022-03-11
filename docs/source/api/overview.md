# Overview

fvGP is an API for flexible HPC single and multi-task Gaussian processes.

The [GP](GP.md) class is the core of most the functionality in fvGP
and therefore in [gpCAM](https://gpcam.readthedocs.io)

fvGP can use [HGDL](https://hgdl.readthedocs.io) optimization on supercomputers for advanced GP use cases.

The [fvGP](fvGP.md) class is a multi-output GP framework
which inherits most of its functionality from the [GP](GP.md) base class.

In the near future, new training engines will be offered, even more specialized
to big-data and many-hyperparameter problems, allowing for ever-more powerful
GPs to be deployed by the users.
