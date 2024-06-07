# Overview

fvGP is an API for flexible HPC single and multi-task Gaussian processes over Euclidean and non-Euclidean (strings, molecules, materials) spaces.

The [GP](GP.md) class presents the core functionality of fvGP
and therefore [gpCAM](https://gpcam.readthedocs.io).

fvGP can use [HGDL](https://hgdl.readthedocs.io) optimization on supercomputers for advanced GP use cases.

The [fvGP](fvGP.md) class is a multi-output GP framework
which inherits most of its functionality from the [GP](GP.md) base class.

The fvGP package holds the world-record for exact scalable GPs. It was run on 5 million data points in 2023.
