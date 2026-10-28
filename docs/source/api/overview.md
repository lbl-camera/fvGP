# API Reference

```{toctree}
:hidden:
:maxdepth: 1

GP.md
fvGP.md
ggmp.md
kernels.md
gpMCMC.md
logging.md
```

fvGP is an API for flexible HPC single and multi-task Gaussian processes over Euclidean and non-Euclidean (strings, molecules, materials) spaces.

The [GP](GP.md) class presents the core functionality of fvGP
and therefore [gpCAM](https://gpcam.readthedocs.io), for single- and multi-task GPs alike:
the shape of `y_data` decides, and `multi_task_mode` how several outputs are modeled.

fvGP can use [HGDL](https://hgdl.readthedocs.io) optimization on supercomputers for advanced GP use cases.

The [fvGP](fvGP.md) class, the multi-task GP of earlier versions, is deprecated since 4.11
and will be removed in 5.0.0.

The fvGP package holds the world-record for exact scalable GPs. It was run on 5 million data points in 2023.
