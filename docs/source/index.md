```{toctree}
:hidden:
:maxdepth: 1

api/overview.md
examples/index.md
```

# fvGP — A Flexible Multi-Task GP Engine

fvGP is a next-generation Gaussian (and Gaussian-related) process engine for flexible,
domain-informed and HPC-ready stochastic function approximation.
It is the backbone of the [gpCAM](https://gpcam.readthedocs.io) API.

The objective of fvGP is to handle the mathematics behind GP training and predictions
while giving the user maximum flexibility in defining kernels, mean functions, and noise models.
The *fv* in fvGP stands for *function-valued* — an extension of multi-task GPs by the notion
of an output space with its own topology, whose metric can be learned via hyperparameter
optimization. [HGDL](https://hgdl.readthedocs.io) provides distributed multi-node asynchronous
constrained optimization for training.

The fvGP package holds the world record for scaling up exact GPs!

## See Also

* [gpCAM](https://gpcam.readthedocs.io)
* [HGDL](https://hgdl.readthedocs.io)
