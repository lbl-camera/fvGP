---
banner: _static/landing.png
banner_height: "40vh"
banner_brightness: .5
---

```{toctree}
---
hidden: true
maxdepth: 2
caption: API
---
api/overview.md
api/GP.md
api/fvGP.md
api/logging.md
```

```{toctree}
---
hidden: true
maxdepth: 2
caption: Examples
---
examples/SingleTaskTest.ipynb
examples/MultiTaskTest.ipynb
examples/gp2ScaleTest.ipynb
examples/NonEuclideanInputSpaces.ipynb
```

# fvGP - A Flexible Multi-Task GP Engine

## fvGP
Welcome to the documentation of the fvGP API.
fvGP is a next-generation Gaussian (and Gaussian-related) process engine for flexible, domain-informed and 
HPC-ready stochastic function approximation. It is the backbone of the [gpCAM](https://gpcam.readthedocs.io) API.
The objective of fvGP is to take care of the mathematics behind GP training and predictions but allow the user to have
maximum flexibility in defining the GP. The fv in fvGP stands for function valued, an extension of multi-task GPs by the notion
of an output space with it's own topology. In this framwork, the output space is assumed to have a non-constant (accoss
input and output space) metric that can be learned via hyperparameter optimization. HGDL provides distributed multi-node asynchronous
constrained function optimization for the training.

## See Also

* [gpCAM](https://gpcam.readthedocs.io)
* [HGDL](https://hgdl.readthedocs.io)
