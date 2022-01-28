# Overview

The [GP](GP.md) class is the core of almost all the functionality in fvGP
and therefore [gpCAM](https://gpcam.readthedocs.io)
It is the base class that is used for multi-task GPs and also for ensemble GPs.

The [gpHGDL](gpHGDL.md) class is based on the gp class but is optimized (torch, autograd, robust kernels) for local
and [HGDL](https://hgdl.readthedocs.io) optimization on supercomputers

The [fvGP](fvGP.md) class is a multi-output GP Framework
which inherits most of its functionality from the [GP](GP.md) base class.

The [EnsembleGP](EnsembleGP.md) class is for ensemble GPs.
Note hyperparameters are now an instance of the hyperparameters class.
See the examples in ./tests/ for more info. 

The [EnsembleFvGP](EnsembleFvGP.md) class is for ensemble multi-task GPs.