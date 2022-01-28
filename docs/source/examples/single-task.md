---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# FVGP Single Task Notebook

In this notebook we will go through a few features of fvGP.
We will be primarily concerned with regression over a single dimension output and single task.
See the multiple_task_test_notebook.ipynb for single dimension and multiple task example.
The extension to multiple dimensions is straight forward.

## Simple Example

Simple setup and training for fvgp single task.

### Import fvgp and relevant libraries

```{code-cell} ipython3
:tags: [remove-output]

import fvgp
from fvgp import gp
import numpy as np
import matplotlib.pyplot as plt
```

### Defining some input data and testing points

```{code-cell} ipython3
:tags: [remove-output]

def test_data_function(x):
    return np.sin(0.1 * x)+np.cos(0.05 * x)
```

```{code-cell} ipython3
:tags: [remove-output]

x_input = np.linspace(-2*np.pi, 10*np.pi,20)
```

```{code-cell} ipython3
:tags: [remove-output]

y_output = test_data_function(x_input)
```

```{code-cell} ipython3
:tags: [remove-output]

x_input_test = np.linspace(-2*np.pi, 10*np.pi, 100)
```

### Setting up the fvgp single task object

NOTE: The input data need to be given in the form (N x input_space_dim).
The output can either be a N array or N x 1 array where N is the number of data points.
See help(gp.GP) for more information.

```{code-cell} ipython3
:tags: [remove-output]

obj = gp.GP(input_space_dim = 1,
            points = x_input.reshape(-1,1),
            values = y_output.reshape(-1,1),
            init_hyperparameters = np.array([10,10]))
```

### Training our gaussian process regression on given data

```{code-cell} ipython3
:tags: [remove-output]

hyper_param_bounds = np.array([[0.0001, 1000000],[ 0.000001, 100]])
##this will block the main thread, even if you use "hgdl", another option is "global" or "local"
print("Blocking main thread...")
obj.train(hyper_param_bounds, method = "hgdl")
print("free again")
print("this also killed the client")
```

### Looking the posterior mean at the test points

```{code-cell} ipython3
:tags: [remove-output]

post_mean= obj.posterior_mean(x_input_test.reshape(-1,1))
```

```{code-cell} ipython3
plt.plot(x_input_test, post_mean['f(x)'], label='gp interpolation')
plt.scatter(x_input, y_output, label='data')
plt.plot(x_input_test,test_data_function(x_input_test), label = '')
plt.legend()
```

## Training Asynchronously

```{code-cell} ipython3
:tags: [remove-output]

obj = gp.GP(input_space_dim = 1,
            points = x_input.reshape(-1,1),
            values = y_output.reshape(-1,1),
            init_hyperparameters = np.array([10,10]),
            variances = np.zeros(y_output.reshape(-1,1).shape))
```

```{code-cell} ipython3
:tags: [remove-output]

hyper_param_bounds = np.array([[0.0001, 100], [ 0.0001, 100]])
```

```{code-cell} ipython3
:tags: [remove-output]

async_obj = obj.train_async(hyper_param_bounds)
```

### Updating asynchronously

Updates hyperparameters to current optimization values

```{code-cell} ipython3
:tags: [remove-output]

obj.update_hyperparameters(async_obj)
```

### Killing training

```{code-cell} ipython3
:tags: [remove-output]

obj.kill_training(async_obj)
```

### Looking the posterior mean at the test points

```{code-cell} ipython3
:tags: [remove-output]

post_mean= obj.posterior_mean(x_input_test.reshape(-1,1))
```

```{code-cell} ipython3
plt.plot(x_input_test, post_mean['f(x)'], label='interpolation')
plt.plot(x_input_test, test_data_function(x_input_test), label='ground truth')
plt.legend()
```

## Custom Kernels

```{code-cell} ipython3
:tags: [remove-output]

def kernel_l1(x1,x2, hp, obj):
    ################################################################
    ###standard anisotropic kernel in an input space with l1########
    ################################################################
    d1 = abs(np.subtract.outer(x1[:,0],x2[:,0])) 
    return hp[0] * np.exp(-d1/hp[1])
```

```{code-cell} ipython3
:tags: [remove-output]

obj = gp.GP(input_space_dim = 1,
            points = x_input.reshape(-1,1),
            values = y_output.reshape(-1,1),
            init_hyperparameters = np.array([10,10]),
            variances = np.zeros(y_output.reshape(-1,1).shape),
            gp_kernel_function = kernel_l1)
```

### Training our gaussian process regression on given data

```{code-cell} ipython3
:tags: [remove-output]

hyper_param_bounds = np.array([[0.0001, 1000],[ 0.0001, 1000]])
obj.train(hyper_param_bounds)
```

### Looking the posterior mean at the test points

```{code-cell} ipython3
:tags: [remove-output]

post_mean= obj.posterior_mean(x_input_test.reshape(-1,1))
```

```{code-cell} ipython3
plt.plot(x_input_test, post_mean['f(x)'], label='interpolation')
plt.plot(x_input_test, test_data_function(x_input_test), label='ground truth')
plt.legend()
```

## Prior Mean Functions

NOTE: The prior mean function must return a 1d vector, e.g., (100,)

```{code-cell} ipython3
:tags: [remove-output]

def example_mean(gp_obj,x,hyperparameters):
    return np.ones(len(x))
```

```{code-cell} ipython3
:tags: [remove-output]

obj = gp.GP(input_space_dim = 1,
            points = x_input.reshape(-1,1),
            values = y_output.reshape(-1,1),
            init_hyperparameters = np.array([10,10]),
            variances = np.zeros(y_output.reshape(-1,1).shape),
            gp_mean_function = example_mean)
```

### Training our gaussian process regression on given data

```{code-cell} ipython3
:tags: [remove-output]

hyper_param_bounds = np.array([[0.0001, 1000],[ 0.0001, 1000]])
obj.train(hyper_param_bounds)
```

### Looking the posterior mean at the test points

```{code-cell} ipython3
:tags: [remove-output]

post_mean= obj.posterior_mean(x_input_test.reshape(-1,1))
```

```{code-cell} ipython3
plt.plot(x_input_test, post_mean['f(x)'], label='interpolation')
plt.plot(x_input_test, test_data_function(x_input_test), label='ground truth')
plt.scatter(x_input_test,post_mean['f(x)'])
plt.legend()
```
