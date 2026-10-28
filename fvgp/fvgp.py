import warnings
from .gp import GP


class fvGP(GP):
    """
    Deprecated since 4.11, removed in 5.0.0: :py:class:`fvgp.GP` does multi-task GPs itself.
    A ``y_data`` of shape (V, No) with No > 1 makes it one, modeled as ``multi_task_mode``
    says; ``fvGP(x, y, ...)`` is ``GP(x, y, ..., multi_task_mode="cartesian")``, and any other
    ``multi_task_mode`` given to fvGP is passed on the same way. See :py:class:`fvgp.GP` for
    all parameters and methods.

    Unlike fvGP before 4.11, a ``y_data`` with a single column is a single-task GP, and the
    data before the Cartesian transform (``fvgp_x_data``, ``fvgp_y_data``,
    ``fvgp_noise_variances``) is no longer kept: :py:attr:`fvgp.GP.x_data` and the like are
    the stored, transformed data.
    """

    def __init__(
        self,
        x_data,
        y_data,
        init_hyperparameters=None,
        noise_variances=None,
        compute_device="cpu",
        kernel_function=None,
        kernel_function_grad=None,
        noise_function=None,
        noise_function_grad=None,
        prior_mean_function=None,
        prior_mean_function_grad=None,
        gp2Scale=False,
        dask_client=None,
        gp2Scale_batch_size=10000,
        gp2Scale_distribution="blockwise",
        linalg_mode=None,
        gradient_per_direction=False,
        args=None,
        multi_task_mode="cartesian"
    ):
        warnings.warn("fvGP is deprecated and will be removed in fvgp 5.0.0: GP handles multi-task GPs. "
                      f"Use GP(..., multi_task_mode={multi_task_mode!r}).", FutureWarning, stacklevel=2)
        super().__init__(
            x_data,
            y_data,
            init_hyperparameters=init_hyperparameters,
            noise_variances=noise_variances,
            compute_device=compute_device,
            kernel_function=kernel_function,
            kernel_function_grad=kernel_function_grad,
            prior_mean_function=prior_mean_function,
            prior_mean_function_grad=prior_mean_function_grad,
            noise_function=noise_function,
            noise_function_grad=noise_function_grad,
            gp2Scale=gp2Scale,
            dask_client=dask_client,
            gp2Scale_batch_size=gp2Scale_batch_size,
            gp2Scale_distribution=gp2Scale_distribution,
            linalg_mode=linalg_mode,
            gradient_per_direction=gradient_per_direction,
            args=args,
            multi_task_mode=multi_task_mode)
