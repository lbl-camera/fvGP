"""Distributed assembly of sparse covariance matrices for gp2Scale.

Every kernel evaluation gp2Scale distributes goes through
:py:func:`distributed_covariance`: the symmetric prior covariance, the rectangular and
symmetric blocks needed when data is appended, and the training-set cross-covariance the
posterior needs.  They differ only in whether the two point sets are the same
(``symmetric``), which is a flag rather than a separate code path.

Two ways of cutting the work across the cluster are available, chosen with
``distribution``:

``"blockwise"``
    Tasks are (row block, column block) pairs.  When ``symmetric`` only the upper
    triangle is scheduled, so the cluster does half the kernel evaluations, and the host
    mirrors the result.  This is the historical behavior and remains the default.

``"rowwise"``
    Tasks are row strips, and each worker returns a *finished* CSR strip.  The COO-to-CSR
    sort therefore happens in parallel on the workers, and host assembly is a plain
    ``vstack`` -- a concatenation of ``data``/``indices`` with an offset ``indptr``, with
    no global COO and no mirroring.  Symmetry cannot be exploited, so the cluster does
    twice the kernel evaluations; in exchange the host stops being the bottleneck and its
    peak memory drops to the finished matrix plus one strip.

Callers own their scatter futures.  Nothing here scatters or releases, which keeps the
scatter-lifecycle rules in :py:class:`fvgp.gp_prior.GPprior` where they are documented.
"""

import itertools
import time
from functools import partial

import dask.distributed as distributed
import numpy as np
import scipy.sparse as sparse
from loguru import logger

_DISTRIBUTIONS = ("blockwise", "rowwise")


def ranges(N, nb):
    """Split ``range(N)`` into ``nb`` chunks given as ``(start, end)`` tuples."""
    if nb == 0: nb = 1
    step = N / nb
    return [(round(step * i), round(step * (i + 1))) for i in range(nb)]


def num_blocks(n, batch_size):
    """Number of chunks ``n`` points are cut into at ``batch_size`` points per chunk."""
    return max(1, n // batch_size)


def index_dtype_for(n1, n2):
    """int32 indices whenever the matrix is small enough for them.

    Halves both the bytes serialized back from every worker and the size of the host
    index arrays, which for a matrix with billions of nonzeros is the difference between
    fitting in memory and not.
    """
    return np.int32 if max(n1, n2) < 2 ** 31 else np.int64


##########################################################################
###################### worker-side functions #############################
##########################################################################
def evaluate_kernel(kernel, x1, x2, hyperparameters, k_n_params, args):
    """Call the kernel with whichever signature it declares.

    Mirrors :py:meth:`fvgp.gp_prior.GPprior.compute_covariances`.  The historical
    gp2Scale workers called ``kernel(x1, x2, hps)`` unconditionally, so a four-argument
    ``args``-taking kernel -- supported everywhere else -- raised ``TypeError`` on the
    worker.
    """
    if k_n_params == 4:
        return kernel(x1, x2, hyperparameters, args)
    elif k_n_params == 3:
        return kernel(x1, x2, hyperparameters)
    else:
        raise Exception("No valid kernel function signature")


def block_to_coo(k, index_dtype):
    """A dense or sparse kernel block as ``(data, rows, cols)`` in block-local indices.

    A kernel that is already support-aware (see
    :py:func:`fvgp.kernels.wendland_anisotropic_gp2Scale_cpu_sparse`) hands us a sparse
    block, and is passed straight through rather than round-tripped via dense.
    """
    if sparse.issparse(k):
        k = k.tocoo()
        return k.data, k.row.astype(index_dtype, copy=False), k.col.astype(index_dtype, copy=False)
    k = np.asarray(k)
    rows, cols = np.nonzero(k)
    return k[rows, cols], rows.astype(index_dtype, copy=False), cols.astype(index_dtype, copy=False)


def block_triplets(range_ij, x1, x2, hyperparameters, kernel,
                   k_n_params, args, symmetric, index_dtype):
    """COO triplets of one block, in *global* matrix coordinates.

    ``x1``/``x2`` arrive as the materialized values of the caller's scatter futures.
    Returning triplets rather than a sparse block keeps the wire format independent of
    where the block sits in the matrix.
    """
    (i_start, i_end), (j_start, j_end) = range_ij
    k = evaluate_kernel(kernel, x1[i_start:i_end], x2[j_start:j_end],
                        hyperparameters, k_n_params, args)
    data, rows, cols = block_to_coo(k, index_dtype)

    # Blocks straddling the diagonal of a symmetric matrix are computed once and mirrored
    # by the host, so only their upper triangle may be reported.
    if symmetric and i_start == j_start and data.size:
        mask = rows <= cols
        data, rows, cols = data[mask], rows[mask], cols[mask]

    return data, rows + index_dtype(i_start), cols + index_dtype(j_start)


def row_strip_csr(range_i, x1, x2, hyperparameters, kernel,
                  k_n_params, args, n2, col_batch_size, index_dtype):
    """One finished CSR row strip, tagged with its first row index.

    The strip is evaluated in column chunks so peak worker memory stays at a single dense
    block, and converted to CSR here so the sort is done by the workers in parallel
    rather than by the host on the assembled whole.
    """
    i_start, i_end = range_i
    x1_block = x1[i_start:i_end]
    data_parts, row_parts, col_parts = [], [], []

    for j_start, j_end in ranges(n2, num_blocks(n2, col_batch_size)):
        k = evaluate_kernel(kernel, x1_block, x2[j_start:j_end],
                            hyperparameters, k_n_params, args)
        data, rows, cols = block_to_coo(k, index_dtype)
        if data.size == 0: continue
        data_parts.append(data)
        row_parts.append(rows)
        col_parts.append(cols + index_dtype(j_start))

    shape = (i_end - i_start, n2)
    if not data_parts:
        return i_start, sparse.csr_matrix(shape)

    strip = sparse.coo_matrix((np.concatenate(data_parts),
                               (np.concatenate(row_parts), np.concatenate(col_parts))),
                              shape=shape)
    return i_start, strip.tocsr()


##########################################################################
###################### host-side assembly ################################
##########################################################################
def _harvest(future_result):
    """Take a result off the wire and drop the client's reference to its future.

    A cancelled task is handed back *as* its exception rather than raised, so without
    this check a scheduler-side cancellation would flow into the assembly and fail there
    as an unrelated shape or type error, far from its cause.
    """
    future, result = future_result
    future.release()
    if isinstance(result, BaseException):
        raise Exception(
            f"A gp2Scale covariance block failed on the cluster: "
            f"{type(result).__name__}: {result}") from result
    return result


def assemble_triplets(harvest, n1, n2, symmetric, index_dtype):
    """Assemble global COO triplets into CSR with a single allocation.

    The parts are sized first, then copied into one preallocated set of arrays, each part
    being written together with its mirror image and dropped immediately afterwards.  COO
    does not care about ordering, so interleaving a block and its mirror costs nothing
    and lets the gathered results be freed as we go.  The alternative -- ``np.hstack``
    over every block and then a second ``np.hstack`` for the mirrored half -- holds three
    to four copies of the matrix at peak.
    """
    parts, total, dtypes = [], 0, []
    for data, rows, cols in harvest:
        if data.size == 0: continue
        parts.append((data, rows, cols))
        dtypes.append(data.dtype)
        total += data.size
        # Only entries off the diagonal get mirrored.  Diagonal entries exist solely in
        # blocks that straddle the diagonal, so this counts zero for every other block.
        if symmetric: total += data.size - int(np.count_nonzero(rows == cols))

    if not parts:
        return sparse.csr_matrix((n1, n2))

    out_data = np.empty(total, dtype=np.result_type(*dtypes))
    out_rows = np.empty(total, dtype=index_dtype)
    out_cols = np.empty(total, dtype=index_dtype)

    position = 0
    while parts:
        data, rows, cols = parts.pop()
        n = data.size
        out_data[position:position + n] = data
        out_rows[position:position + n] = rows
        out_cols[position:position + n] = cols
        position += n
        if symmetric:
            off_diagonal = rows != cols
            n = int(np.count_nonzero(off_diagonal))
            if n:
                out_data[position:position + n] = data[off_diagonal]
                out_rows[position:position + n] = cols[off_diagonal]
                out_cols[position:position + n] = rows[off_diagonal]
                position += n
        del data, rows, cols

    K = sparse.coo_matrix((out_data, (out_rows, out_cols)), shape=(n1, n2))
    del out_data, out_rows, out_cols
    return K.tocsr()


def assemble_row_strips(harvest, n1, n2):
    """Assemble finished CSR row strips in row order."""
    strips = dict(harvest)
    if not strips:
        return sparse.csr_matrix((n1, n2))
    return sparse.vstack([strips[key] for key in sorted(strips)], format="csr")


##########################################################################
###################### the single entry point ############################
##########################################################################
def distributed_covariance(client, kernel, hyperparameters,
                           x1_future, n1, x2_future, n2,
                           batch_size, symmetric=False, distribution="blockwise",
                           k_n_params=3, args=None):
    """Compute ``k(x1, x2)`` across a dask cluster and return it as CSR.

    Parameters
    ----------
    client : distributed.Client
        The cluster the kernel blocks are mapped over.
    kernel : callable
        ``f(x1, x2, hyperparameters)`` or ``f(x1, x2, hyperparameters, args)``, selected
        by ``k_n_params``.  May return a dense array or a sparse block.
    hyperparameters : np.ndarray
    x1_future, x2_future : distributed.Future
        Scattered point sets.  For ``symmetric=True`` these must be the same future, so
        that the workers slice one broadcast copy.
    n1, n2 : int
        Number of points behind each future; the shape of the result.
    batch_size : int
        Target points per chunk along each axis.
    symmetric : bool
        Whether the result is ``k(x, x)``, which lets ``"blockwise"`` schedule only the
        upper triangle.
    distribution : str
        ``"blockwise"`` or ``"rowwise"``; see the module docstring.
    k_n_params : int
        3 or 4, the kernel's arity.
    args : dict or None
        Passed to a four-argument kernel.

    Return
    ------
    Covariance matrix : scipy.sparse.csr_matrix
    """
    if distribution not in _DISTRIBUTIONS:
        raise Exception(f"Unknown gp2Scale distribution `{distribution}`. "
                        f"Choose from: {list(_DISTRIBUTIONS)}")
    if symmetric:
        assert n1 == n2, "a symmetric covariance must be square"
        assert x1_future is x2_future, \
            "a symmetric covariance must be computed from a single scattered point set"

    st = time.time()
    index_dtype = index_dtype_for(n1, n2)
    logger.debug("gp2Scale covariance ({}, symmetric={}) on client {}",
                 distribution, symmetric, client.id)

    if distribution == "blockwise":
        row_ranges = ranges(n1, num_blocks(n1, batch_size))
        col_ranges = row_ranges if symmetric else ranges(n2, num_blocks(n2, batch_size))
        tasks = list(itertools.product(row_ranges, col_ranges))
        # filter the lower triangle; the host mirrors instead
        if symmetric: tasks = [task for task in tasks if task[0][0] <= task[1][0]]
        worker = partial(block_triplets,
                         hyperparameters=hyperparameters, kernel=kernel,
                         k_n_params=k_n_params, args=args,
                         symmetric=symmetric, index_dtype=index_dtype)
    else:
        tasks = ranges(n1, num_blocks(n1, batch_size))
        worker = partial(row_strip_csr,
                         hyperparameters=hyperparameters, kernel=kernel,
                         k_n_params=k_n_params, args=args, n2=n2,
                         col_batch_size=batch_size, index_dtype=index_dtype)

    logger.debug("        gp2Scale covariance init done after {} seconds ({} tasks).",
                 time.time() - st, len(tasks))

    futures = client.map(worker, tasks, [x1_future] * len(tasks), [x2_future] * len(tasks))
    harvest = map(_harvest, distributed.as_completed(futures, with_results=True))

    if distribution == "blockwise":
        K = assemble_triplets(harvest, n1, n2, symmetric, index_dtype)
    else:
        K = assemble_row_strips(harvest, n1, n2)

    logger.debug("        gp2Scale covariance assembled after {} seconds.", time.time() - st)
    logger.debug("        gp2Scale covariance sparsity = {}.", float(K.nnz) / float(n1 * n2))
    return K


def stack_augmented_covariance(K, B, D):
    """Assemble ``[[K, B], [B.T, D]]`` the way scipy can do fastest.

    ``scipy.sparse`` has a fast path for block assembly when *every* block is already
    CSR: it concatenates along each axis instead of converting the lot to COO and
    rebuilding.  ``B.T`` is CSC and a freshly built block is COO, so without the explicit
    conversions here the whole augmented matrix takes the slow path.
    """
    K, B, D = _as_csr(K), _as_csr(B), _as_csr(D)
    return sparse.block_array([[K, B], [_as_csr(B.transpose()), D]], format="csr")


def _as_csr(matrix):
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)
