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
    Tasks are row strips: one kernel call per strip against *every* column, returning a
    finished CSR strip.  The COO-to-CSR sort therefore happens in parallel on the
    workers, and host assembly is a plain ``vstack`` -- a concatenation of
    ``data``/``indices`` with an offset ``indptr``, with no global COO and no mirroring.
    Symmetry cannot be exploited, so the cluster does twice the kernel evaluations; in
    exchange the host stops being the bottleneck and its peak memory drops to the
    finished matrix plus one strip.

    Giving the kernel a whole row at once is the point: a support-aware kernel prunes an
    entire row in one pass, and a kernel that vectorizes or offloads gets one large call
    rather than many small ones.  The cost is that a *dense* kernel materializes
    ``strip_width x n2`` values on the worker, so ``batch_size`` -- the strip width here
    -- is what bounds worker memory.

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


def ranges(n, batch_size):
    """Split ``range(n)`` into ``(start, end)`` chunks of at most ``batch_size``.

    ``batch_size`` is a maximum, not a target: every chunk is exactly ``batch_size``
    except the last, which carries the remainder. 100 000 points at 15 000 give six
    chunks of 15 000 and one of 10 000.

    This used to divide ``n`` into ``n // batch_size`` *equal* chunks, which made
    ``batch_size`` a lower bound instead -- 19 999 points at 15 000 came out as a single
    19 999-wide chunk, a 3.2 GB dense block from a setting that reads like a promise of
    15 000. Peak memory per worker now follows the number the user actually set.
    """
    batch_size = max(1, int(batch_size))
    return [(start, min(start + batch_size, n)) for start in range(0, n, batch_size)]


def task_budget(n1, batch_size, distribution):
    """Entries one task is meant to hold, as ``gp2Scale_batch_size`` already implies.

    The user's ``B`` is not the same physical quantity in the two distributions, and this
    is where that becomes explicit:

    * ``"blockwise"`` -- a task is a ``B x B`` block, so ``B * B`` entries.
    * ``"rowwise"``   -- a task is a strip spanning the data, so ``N * B`` entries.

    At ``B = 10000`` and N = 1e6 those are 0.8 GB and 80 GB respectively: the same number
    sanctions 100x different memory depending on the mode. Two users who wrote the same
    ``B`` have not asked for the same thing, which is why the same problem can be
    distributed under one mode and computed locally under the other. Deriving the budget
    per mode is the point -- it respects what each user actually declared.
    """
    batch_size = max(1, int(batch_size))
    if distribution == "blockwise":
        return batch_size * batch_size
    return n1 * batch_size


def should_distribute(n1, n2, batch_size, distribution):
    """Whether a covariance of this shape is worth sending to the cluster at all.

    A result that fits inside one task's RAM budget is cheaper to compute in a single
    kernel call than to schedule: a 10 000 x 2 cross-covariance took 0.83 ms directly
    against 1.8 s spread over 1000 dask tasks of 20 entries each.
    """
    return n1 * n2 > task_budget(n1, batch_size, distribution)


def strip_width(long_axis, short_axis, budget):
    """Width of a cross-covariance strip: as wide as the budget allows, at least 1.

    A strip spans the long axis, so it can never hold fewer than ``long_axis`` entries --
    when the budget is smaller than that (a small ``B`` under ``"blockwise"``) the width
    clamps to 1 and the strip necessarily exceeds the nominal budget. No chunking can do
    better; the alternative is blocking, which multiplies the task count without reducing
    per-task memory, because the long axis has to be traversed either way.
    """
    return int(max(1, min(short_axis, budget // max(long_axis, 1))))


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
                  k_n_params, args, n2, index_dtype):
    """One finished CSR row strip, tagged with its first row index.

    A strip is the whole row: ``k(x1[i_start:i_end], x2)`` in a single kernel call
    against every column, which is what makes this row-wise rather than block-wise with
    a different task shape. The kernel therefore sees one ``(strip_width x n2)`` call and
    can use the full row -- a support-aware kernel prunes the whole row at once, and a
    kernel with its own vectorization or GPU offload gets one large call instead of
    ``n2 / strip_width`` small ones.

    The price is peak worker memory: a *dense* kernel materializes
    ``strip_width x n2`` values here. ``gp2Scale_batch_size`` is the strip width, and is
    the dial for that -- see the note in :py:func:`distributed_covariance`.

    Converted to CSR here so the sort happens on the workers in parallel rather than on
    the host over the assembled whole.
    """
    i_start, i_end = range_i
    k = evaluate_kernel(kernel, x1[i_start:i_end], x2, hyperparameters, k_n_params, args)
    data, rows, cols = block_to_coo(k, index_dtype)

    shape = (i_end - i_start, n2)
    if data.size == 0:
        return i_start, sparse.csr_matrix(shape)
    return i_start, sparse.coo_matrix((data, (rows, cols)), shape=shape).tocsr()



def col_strip_csc(range_j, x1, x2, hyperparameters, kernel,
                  k_n_params, args, n1, index_dtype):
    """One finished CSC column strip, tagged with its first column index.

    The mirror image of :py:func:`row_strip_csr`, for a covariance that is tall and thin:
    ``k(x1, x2[j_start:j_end])`` in a single kernel call spanning every row. Returned as
    **CSC** on purpose -- ``scipy.sparse`` concatenates CSC along columns (and CSR along
    rows) by splicing ``indptr``, and falls back to rebuilding through COO otherwise.
    """
    j_start, j_end = range_j
    k = evaluate_kernel(kernel, x1, x2[j_start:j_end], hyperparameters, k_n_params, args)
    data, rows, cols = block_to_coo(k, index_dtype)

    shape = (n1, j_end - j_start)
    if data.size == 0:
        return j_start, sparse.csc_matrix(shape)
    return j_start, sparse.coo_matrix((data, (rows, cols)), shape=shape).tocsc()


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


def assemble_col_strips(harvest, n1, n2):
    """Assemble finished CSC column strips in column order, returning CSR.

    ``hstack`` of CSC blocks is the fast path (an ``indptr`` splice); the single
    conversion to CSR at the end is what every caller expects back.
    """
    strips = dict(harvest)
    if not strips:
        return sparse.csr_matrix((n1, n2))
    return sparse.hstack([strips[key] for key in sorted(strips)], format="csc").tocsr()


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
        Maximum points per chunk, not a target: every chunk is exactly ``batch_size``
        except the last along each axis, which carries the remainder.

        Its meaning differs by distribution. For ``"blockwise"`` it is the side of a
        square block, so a worker evaluates ``batch_size x batch_size``. For
        ``"rowwise"`` it is the **strip width**: a worker evaluates
        ``batch_size x n2``, the whole row at once. A dense kernel therefore allocates
        ``batch_size * n2`` values per row-wise task, which is the number to size against
        worker memory -- at n2 = 100 000 and a strip width of 10 000 that is 8 GB in
        float64. Sparse (support-aware) kernels never materialize it.

        Row-wise also produces far fewer tasks -- ``n1 / batch_size`` against
        ``(n1 / batch_size)^2 / 2`` for block-wise -- so on a large cluster it usually
        wants a smaller ``batch_size`` than block-wise would.
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

    # Shape decides the strategy; `batch_size` decides the size. A symmetric covariance --
    # the prior, and k(x_new, x_new) on an append -- is computed the way the user asked
    # for. A cross-covariance is tall or thin, and is always cut into strips whatever
    # `distribution` says: blocking it would multiply the task count without reducing
    # per-task memory, since the long axis has to be traversed either way.
    strategy = distribution if symmetric else "stripwise"

    if strategy == "blockwise":
        row_ranges = ranges(n1, batch_size)
        col_ranges = row_ranges if symmetric else ranges(n2, batch_size)
        tasks = list(itertools.product(row_ranges, col_ranges))
        # filter the lower triangle; the host mirrors instead
        if symmetric: tasks = [task for task in tasks if task[0][0] <= task[1][0]]
        worker = partial(block_triplets,
                         hyperparameters=hyperparameters, kernel=kernel,
                         k_n_params=k_n_params, args=args,
                         symmetric=symmetric, index_dtype=index_dtype)
    elif strategy == "rowwise":
        tasks = ranges(n1, batch_size)
        worker = partial(row_strip_csr,
                         hyperparameters=hyperparameters, kernel=kernel,
                         k_n_params=k_n_params, args=args, n2=n2,
                         index_dtype=index_dtype)
    else:
        # split the short axis; each task spans the long one, as wide as the budget allows
        budget = task_budget(n1, batch_size, distribution)
        if n2 <= n1:
            width = strip_width(n1, n2, budget)
            tasks = ranges(n2, width)
            worker = partial(col_strip_csc,
                             hyperparameters=hyperparameters, kernel=kernel,
                             k_n_params=k_n_params, args=args, n1=n1,
                             index_dtype=index_dtype)
        else:
            width = strip_width(n2, n1, budget)
            tasks = ranges(n1, width)
            worker = partial(row_strip_csr,
                             hyperparameters=hyperparameters, kernel=kernel,
                             k_n_params=k_n_params, args=args, n2=n2,
                             index_dtype=index_dtype)
        strategy = "colstrips" if n2 <= n1 else "rowwise"

    logger.debug("        gp2Scale covariance init done after {} seconds ({} tasks).",
                 time.time() - st, len(tasks))

    futures = client.map(worker, tasks, [x1_future] * len(tasks), [x2_future] * len(tasks))
    harvest = map(_harvest, distributed.as_completed(futures, with_results=True))

    if strategy == "blockwise":
        K = assemble_triplets(harvest, n1, n2, symmetric, index_dtype)
    elif strategy == "colstrips":
        K = assemble_col_strips(harvest, n1, n2)
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
