# Future Work Plan: Exact Sparse Linear Algebra for SLE Kernels in gp2Scale

*Draft, October 27, 2026*

## 1. Objective

The sparse landmark embedding (SLE) produces a sparse embedding, but the SLE covariance matrix with a stationary inner kernel (RBF, Matérn) is dense. This has prevented SLE from using gp2Scale. A second obstacle is that the support radius of the bump function is a hyperparameter, so the sparsity pattern changes during training.

This plan describes an exact, model-preserving decomposition that resolves both obstacles:

$$K_{\mathrm{SLE}} = S + U U^\top,$$

where $S$ is sparse and positive semi-definite (PSD), with a sparsity pattern fixed by landmark overlap, and $U U^\top$ is PSD with low rank. For the RBF inner kernel, the rank is exactly one. For the Matérn inner kernel, the rank is small, and a nonnegative exponential fit makes the factorization exact to within $10^{-9}$ in the tests below. The GP posterior, marginal likelihood, and gradients then require only solves and the log-determinant of the sparse matrix $S + \sigma_n^2 I$, which is the computation gp2Scale already performs. The low-rank part enters through a $Q \times Q$ Woodbury correction, where $Q$ is the rank of $U$.

## 2. Notation

| Symbol | Meaning |
|---|---|
| $\mathcal{X}$, $\rho$ | Input space and its distance measure (graphs, distributions, manifolds); $\rho$ need not be conditionally negative definite (CND) |
| $X = \{x_1,\dots,x_n\}$ | Training inputs; all of them serve as landmarks |
| $\mathbf{y} \in \mathbb{R}^n$ | Observations |
| $\sigma_n^2$ | Noise variance |
| $b(d; a, r, \beta)$ | Bump function, $a\exp\!\big(\beta - \beta/(1 - d^2/r^2)\big)$ for $d < r$, zero otherwise; $a>0$ amplitude, $r>0$ support radius, $\beta>0$ flatness |
| $\boldsymbol{\phi}(x)$ | SLE embedding, $\big[b(\rho(x,x_1)),\dots,b(\rho(x,x_n))\big]^\top \in \mathbb{R}^n$ |
| $\Phi$ | $n \times n$ matrix with rows $\boldsymbol{\phi}(x_i)^\top$; sparse |
| $q(x)$ | Squared embedding norm, $\lVert\boldsymbol{\phi}(x)\rVert^2$ |
| $\kappa$ | Inner kernel acting on the embedding, $k_{\mathrm{SLE}}(x,x') = \kappa\big(\lVert\boldsymbol{\phi}(x) - \boldsymbol{\phi}(x')\rVert\big)$ |
| $\sigma^2$, $\ell$ | Signal variance and length scale of $\kappa$ |
| $S$ | Sparse PSD part of the Gram matrix; $s(x,x')$ is its kernel |
| $U$ | $n \times Q$ low-rank factor; $\mathbf{u}(x) \in \mathbb{R}^Q$ is its row for input $x$ |
| $A$ | $S + \sigma_n^2 I$, sparse system matrix |
| $C$ | $K_{\mathrm{SLE}} + \sigma_n^2 I = A + UU^\top$, dense full covariance |
| $x_*$, $\mathbf{s}_*$ | Test input and sparse cross-covariance vector $[s(x_*,x_1),\dots,s(x_*,x_n)]^\top$ |
| $\circ$ | Entrywise (Hadamard) operations |

## 3. Results Established So Far

All numerical checks used shortest-path distances on a connected random geometric graph with 388 nodes. On this metric the double-centered squared distance matrix has a smallest eigenvalue of $-655$, so the metric is strongly non-Euclidean.

### 3.1 Background: compact support without dimension

A radial function $\varphi(\rho)$ is positive definite on $(\mathcal{X},\rho)$ only if it maps every distance matrix realizable in $\mathcal{X}$ to a PSD matrix.

- **Euclidean case.** In $\mathbb{R}^d$, the set of realizable distance matrices grows with $d$, and Wendland functions are positive definite only up to a design dimension.
- **Spaces that contain all finite Euclidean configurations.** Examples include $W_2$ on distributions over $\mathbb{R}$ and $L^2$ function spaces. Here no compactly supported $\varphi(\rho)$ is positive definite, by Schoenberg's characterization of functions positive definite on every $\mathbb{R}^d$.
- **Non-CND metrics.** Even the Gaussian $\varphi$ can fail.

A dimension-free, compactly supported, PSD kernel is obtained by convolution over landmarks:

$$k_{\mathrm{lin}}(x,x') = \boldsymbol{\phi}(x)^\top\boldsymbol{\phi}(x').$$

- This is SLE with a linear inner kernel. It is PSD for any $\rho$.
- It vanishes whenever $\rho(x,x') \ge 2r$, by the triangle inequality.

**Check.** With 80 landmarks, a Wendland profile, and support radius 4, the plain Wendland function of $\rho$ gives a smallest eigenvalue of $-0.67$. The landmark kernel gives $-6\times10^{-15}$, with 21% nonzero entries and support at most $6 \le 2r = 8$.

### 3.2 SLE-RBF: sparse plus rank one

Expanding the squared norm gives

$$k_{\mathrm{SLE\text{-}RBF}}(x,x') = \sigma^2 g(x) g(x') \exp\!\big(\boldsymbol{\phi}(x)^\top\boldsymbol{\phi}(x')/\ell^2\big), \qquad g(x) = e^{-q(x)/(2\ell^2)}.$$

Subtracting the constant term of the exponential yields

$$K = \sigma^2\mathbf{g}\mathbf{g}^\top + S, \qquad S = \sigma^2\operatorname{diag}(\mathbf{g})\big[\exp^{\circ}(\Phi\Phi^\top/\ell^2) - \mathbf{1}\mathbf{1}^\top\big]\operatorname{diag}(\mathbf{g}),$$

where $\mathbf{g} = [g(x_1),\dots,g(x_n)]^\top$ and $\mathbf{1}$ is the all-ones vector.

- **$S$ is PSD.** $\exp^{\circ}(M) - \mathbf{1}\mathbf{1}^\top = \sum_{p\ge1} M^{\circ p}/p!$, where $M^{\circ p}$ is the $p$-fold entrywise power. Each term is PSD by the Schur product theorem.
- **Sparsity pattern.** $S_{ij} \ne 0$ exactly when $(\Phi\Phi^\top)_{ij} > 0$, that is, when $x_i$ and $x_j$ share a landmark within radius $r$. This implies $\rho(x_i,x_j) < 2r$.
- **Check.** $S$ has 14.4% nonzero entries. The reconstruction error is $8\times10^{-16}$, and the smallest eigenvalue of $S$ is $0.0061$.

### 3.3 SLE-Matérn: sparse plus low rank

This corrects an earlier assessment that the Matérn inner kernel admits no exact split.

**Mixture representation.** Every inner kernel that is positive definite on every $\mathbb{R}^m$ is a Gaussian scale mixture, $\kappa(d) = \int_0^\infty e^{-t d^2}\,\mu(dt)$, with a nonnegative measure $\mu$ (Schoenberg, 1938). This includes the full Matérn family and the RBF kernel.

**Decomposition.** Substituting $d^2 = q(x) + q(x') - 2\boldsymbol{\phi}(x)^\top\boldsymbol{\phi}(x')$ gives

$$k_{\mathrm{SLE}}(x,x') = \underbrace{\int e^{-t q(x)} e^{-t q(x')}\,\mu(dt)}_{F(x,x') \,=\, \kappa\left(\sqrt{q(x)+q(x')}\right)} + \underbrace{\int e^{-t q(x)} e^{-t q(x')}\big(e^{2t\,\boldsymbol{\phi}(x)^\top\boldsymbol{\phi}(x')} - 1\big)\,\mu(dt)}_{s(x,x')}.$$

**Properties of the far-field part $F$.**

- It is PSD, as a nonnegative mixture of rank-one kernels.
- It depends on the inputs only through the scalars $q(x)$ and $q(x')$.
- It is numerically low rank.

**Properties of the remainder $s$.**

- It is PSD, by the same Schur argument applied for each $t$.
- It has the same sparsity pattern as in Section 3.2.

**Explicit low-rank factor.** A nonnegative fit $\kappa(\sqrt{z}) \approx \sum_{j=1}^{Q} \omega_j e^{-t_j z}$, with weights $\omega_j \ge 0$ and decay rates $t_j > 0$, yields

$$\mathbf{u}(x) = \big[\sqrt{\omega_1}e^{-t_1 q(x)},\dots,\sqrt{\omega_Q}e^{-t_Q q(x)}\big]^\top.$$

This factor is available at test points, and $S + UU^\top$ stays PSD by construction.

**Check (Matérn $\nu = 3/2$, $\ell = 1.5$).**

- The remainder is PSD, with smallest eigenvalue 0.0162.
- Its sparsity pattern is identical to that of $\Phi\Phi^\top > 0$.
- The eigenvalues of $F$ decay quickly. The relative Frobenius error of the best rank-$Q$ approximation is $2.4\times10^{-2}$ for $Q = 1$, $9.5\times10^{-4}$ for $Q = 2$, $1.9\times10^{-6}$ for $Q = 4$, and $1.9\times10^{-9}$ for $Q = 6$.
- A nonnegative least-squares (NNLS) fit on a fixed grid of 60 rates kept $Q = 21$ terms and reproduced the exact SLE-Matérn Gram matrix to $1.3\times10^{-9}$.

### 3.4 Exact posterior and marginal likelihood

The prior is $f = f_S + \mathbf{u}(\cdot)^\top\mathbf{w}$, with $f_S \sim \mathcal{GP}(0,s)$ and $\mathbf{w} \sim \mathcal{N}(0, I_Q)$ independent.

Define:
- $\boldsymbol{\alpha} = A^{-1}\mathbf{y}$ and $P = A^{-1}U$, which take $Q+1$ sparse solves with one factorization;
- $M = I_Q + U^\top P$;
- $\boldsymbol{\eta} = U^\top\boldsymbol{\alpha}$ and $\hat{\mathbf{w}} = M^{-1}\boldsymbol{\eta}$, the posterior mean of $\mathbf{w}$, whose posterior covariance is $M^{-1}$.

**Marginal likelihood.** By Woodbury and the matrix determinant lemma,

$$\log p(\mathbf{y}) = -\tfrac12\big[\mathbf{y}^\top\boldsymbol{\alpha} - \boldsymbol{\eta}^\top\hat{\mathbf{w}}\big] - \tfrac12\big[\log\det A + \log\det M\big] - \tfrac{n}{2}\log 2\pi.$$

**Prediction.** With the residualized features $\mathbf{h}(x_*) = \mathbf{u}(x_*) - P^\top\mathbf{s}_*$:

$$\mathbb{E}[f(x_*)\mid\mathbf{y}] = \mathbf{s}_*^\top\boldsymbol{\alpha} + \mathbf{h}(x_*)^\top\hat{\mathbf{w}},$$

$$\operatorname{Var}[f(x_*)\mid\mathbf{y}] = s(x_*,x_*) - \mathbf{s}_*^\top A^{-1}\mathbf{s}_* + \mathbf{h}(x_*)^\top M^{-1}\mathbf{h}(x_*).$$

The first terms are the sparse GP posterior that gp2Scale already computes. The low-rank correction uses only $Q$-dimensional algebra.

**Not valid: adding separately conditioned posteriors.** The components are coupled through the data. Per component, the posterior means are $\mathbf{u}(x_*)^\top\hat{\mathbf{w}}$ and $\mathbf{s}_*^\top A^{-1}(\mathbf{y} - U\hat{\mathbf{w}})$, which is the backfitting fixed point in closed form.

**Check (SLE-RBF, 300 training and 88 test nodes).**

| Quantity | Max. absolute error vs. dense posterior |
|---|---|
| Decomposed mean | $7\times10^{-15}$ |
| Decomposed covariance | $2\times10^{-15}$ |
| Naive sum of separate posteriors (mean) | 0.31 (mean magnitude up to 1.15) |

**Check (SLE-Matérn, $Q = 21$, sparse LU for $A$).**

| Quantity | Error vs. dense computation |
|---|---|
| Log marginal likelihood | $6\times10^{-14}$ |
| Posterior mean | $3\times10^{-15}$ |
| Posterior variance | $2\times10^{-15}$ |

### 3.5 Gradients

For a hyperparameter $\theta$, with $\partial_\theta$ denoting differentiation with respect to $\theta$:

$$\partial_\theta(U^\top A^{-1}U) = (\partial_\theta U)^\top P + P^\top(\partial_\theta U) - P^\top(\partial_\theta S)P,$$

$$\partial_\theta(U^\top A^{-1}\mathbf{y}) = (\partial_\theta U)^\top\boldsymbol{\alpha} - P^\top(\partial_\theta S)\boldsymbol{\alpha}.$$

The terms $\partial_\theta(\mathbf{y}^\top\boldsymbol{\alpha}) = -\boldsymbol{\alpha}^\top(\partial_\theta S)\boldsymbol{\alpha}$ and $\partial_\theta\log\det A = \operatorname{tr}(A^{-1}\partial_\theta S)$ are the gradients the sparse GP already computes.

The additional cost is $Q+1$ sparse matrix-vector products with $\partial_\theta S$, which shares the sparsity pattern of $S$. No solves are needed beyond the $Q+1$ from the forward pass.

### 3.6 The support radius and the sparsity pattern

- **Pattern-preserving parameters.** The amplitude $a$, the flatness $\beta$, the length scale $\ell$, and the signal variance $\sigma^2$ change the values of $S$ but not its pattern.
- **Smoothness in $r$.** For fixed $d$, $b(d;a,r,\beta)$ is $C^\infty$ in $r$, and it tends to zero with all derivatives as $r \downarrow d$. Entries enter and leave the pattern smoothly, so the marginal likelihood has no discontinuity in $r$.
- **Fixed allocation.** Allocating the sparse structure once at the upper bound $r_{\max}$ of the search range fixes the structure for the entire optimization.

## 4. Work Packages

### WP1. Interface audit of gp2Scale

Confirm that the gp2Scale code path exposes the following, and add thin hooks where it does not:

1. a solve with $A$ for an arbitrary block right-hand side;
2. $\log\det A$ and its gradient;
3. user-supplied sparse kernel blocks.

If gp2Scale uses an iterative solver and a stochastic log-determinant on a given path, determine whether a block solver handles the $Q+1$ right-hand sides at the cost of roughly one solve.

**Deliverable:** a short interface note.

### WP2. Sparse kernel assembly

- Compute the pattern of $\Phi\Phi^\top$ at $r_{\max}$ from landmark neighborhoods, using distances below $r_{\max}$ only.
- Assemble $S$ blockwise on this fixed pattern for the RBF and the Matérn remainder kernel.
- Assemble $\partial_\theta S$ on the same pattern.

**Acceptance:** agreement with dense assembly to machine precision on test problems.

### WP3. Low-rank factor

| Inner kernel | Factor |
|---|---|
| RBF | $U = \sigma\mathbf{g}$, $Q = 1$, closed form |
| Matérn | Nonnegative exponential fit of $\kappa(\sqrt{z})$ on $z \in [0, 2\max_x q(x)]$ |

Tasks for the Matérn factor:

- Reduce $Q$ from the NNLS value of 21 toward the 4–6 terms the eigenvalue decay indicates. Candidate methods are sparse NNLS, greedy rate selection, and Gauss-type quadrature of $\mu$.
- Parameterize the rates as $t_j = \tau_j/\ell^2$, so that a single fit for $\ell = 1$ serves all length scales.
- Record the uniform fit error, which bounds the deviation from exact SLE-Matérn.

**Acceptance:** Gram-matrix error below $10^{-8}$ and $Q \le 8$ for $\nu \in \{1/2, 3/2, 5/2\}$.

### WP4. Woodbury wrapper in gpCAM

Implement Section 3.4 (the prototype is in the Appendix) and the gradients of Section 3.5 as a wrapper around the gp2Scale solver.

**Acceptance:** log marginal likelihood, posterior mean, posterior variance, and gradients match dense computations to $10^{-10}$ relative error up to $n \approx 10^4$.

### WP5. Scaling study on Perlmutter

- Measure the nonzero fraction of $S$, the time per likelihood evaluation, and memory as functions of $n$ and $r$.
- Compare against dense SLE and the earlier masking-kernel construction, which modifies the model; the present split leaves the published SLE kernel unchanged.
- Datasets: the graph and Wasserstein benchmarks from the NeurIPS paper, then a large graph set (molecular graphs) at $n \ge 10^5$.

### WP6. Theory note

Planned contents:

1. A proposition: every SLE kernel whose inner kernel is a Gaussian scale mixture (or a dot-product kernel with nonnegative power-series coefficients) splits into a sparse PSD part supported on landmark overlap and a PSD far-field part that depends only on $q(x)$ and $q(x')$.
2. A bound on the numerical rank of the far-field part, using the smoothness of $\kappa(\sqrt{z})$ on a bounded interval.
3. The exact posterior and likelihood formulas.

**Target:** a section in a follow-up paper on SLE at scale, or the gpCAM/gp2Scale platform paper.

## 5. Open Questions and Risks

- **Density of $S$.** With all training points as landmarks, the pattern of $\Phi\Phi^\top$ reaches neighbors at distance up to $2r$. When the learned $r$ is large relative to the data spacing, $S$ becomes dense. Options: cap $r_{\max}$ by a sparsity budget, or use a landmark subset with weights (Section 3.1) to control the pattern directly.
- **Non-stationary SLE.** The split requires only nonnegative, compactly supported embedding entries. It should carry over to spatially varying bump parameters; this needs verification.
- **Conditioning.** The decomposition does not change the conditioning of $C$. The $Q \times Q$ matrix $M$ is well conditioned by construction, since $M \succeq I_Q$.
- **Test-time cost.** Each prediction requires $\boldsymbol{\phi}(x_*)$, which needs distances from $x_*$ to the landmarks within $r$, plus one sparse solve for the variance. Batched test points share the factorization.
- **Matérn fit accuracy across $\ell$.** The rescaled-rate parameterization must hold uniformly over the hyperparameter bounds.

## 6. References

- Bochner, S. (1933). Monotone Funktionen, Stieltjessche Integrale und harmonische Analyse. *Mathematische Annalen*, 108, 378–410.
- Duvenaud, D., Nickisch, H., and Rasmussen, C. E. (2011). Additive Gaussian processes. *Advances in Neural Information Processing Systems*, 24.
- Gneiting, T. (2002). Compactly supported correlation functions. *Journal of Multivariate Analysis*, 83, 493–508.
- O'Hagan, A. (1978). Curve fitting and optimal design for prediction. *Journal of the Royal Statistical Society, Series B*, 40, 1–42.
- Rasmussen, C. E., and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*, Section 2.7. MIT Press.
- Schoenberg, I. J. (1938). Metric spaces and completely monotone functions. *Annals of Mathematics*, 39, 811–841.
- Wendland, H. (1995). Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree. *Advances in Computational Mathematics*, 4, 389–396.

## Appendix: Prototype Wrapper

Verified against dense computations (Section 3.4). `solve_A` and `logdet_A` come from the sparse solver.

```python
import numpy as np

def fit_sparse_plus_lowrank(solve_A, logdet_A, y, U):
    """Exact GP quantities for covariance K = S + U U^T.
    solve_A(B): returns A^{-1} B with A = S + sigma_n^2 I (sparse); B is (n,) or (n, k).
    logdet_A: log det A.  y: (n,) data.  U: (n, Q) low-rank factor at training points."""
    n, Q = U.shape
    alpha = solve_A(y)                        # A^{-1} y
    P = solve_A(U)                            # A^{-1} U
    M = np.eye(Q) + U.T @ P                   # I_Q + U^T A^{-1} U
    L = np.linalg.cholesky(M)
    eta = U.T @ alpha                         # U^T A^{-1} y
    w_hat = np.linalg.solve(M, eta)           # posterior mean of low-rank weights
    quad = y @ alpha - eta @ w_hat            # y^T C^{-1} y
    logdet_C = logdet_A + 2.0 * np.log(np.diag(L)).sum()
    log_ml = -0.5 * quad - 0.5 * logdet_C - 0.5 * n * np.log(2.0 * np.pi)
    return dict(alpha=alpha, P=P, M=M, w_hat=w_hat, log_ml=log_ml)

def predict_sparse_plus_lowrank(fit, s_star, U_star, s_starstar_diag, solve_A):
    """s_star: (m, n) sparse cross-covariances S(x*, X).
    U_star: (m, Q) low-rank factor at test points.
    s_starstar_diag: (m,) diagonal of S(x*, x*)."""
    H = U_star - s_star @ fit["P"]            # residualized low-rank features
    mean = s_star @ fit["alpha"] + H @ fit["w_hat"]
    var_S = s_starstar_diag - np.einsum("ij,ji->i", s_star, solve_A(s_star.T))
    var = var_S + np.einsum("ij,ji->i", H, np.linalg.solve(fit["M"], H.T))
    return mean, var
```

Low-rank factors at an input $x$, with $q(x) = \lVert\boldsymbol{\phi}(x)\rVert^2$:

- RBF: $\mathbf{u}(x) = \sigma e^{-q(x)/(2\ell^2)}$, so $Q = 1$.
- Matérn: $\mathbf{u}(x) = \big[\sqrt{\omega_j}\,e^{-t_j q(x)}\big]_{j=1}^{Q}$, from the nonnegative fit of $\kappa(\sqrt{z})$.
