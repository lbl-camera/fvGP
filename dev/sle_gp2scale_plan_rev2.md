# Exact Sparse-Plus-Low-Rank Linear Algebra for SLE Kernels in gp2Scale

*Revised draft, October 29, 2026. Supersedes the draft of October 27, 2026.*

## 0. Changes from the October 27 Draft

1. **Sparsity pattern.** The pattern is defined by shared landmarks, which holds for any distance measure. The bound $\rho(x_i,x_j) < 2r$ is now a corollary that requires the triangle inequality.
2. **Non-stationary SLE.** The decomposition holds for every embedding, so spatially varying bump parameters are covered (Corollary 2). The corresponding risk item is closed.
3. **Far-field fit.** Every far-field entry that enters training or prediction has argument $z \ge a^2$, where $a$ is the bump amplitude. The exponential-sum fit is therefore performed once per inner kernel on $[w_{\mathrm{lo}}, \infty)$ in the rescaled variable $w = z/\ell^2$, and one fit serves the entire hyperparameter search. Test-point prior variances use the exact value $\kappa(0)$.
4. **Predictive variance.** The variance formula is restated with the exact prior variance. It requires the fitted factor only through cross-covariances.
5. **Gradients.** The noise-variance gradient and the identity $U^\top C^{-1} = M^{-1}P^\top$ are added, which completes the trace terms.
6. **Density control.** The landmark-subset option with a linear inner kernel is removed, because it changes the model and deflates predictive variance. Density is controlled by a sparsity budget on the upper radius bound $r_{\max}$.
7. **Prior properties.** Section 3.7 records the density interpretation of the far-field term, the behavior at points outside every support, and rank control through the amplitude field.
8. **Rank target.** The target $Q \le 8$ is replaced by a fit-accuracy criterion. Universal fits need $Q \approx 20$–$60$, while the data-dependent numerical rank is about 8. An optional compression step closes this gap.
9. **Implementation.** Section 5 gives a step-by-step implementation guide. Appendix A contains a reference implementation validated against dense computations (Section 4.2).

## 1. Objective

The sparse landmark embedding (SLE) kernel is sparse in its embedding, but its covariance matrix is dense for stationary inner kernels such as RBF and Matérn. This has prevented SLE from running on gp2Scale. The support radius $r$ is also a hyperparameter, so the sparsity pattern changes during training.

This plan describes an exact, model-preserving decomposition that removes both obstacles:

$$K_{\mathrm{SLE}} = S + UU^\top.$$

- $S$ is sparse and positive semi-definite (PSD), with a pattern fixed by landmark overlap.
- $UU^\top$ is PSD with $Q$ columns. $Q = 1$ exactly for the RBF inner kernel. For Matérn kernels a nonnegative exponential fit achieves entrywise error below $10^{-10}\sigma^2$.
- The likelihood, its gradients, and the posterior require solves with, and the log-determinant of, the sparse matrix $S + \sigma_n^2 I$. This is the computation gp2Scale performs. The low-rank part enters through a $Q \times Q$ Woodbury correction.

## 2. Notation

| Symbol | Meaning |
|---|---|
| $\mathcal{X}$, $\rho$ | Input space and its distance measure; $\rho$ need not be conditionally negative definite (CND). We assume $\rho(x,x) = 0$ |
| $X = \{x_1,\dots,x_n\}$ | Training inputs; all of them serve as landmarks |
| $x_*$, $m$ | A test input; number of test inputs |
| $\mathbf{y} \in \mathbb{R}^n$ | Observations |
| $\sigma_n^2$ | Noise variance |
| $b(d; a, r, \beta)$ | Bump function, $a\exp\!\big(\beta - \beta/(1-d^2/r^2)\big)$ for $d < r$, zero otherwise; $a>0$ amplitude, $r>0$ support radius, $\beta>0$ flatness |
| $r_{\max}$ | Upper bound of $r$ during optimization; fixes the sparsity structure |
| $\boldsymbol{\phi}(x)$ | SLE embedding $\big[b(\rho(x,x_1)),\dots,b(\rho(x,x_n))\big]^\top \in \mathbb{R}^n$ |
| $\Phi$ | Sparse $n\times n$ matrix with rows $\boldsymbol{\phi}(x_i)^\top$ |
| $\mathcal{L}_l$ | Members of landmark $l$: training points $x_i$ with $\rho(x_i,x_l) < r_{\max}$ |
| $G = \Phi\Phi^\top$ | Embedding Gram matrix; $G_{ij} = \boldsymbol{\phi}(x_i)^\top\boldsymbol{\phi}(x_j)$ |
| $\mathcal{P}$ | Sparsity pattern: index pairs $(i,j)$ that share a landmark at $r_{\max}$ |
| $T$ | Number of overlap triplets, $T = \sum_l \lvert\mathcal{L}_l\rvert^2$ |
| $q(x)$, $q_i$ | Squared embedding norm $\lVert\boldsymbol{\phi}(x)\rVert^2$; $q_i = G_{ii}$ |
| $\kappa$ | Inner kernel, $k_{\mathrm{SLE}}(x,x') = \kappa\big(\lVert\boldsymbol{\phi}(x)-\boldsymbol{\phi}(x')\rVert\big)$ |
| $\sigma^2$, $\ell$, $\nu$ | Signal variance, length scale, and Matérn smoothness of $\kappa$ |
| $\kappa_1(w)$ | Unit-variance, unit-length-scale profile: $\kappa(d) = \sigma^2\kappa_1(d^2/\ell^2)$; $\kappa_1'$ is its derivative in $w$ |
| $\mu$, $\tau$ | Mixing measure and precision in $\kappa_1(w) = \int_0^\infty e^{-\tau w}\,\mu(d\tau)$ |
| $w_{ij}$, $w^F_{ij}$ | $\lVert\boldsymbol{\phi}(x_i)-\boldsymbol{\phi}(x_j)\rVert^2/\ell^2$ and $(q_i+q_j)/\ell^2$ |
| $F$ | Far-field matrix, $F_{ij} = \sigma^2\kappa_1(w^F_{ij})$ |
| $S$, $s(x,x')$ | Sparse PSD part, $S = K - F$, and its kernel |
| $\omega_j$, $\tau_j$, $Q$ | Weights, rates, and number of terms of the exponential fit of $\kappa_1$ |
| $\varepsilon_{\mathrm{fit}}$, $w_{\mathrm{lo}}$ | Uniform fit error and lower end of the fit domain |
| $U$, $\mathbf{u}(x)$ | $n \times Q$ far-field factor; $\mathbf{u}(x) = \big[\sigma\sqrt{\omega_j}\,e^{-\tau_j q(x)/\ell^2}\big]_{j=1}^Q$ is its row for input $x$ |
| $A$ | $S + \sigma_n^2 I$, sparse system matrix |
| $C$ | $K + \sigma_n^2 I = A + UU^\top$, dense full covariance (never formed) |
| $\boldsymbol{\alpha}$, $P$, $M$ | $A^{-1}\mathbf{y}$, $A^{-1}U$, $I_Q + U^\top P$ |
| $\boldsymbol{\eta}$, $\hat{\mathbf{w}}$, $\boldsymbol{\gamma}$ | $U^\top\boldsymbol{\alpha}$, $M^{-1}\boldsymbol{\eta}$, $C^{-1}\mathbf{y} = \boldsymbol{\alpha} - P\hat{\mathbf{w}}$ |
| $\mathbf{s}_*$, $\mathbf{u}_*$, $\mathbf{h}_*$ | $[s(x_*,x_i)]_{i=1}^n$, $\mathbf{u}(x_*)$, and $\mathbf{u}_* - P^\top\mathbf{s}_*$ |
| $\theta$, $\partial_\theta$ | A hyperparameter and differentiation with respect to it |
| $\circ$ | Entrywise (Hadamard) operations; $G^{\circ k}$ is the $k$-fold entrywise power |

## 3. Theory

### 3.1 Background: compact support without dimension

A radial function $\varphi(\rho)$ is positive definite on $(\mathcal{X},\rho)$ only if it maps every distance matrix realizable in $\mathcal{X}$ to a PSD matrix.

- **Euclidean case.** In $\mathbb{R}^d$ the realizable distance matrices grow with $d$, and Wendland functions are positive definite only up to a design dimension.
- **Spaces containing all finite Euclidean configurations.** Examples are $W_2$ on distributions over $\mathbb{R}$ and $L^2$ function spaces. No compactly supported $\varphi(\rho)$ is positive definite on such spaces, because a function positive definite on every $\mathbb{R}^d$ is a Gaussian scale mixture (Schoenberg, 1938) and is strictly positive.
- **Non-CND distances.** Even the Gaussian $\varphi$ can fail.

The landmark construction $k_{\mathrm{lin}}(x,x') = \boldsymbol{\phi}(x)^\top\boldsymbol{\phi}(x')$ is PSD for any $\rho$ and compactly supported in the landmark sense: it vanishes when no landmark lies within $r$ of both inputs. For a metric $\rho$ this implies $k_{\mathrm{lin}}(x,x') = 0$ whenever $\rho(x,x') \ge 2r$. The rank of $k_{\mathrm{lin}}$ is at most the number of landmarks, and its prior variance $q(x)$ vanishes outside all supports. It serves here as background only and is not used as a model.

### 3.2 The decomposition

**Theorem 1.** Let $\boldsymbol{\phi}: \mathcal{X} \to \mathbb{R}^n$ be any map, and let $\kappa(d) = \sigma^2\kappa_1(d^2/\ell^2)$ with $\kappa_1(w) = \int_0^\infty e^{-\tau w}\mu(d\tau)$ for a finite nonnegative measure $\mu$. Then

$$K = F + S, \qquad F_{ij} = \kappa\big(\sqrt{q_i + q_j}\big), \qquad S_{ij} = \kappa\big(\lVert\boldsymbol{\phi}(x_i)-\boldsymbol{\phi}(x_j)\rVert\big) - F_{ij},$$

where $F$ and $S$ are both PSD and $S_{ij} = 0$ whenever $G_{ij} = 0$.

*Proof.* The squared embedding distance is $q_i + q_j - 2G_{ij}$, so for each $\tau$

$$e^{-\tau(q_i+q_j-2G_{ij})/\ell^2} = g_i\,g_j\,e^{2\tau G_{ij}/\ell^2}, \qquad g_i = e^{-\tau q_i/\ell^2}.$$

Write $e^{2\tau G_{ij}/\ell^2} = 1 + \big(e^{2\tau G_{ij}/\ell^2} - 1\big)$.

- The first term gives the rank-one PSD matrix $\mathbf{g}\mathbf{g}^\top$. Integrating against $\mu$ yields $F$, which is PSD as a nonnegative mixture of PSD matrices.
- The second term is $\operatorname{diag}(\mathbf{g})\big[\sum_{k\ge 1}(2\tau/\ell^2)^k G^{\circ k}/k!\big]\operatorname{diag}(\mathbf{g})$. $G$ is a Gram matrix, so each $G^{\circ k}$ is PSD by the Schur product theorem. Congruence by $\operatorname{diag}(\mathbf{g})$ and integration against $\mu$ preserve PSD, which gives $S$. Every term vanishes where $G_{ij} = 0$. $\square$

The class of inner kernels covered is exactly the class that is positive definite on $\mathbb{R}^n$ for every $n$ (Schoenberg, 1938). SLE requires this class, because the embedding dimension grows with the data. The class contains RBF and the full Matérn family.

**Corollary 1 (RBF).** For $\kappa_1(w) = e^{-w/2}$, $\mu$ is a point mass at $\tau = 1/2$, so $F = \sigma^2\mathbf{g}\mathbf{g}^\top$ with $g_i = e^{-q_i/(2\ell^2)}$, and

$$S = \sigma^2\operatorname{diag}(\mathbf{g})\big[\exp^{\circ}(G/\ell^2) - \mathbf{1}\mathbf{1}^\top\big]\operatorname{diag}(\mathbf{g}).$$

**Corollary 2 (non-stationary SLE).** Theorem 1 uses only that $G$ is a Gram matrix. Spatially varying $a(\cdot)$, $r(\cdot)$, and $\beta(\cdot)$ are therefore covered without change.

**Corollary 3 (pattern).** Bump values are strictly positive on their support, so no cancellation occurs in $G_{ij} = \sum_l \phi_l(x_i)\phi_l(x_j)$. Hence $G_{ij} > 0$ exactly when some landmark $x_l$ satisfies $\rho(x_i,x_l) < r_l$ and $\rho(x_j,x_l) < r_l$, where $r_l$ is the radius of landmark $l$. This characterization holds for any $\rho$. If $\rho$ satisfies the triangle inequality, it implies $\rho(x_i,x_j) < 2\max_l r_l$.

**Corollary 4 (dot-product inner kernels).** For $k(x,x') = \sum_{k\ge0} c_k\,(\boldsymbol{\phi}(x)^\top\boldsymbol{\phi}(x'))^k$ with $c_k \ge 0$, the split is $c_0\mathbf{1}\mathbf{1}^\top$ plus a sparse PSD remainder on the same pattern.

### 3.3 The far-field factor

**Exact case.** For RBF, $U = \sigma\mathbf{g}$ and $Q = 1$.

**Matérn and other mixtures.** Fit $\kappa_1(w) \approx \sum_{j=1}^Q \omega_j e^{-\tau_j w}$ with $\omega_j \ge 0$ and $\tau_j > 0$. Then $\mathbf{u}(x) = \big[\sigma\sqrt{\omega_j}\,e^{-\tau_j q(x)/\ell^2}\big]_j$, and $UU^\top$ is PSD by construction. The rates are expressed in $w = z/\ell^2$, so one fit serves every $\ell$.

**Lemma 1 (fit domain).** Assume $\rho(x,x) = 0$. Every training point is its own landmark, so $\phi_i(x_i) = b(0) = a$ and $q_i \ge a^2$. (With a non-stationary amplitude, $q_i \ge a(x_i)^2$.) Consequently:

- training–training entries of $F$ have $z = q_i + q_j \ge 2a^2$;
- test–training entries have $z = q(x_*) + q_j \ge a^2$, including test points outside every support, where $q(x_*) = 0$.

Only the test prior variance $\kappa(0)$ involves $z$ near zero, and it is known exactly (Section 3.4). The fit is therefore needed on $w \in [w_{\mathrm{lo}}, \infty)$ with

$$w_{\mathrm{lo}} = a_{\min}^2 / \ell_{\max}^2,$$

where $a_{\min}$ and $\ell_{\max}$ are the hyperparameter bounds. One fit per inner kernel and per bound box serves the whole optimization. Beyond the largest grid point $w_{\mathrm{hi}}$, both $\kappa_1$ and the fit are decreasing and below $\varepsilon_{\mathrm{fit}}$. The uniform error on $[w_{\mathrm{lo}},\infty)$ thus follows from the error on a grid over $[w_{\mathrm{lo}}, w_{\mathrm{hi}}]$.

**Model error.** Write $\tilde{K} = S + UU^\top$. Then $\tilde{K} - K = UU^\top - F$, and Lemma 1 gives

$$\max_{ij}\lvert\tilde{K}_{ij} - K_{ij}\rvert \le \sigma^2\varepsilon_{\mathrm{fit}}, \qquad \lVert\tilde{K} - K\rVert_2 \le n\,\sigma^2\varepsilon_{\mathrm{fit}}.$$

$\tilde{K}$ is PSD, since both of its parts are PSD. A sufficient accuracy target is

$$n\,\sigma^2\varepsilon_{\mathrm{fit}} \le 10^{-3}\,\sigma_n^2,$$

which keeps the spectral perturbation three orders below the noise floor. For $n = 10^5$ and $\sigma_n^2/\sigma^2 = 10^{-2}$, this gives $\varepsilon_{\mathrm{fit}} \le 10^{-10}$.

**Measured fit sizes.** The fits use nonnegative least squares (NNLS) on 160 log-spaced rates followed by greedy pruning (Appendix A, `fit_far_field`).

| Tolerance $\varepsilon_{\mathrm{fit}}$ | $\nu$ | $w_{\mathrm{lo}} = 0$ | $10^{-3}$ | $10^{-1}$ | $1$ |
|---|---|---|---|---|---|
| $10^{-8}$ | 1/2 | 101 | 52 | 27 | 22 |
| $10^{-8}$ | 3/2 | 39 | 39 | 27 | 19 |
| $10^{-8}$ | 5/2 | 28 | 29 | 24 | 19 |
| $10^{-10}$ | 1/2 | 107 | 66 | 43 | 27 |
| $10^{-10}$ | 3/2 | 58 | 51 | 32 | 25 |
| $10^{-10}$ | 5/2 | 37 | 45 | 33 | 21 |

Lemma 1 reduces $Q$ substantially for $\nu = 1/2$, whose profile $e^{-\sqrt{w}}$ has a square-root singularity at $w = 0$. Greedy pruning does not minimize $Q$. Gauss-type quadrature of $\mu$ or the exponential-sum methods of Beylkin and Monzón (2005) should lower these counts.

**Optional compression.** On a given dataset, the factor $U$ has much lower numerical rank than $Q$, because the values $q_i$ occupy a narrow range. In Section 4.2, $Q = 30$–$35$ terms gave numerical rank 8 at relative tolerance $10^{-10}$. Compression proceeds as follows:

1. Compute the eigendecomposition $U^\top U = V\Lambda V^\top$, with eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots$, at cost $O(nQ^2)$.
2. Keep the leading $p$ eigenvectors $V_p$ and set $U_p = UV_p$.

The result is PSD, and $\lVert UU^\top - U_pU_p^\top\rVert_2 = \lambda_{p+1}$. The model depends on $V_p$ only through the projector $\Pi = V_pV_p^\top$, so gradients need only the components of $\partial V_p$ outside the retained subspace:

$$\partial\mathbf{v}_k = \sum_{m > p}\mathbf{v}_m\,\frac{\mathbf{v}_m^\top(\partial B)\,\mathbf{v}_k}{\lambda_k - \lambda_m}, \qquad \partial B = (\partial U)^\top U + U^\top\partial U,$$

which requires a spectral gap $\lambda_p > \lambda_{p+1}$. Compression pays off when each right-hand side costs an iterative solve. With a sparse direct solver, the $Q+1$ triangular solves are usually cheap relative to the factorization. The reference implementation does not include compression.

### 3.4 Exact posterior and marginal likelihood

The prior is $f = f_S + \mathbf{u}(\cdot)^\top\mathbf{w}$, with $f_S \sim \mathcal{GP}(0, s)$ and $\mathbf{w} \sim \mathcal{N}(0, I_Q)$ independent. The weights $\mathbf{w}$ have posterior mean $\hat{\mathbf{w}}$ and posterior covariance $M^{-1}$. All of $\boldsymbol{\alpha}$ and $P$ follow from $Q+1$ solves with one factorization of $A$.

**Marginal likelihood.** By Woodbury and the matrix determinant lemma,

$$\log p(\mathbf{y}) = -\tfrac12\big[\mathbf{y}^\top\boldsymbol{\alpha} - \boldsymbol{\eta}^\top\hat{\mathbf{w}}\big] - \tfrac12\big[\log\det A + \log\det M\big] - \tfrac{n}{2}\log 2\pi.$$

**Prediction.**

$$\mathbb{E}[f(x_*)\mid\mathbf{y}] = \mathbf{s}_*^\top\boldsymbol{\alpha} + \mathbf{h}_*^\top\hat{\mathbf{w}},$$

$$\operatorname{Var}[f(x_*)\mid\mathbf{y}] = \kappa(0) - \Big[\mathbf{s}_*^\top A^{-1}\mathbf{s}_* + \mathbf{u}_*^\top\mathbf{u}_* - \mathbf{h}_*^\top M^{-1}\mathbf{h}_*\Big].$$

The bracket equals $\mathbf{k}_*^\top C^{-1}\mathbf{k}_*$ with $\mathbf{k}_* = \mathbf{s}_* + U\mathbf{u}_*$. This is an algebraic identity, so the bracket depends on $\mathbf{u}_*$ only through the cross-covariances $U\mathbf{u}_*$, whose arguments satisfy Lemma 1. For the noisy observation $y_*$, add $\sigma_n^2$ to the variance. This formula agrees with the October 27 version when $s(x_*,x_*)$ is defined as $\kappa(0) - \mathbf{u}_*^\top\mathbf{u}_*$.

**Separately conditioned posteriors cannot be added.** The components are coupled through the data. The per-component posterior means are $\mathbf{u}_*^\top\hat{\mathbf{w}}$ and $\mathbf{s}_*^\top A^{-1}(\mathbf{y} - U\hat{\mathbf{w}})$, which is the closed-form backfitting fixed point.

### 3.5 Gradients

For any hyperparameter $\theta$,

$$\partial_\theta\log p(\mathbf{y}) = \tfrac12\,\boldsymbol{\gamma}^\top(\partial_\theta C)\,\boldsymbol{\gamma} - \tfrac12\operatorname{tr}\!\big(C^{-1}\partial_\theta C\big), \qquad \partial_\theta C = \partial_\theta S + (\partial_\theta U)U^\top + U(\partial_\theta U)^\top.$$

Using $C^{-1} = A^{-1} - PM^{-1}P^\top$ and $U^\top C^{-1} = P^\top - (M - I)M^{-1}P^\top = M^{-1}P^\top$:

$$\boldsymbol{\gamma}^\top(\partial_\theta C)\boldsymbol{\gamma} = \boldsymbol{\gamma}^\top(\partial_\theta S)\boldsymbol{\gamma} + 2\,\big((\partial_\theta U)^\top\boldsymbol{\gamma}\big)^\top\big(U^\top\boldsymbol{\gamma}\big),$$

$$\operatorname{tr}\!\big(C^{-1}\partial_\theta C\big) = \operatorname{tr}\!\big(A^{-1}\partial_\theta S\big) - \operatorname{tr}\!\big(M^{-1}P^\top(\partial_\theta S)P\big) + 2\operatorname{tr}\!\big(M^{-1}P^\top\partial_\theta U\big).$$

For $\theta = \sigma_n^2$, $\partial_\theta C = I$, and $\operatorname{tr}(C^{-1}) = \operatorname{tr}(A^{-1}) - \operatorname{tr}(M^{-1}P^\top P)$.

The term $\operatorname{tr}(A^{-1}\partial_\theta S)$ is the gradient term the sparse GP computes. The remaining terms need $Q$ sparse matrix–vector products with $\partial_\theta S$ and $O(nQ^2)$ dense work, and no solves beyond the $Q+1$ of the forward pass.

**Entrywise derivatives.** On the pattern $\mathcal{P}$:

- $\partial_{\sigma^2}S = S/\sigma^2$.
- $\partial_\ell S_{ij} = \sigma^2\big[\kappa_1'(w_{ij})(-2w_{ij}/\ell) - \kappa_1'(w^F_{ij})(-2w^F_{ij}/\ell)\big]$.
- For $\theta \in \{a, r, \beta\}$: $\partial_\theta S_{ij} = (\sigma^2/\ell^2)\big[\kappa_1'(w_{ij})\,(\partial q_i + \partial q_j - 2\,\partial G_{ij}) - \kappa_1'(w^F_{ij})\,(\partial q_i + \partial q_j)\big]$, with $\partial G_{ij} = \sum_l(\partial\phi_{il}\,\phi_{jl} + \phi_{il}\,\partial\phi_{jl})$ and $\partial q_i = \partial G_{ii}$.

The factor derivatives are:

- $\partial_{\sigma^2}U = U/(2\sigma^2)$.
- $\partial_\ell U_{ij} = U_{ij}\cdot 2\tau_j q_i/\ell^3$.
- $\partial_\theta U_{ij} = -U_{ij}\,\tau_j\,\partial q_i/\ell^2$ for $\theta \in \{a, r, \beta\}$.

The bump derivatives, with $u = \rho^2/r^2 < 1$, are:

- $\partial_a b = b/a$;
- $\partial_\beta b = -b\,u/(1-u)$;
- $\partial_r b = 2\beta\rho^2 b/\big(r^3(1-u)^2\big)$.

The profile derivatives are:

| Kernel | $\kappa_1(w)$ | $\kappa_1'(w)$ |
|---|---|---|
| RBF | $e^{-w/2}$ | $-\tfrac12 e^{-w/2}$ |
| Matérn 3/2 | $(1+\sqrt{3w})\,e^{-\sqrt{3w}}$ | $-\tfrac32\,e^{-\sqrt{3w}}$ |
| Matérn 5/2 | $(1+\sqrt{5w}+\tfrac53 w)\,e^{-\sqrt{5w}}$ | $-\tfrac56(1+\sqrt{5w})\,e^{-\sqrt{5w}}$ |
| Matérn 1/2 | $e^{-\sqrt{w}}$ | $-e^{-\sqrt{w}}/(2\sqrt{w})$, singular at $w = 0$ |

### 3.6 The support radius and the sparsity pattern

- **Pattern-preserving parameters.** The amplitude $a$, the flatness $\beta$, the length scale $\ell$, the smoothness $\nu$, and the signal variance $\sigma^2$ change the values of $S$ and leave its pattern unchanged.
- **Smoothness in $r$.** For fixed $\rho$, $b(\rho; a, r, \beta)$ is $C^\infty$ in $r$, and it tends to zero with all derivatives as $r \downarrow \rho$. Entries enter and leave the support smoothly, so the marginal likelihood is continuous and differentiable in $r$.
- **Fixed allocation.** The pattern $\mathcal{P}$ at $r_{\max}$ contains the pattern at every $r \le r_{\max}$. Allocating it once and storing exact zeros for inactive pairs fixes the structure for the whole optimization. The optimizer bound for $r$ must equal $r_{\max}$.

### 3.7 Prior properties

- **Density interpretation.** $q(x) = \sum_i b(\rho(x,x_i))^2$ is an unnormalized kernel density estimate with kernel $b^2$. The far-field component $\mathbf{u}(x)^\top\mathbf{w}$ is a Gaussian process indexed by local data density. Covariance between inputs that share no landmark is $\kappa(\sqrt{q(x)+q(x')})$, which decreases with density at both inputs.
- **Points outside every support.** These map to $\boldsymbol{\phi} = \mathbf{0}$. Any two such points have prior correlation one and identical posteriors (verified in Section 4.2), and each has covariance $\kappa(\sqrt{q_j})$ with training point $x_j$. Extrapolation far from the data reduces to one shared random variable.
- **Rank control.** The numerical rank of $F$ grows with the dynamic range of $q$. A non-stationary amplitude field $a(\cdot)$ that equalizes $q_i$ drives $F$ toward rank one.
- **Density of the pattern.** $\lvert\mathcal{P}\rvert \le T = \sum_l\lvert\mathcal{L}_l\rvert^2$. With a fixed radius and growing data density, $T$ grows as $n^2$. Sparse scaling requires $r_{\max}$ to shrink with density, or a non-stationary radius.

## 4. Numerical Checks

### 4.1 Checks reported in the October 27 draft

These checks used shortest-path distances on a connected random geometric graph with 388 nodes, whose double-centered squared distance matrix has smallest eigenvalue $-655$. They were not rerun for this revision.

- **Landmark kernel.** 80 landmarks, Wendland profile, radius 4: the plain Wendland function of $\rho$ gave smallest eigenvalue $-0.67$. $k_{\mathrm{lin}}$ gave $-6\times10^{-15}$, with 21% nonzero entries.
- **SLE-RBF split.** 14.4% nonzero entries in $S$, reconstruction error $8\times10^{-16}$, smallest eigenvalue of $S$ equal to $0.0061$. Decomposed posterior mean and covariance matched dense computations to $7\times10^{-15}$ and $2\times10^{-15}$. The naive sum of separate posteriors had mean error 0.31.
- **SLE-Matérn 3/2 split.** Remainder PSD (smallest eigenvalue 0.0162), with pattern identical to $\Phi\Phi^\top > 0$. A 21-term NNLS fit reproduced the Gram matrix to $1.3\times10^{-9}$. The log marginal likelihood, posterior mean, and posterior variance matched dense computations to $6\times10^{-14}$, $3\times10^{-15}$, and $2\times10^{-15}$.

### 4.2 Checks of this revision (reference implementation, Appendix A)

**Setup.** The metric is shortest paths on a random geometric graph with 700 nodes, whose double-centered squared distance matrix has smallest eigenvalue $-1.27$. The data are 450 training points and 100 test points with $\sigma^2 = 1.3$, $\ell = 0.9$, $a = 1$, $\beta = 1$, $\sigma_n^2 = 10^{-2}$, $r = 0.16$, and $r_{\max} = 0.22$. The pattern at $r_{\max}$ holds 34% of entries, and 18.7% are nonzero at $r$, which exercises the fixed-allocation scheme. The far-field fit uses $w_{\mathrm{lo}} = 1/16$ and $\varepsilon_{\mathrm{fit}} = 10^{-10}$. Gradients are compared with central finite differences of the dense log marginal likelihood for $\sigma^2$, $\ell$, $a$, $r$, $\beta$, and $\sigma_n^2$.

| Quantity | RBF ($Q=1$) | Matérn 3/2 ($Q=35$) | Matérn 5/2 ($Q=30$) |
|---|---|---|---|
| Smallest eigenvalue of $S$ | $6.8\times10^{-4}$ | $4.8\times10^{-3}$ | $2.2\times10^{-3}$ |
| $\max\lvert S + UU^\top - K\rvert$ | $1.0\times10^{-14}$ | $9.0\times10^{-12}$ | $4.0\times10^{-12}$ |
| Log marginal likelihood error | $2.8\times10^{-13}$ | $1.1\times10^{-9}$ | $7.5\times10^{-10}$ |
| Max. relative gradient error | $4.4\times10^{-9}$ | $4.0\times10^{-8}$ | $1.1\times10^{-8}$ |
| Posterior mean error | $2.5\times10^{-14}$ | $9.0\times10^{-11}$ | $4.4\times10^{-11}$ |
| Posterior variance error | $2.8\times10^{-14}$ | $7.0\times10^{-12}$ | $3.6\times10^{-12}$ |
| Numerical rank of $U$ at $10^{-10}$ | 1 | 8 | 8 |

The gradient errors are at the accuracy of the finite differences. A second check placed 10 test points outside every support ($q(x_*) = 0$). Mean and variance matched the dense computation to $9.0\times10^{-11}$ and $7.0\times10^{-12}$, and two such points had posterior correlation 1.000000, as Section 3.7 predicts.

## 5. Implementation Guide

### 5.1 Architecture

The computation splits into three phases, distinguished by how often they run.

| Phase | Runs | Content | Reference code |
|---|---|---|---|
| Structure | Once per training run | Neighbor search at $r_{\max}$, overlap triplets, pattern $\mathcal{P}$, far-field fit | `SLEStructure`, `fit_far_field` |
| Evaluation | Once per hyperparameter vector | Bump values, $G$, $q$, $S$, $U$, derivatives, sparse factorization, Woodbury algebra | `assemble`, `fit` |
| Prediction | Once per test batch | Test embedding, $\mathbf{s}_*$, $\mathbf{u}_*$, mean, variance | `test_embedding`, `predict` |

The design rule is that every sparse array has a fixed pattern during the run. The values change with the hyperparameters, and the indices never do.

### 5.2 Step 1: Neighbor structure at $r_{\max}$

**Goal.** For each training point $x_i$, store all landmarks $x_l$ with $\rho(x_i,x_l) < r_{\max}$, together with the distances. The result is a sparse $n \times n$ matrix whose stored values are distances. The diagonal (distance zero) must be stored explicitly, because it carries $q_i \ge a^2$.

**Distance access.**
- **Precomputed distance matrix** (the setting of the NeurIPS experiments): threshold each row.
- **Metric $\rho$**: range queries with a cover tree or vantage-point tree. These require only the distance function and the triangle inequality, and they avoid the $O(n^2)$ distance evaluations.
- **Non-metric $\rho$**: tree pruning is invalid. Use exhaustive evaluation, or filter candidates with a cheap proxy distance and evaluate $\rho$ exactly on the survivors. A proxy filter must be conservative (no false negatives) for the model to stay exact.

**Storage caveat.** SciPy drops explicit zeros in several sparse operations, including matrix products. Keep distances in a plain array aligned with the index structure (as `SLEStructure.rho` does), and avoid sparse products that could remove pattern entries.

### 5.3 Step 2: Choosing $r_{\max}$ by a sparsity budget

The upper radius bound fixes both memory and cost, so it is chosen from a memory budget. The bound used in the NeurIPS experiments (the maximum pairwise distance) makes $S$ dense.

1. For a candidate radius $\bar{r}$, compute the landmark degrees $\lvert\mathcal{L}_l(\bar{r})\rvert$ from the neighbor counts.
2. Evaluate $T(\bar{r}) = \sum_l\lvert\mathcal{L}_l(\bar{r})\rvert^2$, an upper bound on $\lvert\mathcal{P}\rvert$ and the exact number of overlap triplets.
3. $T$ is nondecreasing in $\bar{r}$. Bisect on $\bar{r}$ until $T$ meets the memory budget (for example, a multiple of the nonzeros gp2Scale handles for the same $n$).
4. Set $r_{\max}$ to the result, and set the optimizer bound of $r$ to $r_{\max}$.

Monitor the fraction of pattern entries that are nonzero at the current $r$. If the optimizer presses against $r_{\max}$, the likelihood favors longer correlation than the budget allows. This should be reported as a constraint of the run.

### 5.4 Step 3: Overlap triplets and the fixed pattern

$G$ is assembled from landmark outer products, $G = \sum_l \boldsymbol{\phi}_{:,l}\boldsymbol{\phi}_{:,l}^\top$. For each landmark $l$ and each ordered pair of members $(x_i, x_j) \in \mathcal{L}_l \times \mathcal{L}_l$, store one triplet:

- `trip_pos`: the position $k$ of $(i,j)$ in the pattern $\mathcal{P}$;
- `trip_a`, `trip_b`: the positions of $\phi_{il}$ and $\phi_{jl}$ in the stored embedding values.

The pattern itself is the set of unique pairs $(i,j)$ among all triplets. One sort by the key $i\,n + j$ produces both the pattern and `trip_pos` (`np.unique` with `return_inverse`). Record the positions of the diagonal entries, which give $q_i = G_{ii}$.

Assembly is then a scatter-add with fixed indices:

$$G_k = \sum_{t:\ \mathrm{trip\_pos}_t = k}\phi[\mathrm{trip\_a}_t]\,\phi[\mathrm{trip\_b}_t],$$

implemented with `np.bincount`. Derivatives use the same indices (Section 3.5). Memory is $O(T)$ integers.

**Distribution.** Partition $\mathcal{P}$ into the block pairs gp2Scale already uses, and give each worker the triplets whose `trip_pos` falls in its block. Triplets of one landmark may span several blocks. Each worker needs the embedding values of its rows and columns and the vector $q$. Production code should build the triplets in compiled code (Numba or C++), since the reference loop over landmarks is Python.

### 5.5 Step 4: Per-evaluation assembly of $S$ and $\partial S$

For a hyperparameter vector:

1. Evaluate bump values and their derivatives on the stored distances (`bump`). Entries with $\rho \ge r$ become exact zeros, and the pattern is untouched.
2. Assemble $G$ and $\partial G$ on $\mathcal{P}$ with the triplets. Read $q$ and $\partial q$ from the diagonal positions.
3. Compute $w_{ij} = \max(q_i + q_j - 2G_{ij}, 0)/\ell^2$ and $w^F_{ij} = (q_i+q_j)/\ell^2$. Clipping at zero removes round-off in near-duplicate embeddings.
4. Set $S_{ij} = \sigma^2[\kappa_1(w_{ij}) - \kappa_1(w^F_{ij})]$ and $\partial S$ by the formulas of Section 3.5.

**Accuracy.** When $G_{ij} \ll q_i + q_j$, the difference in step 4 cancels. The absolute error stays at machine precision times $\sigma^2$, which suffices for the solver. For RBF, the cancellation-free form $S_{ij} = \sigma^2 e^{-(q_i+q_j)/(2\ell^2)}\operatorname{expm1}(G_{ij}/\ell^2)$ is available. For Matérn 1/2, $\kappa_1'$ is singular at $w = 0$. On the diagonal the factor multiplying it is exactly zero, so set that product to zero. Off-diagonal pairs with identical embeddings need the same treatment.

### 5.6 Step 5: Far-field factor

1. **Once per inner kernel and bound box.** Compute $w_{\mathrm{lo}} = a_{\min}^2/\ell_{\max}^2$ and fit $\kappa_1$ on $[w_{\mathrm{lo}},\infty)$ to $\varepsilon_{\mathrm{fit}}$ from Section 3.3 (`fit_far_field`). Store $(\omega_j, \tau_j)$. With a non-stationary amplitude, use $\min_x a(x)$ over the bounds. Cache fits in a small library keyed by $(\nu, w_{\mathrm{lo}}, \varepsilon_{\mathrm{fit}})$.
2. **Per evaluation.** Form $U_{ij} = \sigma\sqrt{\omega_j}\,e^{-\tau_j q_i/\ell^2}$ and its derivatives (`assemble`). The cost is $O(nQ)$, and no $n \times n$ object is formed.
3. **Optional.** Compress $U$ as in Section 3.3 when the solver is iterative.

For RBF, skip the fit: $\omega_1 = 1$ and $\tau_1 = 1/2$.

### 5.7 Step 6: Likelihood and gradient

The wrapper (`fit` in Appendix A) needs three solver hooks:

| Hook | Use | Direct path | Iterative path |
|---|---|---|---|
| `solve(B)` | $A^{-1}\mathbf{y}$, $A^{-1}U$, $A^{-1}\mathbf{s}_*$ | Sparse Cholesky (CHOLMOD), one factorization, $Q+1$ triangular solve pairs | Block conjugate gradients on $Q+1$ right-hand sides |
| `logdet()` | $\log\det A$ | Sum of log-diagonal of the Cholesky factor | Stochastic Lanczos quadrature |
| `trace_inv(D)` | $\operatorname{tr}(A^{-1}\partial_\theta S)$, $\operatorname{tr}(A^{-1})$ | Selected inversion (Takahashi recurrences) on the pattern of the factor, which contains $\mathcal{P}$ | Hutchinson estimator with the same probe vectors for all $\theta$ |

With the iterative path the likelihood and gradients become stochastic. Fixed probe vectors across one optimization run (common random numbers) keep the objective smooth. The reference `SparseSolver` uses SciPy's `splu` and computes traces by dense solves. It is suitable for tests up to a few thousand points only.

Sequence per evaluation:

1. Factor $A = S + \sigma_n^2 I$.
2. Solve for $\boldsymbol{\alpha}$ and $P$.
3. Form $M$, its Cholesky factor, $\boldsymbol{\eta}$, $\hat{\mathbf{w}}$, and $\boldsymbol{\gamma}$.
4. Evaluate $\log p(\mathbf{y})$ (Section 3.4).
5. For each $\theta$: one sparse product $(\partial_\theta S)\boldsymbol{\gamma}$, one product $(\partial_\theta S)P$, one trace hook call, and $O(nQ^2)$ dense work (Section 3.5).

### 5.8 Step 7: Prediction

1. **Test embedding.** For each test point, find landmarks with $\rho(x_*, x_l) < r$ at the trained $r$ and evaluate bumps. Points with no landmark in range receive the zero embedding and remain valid inputs.
2. **Cross terms.** Compute $\mathbf{s}_*$ on the nonzeros of $\boldsymbol{\phi}(x_*)^\top\Phi^\top$. Pairs with no shared landmark have $s = 0$ exactly, so no fixed pattern is needed at test time. Compute $\mathbf{u}_*$ from $q(x_*)$.
3. **Mean.** $\mathbf{s}_*^\top\boldsymbol{\alpha} + \mathbf{h}_*^\top\hat{\mathbf{w}}$, at cost $O(\operatorname{nnz}(\mathbf{s}_*) + Q)$ per point after training.
4. **Variance.** $\kappa(0) - [\mathbf{s}_*^\top A^{-1}\mathbf{s}_* + \mathbf{u}_*^\top\mathbf{u}_* - \mathbf{h}_*^\top M^{-1}\mathbf{h}_*]$, with one solve per test point. Batch test points as block right-hand sides that reuse the training factorization.

### 5.9 Step 8: Integration with gp2Scale and gpCAM

- **Inputs as indices.** As in the NeurIPS experiments, pass integer indices as GP inputs. The kernel callable looks up embedding rows, $q$, and distances by index.
- **Block kernel.** gp2Scale distributes covariance blocks over workers. The block callable returns $S$ restricted to the block pair, computed from the worker's triplets, embedding values, and $q$. The far-field factor never passes through the block kernel.
- **Block scheduling.** Block pair $(I,J)$ is nonzero exactly when some landmark has members in both blocks. Form the boolean block–landmark incidence matrix $B$ (blocks by landmarks) and the boolean product $BB^\top$. Schedule only its nonzero block pairs. This is the block-level analogue of Corollary 3.
- **Woodbury wrapper.** In gpCAM, wrap the gp2Scale linear-algebra path: the wrapper calls the three hooks of Section 5.7 and adds the $Q$-dimensional terms. $U$ is formed on the driver at cost $O(nQ)$ and enters the solver as a block right-hand side.
- **Hyperparameter vector.** $(\sigma^2, \ell, a, r, \beta, \sigma_n^2)$, extended by the parameters of $a(\cdot)$, $r(\cdot)$, and $\beta(\cdot)$ in the non-stationary case. The bound for $r$ is $r_{\max}$, and the far-field fit depends on the bounds of $a$ and $\ell$.

### 5.10 Step 9: Validation protocol

Run these tests on problems small enough for dense reference computations ($n \le 2000$):

1. **Bump derivatives** against finite differences in $a$, $r$, $\beta$, including distances close to $r$.
2. **Fixed pattern.** $G$ on $\mathcal{P}$ equals dense $\Phi\Phi^\top$ for several $r \le r_{\max}$, with zeros at inactive pattern entries.
3. **Reconstruction.** $\max\lvert S + UU^\top - K\rvert \le \sigma^2\varepsilon_{\mathrm{fit}}$.
4. **Positivity.** Smallest eigenvalue of $S$ above $-10^{-12}\sigma^2$.
5. **Likelihood and gradients** against the dense computation and central finite differences of it, for every hyperparameter.
6. **Prediction** against the dense computation, including points outside every support.
7. **Scaling.** Time and memory per evaluation as functions of $n$ and $r_{\max}$.

Appendix B contains the script that produced Section 4.2 and covers tests 3–6.

## 6. Work Packages

### WP1. Interface audit of gp2Scale

Confirm that the gp2Scale code path exposes the three hooks of Section 5.7 (block solve, log-determinant, trace of $A^{-1}$ against a sparse matrix) and accepts user-supplied sparse kernel blocks with a prescribed block schedule. Add thin hooks where it does not. For iterative paths, determine whether a block solver handles $Q+1$ right-hand sides at the cost of roughly one solve.

**Deliverable:** a short interface note.

### WP2. Structure and sparse assembly

- Neighbor search at $r_{\max}$ for precomputed, metric, and non-metric distances (Section 5.2).
- $r_{\max}$ selection from the sparsity budget (Section 5.3).
- Overlap triplets and pattern in compiled code, with distribution by block pair (Section 5.4).
- Assembly of $S$ and $\partial_\theta S$ (Section 5.5).

**Acceptance:** tests 1–4 of Section 5.10 at machine precision.

### WP3. Far-field factor

| Inner kernel | Factor |
|---|---|
| RBF | $U = \sigma\mathbf{g}$, $Q = 1$, closed form |
| Matérn, $\nu \in \{1/2, 3/2, 5/2\}$ | Nonnegative exponential fit of $\kappa_1$ on $[w_{\mathrm{lo}},\infty)$, $w_{\mathrm{lo}} = a_{\min}^2/\ell_{\max}^2$ |

Tasks:
- Reduce $Q$ below the greedy NNLS counts of Section 3.3 with Gauss-type quadrature of $\mu$ or Beylkin–Monzón exponential sums.
- Build the fit library and the accuracy criterion $n\sigma^2\varepsilon_{\mathrm{fit}} \le 10^{-3}\sigma_n^2$ into the code.
- Implement and test optional compression with the gradient formula of Section 3.3.

**Acceptance:** $\varepsilon_{\mathrm{fit}}$ meets the criterion for $n$ up to $10^6$. Compressed and uncompressed likelihoods agree to $10^{-8}$ relative error.

### WP4. Woodbury wrapper in gpCAM

Implement Sections 3.4 and 3.5 around the gp2Scale solver, following the reference functions `fit` and `predict`.

**Acceptance:** log marginal likelihood, posterior mean, posterior variance, and gradients match dense computations to $10^{-8}$ relative error up to $n \approx 10^4$. The reference implementation reached $10^{-9}$–$10^{-14}$ at $n = 450$ (Section 4.2).

### WP5. Scaling study on Perlmutter

- Measure the nonzero fraction of $S$, the time per likelihood evaluation, and memory as functions of $n$ and $r_{\max}$.
- Compare against dense SLE and against the masking-kernel construction, which modifies the model. The present split leaves the published SLE kernel unchanged.
- Datasets: the manifold and Wasserstein benchmarks of the NeurIPS paper, then a molecular graph set with $n \ge 10^5$.

### WP6. Theory note

Planned contents:

1. Theorem 1 with Corollaries 1–4.
2. Lemma 1, the model-error bound, and a bound on the numerical rank of $F$ from the analyticity of $\kappa_1$ on $[w_{\mathrm{lo}},\infty)$ and the range of $q$.
3. The exact posterior, likelihood, and gradient formulas.
4. The prior properties of Section 3.7.

**Target:** a section in a follow-up paper on SLE at scale, or the gpCAM/gp2Scale platform paper.

## 7. Open Questions and Risks

- **Density of $S$.** With all training points as landmarks, $\lvert\mathcal{P}\rvert$ grows as $n^2$ at fixed radius. The budget of Section 5.3 bounds memory, but a binding budget restricts the correlation length the model can learn. A non-stationary radius $r(\cdot)$ is the principled remedy, since it adapts the support to local density.
- **Neighbor search for non-metric distances.** Without the triangle inequality, tree pruning fails and exact search costs $O(n^2)$ distance evaluations. Conservative proxy filters are distance-specific.
- **Fit size for $\nu = 1/2$.** $Q$ stays large when $w_{\mathrm{lo}}$ is small. Compression or improved fits (WP3) address this.
- **Stochastic trace estimation.** On the iterative path, gradient noise can slow optimization. Fixed probes and a direct path for moderate $n$ mitigate it.
- **Far-field prior behavior.** The density dependence of the far-field covariance (Section 3.7) is a modeling property of SLE. Its effect on calibration under strongly varying sampling density should be tested.
- **Conditioning.** The decomposition leaves the conditioning of $C$ unchanged. $M \succeq I_Q$ is well conditioned by construction.
- **Assumption $\rho(x,x) = 0$.** Lemma 1 uses it. For noisy dissimilarities with $\rho(x,x) > 0$, replace $a^2$ by $\min_i b(\rho(x_i,x_i))^2$, or by zero.

## 8. References

- Beylkin, G., and Monzón, L. (2005). On approximation of functions by exponential sums. *Applied and Computational Harmonic Analysis*, 19, 17–48.
- Bochner, S. (1933). Monotone Funktionen, Stieltjessche Integrale und harmonische Analyse. *Mathematische Annalen*, 108, 378–410.
- Duvenaud, D., Nickisch, H., and Rasmussen, C. E. (2011). Additive Gaussian processes. *Advances in Neural Information Processing Systems*, 24.
- Gneiting, T. (2002). Compactly supported correlation functions. *Journal of Multivariate Analysis*, 83, 493–508.
- Hutchinson, M. F. (1989). A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. *Communications in Statistics – Simulation and Computation*, 18, 1059–1076.
- Lawson, C. L., and Hanson, R. J. (1974). *Solving Least Squares Problems*. Prentice-Hall.
- Noack, M. M., Alghalayini, M. B., and Risser, M. D. (2026). A general kernel framework for non-CND distance measures using $|\mathcal{D}|$-dimensional sparse landmark embeddings. arXiv:2609.19083.
- O'Hagan, A. (1978). Curve fitting and optimal design for prediction. *Journal of the Royal Statistical Society, Series B*, 40, 1–42.
- Rasmussen, C. E., and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*, Section 2.7. MIT Press.
- Schoenberg, I. J. (1938). Metric spaces and completely monotone functions. *Annals of Mathematics*, 39, 811–841.
- Takahashi, K., Fagan, J., and Chen, M.-S. (1973). Formation of a sparse bus impedance matrix and its application to short circuit study. *Proceedings of the 8th PICA Conference*, 63–69.
- Ubaru, S., Chen, J., and Saad, Y. (2017). Fast estimation of $\operatorname{tr}(f(A))$ via stochastic Lanczos quadrature. *SIAM Journal on Matrix Analysis and Applications*, 38, 1075–1099.
- Wendland, H. (1995). Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree. *Advances in Computational Mathematics*, 4, 389–396.

## Appendix A: Reference Implementation (`sle_split_ref.py`)

Validated against dense computations in Section 4.2. `SparseSolver` is the hook to replace with the gp2Scale solver. Its `trace_inv` uses dense solves and is meant for tests only.

```python
"""Reference implementation: exact sparse-plus-low-rank SLE GP.

K = S + U U^T, S sparse PSD on the landmark-overlap pattern, U U^T PSD far field.
All hyperparameters live in a dict hp = {sig2, ell, a, r, beta, noise}.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import nnls

# ---------------------------------------------------------------------------
# 1. Inner kernels kappa_1(w), w = d^2 / ell^2, unit variance, and dkappa_1/dw
# ---------------------------------------------------------------------------
def kappa1(w, nu):
    w = np.maximum(w, 0.0)
    if nu == "rbf":
        return np.exp(-0.5 * w)
    c = np.sqrt(w)
    if nu == 0.5:
        return np.exp(-c)
    if nu == 1.5:
        c = np.sqrt(3.0) * c
        return (1.0 + c) * np.exp(-c)
    if nu == 2.5:
        c = np.sqrt(5.0) * c
        return (1.0 + c + c * c / 3.0) * np.exp(-c)
    raise ValueError(nu)


def dkappa1(w, nu):
    w = np.maximum(w, 0.0)
    if nu == "rbf":
        return -0.5 * np.exp(-0.5 * w)
    if nu == 1.5:
        return -1.5 * np.exp(-np.sqrt(3.0 * w))
    if nu == 2.5:
        c = np.sqrt(5.0 * w)
        return -(5.0 / 6.0) * (1.0 + c) * np.exp(-c)
    raise ValueError("gradients implemented for rbf, 1.5, 2.5")


# ---------------------------------------------------------------------------
# 2. Far-field factor: kappa_1(w) ~ sum_j omega_j exp(-tau_j w) on [w_lo, inf)
# ---------------------------------------------------------------------------
def fit_far_field(nu, w_lo, tol=1e-10, n_rates=160, n_grid=4000):
    """Nonnegative exponential-sum fit, valid for every ell (rates scale as tau/ell^2).
    Returns (omega, tau, sup_error). Uniform absolute error is checked on [w_lo, w_hi];
    beyond w_hi both kappa_1 and the fit are decreasing and below tol."""
    if nu == "rbf":
        return np.array([1.0]), np.array([0.5]), 0.0
    w_hi = w_lo + 1.0
    while kappa1(w_hi, nu) > 0.1 * tol:
        w_hi *= 2.0
    lo = max(w_lo, 1e-12)
    w = np.unique(np.concatenate([np.geomspace(lo, w_hi, n_grid), np.linspace(w_lo, w_hi, n_grid)]))
    f = kappa1(w, nu)
    tau_all = np.geomspace(0.05 / w_hi, 50.0 / lo, n_rates)
    E = np.exp(-np.outer(w, tau_all))
    active = np.arange(n_rates)

    def solve(idx):
        om, _ = nnls(E[:, idx], f, maxiter=2000 * len(idx))
        keep = om > 0
        idx, om = idx[keep], om[keep]
        err = np.abs(E[:, idx] @ om - f).max()
        return idx, om, err

    active, omega, err = solve(active)
    # greedy pruning: drop the smallest contribution while the error stays below tol
    while len(active) > 1:
        contrib = omega * E[:, active].max(axis=0)
        trial = np.delete(active, np.argmin(contrib))
        t_idx, t_om, t_err = solve(trial)
        if t_err > tol:
            break
        active, omega, err = t_idx, t_om, t_err
    tail = max(kappa1(w_hi, nu), np.exp(-w_hi * tau_all[active]) @ omega)
    return omega, tau_all[active], max(err, tail)


# ---------------------------------------------------------------------------
# 3. Bump function and its hyperparameter derivatives
# ---------------------------------------------------------------------------
def bump(rho, a, r, beta):
    """Returns b and dict of partial derivatives w.r.t. a, r, beta (zero outside support)."""
    b = np.zeros_like(rho)
    db = {k: np.zeros_like(rho) for k in ("a", "r", "beta")}
    m = rho < r
    u = rho[m] ** 2 / r**2
    e = np.exp(beta - beta / (1.0 - u))
    b[m] = a * e
    db["a"][m] = e
    db["beta"][m] = b[m] * (-u / (1.0 - u))
    db["r"][m] = b[m] * 2.0 * beta * rho[m] ** 2 / (r**3 * (1.0 - u) ** 2)
    return b, db


# ---------------------------------------------------------------------------
# 4. Fixed structure at r_max: neighbor lists, pattern of G, overlap triplets
# ---------------------------------------------------------------------------
class SLEStructure:
    """Built once per optimization run.

    nbr : CSR (n x n) whose stored entries are rho(x_i, x_l) for all pairs with rho < r_max
          (diagonal stored explicitly; explicit zeros allowed in .data).
    """

    def __init__(self, nbr):
        nbr = sp.csr_matrix(nbr)
        nbr.sort_indices()
        self.n = nbr.shape[0]
        self.rho = nbr.data.copy()               # distances, one per stored embedding entry
        self.row = np.repeat(np.arange(self.n), np.diff(nbr.indptr))
        self.col = nbr.indices.copy()            # landmark index
        # members of each landmark l, as positions into self.rho
        order = np.argsort(self.col, kind="stable")
        counts = np.bincount(self.col, minlength=self.n)
        starts = np.concatenate([[0], np.cumsum(counts)])
        ti, tj, pa, pb = [], [], [], []
        for l in range(self.n):
            pos = order[starts[l]:starts[l + 1]]
            if pos.size == 0:
                continue
            A, B = np.meshgrid(pos, pos, indexing="ij")
            pa.append(A.ravel()); pb.append(B.ravel())
            ti.append(self.row[A.ravel()]); tj.append(self.row[B.ravel()])
        ti, tj = np.concatenate(ti), np.concatenate(tj)
        self.trip_a, self.trip_b = np.concatenate(pa), np.concatenate(pb)
        key = ti.astype(np.int64) * self.n + tj
        ukey, self.trip_pos = np.unique(key, return_inverse=True)
        self.pi = (ukey // self.n).astype(np.int64)   # pattern rows
        self.pj = (ukey % self.n).astype(np.int64)    # pattern cols
        self.nnz = ukey.size
        self.diag_pos = np.flatnonzero(self.pi == self.pj)
        assert self.diag_pos.size == self.n          # every point is its own landmark

    def gram_values(self, phi, dphi=None):
        """G = Phi Phi^T on the fixed pattern; optionally dG for each entry of dphi."""
        G = np.bincount(self.trip_pos, phi[self.trip_a] * phi[self.trip_b], minlength=self.nnz)
        if dphi is None:
            return G
        dG = {k: np.bincount(self.trip_pos,
                             v[self.trip_a] * phi[self.trip_b] + phi[self.trip_a] * v[self.trip_b],
                             minlength=self.nnz) for k, v in dphi.items()}
        return G, dG

    def to_csr(self, vals):
        return sp.csr_matrix((vals, (self.pi, self.pj)), shape=(self.n, self.n))


# ---------------------------------------------------------------------------
# 5. Assembly of S, U and their derivatives at hyperparameters hp
# ---------------------------------------------------------------------------
def assemble(st, hp, nu, omega, tau, grad=False):
    sig2, ell = hp["sig2"], hp["ell"]
    phi, dphi = bump(st.rho, hp["a"], hp["r"], hp["beta"])
    if grad:
        G, dG = st.gram_values(phi, dphi)
    else:
        G = st.gram_values(phi)
    q = G[st.diag_pos]                              # q_i = G_ii (diag_pos sorted by row)
    qi, qj = q[st.pi], q[st.pj]
    w = np.maximum(qi + qj - 2.0 * G, 0.0) / ell**2  # embedding distance^2 / ell^2
    wF = (qi + qj) / ell**2                          # far-field argument
    s = sig2 * (kappa1(w, nu) - kappa1(wF, nu))
    E = np.exp(-np.outer(q, tau) / ell**2)          # n x Q
    U = np.sqrt(sig2 * omega)[None, :] * E
    out = dict(S=st.to_csr(s), U=U, q=q)
    if not grad:
        return out
    k1, k1F = dkappa1(w, nu), dkappa1(wF, nu)
    dS, dU = {}, {}
    dS["sig2"] = s / sig2
    dU["sig2"] = U / (2.0 * sig2)
    dS["ell"] = sig2 * (k1 * (-2.0 * w / ell) - k1F * (-2.0 * wF / ell))
    dU["ell"] = U * (2.0 * np.outer(q, tau) / ell**3)
    for k in ("a", "r", "beta"):
        dq = dG[k][st.diag_pos]
        dd2 = dq[st.pi] + dq[st.pj] - 2.0 * dG[k]
        dS[k] = sig2 * (k1 * dd2 - k1F * (dq[st.pi] + dq[st.pj])) / ell**2
        dU[k] = U * (-np.outer(dq, tau) / ell**2)
    out["dS"] = {k: st.to_csr(v) for k, v in dS.items()}
    out["dU"] = dU
    return out


# ---------------------------------------------------------------------------
# 6. Sparse solver hook (replace with the gp2Scale solver)
# ---------------------------------------------------------------------------
class SparseSolver:
    def __init__(self, A):
        self.A = sp.csc_matrix(A)
        self.lu = spla.splu(self.A)

    def solve(self, B):
        return self.lu.solve(np.asarray(B, dtype=float))

    def logdet(self):
        return np.log(np.abs(self.lu.U.diagonal())).sum()   # A SPD; L has unit diagonal

    def trace_inv(self, D):
        """tr(A^{-1} D). Reference version via dense solves; production: selected
        inversion on the pattern of D, or a stochastic trace estimator."""
        return np.trace(self.solve(D.toarray()))


# ---------------------------------------------------------------------------
# 7. Woodbury likelihood, gradient, prediction
# ---------------------------------------------------------------------------
def fit(S, U, y, noise, grad_terms=None):
    n, Q = U.shape
    A = S + noise * sp.identity(n, format="csr")
    sol = SparseSolver(A)
    alpha = sol.solve(y)
    P = sol.solve(U)
    M = np.eye(Q) + U.T @ P
    L = np.linalg.cholesky(M)
    eta = U.T @ alpha
    w_hat = np.linalg.solve(M, eta)
    gamma = alpha - P @ w_hat                       # C^{-1} y
    log_ml = -0.5 * (y @ alpha - eta @ w_hat) - 0.5 * (sol.logdet() + 2 * np.log(np.diag(L)).sum()) \
        - 0.5 * n * np.log(2 * np.pi)
    res = dict(sol=sol, alpha=alpha, P=P, M=M, w_hat=w_hat, gamma=gamma, log_ml=log_ml)
    if grad_terms is None:
        return res
    dS, dU = grad_terms
    Minv = np.linalg.inv(M)
    Ug = U.T @ gamma
    g = {}
    for k in dS:
        dSk, dUk = dS[k], dU[k]
        quad = gamma @ (dSk @ gamma) + 2.0 * (gamma @ dUk) @ Ug
        tr = sol.trace_inv(dSk) - np.trace(Minv @ (P.T @ (dSk @ P))) + 2.0 * np.trace(Minv @ (P.T @ dUk))
        g[k] = 0.5 * quad - 0.5 * tr
    # noise variance: dC = I
    g["noise"] = 0.5 * gamma @ gamma - 0.5 * (sol.trace_inv(sp.identity(n, format="csr"))
                                            - np.trace(Minv @ (P.T @ P)))
    res["grad"] = g
    return res


def test_embedding(rho_test, hp):
    """rho_test: CSR (m x n), stored entries = distances from test points to landmarks < r."""
    rho_test = sp.csr_matrix(rho_test)
    b, _ = bump(rho_test.data, hp["a"], hp["r"], hp["beta"])
    return sp.csr_matrix((b, rho_test.indices, rho_test.indptr), shape=rho_test.shape)


def predict(res, phi_star, phi_train_csr, q, hp, nu, omega, tau, U):
    sig2, ell = hp["sig2"], hp["ell"]
    q_star = np.asarray(phi_star.multiply(phi_star).sum(axis=1)).ravel()
    Gs = (phi_star @ phi_train_csr.T).tocoo()       # nonzeros only: zero G gives s = 0
    w = np.maximum(q_star[Gs.row] + q[Gs.col] - 2 * Gs.data, 0) / ell**2
    wF = (q_star[Gs.row] + q[Gs.col]) / ell**2
    s_star = sp.csr_matrix((sig2 * (kappa1(w, nu) - kappa1(wF, nu)), (Gs.row, Gs.col)),
                           shape=(phi_star.shape[0], len(q)))
    u_star = np.sqrt(sig2 * omega)[None, :] * np.exp(-np.outer(q_star, tau) / ell**2)
    H = u_star - s_star @ res["P"]
    mean = s_star @ res["alpha"] + H @ res["w_hat"]
    AinvS = res["sol"].solve(s_star.T.toarray())    # batched right-hand sides
    explained = np.einsum("ij,ji->i", s_star.toarray(), AinvS) + np.sum(u_star**2, axis=1) \
        - np.einsum("ij,ji->i", H, np.linalg.solve(res["M"], H.T))
    var = sig2 * kappa1(0.0, nu) - explained         # exact prior variance kappa(0)
    return mean, var
```

## Appendix B: Validation Script (`test_ref.py`)

Produces the results of Section 4.2. The dense reference forms the full SLE Gram matrix, so the script is limited to small $n$.

```python
import numpy as np, scipy.sparse as sp, time
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.spatial.distance import cdist
import sle_split_ref as R

rng = np.random.default_rng(7)
# --- non-Euclidean metric: shortest paths on a random geometric graph
P = rng.uniform(0, 1, (700, 2))
E = cdist(P, P); Adj = np.where(E < 0.09, E, 0.0)
nc, lab = connected_components(sp.csr_matrix(Adj), directed=False)
keep = np.flatnonzero(lab == np.bincount(lab).argmax())
D = shortest_path(sp.csr_matrix(Adj[np.ix_(keep, keep)]), directed=False)
J = np.eye(len(keep)) - 1 / len(keep)
print("points:", len(keep), " min eig of -J D^2 J / 2:", np.linalg.eigvalsh(-0.5 * J @ D**2 @ J).min().round(2))
perm = rng.permutation(len(keep)); tr, te = perm[:450], perm[450:550]
Dtt, Dst = D[np.ix_(tr, tr)], D[np.ix_(te, tr)]
y = np.sin(4 * D[tr, 0]) + 0.05 * rng.standard_normal(len(tr))

def nbr_csr(Dm, rad):
    i, j = np.nonzero(Dm < rad)
    return sp.csr_matrix((Dm[i, j] + 0.0, (i, j)), shape=Dm.shape)

hp = dict(sig2=1.3, ell=0.9, a=1.0, r=0.16, beta=1.0, noise=1e-2)
r_max = 0.22
st = R.SLEStructure(nbr_csr(Dtt, r_max))
print(f"pattern nnz/n^2 at r_max: {st.nnz/len(tr)**2:.3f}, triplets: {st.trip_pos.size}")

def dense_K(DA, DB, hp, nu):
    PA, _ = R.bump(DA, hp["a"], hp["r"], hp["beta"]); PB, _ = R.bump(DB, hp["a"], hp["r"], hp["beta"])
    d2 = (PA**2).sum(1)[:, None] + (PB**2).sum(1)[None, :] - 2 * PA @ PB.T
    return hp["sig2"] * R.kappa1(np.maximum(d2, 0) / hp["ell"]**2, nu)

def dense_logml(hp, nu):
    C = dense_K(Dtt, Dtt, hp, nu) + hp["noise"] * np.eye(len(tr))
    L = np.linalg.cholesky(C); a = np.linalg.solve(C, y)
    return -0.5 * y @ a - np.log(np.diag(L)).sum() - 0.5 * len(y) * np.log(2 * np.pi)

for nu in ["rbf", 1.5, 2.5]:
    w_lo = hp["a"]**2 / 4.0**2       # a_min^2 / ell_max^2 over assumed bounds
    om, ta, ferr = R.fit_far_field(nu, w_lo, tol=1e-10)
    out = R.assemble(st, hp, nu, om, ta, grad=True)
    S, U = out["S"], out["U"]
    Kd = dense_K(Dtt, Dtt, hp, nu)
    rec = np.abs(S.toarray() + U @ U.T - Kd).max()
    Sd = S.toarray(); minS = np.linalg.eigvalsh(Sd).min()
    t = time.time(); res = R.fit(S, U, y, hp["noise"], (out["dS"], out["dU"])); t = time.time() - t
    ll_ref = dense_logml(hp, nu)
    # gradients by central differences of the dense reference
    gerr = []
    for k in ["sig2", "ell", "a", "r", "beta", "noise"]:
        h = 1e-5 * hp[k]; hp_p = dict(hp); hp_m = dict(hp); hp_p[k] += h; hp_m[k] -= h
        fd = (dense_logml(hp_p, nu) - dense_logml(hp_m, nu)) / (2 * h)
        gerr.append(abs(res["grad"][k] - fd) / max(1.0, abs(fd)))
    # prediction
    phi_tr = R.test_embedding(nbr_csr(Dtt, hp["r"]), hp)
    phi_te = R.test_embedding(nbr_csr(Dst, hp["r"]), hp)
    mean, var = R.predict(res, phi_te, phi_tr, out["q"], hp, nu, om, ta, U)
    C = Kd + hp["noise"] * np.eye(len(tr)); Ks = dense_K(Dst, Dtt, hp, nu)
    mref = Ks @ np.linalg.solve(C, y)
    vref = hp["sig2"] * R.kappa1(0.0, nu) - np.einsum("ij,ji->i", Ks, np.linalg.solve(C, Ks.T))
    lam = np.linalg.eigvalsh(U.T @ U)[::-1]
    Fex = hp["sig2"] * R.kappa1((out["q"][:, None] + out["q"][None, :]) / hp["ell"]**2, nu)
    sv = np.linalg.eigvalsh(Fex)[::-1]
    p_c = int(np.sum(lam > 1e-10 * lam[0])); p_F = int(np.sum(sv > 1e-10 * sv[0]))
    n_iso = int((np.asarray(phi_te.sum(1)).ravel() == 0).sum())
    print(f"\nnu={nu}: Q={U.shape[1]} fit err={ferr:.1e}  nnz(S)/n^2={S.nnz/len(tr)**2:.3f} "
          f"(nonzero values {np.count_nonzero(S.data)/len(tr)**2:.3f})  min eig S={minS:.2e}")
    print(f"  max|S+UU^T-K|={rec:.1e}  log-ML err={abs(res['log_ml']-ll_ref):.1e}  "
          f"max rel grad err={max(gerr):.1e}  fit time={t:.2f}s")
    print(f"  mean err={np.abs(mean-mref).max():.1e}  var err={np.abs(var-vref).max():.1e}  "
          f"(isolated test points: {n_iso})")
    print(f"  compression: eigvals of U^T U above 1e-10*max: {p_c};  rank_1e-10 of exact F: {p_F}")

# isolated test points: all distances beyond r (phi* = 0), Matern 3/2
nu = 1.5; om, ta, _ = R.fit_far_field(nu, hp["a"]**2 / 16.0, tol=1e-10)
out = R.assemble(st, hp, nu, om, ta); S, U = out["S"], out["U"]
res = R.fit(S, U, y, hp["noise"])
Dfar = Dst.copy(); Dfar[:10] = Dfar[:10] + 1.0          # 10 isolated rows
phi_tr = R.test_embedding(nbr_csr(Dtt, hp["r"]), hp); phi_te = R.test_embedding(nbr_csr(Dfar, hp["r"]), hp)
mean, var = R.predict(res, phi_te, phi_tr, out["q"], hp, nu, om, ta, U)
Kd = dense_K(Dtt, Dtt, hp, nu); C = Kd + hp["noise"] * np.eye(len(tr)); Ks = dense_K(Dfar, Dtt, hp, nu)
vref = hp["sig2"] - np.einsum("ij,ji->i", Ks, np.linalg.solve(C, Ks.T))
print(f"\nisolated rows: {int((np.asarray(phi_te.sum(1)).ravel()==0).sum())}, "
      f"mean err={np.abs(mean - Ks @ np.linalg.solve(C, y)).max():.1e}, var err={np.abs(var-vref).max():.1e}, "
      f"posterior corr of two isolated points (dense)="
      f"{(dense_K(Dfar[:2],Dfar[:2],hp,nu)[0,1]-Ks[0]@np.linalg.solve(C,Ks[1]))/np.sqrt(vref[0]*vref[1]):.6f}")
```
