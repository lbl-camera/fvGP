# Test-Point Bumps for Posterior Covariances in the SLE Kernel

*Design note and implementation guide. Status: validated in 1D (November 1, 2026); not yet implemented in gpCAM.*

---

## 1. Purpose and scope

The Sparse Landmark Embedding (SLE) kernel uses every training point as a landmark. Beyond the support of all landmark bumps, the embedding of a point is the zero vector. At the free marginal-likelihood optimum, native SLE produces calibrated posterior **marginals** in such regions (coverage 0.95 on an extrapolation test). Its posterior **covariances** fail there: every unsupported point receives the identical embedding, so the posterior correlation between any two unsupported points equals 1, independent of their separation.

This note specifies an inference-time correction that

- leaves the training covariance, the marginal likelihood, and the trained hyperparameters untouched (no refactorization);
- requires no knowledge of the prediction points during training;
- introduces no new hyperparameters and no separation distance or grid spacing of the query set;
- gives posterior marginals and test–training covariances that are exactly independent of the query set;
- produces test–test posterior correlations that decay with separation.

**When to use it.** If only posterior means and marginal variances are needed, native SLE suffices. The correction applies when joint posterior quantities are needed: posterior covariances, joint sampling, joint log scores, batch acquisition functions in Bayesian optimization, or any downstream quantity that integrates over the joint posterior.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $X = \{x_1,\dots,x_n\}$ | training inputs; each is also a landmark |
| $y \in \mathbb{R}^n$ | training observations |
| $Z = \{z_1,\dots,z_m\}$ | prediction (test) points, known only at inference time |
| $\rho(\cdot,\cdot)$ | the distance measure on the input space (need not be conditionally negative definite) |
| $b(d; a, r, \beta)$ | bump function, $b = a\exp\!\big(\beta - \beta/(1-d^2/r^2)\big)$ for $d<r$, else $0$; $b(0)=a$ |
| $a, r, \beta$ | bump amplitude, support radius, and shape parameter (trained or fixed as in native SLE) |
| $\varphi(x) \in \mathbb{R}^n$ | SLE embedding, $\varphi_l(x) = b(\rho(x,x_l); a, r, \beta)$, $l = 1,\dots,n$ |
| $q(x) = \lVert\varphi(x)\rVert^2$ | squared embedding norm (the "support" of $x$) |
| $q_i = q(x_i)$ | squared embedding norm of training point $i$ |
| $q_{\mathrm{ref}}$ | reference norm, a statistic of $\{q_i\}$ (Section 5) |
| $\kappa(w)$ | inner kernel as a function of a squared distance $w$; for Matérn 3/2, $\kappa(w) = (1+\sqrt{3w})\,e^{-\sqrt{3w}}$ |
| $\sigma^2, \ell$ | signal variance and length scale of the inner kernel |
| $\sigma_n^2$ | observation-noise variance |
| $A(z) \ge 0$ | amplitude of the test-point coordinates of $z$ (Section 4.2) |
| $u(z)$ | unit direction of the test-point coordinates of $z$ (Section 4.3) |
| $\psi(z) = A(z)\,u(z)$ | test-point coordinates of $z$ |
| $\varrho(z,z')$ | overlap $\langle u(z), u(z')\rangle \in [0,1]$ |
| $K_{XX}, K_{ZX}, K_{ZZ}$ | prior covariance blocks between the indicated sets |
| $\mu(Z), \Sigma(Z)$ | posterior mean vector and covariance matrix at $Z$ |

---

## 3. The defect in native SLE

Native SLE uses

$$
k(x,x') = \sigma^2\,\kappa\!\left(\frac{\lVert\varphi(x)-\varphi(x')\rVert^2}{\ell^2}\right).
$$

If $z$ and $z'$ both lie outside every landmark support, $\varphi(z) = \varphi(z') = 0$. Then $k(z,z') = \sigma^2\kappa(0) = k(z,z)$, and $k(z,x_i) = k(z',x_i)$ for every $i$. The two points are indistinguishable to the model, so their posterior correlation is exactly 1 and every posterior sample is constant beyond the data. Marginals remain sensible at the free optimum ($r \approx 0.075$, $\lambda = \ell/\sqrt{2\bar q} \approx 0.5$, where $\bar q$ is the median of $\{q_i\}$). Joint scores collapse: joint negative log predictive density (NLPD) per point $\approx 28$, Mahalanobis distance per point $\approx 59$ (ideal value 1).

---

## 4. Construction

### 4.1 Augmented embedding

At inference time, each test point $z$ receives additional coordinates $\psi(z)$. Every training point has $\psi(x_i) = 0$. The augmented embedding is

$$
\Phi(x) = \big[\varphi(x),\ 0\big] \quad (x \in X), \qquad
\Phi(z) = \big[\varphi(z),\ \psi(z)\big] \quad (z \in Z),
$$

and the kernel is the native inner kernel applied to augmented squared distances:

$$
k(\cdot,\cdot) = \sigma^2\,\kappa\!\left(\frac{\lVert\Phi(\cdot)-\Phi(\cdot)\rVert^2}{\ell^2}\right).
$$

Because $\psi(x_i) = 0$, the training block $K_{XX}$ is identical to native SLE. The prior is never re-evaluated at training points.

### 4.2 Amplitude

$$
A(z)^2 = \big(q_{\mathrm{ref}} - q(z)\big)_+ ,
$$

where $(\cdot)_+ = \max(\cdot, 0)$. The amplitude restores the embedding norm a point would have if its support matched the reference level: $\lVert\Phi(z)\rVert^2 = \max\big(q(z), q_{\mathrm{ref}}\big)$. Points with $q(z) \ge q_{\mathrm{ref}}$ receive $A(z) = 0$ and are treated exactly as in native SLE. $A$ is a continuous function of $z$ because $q$ is continuous.

### 4.3 Direction

The test-point coordinates live on new axes, one per test point. The coordinate of $z_i$ on the axis of $z_j$ is the unit-amplitude bump of the learned radius,

$$
B_{ij} = b\big(\rho(z_i, z_j); 1, r, \beta\big),
$$

normalized per row:

$$
u(z_i) = \frac{B_{i\cdot}}{\lVert B_{i\cdot}\rVert}, \qquad \psi(z_i) = A(z_i)\,u(z_i).
$$

$\lVert B_{i\cdot}\rVert \ge B_{ii} = 1$, so the normalization never divides by zero. Normalization fixes $\lVert\psi(z)\rVert = A(z)$ regardless of how many other test points lie nearby. This is what removes the dependence on the query density.

### 4.4 Closed form

Define the overlap

$$
\varrho(z,z') = \langle u(z), u(z')\rangle = \frac{\sum_{k} B_{zk} B_{z'k}}{\lVert B_{z\cdot}\rVert\,\lVert B_{z'\cdot}\rVert}.
$$

The three covariance blocks are

$$
\begin{aligned}
k(x_i, x_j) &= \sigma^2\,\kappa\!\left(\frac{\lVert\varphi(x_i)-\varphi(x_j)\rVert^2}{\ell^2}\right) &&\text{(unchanged)}\\[4pt]
k(z, x_i) &= \sigma^2\,\kappa\!\left(\frac{\lVert\varphi(z)-\varphi(x_i)\rVert^2 + A(z)^2}{\ell^2}\right)\\[4pt]
k(z, z') &= \sigma^2\,\kappa\!\left(\frac{\lVert\varphi(z)-\varphi(z')\rVert^2 + A(z)^2 + A(z')^2 - 2A(z)A(z')\,\varrho(z,z')}{\ell^2}\right)
\end{aligned}
$$

with $k(z,z) = \sigma^2\kappa(0) = \sigma^2$. Squared embedding distances are computed as $\lVert\varphi(x)-\varphi(x')\rVert^2 = q(x) + q(x') - 2\,\varphi(x)^\top\varphi(x')$.

The posterior follows the standard formulas

$$
\mu(Z) = K_{ZX}\,(K_{XX}+\sigma_n^2 I)^{-1} y, \qquad
\Sigma(Z) = K_{ZZ} - K_{ZX}\,(K_{XX}+\sigma_n^2 I)^{-1} K_{XZ}.
$$

### 4.5 Properties

1. **Positive semi-definite.** The kernel is a Matérn (or any radial positive-definite) function of Euclidean distances between explicit feature vectors $\Phi$, so every joint Gram matrix over $X \cup Z$ is PSD.
2. **Training unchanged.** $K_{XX}$, the marginal likelihood, its gradients, the hyperparameters, and the Cholesky (or sparse) factor are exactly those of native SLE.
3. **Query-independent marginals.** $k(z,z) = \sigma^2$ and $k(z,x_i)$ depend only on $z$ through $\varphi(z)$ and $A(z)$. Therefore $\mu(z)$ and $\Sigma_{zz}$ are identical whether $z$ is predicted alone or inside any query set (verified to $10^{-13}$).
4. **No new hyperparameters.** $q_{\mathrm{ref}}$ is a statistic of the training norms. The direction uses the learned radius $r$ and shape $\beta$. The amplitude scale is fixed by normalization.
5. **No separation distance $h$.** On regular grids, row normalization cancels the grid spacing in $\varrho$. Correlations at fixed separations are identical for $h = 0.01$ and $h = 0.0025$ (Section 8).
6. **Consistent reduction.** Where $q(z) \ge q_{\mathrm{ref}}$, $A(z) = 0$ and the model equals native SLE.
7. **Distance-aware decorrelation.** For two unsupported points, the augmented squared distance is $2A^2(1-\varrho)$. It is zero for coincident points and rises to $2A^2$ once their bumps stop overlapping (separation $\ge 2r$ in the continuum limit).

---

## 5. Choice of the reference norm $q_{\mathrm{ref}}$

$q_{\mathrm{ref}}$ sets the support level below which a test point receives extra coordinates. The test points that receive them are pushed equally far from **all** training points, including their nearest neighbors. A $q_{\mathrm{ref}}$ that is too large therefore degrades in-data predictions.

| $q_{\mathrm{ref}}$ | Value in the test | In-data NLPD | Comment |
|---|---|---|---|
| minimum of $\{q_i\}$ | 68.7 (domain boundary) | −0.88 (unchanged) | default; no in-data test point is modified |
| minimum over interior training points | 128.2 | −0.86 | modifies 1% of in-data test points; better joint calibration |
| median of $\{q_i\}$ | 146.8 | −0.34 | about half of in-data test points are pushed away from the data; rejected |

**Recommendation.** Use $q_{\mathrm{ref}} = \min_i q_i$ by default. Offer a low quantile of $\{q_i\}$ (e.g., the 1st percentile) as an option. "Interior" has no general meaning in arbitrary metric spaces, and a low quantile is a portable surrogate for it.

---

## 6. Alternatives considered and rejected

These variants were tested on the same problems before arriving at Section 4. They are recorded so the reasons for the final design are explicit.

| Variant | Failure |
|---|---|
| Test-point bumps evaluated at training points too (consistent bumps) | Training covariance and evidence then depend on the query set; requires refactorization per query set |
| Unit amplitude ($A = 1$) | Too weak against $q \approx 150$; no measurable effect |
| Norm-matched amplitude with $q_{\mathrm{ref}}$ = median | Damages in-data NLPD (Section 5) |
| Zero-radius test-point bump (pure deficit axis per point) | No overlap between neighboring test points; distance-unaware decorrelation |
| Single shared deficit axis | All unsupported points share one coordinate; correlation 1 persists |
| Unnormalized test-point bumps | Norm grows with query density; marginals depend on the grid spacing $h$ |
| Deficit field on fixed nodes with quadrature weights $A(t)^2 = c\,h\,(1 - q(t)/\bar q)_+$ | Converges in $h$ but needs nodes covering the prediction domain (extra landmarks) and a known $h$ |

---

## 7. Implementation guide

### 7.1 Algorithm

Training is native SLE. At prediction time, for a query set $Z$:

1. **Training embedding cross terms.** Compute the sparse matrix $P_Z \in \mathbb{R}^{m\times n}$, $(P_Z)_{il} = b(\rho(z_i,x_l); a, r, \beta)$, using a radius-$r$ neighbor search (KD-tree, ball tree, or the gp2Scale distance machinery). Compute $q(z_i) = \lVert (P_Z)_{i\cdot}\rVert^2$.
2. **Amplitude.** $A_i = \sqrt{(q_{\mathrm{ref}} - q(z_i))_+}$. Store $q_{\mathrm{ref}}$ with the trained model.
3. **Fast path.** If only marginals are requested, skip steps 4–5. $k(z,x_i)$ needs only $A(z)$, and $k(z,z) = \sigma^2$.
4. **Test–test overlap.** Restrict to the index set $U = \{i : A_i > 0\}$; all other rows of $\psi$ vanish. Compute the sparse matrix $B_{UU}$ with unit-amplitude bumps of radius $r$ among the points of $U$, normalize its rows to obtain $V$, and form $R = V V^\top$ (the overlap $\varrho$ on $U$).
5. **Augmented test–test squared distances.** $D_{ZZ} = \lVert\varphi(z)-\varphi(z')\rVert^2$ plus, on $U\times U$, $A_i^2 + A_j^2 - 2A_iA_jR_{ij}$; and plus $A_i^2$ or $A_j^2$ in the blocks where only one point lies in $U$.
6. **Kernel blocks.** $K_{ZX} = \sigma^2\kappa\big((D_{ZX} + A^2\mathbf{1}^\top)/\ell^2\big)$ and $K_{ZZ} = \sigma^2\kappa(D_{ZZ}/\ell^2)$.
7. **Posterior.** Use the existing training factorization for $\mu$ and $\Sigma$. Symmetrize $\Sigma$ and clip tiny negative eigenvalues before sampling.

### 7.2 Reference code (dense, NumPy)

```python
import numpy as np
import scipy.linalg as sla

def bump(d, a, r, beta):
    """b(d; a, r, beta) = a exp(beta - beta / (1 - d^2/r^2)) for d < r, else 0."""
    out = np.zeros_like(d, dtype=float)
    m = d < r
    out[m] = a * np.exp(beta - beta / (1.0 - (d[m] / r) ** 2))
    return out

def kappa_matern32(w):
    """Matern 3/2 as a function of the scaled squared distance w."""
    s = np.sqrt(3.0 * np.maximum(w, 0.0))
    return (1.0 + s) * np.exp(-s)

def sqdist(F, G):
    """Squared Euclidean distances between rows of F and G."""
    return np.maximum((F**2).sum(1)[:, None] + (G**2).sum(1)[None, :] - 2.0 * F @ G.T, 0.0)

class SLETestPointBumps:
    """Posterior of a trained SLE model with test-point bumps (training untouched).

    dist(A, B): pairwise distance matrix under the SLE distance measure.
    a, r, beta: learned bump parameters; sig2, ell, noise: learned kernel parameters.
    """
    def __init__(self, X, y, dist, a, r, beta, sig2, ell, noise, qref="min"):
        self.X, self.dist = X, dist
        self.a, self.r, self.beta = a, r, beta
        self.sig2, self.ell = sig2, ell
        self.P = bump(dist(X, X), a, r, beta)                 # sparse in practice
        q = (self.P**2).sum(1)
        self.qref = q.min() if qref == "min" else np.quantile(q, qref)
        K = sig2 * kappa_matern32(sqdist(self.P, self.P) / ell**2)
        self.cf = sla.cho_factor(K + noise * np.eye(len(X)))  # native SLE factor
        self.alpha = sla.cho_solve(self.cf, y)

    def predict(self, Z, full_cov=True):
        Pz = bump(self.dist(Z, self.X), self.a, self.r, self.beta)
        qz = (Pz**2).sum(1)
        A = np.sqrt(np.maximum(self.qref - qz, 0.0))
        # test-training block: psi(x_i) = 0, so ||Phi(z) - Phi(x_i)||^2 = ||phi(z) - phi(x_i)||^2 + A(z)^2
        Kzx = self.sig2 * kappa_matern32((sqdist(Pz, self.P) + A[:, None]**2) / self.ell**2)
        mu = Kzx @ self.alpha
        if not full_cov:                                     # fast path: marginals only
            var = self.sig2 - np.einsum("ij,ji->i", Kzx, sla.cho_solve(self.cf, Kzx.T))
            return mu, var
        B = bump(self.dist(Z, Z), 1.0, self.r, self.beta)    # unit-amplitude test-point bumps
        U = B / np.linalg.norm(B, axis=1, keepdims=True)     # row-normalized directions u(z)
        Psi = A[:, None] * U                                 # psi(z) = A(z) u(z)
        Dzz = sqdist(Pz, Pz) + sqdist(Psi, Psi)
        Kzz = self.sig2 * kappa_matern32(Dzz / self.ell**2)
        S = Kzz - Kzx @ sla.cho_solve(self.cf, Kzx.T)
        return mu, 0.5 * (S + S.T)
```

The validated experiment scripts are `normalized_query.py` and `nq_baseline.py`, with shared setup in `joint_cov.py` and hyperparameters in `extrapolation.json`. All are in the session scratchpad.

### 7.3 Sparse and scalable version

- $P_Z$ and $B$ are sparse with the same radius-$r$ pattern as the training embedding. Build them with the same neighbor search used for training.
- $D_{ZZ}$ is dense in general because $\kappa$ has global support. For large $m$, compute $\Sigma$ only on requested blocks (e.g., a batch of candidate points), or draw joint samples through a low-rank plus sparse representation of $K_{ZZ}$ mirroring the training decomposition $K = S + F$.
- The cost beyond native SLE is the sparse product $V V^\top$ on the set $U$ of unsupported test points, plus one extra term in the squared distances. The training solve is unchanged.

### 7.4 Integration into gpCAM / fvGP

- Store $q_{\mathrm{ref}}$ (and the chosen rule: `"min"` or a quantile) with the trained SLE kernel object.
- Expose the correction through the existing posterior-covariance call, e.g. `posterior_covariance(x_pred, test_point_bumps=True)`. Posterior mean and marginal variance calls need only $A(z)$.
- Keep $q_{\mathrm{ref}}$ fixed after training; recompute it only when the training data or bump parameters change.

### 7.5 Unit tests

1. **Training invariance.** $K_{XX}$ and the log marginal likelihood equal native SLE to machine precision.
2. **Query independence of marginals.** $\mu(z)$ and $\Sigma_{zz}$ for a point predicted alone equal those from inside a large query set (observed differences $\sim 10^{-13}$).
3. **Reduction.** For a query set entirely inside the data with $q(z) \ge q_{\mathrm{ref}}$, the output equals native SLE.
4. **Grid-spacing invariance.** On uniform grids with spacings $h$ and $h/4$, correlations at fixed separations agree to three digits.
5. **PSD.** The joint Gram matrix over $X\cup Z$ has no eigenvalue below $-10^{-10}\sigma^2$.
6. **Decorrelation.** Beyond the data, the posterior correlation between two test points decreases monotonically with separation up to $2r$ and is below 1 for all separations greater than 0.

---

## 8. Validation (1D extrapolation test)

**Setup.** $f(x) = \sin(6\pi x) + 0.4\sin(60\pi x) + 0.3x$; $n = 2{,}000$ training points uniform on $[0,1]$; noise standard deviation 0.1. Matérn 3/2 inner kernel with hyperparameters from the free marginal-likelihood fit: $\sigma^2 = 1.074$, $\ell = 8.782$, $r = 0.0747$, $\sigma_n^2 = 0.0095$ ($\lambda = 0.51$, $\bar q = 146.8$). Queries on a uniform grid over $[0.8, 2]$ with spacing $h$; joint scores on the far region $[1.1, 2]$. In-data scores on 300 random points in $[0.02, 0.98]$. Correlations are with $z_0 = 1.5$ at separations 0.01, 0.05, 0.1, 0.15, 0.3.

| Variant | $h$ | In-data NLPD | Far cov. 95% | Far NLPD | Far latent sd | Joint NLPD/pt | Mahalanobis/$m$ | Correlation vs. separation |
|---|---|---|---|---|---|---|---|---|
| Native SLE | 0.01 | −0.88 | 0.95 | 1.15 | 0.70 | 27.8 | 58.4 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 |
| Native SLE | 0.0025 | −0.88 | 0.95 | 1.16 | 0.70 | 28.5 | 59.8 | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 |
| $q_{\mathrm{ref}}$ = min | 0.01 | −0.88 | 1.00 | 1.19 | 0.90 | 1.20 | 2.78 | 0.93 / 0.39 / 0.13 / 0.11 / 0.11 |
| $q_{\mathrm{ref}}$ = min | 0.0025 | −0.88 | 1.00 | 1.19 | 0.90 | −0.14 | 1.58 | 0.93 / 0.39 / 0.13 / 0.11 / 0.11 |
| $q_{\mathrm{ref}}$ = interior min | 0.01 | −0.86 | 1.00 | 1.22 | 0.96 | 0.84 | 1.58 | 0.89 / 0.28 / 0.06 / 0.05 / 0.05 |
| $q_{\mathrm{ref}}$ = interior min | 0.0025 | −0.86 | 1.00 | 1.23 | 0.96 | −0.28 | 1.07 | 0.89 / 0.28 / 0.06 / 0.05 / 0.05 |

The true function has standard deviation 0.75 on the far region. Joint NLPD per point decreases with denser grids because neighboring points carry redundant information; the grid-independent diagnostics are the marginal scores and the correlation profile. The Mahalanobis distance per point should be near 1.

**Reading.** In-data predictions are unchanged for $q_{\mathrm{ref}} = \min_i q_i$. Joint calibration beyond the data improves from Mahalanobis/$m \approx 59$ to 1.6–2.8. Correlations decay with separation and plateau at a small value beyond $2r$, and the profile does not depend on $h$. Far marginals widen slightly (latent sd 0.70 → 0.90), with coverage 1.00 and NLPD 1.19 vs. 1.15.

---

## 9. Open issues

1. **Coupled effects.** The same amplitude $A$ controls both the marginal widening and the test–test decorrelation. A separate scale for the direction term would decouple them, but it would also change the marginals. No variant with separate control has been tested.
2. **Sparse query sets.** $\varrho$ is computed from the query set itself. For query spacing comparable to or larger than $r$, rows of $B$ share few entries and correlations fall toward the plateau value early. The continuum limit
   $$
   \varrho_\infty(z,z') = \frac{\int b(\rho(t,z))\,b(\rho(t,z'))\,d\mu(t)}{\sqrt{\int b(\rho(t,z))^2 d\mu(t)\,\int b(\rho(t,z'))^2 d\mu(t)}}
   $$
   for a reference measure $\mu$ is fully query-independent. In $\mathbb{R}^d$ with Lebesgue measure it depends only on $\lVert z-z'\rVert$ and can be tabulated once. In general metric spaces (the main SLE use case), a reference measure must be chosen; the query set's empirical measure (the tested version) is one choice.
3. **Mahalanobis above 1 at $q_{\mathrm{ref}}$ = min, $h = 0.01$.** Short-range correlations are slightly too strong in this configuration. The interior-min variant is closer to calibrated (1.07–1.58).
4. **Plateau correlation.** Correlations beyond $2r$ plateau at 0.05–0.11. This value is inherited from the inner kernel at the augmented distance $\sqrt{2}A$ and matches the correlation between unrelated training points only approximately.
5. **Higher dimensions.** All validation is 1D. In higher dimensions, $q(z)$ decays differently near the data boundary, and the $q_{\mathrm{ref}}$ rule should be rechecked. Expect boundary training points to have far lower $q$ than interior ones.
6. **Interaction with other fixes.** The cosine (normalized) SLE with a shared deficit axis gives calibrated marginals at small radii, but also correlation 1 between unsupported points. The test-point bumps can replace its shared deficit axis; this combination is untested.
7. **Adaptive radius.** If landmark radii become point-dependent, $r_l = \alpha\,d_k(x_l)$, with $d_k$ the distance to the $k$-th nearest training point, the test-point bump radius needs a rule (e.g., $\alpha\,d_k(z)$). This was tested earlier only in the re-evaluated (consistent) variant.
