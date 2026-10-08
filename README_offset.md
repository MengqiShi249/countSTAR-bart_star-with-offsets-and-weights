# countSTAR: `bart_star()` with offsets

This repository extends `bart_star()` from [countSTAR](https://github.com/bking124/countSTAR) to support **model offsets** (exposures) for count-valued BART regression. **Regression weights** are documented separately in `README_weight.md`. The motivating application is national density and distribution modelling for Canadian birds, but the extension applies to any STAR model with a known, mechanistic exposure term. The new version `bart_star()` with offsets does not require any modification to `countSTAR` helper functions.

## Motivation

`bart_star()` fits a **S**imultaneous **T**ransformation **A**nd **R**ounding (STAR) model with a BART latent mean via `dbarts`. It captures nonlinearities and interactions (e.g. thresholds in temperature, elevation, distance to water) that GLMs/GAMs miss, and returns calibrated posterior uncertainty without the bootstrapping that boosted regression trees require.

**Offsets.** Each point count has a known offset $e_i > 0$ (e.g. detection proportion or survey effort). The usual log-link model is $\text{count}_i \sim \text{Poisson}\big(e_i \exp(\eta_i)\big)$, $\eta_i = \mu(x_i)$, so $\mu(x_i)$ is a log density rather than a log expected count. Because $e_i$ acts multiplicatively and mechanistically, it enters as an offset, not as a covariate. Offsets turn a model for counts into a model for absolute density, which allows population-size estimation. In STAR, $g$ need not be $\log$, so the offset cannot be absorbed into the linear predictor. Replacing $y$ by $y/e$ as the response is also incorrect, even for $g = \log$ (see below).

## Background: the STAR model

$$y_i = h(y_i^\ast) = \lfloor y_i^\ast\rfloor, \qquad z_i = g(y_i^\ast), \qquad z_i = \mu(x_i) + \varepsilon_i, \quad \varepsilon_i \sim N(0, \sigma^2)$$

Here $h$ is the floor (`round_floor()`: negatives mapped to 0, capped at `y_max`), $g$ is increasing, and $\mu(\cdot)$ is a BART sum of trees. Since $g$ is monotone,

$$\Pr(y_i = j \mid x_i) = \Pr\big(g(a_j) \le z_i < g(a_{j+1})\big), \qquad a_j = j, \quad a_0 = -\infty, \quad a_{y_{\max}+1} = \infty$$

with $a_j$ given by `a_j()`. The Gibbs sampler:

1. **Sample $z$:** draw $z_i \mid y_i$ from $N(\mu(x_i), \sigma^2)$ truncated to $[g(a_{y_i}), g(a_{y_i+1}))$.
2. **Sample $(\mu, \sigma)$:** update the BART fit (`dbarts`) with response $z$.
3. **(Optional) update the transformation**, if $g$ is being learned rather than fixed.
4. **Predict:** draw $\tilde z \sim N(\mu, \sigma^2)$ and use $g^{-1}$ and rounding to generate posterior predictive draws $\tilde y = h(g^{-1}(\tilde z))$.

## Model with offsets

The rounding is unchanged. The transformation acts on the latent count per unit exposure:

$$y_i = \lfloor y_i^\ast\rfloor, \qquad z_i = g\left(\frac{y_i^\ast}{e_i}\right) = \mu(x_i) + \varepsilon_i, \quad \varepsilon_i \sim N(0, \sigma^2)$$

Here $\mu$ and $\sigma$ are on the scale of the transformed density $g(y^\ast/e)$. No rescaling of the offsets or parameters is involved, which keeps the prior and only partial change in likelihood. Also, the regression is only for latent $z$ and covariate $x$, and $y/e$ is never used as response.

### Likelihood

For $e_i > 0$ and monotone $g$,

$$y_i = j \iff a_j \le y_i^\ast < a_{j+1} \iff g\left(\frac{a_j}{e_i}\right) \le z_i < g\left(\frac{a_{j+1}}{e_i}\right)$$

$$\ell(\mu, \sigma, g) = \sum_{i=1}^n \log\left[\Phi\left(\frac{g(a_{y_i+1}/e_i) - \mu_i}{\sigma}\right) - \Phi\left(\frac{g(a_{y_i}/e_i) - \mu_i}{\sigma}\right)\right]$$

with $\pm\infty/e_i = \pm\infty$. This is what `logLikeRcpp` / `logLikePointRcpp` compute on the training data.

**Why not $y/e$.** For $j \ge 1$, the interval for $y_i = j$ on the density scale is $[j/e_i, (j+1)/e_i)$, of width $1/e_i$. Using $y/e$ as the response gives $[j/e_i, j/e_i + 1)$, which is wrong whenever $e_i \neq 1$.

**Special case $g = \log$.** $\log y_i^\ast = \mu(x_i) + \log e_i + \varepsilon_i$ is the familiar additive log-offset.

### Gibbs sampler with offsets

Only the cut points change, from $a_j$ to $a_j/e_i$:

| Step | With offsets |
|---|---|
| Block 1: sample $z$ | $N(\mu_i, \sigma^2)$ truncated to $[g(a_{y_i}/e_i),\thinspace g(a_{y_i+1}/e_i))$ |
| Block 2: sample $(\mu, \sigma)$ | unchanged |
| Block 3: Box–Cox $\lambda$ | slice sampler on $\ell(\lambda) + \log \pi(\lambda)$ with $g_\lambda(a_{y_i}/e_i)$, $g_\lambda(a_{y_i+1}/e_i)$; $\pi(\lambda) = N(1/2, 1)$ truncated to $[0, 3]$ |
| Pointwise log-likelihood / WAIC | offset-adjusted bounds |
| Test log-predictive score | $g(a_{y_j}/e_j^{\text{test}})$, $g(a_{y_j+1}/e_j^{\text{test}})$ |

The scaled cut points are computed once, before the MCMC loop. For Box–Cox, only $g_\lambda$ is re-applied at each iteration.

### Prediction

$$\tilde z_i \sim N(\mu_i, \sigma^2), \qquad \tilde y_i = h\big(e_i\thinspace g^{-1}(\tilde z_i)\big)$$

Here $e_i$ is `offsets` for the training draws (`post.pred`) and `offsets_test` for the test draws (`post.pred.test`, with $\mu_i$ the BART prediction at $x_i^{\text{test}}$). $g^{-1}$ is exact for the Box–Cox family and the grid approximation `g_inv_approx()` for `np`, `pois` and `neg-bin`. Setting $e^{\text{test}}$ to a survey's exposure gives predicted counts for that survey. With $e^{\text{test}} = 1$, $\tilde y$ is the **rounded count at unit exposure**. This is not the continuous density $y^\ast/e$. See [Interpreting outputs](#interpreting-outputs).

### Conditional expectation

$$\mathbb{E}(y_i \mid \mu_i, \sigma, e_i) \approx \sum_{j=0}^{J_i - 1} j \left[\Phi\left(\frac{g(a_{j+1}/e_i) - \mu_i}{\sigma}\right) - \Phi\left(\frac{g(a_j/e_i) - \mu_i}{\sigma}\right)\right],$$

```math
J_i = \min\{\lceil h(e_i\, g^{-1}(q_{0.9999}))\rceil,\ \lceil 2 \max_k (y_k/e_k)\, e_i \rceil\}
```

where $q_{0.9999}$ is the 0.9999 quantile of $N(\mu_i, \sigma^2)$. The cap is stated on the density scale and reduces to the original $2\max(y)$ for unit offsets. The cut-point grid $g(a_j/e)$ depends on $e$, so `expectation_gRcpp()` is called once per distinct offset (grouped with `split()`, i.e. to 15 significant digits). With constant offsets this is a single call, as in the original.

## Transformation $g$ on the density scale

The offsets enter the likelihood only through the cut points $a_j/e_i$, and $\mu$ and $\sigma$ are never rescaled. The transformations differ only in how $g$ is defined on the density scale $y^\ast/e$.

### Case 1: fixed $g$ (`identity`, `log`, `sqrt`)

$g = g_\lambda$ ($\lambda = 1, 0, 1/2$) is applied directly to $y^\ast/e$, so $\mu$ and $\sigma$ are on the scale of $g_\lambda(y^\ast/e)$ in the units of $e$, and the priors are those of the original `bart_star()` on that scale. Changing the units of $e$ changes $\mu$ as it should. For example, for $g = \log$, $e \to ce$ shifts $\mu$ by $-\log c$, with identical count-scale predictions.

### Case 2: $g$ estimated once (`np`, `pois`, `neg-bin`)

`g_cdf(v, distribution)` (unmodified) estimates a transformation from a vector of non-negative values $v$ (the counts $y$ in the original `bart_star()`):

```math
g_v(t) = \Phi^{-1}\big(\hat F_v(t - 1)\big), \qquad t \in \{\text{unique positive } v_i\}
```

It drops knots where this is non-finite and interpolates with `splinefun(method = "monoH.FC")`, which extrapolates linearly. The estimated CDFs are

$$\hat F^{\text{np}}_v(t) = \tfrac{n}{n+1}\widehat{\text{ecdf}}_v(t), \qquad \hat F^{\text{pois}}_v(t) = F_{\text{Pois}}(t; \hat\mu_v), \qquad \hat F^{\text{nb}}_v(t) = F_{\text{NB}}\Big(t;\ \hat r = \tfrac{\hat\mu_v^2}{\hat\sigma_v^2 - \hat\mu_v},\ \hat p = \tfrac{\hat\mu_v}{\hat\sigma_v^2}\Big),$$

where $\hat\mu_v$ and $\hat\sigma_v$ are the sample mean and SD of $v$. For `neg-bin` with underdispersed data ($\hat\mu_v \ge \hat\sigma_v^2$), the code warns and uses $\hat\sigma_v = 1.1\sqrt{\hat\mu_v}$. For integer counts, $\hat F_v(t-1)$ is the estimated probability of a value strictly below $t$.

The sampler evaluates $g$ on the density scale $y^\ast/e$, while a count-scale $g_y$ would place those evaluation points relative to its knots according to the arbitrary units of $e$ (most could fall in the linear extrapolation region). The code therefore defines $g$ on the density scale in two ways.

**`np`: empirical CDF of the observed densities.** Let $d_i = y_i/e_i$ (rounded to 12 significant digits) and let $\delta$ be the smallest gap between distinct $d_i$. The code sets $M = 1$ if $\delta \ge 1$ and $M = 2/\delta$ otherwise, and uses

$$g(t) = g_{M}(M t),$$

where $g_M$ = `g_cdf(M*d, "np")`. Since $M\delta \ge 1$, the shifted point $M d_k - 1$ lies in $[M d_{k-1}, M d_k)$, so the " $t - 1$ " in `g_cdf()` again means "strictly below". At every distinct positive density $d_k$,

$$g(d_k) = \Phi^{-1}\Big(\tfrac{n}{n+1}\hat F_d(d_k^-)\Big), \qquad \hat F_d(t^-) = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}(d_i < t),$$

which is the original nonparametric transformation applied to the observed densities, with the same monotone interpolation between knots. With $e \equiv 1$, $d = y$ and $M = 1$, so $g$ is exactly the original `g_cdf(y, "np")`. The construction is invariant to the units of $e$: $e \to ce$ gives $g(t) \to g(ct)$, so $z_i = g(y_i^\ast/e_i)$ is unchanged. The densities are used only to estimate $g$. The likelihood still uses the cut points $a_j/e_i$, not $y/e$ as the response.

**`pois`, `neg-bin`: count CDF at a reference exposure.** These are count distributions moment-matched to the counts, so $g_y = $ `g_cdf(y, transformation)` is estimated from $y$ and mapped to the density scale with a reference exposure:

$$g(t) = g_y(\bar e\thinspace t), \qquad \bar e = \exp\left(\frac{1}{n}\sum_{i=1}^n \log e_i\right),$$

so that $z_i = g_y(\bar e\thinspace y_i^\ast/e_i)$. An observation at the typical exposure $e_i = \bar e$ is evaluated at its own count, where $g_y$ was estimated. Since $g$ is itself estimated, this defines the latent scale for these transformations. It is invariant to the units of $e$ ($\bar e\thinspace y^\ast/e$ is unchanged by $e \to ce$), and $g = g_y$ exactly when $e \equiv 1$.

For all three, nothing else (offsets, $\mu$, $\sigma$, priors, the other transformations) is rescaled, and $g^{-1}$ is `g_inv_approx(g, t_grid)` on a density-scale grid (see Initialization).

Remark: For `pois` and `neg-bin`, $\bar e$ enters only inside the estimated $g$. The mapping is exact when the offsets are constant. Otherwise it evaluates every observation through the count-scale CDF at the typical exposure $\bar e$, an approximation to the density-scale transformation $\Phi^{-1} \circ F_{y^\ast/e}$. A density-scale version would avoid $\bar e$, but the densities $y/e$ are not Poisson or negative-binomial, so there is no direct moment-matched analogue. `np` does not need $\bar e$, because its empirical CDF is computed directly on $y/e$.

### Case 3: $g$ sampled every iteration (`box-cox`)

$\lambda$ is updated by the slice sampler with target

$$\pi(\lambda \mid \mu, \sigma, y, e) \propto \exp\lbrace \ell(\lambda)\rbrace \thinspace \pi(\lambda), \qquad \ell(\lambda) = \sum_i \log\left[\Phi\left(\frac{g_\lambda(a_{y_i+1}/e_i) - \mu_i}{\sigma}\right) - \Phi\left(\frac{g_\lambda(a_{y_i}/e_i) - \mu_i}{\sigma}\right)\right]$$

The scaled cut points are stored once and only $g_\lambda$ is re-applied after each update.

### `ispline` is not supported with offsets

The I-spline sampler matches cut points exactly (`match(a_y, t_g)`) on an integer grid, which is incompatible with continuous, observation-specific cut points $a_j/e_i$. The code stops with an error if any training or test offset differs from 1. With unit offsets, the call is passed to the original I-spline sampler.

## Initialization

Block 1 samples the truncated normal by inverse CDF. Let $G_{l,i}$ and $G_{u,i}$ be the $N(\mu_i, \sigma^2)$ CDF at the interval endpoints, i.e. the interval on the probability scale:

$$G_{l,i} = \Phi\left(\frac{z_{\text{lower},i} - \mu_i}{\sigma}\right), \qquad G_{u,i} = \Phi\left(\frac{z_{\text{upper},i} - \mu_i}{\sigma}\right), \qquad z_i = \mu_i + \sigma\thinspace \Phi^{-1}\big(G_{l,i} + u_i (G_{u,i} - G_{l,i})\big),$$

with $u_i \sim \text{Unif}(0, 1)$ and the result clamped to $`[z_{\text{lower},i}, z_{\text{upper},i}]`$ (`G_lower_i`, `G_upper_i` in `rtruncnormRcpp()`). If $\mu_i$ lies many $\sigma$ from its interval, $G_{l,i} = G_{u,i} \in \lbrace 0, 1\rbrace$ numerically and the draw collapses to an endpoint, which may be infinite (e.g. $-\infty$ for $y_i = 0$). The starting values therefore use the original rules, evaluated on the density scale $y/e$ where the intervals live:

| Quantity | Rule |
|---|---|
| Latent $z^{(0)}$ | $g((y_i + \lvert\xi_i\rvert)/e_i)$, $\xi_i \sim N(0, 1)$ |
| `sigest` (not `box-cox`) | $\mathrm{sd}\lbrace g((y_i + 1)/e_i)\rbrace$ |
| `sigest` (`box-cox`) | original pilot MCMC if all $e_i = 1$; otherwise $\mathrm{sd}\lbrace g_{\lambda^{(0)}}((y_i + 1)/e_i)\rbrace$, $\lambda^{(0)} \sim \text{Unif}(0, 1)$ |
| Grid for $g^{-1}$ (`np`, `pois`, `neg-bin`) | 250 equally spaced points on $[0, \min\lbrace 2\max_i(y_i/e_i),\ y_{\max}/\min_i e_i\rbrace]$ plus 250 quantiles of unique $(y_i + 1)/e_i$, $y_i < y_{\max}$ |

The `sigest` rules apply only when `sigest = NULL` (the default). `sigest` matters most: it is computed once but fixes the $\sigma$ prior (via `sigquant`) for the whole run. The `box-cox` pilot MCMC fits a mean-only model to the raw counts and cannot take offsets, so with offsets it is replaced by the same rule as the other transformations. Since the half-normal start is the original rule applied on the scale $y/e$, it sits in the same relative position within each interval as in the original sampler. With $e \equiv 1$, every rule reduces to the original code.

## Implementation

All changes are in **`bart_star_offset.R`**. No helper is added or modified: `a_j`, `g_bc`, `g_inv_bc`, `g_cdf`, `g_inv_approx`, `round_floor`, `rtruncnormRcpp`, `logLikeRcpp`, `logLikePointRcpp`, `expectation_gRcpp`, `uni.slice`, `genMCMC_star` and `bart_star_ispline` are called unchanged.

### Interface

```r
bart_star(y, X, X_test = NULL, y_test = NULL,
          offsets      = 1,                     # training offsets e_i > 0 (scalar or length(y))
          offsets_test = 1,                     # test offsets (scalar or nrow(X_test))
          transformation = 'np', ...)           # all other arguments as before
```

- A scalar is recycled, otherwise one value per observation.
- Values must be finite and strictly positive, with lengths `length(y)` and `nrow(X_test)`. Otherwise an error is raised. `offsets_test` is validated only when `X_test` is supplied and is otherwise ignored, apart from the `ispline` check.
- The returned list has the same elements as the original. `post.pred`, `post.log.like.point`, `fitted.values` and `post.fitted.values` are at `offsets`. `post.pred.test`, `post.fitted.values.test` and `post.log.pred.test` are at `offsets_test`.

### Interpreting outputs

- **Latent scale.** `post.mu.test` and `post.sigma` are on the scale of $g(y^\ast/e)$. For `identity`, `log`, `sqrt` and `box-cox`, this is $g_\lambda$ of the density in the units of $e$. For `np`, it is the transformation estimated from the empirical CDF of $y/e$. For `pois` and `neg-bin`, it is $g(t) = g_y(\bar e\thinspace t)$.
- **Continuous density.** For $g = \log$, $y^\ast/e \mid x, \mu, \sigma$ is log-normal, so $\mathbb E(y^\ast/e \mid x, \mu, \sigma) = \exp(\mu(x) + \sigma^2/2)$.
- **Counts.** Count-scale outputs are at the supplied exposures (see Prediction).

### Code changes (marked `# OFFSETS:`)

| Location | Change |
|---|---|
| Argument checks | validate and recycle `offsets` / `offsets_test` |
| `ispline` dispatch | error if any `offsets` or `offsets_test` $\neq 1$; otherwise drop the offset arguments and call the original sampler |
| `g` for `np` | `d = signif(y/offsets, 12)`; `M = 1` if the smallest gap of `unique(d)` is $\ge 1$, else `2/gap`; `g_M = g_cdf(M*d, 'np')`; `g = function(t) g_M(M*t)` |
| `g` for `pois`, `neg-bin` | `e_bar = exp(mean(log(offsets)))`; `g_y = g_cdf(y, transformation)`; `g = function(t) g_y(e_bar*t)` |
| Grid for `g_inv_approx()` | built from `y/offsets` and `(y + 1)/offsets` |
| Initial `z_star` | `g((y + abs(rnorm(n)))/offsets)` |
| `sigest` | `sd(g((y + 1)/offsets))`; pilot MCMC for `box-cox` only if `all(offsets == 1)` |
| Cut points | `a_y = a_j(y, y_max)/offsets`, `a_yp1 = a_j(y + 1, y_max)/offsets` (Blocks 1, 3, pointwise log-likelihood) |
| Predictive draws | `round_floor(g_inv(z_tilde) * offsets, y_max)`; likewise with `offsets_test` |
| Conditional expectation | `Jmax` at exposure $e_i$, capped at `ceiling(2*max(y/offsets)*offsets)`; one `expectation_gRcpp()` call per distinct offset with grid `g(a_j(0:J)/e)` (training and test) |
| Test log-predictive score | `g(a_j(y_test)/offsets_test)`, `g(a_j(y_test + 1)/offsets_test)` |

### Example

```r
library(countSTAR)
source("bart_star_offset.R")

fit = bart_star(y = y, X = X, offsets = e,
                X_test = X_new, offsets_test = 1,
                transformation = 'log', save_y_hat = TRUE)

fit$fitted.values                        # E(y_i | x_i, e_i): expected counts at observed exposures
colMeans(fit$post.fitted.values.test)    # E(y | x_new, e = 1): expected count at unit exposure
```

### Notes

- **Checks.** With `offsets = 1`, the output is `identical()` to the original `bart_star()` (same seed) for all seven transformations. With offsets, `np`, `pois` and `neg-bin` results are unchanged by $e \to ce$ by construction (up to the $g^{-1}$ grid), and `log` results are identical except that $\mu$ shifts by $-\log c$.
- **Cost.** Only the conditional expectation scales with the number of distinct offsets (one `expectation_gRcpp()` call each). Sampling, prediction and the log-likelihood add no per-iteration cost. Runtime in the simulation study (`README.md`) was acceptable.
- **Not supported:** `ispline` with non-unit offsets; regression weights (`README_weight.md`).
