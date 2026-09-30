# countSTAR: `bart_star()` with offsets

This repository extends `bart_star()` from [countSTAR](https://github.com/bking124/countSTAR) to support **model offsets** (exposures) for count-valued BART regression. **Regression weights** are documented separately in `README_weight.md`. The motivating application is national density and distribution modelling for Canadian birds, but the extension applies to any STAR model with a known, mechanistic exposure term.

## Motivation

`bart_star()` fits a **S**imultaneous **T**ransformation **A**nd **R**ounding (STAR) model with a BART latent mean via `dbarts`. It captures nonlinearities and interactions (e.g. thresholds in temperature, elevation, distance to water) that GLMs/GAMs miss, and returns calibrated posterior uncertainty without the bootstrapping that boosted regression trees require.

**Offsets.** Each point count has a known offset $e_i > 0$ (e.g. detection proportion or survey effort). The usual log-link model is $\text{count}_i \sim \text{Poisson}\big(e_i \exp(\eta_i)\big)$, $\eta_i = \mu(x_i)$, so $\mu(x_i)$ is a log density rather than a log expected count. Because $e_i$ acts multiplicatively and mechanistically, it enters as an offset, not as a covariate. Offsets turn a model for counts into a model for absolute density, which allows population-size estimation. In STAR, $g$ need not be $\log$, so the offset cannot be absorbed into the linear predictor. Replacing $y$ by $y/e$ is also incorrect, even for $g = \log$ (see below).

## Background: the STAR model

$$y_i = h(y_i^\ast) = \lfloor y_i^\ast\rfloor, \qquad z_i = g(y_i^\ast), \qquad z_i = \mu(x_i) + \varepsilon_i, \quad \varepsilon_i \sim N(0, \sigma^2)$$

Here $h$ is the floor (negatives mapped to 0, capped at `y_max`), $g$ is increasing, and $\mu(\cdot)$ is a BART sum of trees. Since $g$ is monotone,

$$\Pr(y_i = j \mid x_i) = \Pr\big(g(a_j) \le z_i < g(a_{j+1})\big), \qquad a_j = j, \quad a_0 = -\infty, \quad a_{y_{\max}+1} = \infty$$

with $a_j$ given by `a_j()`. The Gibbs sampler:

1. **Sample $z^\ast$:** draw $z_i \mid y_i$ from $N(\mu(x_i), \sigma^2)$ truncated to $[g(a_{y_i}), g(a_{y_i+1}))$.
2. **Sample $(\mu, \sigma)$:** update the BART fit (`dbarts`) with response $z^\ast$.
3. **(Optional) update the transformation**: if $g$ is being learned rather than fixed.
4. **Predict:** repeat, draw $\tilde z \sim N(\mu, \sigma^2)$. Use $g^{-1}$ and rounding to generate posterior predictive draws $\tilde y = h(g^{-1}(\tilde z))$.

## Model with offsets

The rounding is unchanged. The transformation acts on the latent count per unit exposure:

$$y_i = \lfloor y_i^\ast\rfloor, \qquad z_i = g\left(\frac{y_i^\ast}{e_i}\right) = \mu(x_i) + \varepsilon_i, \quad \varepsilon_i \sim N(0, \sigma^2)$$

*Notation.* This section uses $(e, g)$. The code uses the normalized pair $(\tilde e, g_0)$ defined in [Normalization](#normalization) (to address the issue with `g_cdf()` knots). All formulas hold unchanged with $(\tilde e, g_0)$ in place of $(e, g)$.

### Likelihood

For $e_i > 0$ and monotone $g$,

$$y_i = j \iff a_j \le y_i^\ast < a_{j+1} \iff g\left(\frac{a_j}{e_i}\right) \le z_i < g\left(\frac{a_{j+1}}{e_i}\right)$$

$$\ell(\mu, \sigma, g) = \sum_{i=1}^n \log\left[\Phi\left(\frac{g(a_{y_i+1}/e_i) - \mu_i}{\sigma}\right) - \Phi\left(\frac{g(a_{y_i}/e_i) - \mu_i}{\sigma}\right)\right]$$

with $\pm\infty/e_i = \pm\infty$. This is what `logLikeRcpp` / `logLikePointRcpp` compute on the training data.

**Why not $y/e$.** The interval for $y_i = j$ is $[j/e_i, (j+1)/e_i)$, of width $1/e_i$. Using $y/e$ as data gives $[j/e_i, j/e_i + 1)$, which is wrong whenever $e_i \neq 1$.

**Special case $g = \log$.** $\log y_i^\ast = \mu(x_i) + \log e_i + \varepsilon_i$ is the familiar additive log-offset.

### Gibbs sampler with offsets

Only the cut points change, from $a_j$ to $a_j/e_i$:

| Step | With offsets |
|---|---|
| Block 1: sample $z$ | $N(\mu_i, \sigma^2)$ truncated to $[g(a_{y_i}/e_i),\thinspace g(a_{y_i+1}/e_i))$ |
| Block 2: sample $(\mu, \sigma)$ | unchanged |
| Block 3: Box-Cox $\lambda$ | slice sampler on $\ell(\lambda) + \log \pi(\lambda)$ with $g_\lambda(a_{y_i}/e_i)$, $g_\lambda(a_{y_i+1}/e_i)$; $\pi(\lambda) = N(1/2, 1)$ truncated to $[0, 3]$ |
| Pointwise log-likelihood / WAIC | offset-adjusted bounds |
| Test log-predictive score | $g(a_{y_j}/e_j^{\text{test}})$, $g(a_{y_j+1}/e_j^{\text{test}})$ |

The scaled cut points are computed once, before the MCMC loop. For Box-Cox, only $g_\lambda$ is re-applied at each iteration.

### Prediction

$$\tilde z_i \sim N(\mu_i, \sigma^2), \qquad \tilde y_i = h\big(e^{\text{test}}_i\thinspace g^{-1}(\tilde z_i)\big)$$

Setting $e^{\text{test}}$ to a survey's exposure gives predicted counts for that survey. With $e^{\text{test}} = 1$, $\tilde y$ is the **rounded count at unit exposure** (in the units of $e$). This is not the continuous density $y^\ast/e$. See [Interpreting outputs](#interpreting-outputs).

### Conditional expectation

$$\mathbb{E}(y_i \mid \mu_i, \sigma, e_i) \approx \sum_{j=0}^{J_i - 1} j \left[\Phi\left(\frac{g(a_{j+1}/e_i) - \mu_i}{\sigma}\right) - \Phi\left(\frac{g(a_j/e_i) - \mu_i}{\sigma}\right)\right],$$

```math
J_i = \min\{\lceil h(e_i\, g^{-1}(q_{0.9999}))\rceil,\ \lceil 2 \max_k (y_k/e_k)\, e_i \rceil\}
```

where $q_{0.9999}$ is the 0.9999 quantile of $N(\mu_i, \sigma^2)$. The cap is stated on the density scale and reduces to $2\max(y)$ for unit offsets. The cut-point grid $g(a_j/e)$ depends on $e$, so `expectation_gRcpp()` is called once per distinct offset (grouped with `split()`, i.e. to 15 significant digits). With constant offsets this is a single call, as in the original.

## Transformation $g$ and normalization of the offsets

### Normalization

This step is to address the issue with `g_cdf()` without modifying the original helper function. For the transformations estimated from the data (`np`, `pois`, `neg-bin`), `g_cdf(y)` is fitted to the raw counts, with knots at the distinct positive counts, but the sampler evaluates it at the offset-adjusted values $y^\ast/e$. Without normalization, where these values fall relative to the knots depends on the arbitrary units or range of $e$, which can push most of them into the linear extrapolation region and change the fitted model. Rescaling the offsets so that a typical observation has relative exposure 1 keeps $g$ evaluated where it was estimated and makes the fit invariant to the units of $e$. For the Box-Cox family (`identity`, `log`, `sqrt`, `box-cox`), a rescaling of $e$ only shifts and scales $z$, which $(\mu, \sigma)$ absorb, so the normalization does not change the likelihood there (see Case 1). The code replaces offsets by **relative exposures**, using the geometric mean of the training offsets:

$$\tilde e_i = \frac{e_i}{\bar e}, \qquad \tilde e^{\text{test}}_j = \frac{e^{\text{test}}_j}{\bar e}, \qquad \bar e = \exp\left(\frac{1}{n}\sum_{i=1}^n \log e_i\right)$$

This is a reparametrization of $g$:

$$z_i = g_0\left(\frac{y_i^\ast}{\tilde e_i}\right) = g\left(\frac{y_i^\ast}{e_i}\right), \qquad g_0(t) = g(\frac{t}{\bar e})$$

Consequences:

1. **Invariance.** Changing $e \to ce$ leaves $\tilde e$ unchanged. Test offsets must be given in the same units.
2. **Data-estimated $g$.** An observation at typical exposure ($\tilde e_i = 1$) has the original cut points $a_j$, so $g$ is evaluated where it was estimated, whatever the units of $e$.

### Case 1: fixed $g$ (`identity`, `log`, `sqrt`)

For $\lambda > 0$ and $c > 0$, $g_\lambda(t/c) = c^{-\lambda} g_\lambda(t) + (c^{-\lambda} - 1)/\lambda$; for $\lambda = 0$, $\log(t/c) = \log t - \log c$. The affine map $z \mapsto az + b$ ($a > 0$) is absorbed by $(\mu, \sigma) \mapsto (a\mu + b, a\sigma)$. The likelihood therefore does not depend on $\bar e$, and the normalization affects only the scale of the BART prior (approximately, since `dbarts` sets its prior scale from the initial response and `sigest`).

### Case 2: $g$ estimated once (`np`, `pois`, `neg-bin`)

`g_cdf(y, distribution)` (unmodified) sets

```math
g(j) = \Phi^{-1}\big(\hat F_y(j - 1)\big), \qquad j \in \{\text{unique positive } y_i\}
```

dropping knots where this is non-finite, and interpolates with `splinefun(method = "monoH.FC")` (linear extrapolation). The estimated CDFs are

$$\hat F^{\text{np}}_y(t) = \tfrac{n}{n+1}\widehat{\text{ecdf}}_y(t), \qquad \hat F^{\text{pois}}_y(t) = F_{\text{Pois}}(t; \hat\mu_y), \qquad \hat F^{\text{nb}}_y(t) = F_{\text{NB}}\Big(t;\ \hat r = \tfrac{\hat\mu_y^2}{\hat\sigma_y^2 - \hat\mu_y},\ \hat p = \tfrac{\hat\mu_y}{\hat\sigma_y^2}\Big),$$

where $\hat\mu_y$ and $\hat\sigma_y$ are the sample mean and SD of $y$. For `neg-bin` with underdispersed data ($\hat\mu_y \ge \hat\sigma_y^2$), the code warns and uses $\hat\sigma_y = 1.1\sqrt{\hat\mu_y}$. $g$ is used as $g_0$, i.e. calibrated at the reference exposure $\bar e$, evaluated at $a_j/\tilde e_i$, and held fixed.

### Case 3: $g$ sampled every iteration (`box-cox`)

$$\pi(\lambda \mid \mu, \sigma, y, e) \propto \exp\lbrace \ell(\lambda)\rbrace \thinspace \pi(\lambda), \qquad \ell(\lambda) = \sum_i \log\left[\Phi\left(\frac{g_\lambda(a_{y_i+1}/\tilde e_i) - \mu_i}{\sigma}\right) - \Phi\left(\frac{g_\lambda(a_{y_i}/\tilde e_i) - \mu_i}{\sigma}\right)\right]$$

By Case 1, $\ell(\lambda)$ is invariant to $\bar e$ after a $\lambda$-dependent rescaling of $(\mu, \sigma)$. Because the BART prior scale is fixed, the posterior of $\lambda$ is only approximately invariant.

### `ispline` is not supported with offsets

The I-spline sampler matches cut points exactly (`match(a_y, t_g)`) on an integer grid, which is incompatible with continuous, observation-specific cut points $a_j/e_i$. The code stops with an error if **any raw training or test offset** differs from 1. With unit offsets, the call is passed to the original I-spline sampler.

## Initialization

Block 1 samples the truncated normal by inverse CDF. Let $G_{l,i}$ and $G_{u,i}$ be the $N(\mu_i, \sigma^2)$ CDF at the interval endpoints, i.e. the interval on the probability scale:

$$G_{l,i} = \Phi\left(\frac{z_{\text{lower},i} - \mu_i}{\sigma}\right), \qquad G_{u,i} = \Phi\left(\frac{z_{\text{upper},i} - \mu_i}{\sigma}\right), \qquad z_i = \mu_i + \sigma\thinspace \Phi^{-1}\big(G_{l,i} + u_i (G_{u,i} - G_{l,i})\big),$$

with $u_i \sim \text{Unif}(0, 1)$ and the result clamped to $[z_{\text{lower},i}, z_{\text{upper},i}]$ (`G_lower_i`, `G_upper_i` in `rtruncnormRcpp()`). If $\mu_i$ lies many $\sigma$ from its interval, $G_{l,i} = G_{u,i} \in \lbrace 0, 1\rbrace$ numerically and the draw collapses to an endpoint, which may be infinite (e.g. $-\infty$ for $y_i = 0$). With varying offsets the interval locations shift with $\tilde e_i$, so the starting values are computed on the $y/\tilde e$ scale. `sigest` matters most: it is computed once but fixes the $\sigma$ prior (via `sigquant`) for the whole run.

- **Constant offsets:** $\tilde e_i \equiv 1$, so the offset-adjusted scale is the count scale and the original initialization is kept.
- **Varying offsets:** intervals shift with $\tilde e_i$, so each start is placed inside its own interval and `sigest` is computed on the $y/\tilde e$ scale.

| Quantity | Constant offsets | Varying offsets |
|---|---|---|
| Latent $z^{(0)}$ | $g((y_i + \lvert\xi_i\rvert)/\tilde e_i)$, $\xi_i \sim N(0, 1)$ (original) | $g((y_i + U_i)/\tilde e_i)$, $U_i \sim \text{Unif}(0, 1)$ |
| `sigest` (not `box-cox`) | $\mathrm{sd}\lbrace g((y_i + 1)/\tilde e_i)\rbrace$ | same |
| `sigest` (`box-cox`) | original pilot MCMC (mean-only STAR model) | $\mathrm{sd}\lbrace g_{\lambda^{(0)}}((y_i + 1)/\tilde e_i)\rbrace$, $\lambda^{(0)} \sim \text{Unif}(0, 1)$ |
| Grid for $g^{-1}$ (`np`, `pois`, `neg-bin`) | 250 equally spaced points on $[0, \min\lbrace 2\max_i(y_i/\tilde e_i),\ y_{\max}/\min_i \tilde e_i\rbrace]$ plus 250 quantiles of unique $(y_i + 1)/\tilde e_i$, $y_i < y_{\max}$ | same |

**Start inside the interval.** For $U_i \in (0, 1)$, $y_i/\tilde e_i < (y_i + U_i)/\tilde e_i < (y_i + 1)/\tilde e_i$, so $z_i^{(0)}$ lies strictly inside $\big(g(a_{y_i}/\tilde e_i),\thinspace g(a_{y_i+1}/\tilde e_i)\big)$. The half-normal start exceeds $y_i + 1$ with probability $2\lbrace 1 - \Phi(1)\rbrace \approx 0.32$, which is outside the interval when $y_i < y_{\max}$.

**Box-Cox `sigest`.** The pilot MCMC fits raw counts, whose spread includes exposure variation that the offsets already explain, so it overstates $\sigma$. With varying offsets the code uses the same rule as the other transformations instead.

## Implementation

All changes are in **`bart_star_offset.R`**. No helper is added or modified: `a_j`, `g_bc`, `g_inv_bc`, `g_cdf`, `g_inv_approx`, `round_floor`, `rtruncnormRcpp`, `logLikeRcpp`, `logLikePointRcpp`, `expectation_gRcpp`, `uni.slice` and `genMCMC_star` are called unchanged.

### Interface

```r
bart_star(y, X, X_test = NULL, y_test = NULL,
          offsets      = rep(1, length(y)),     # training offsets e_i > 0
          offsets_test = rep(1, NROW(X_test)),  # test offsets
          transformation = 'np', ...)           # all other arguments as before
```

- One value per observation. A scalar is recycled (e.g. `offsets_test = 1`).
- Values must be finite and strictly positive, with lengths `length(y)` and `nrow(X_test)`. Otherwise an error is raised. `offsets_test` is checked only when `X_test` is supplied, and is otherwise ignored.
- The returned list has the same elements as the original. `post.pred`, `fitted.values` and `post.fitted.values` are at `offsets`; `post.pred.test`, `post.fitted.values.test` and `post.log.pred.test` are at `offsets_test`.

### Interpreting outputs

- **Latent scale.** The code applies the named transformation (e.g. $g_\lambda$) as $g_0$, so `post.mu.test` and $\sigma$ are relative to $\bar e$. Let $(\mu, \sigma)$ denote the parameters when $g_\lambda$ is applied directly to $y^\ast/e$. Then for $g = \log$: $\mu^{\text{code}} = \mu + \log\bar e$. For $\lambda > 0$: $\mu^{\text{code}} = \bar e^{\lambda}\mu + (\bar e^{\lambda} - 1)/\lambda$ and $\sigma^{\text{code}} = \bar e^{\lambda}\sigma$. Count-scale outputs need no correction.
- **Continuous density.** Since $y^\ast/e = g_0^{-1}(z)/\bar e$, for $g = \log$: $\mathbb E(y^\ast/e \mid x) = \exp(\mu^{\text{code}} + \sigma^2/2)/\bar e$. The code does not return $\bar e$; compute it as `exp(mean(log(offsets)))`.

### Code changes (marked `# OFFSETS:`)

| Location | Change |
|---|---|
| Argument checks | validate and recycle `offsets` / `offsets_test` |
| `ispline` dispatch | error if any raw `offsets` or `offsets_test` $\neq 1$; otherwise drop the offset arguments and call the original sampler |
| Normalization | `offsets_vary` (raw values); `e_bar = exp(mean(log(offsets)))`; divide `offsets` and `offsets_test` by `e_bar` |
| Grid for `g_inv_approx()` | built from `y/offsets` and `(y + 1)/offsets` |
| Initial `z_star` | varying: `g((y + runif(n))/offsets)`; constant: `g((y + abs(rnorm(n)))/offsets)` |
| `sigest` | `sd(g((y + 1)/offsets))`; pilot MCMC for `box-cox` only with constant offsets |
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

- **Cost.** Only the conditional expectation scales with the number of distinct offsets (one `expectation_gRcpp()` call each). Sampling, prediction and the log-likelihood add no per-iteration cost.
- **Finite `y_max` (inherited from the original).** The expectation grid `a_j(0:J)` and the test score `a_j(y_test + 1)` omit `y_max`, so their top interval ends at $(y_{\max}+1)/e$ rather than $\infty$. The training cut points are exact.
- **Not supported:** `ispline` with non-unit offsets; regression weights (`README_weight.md`).
