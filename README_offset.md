# countSTAR: `bart_star()` with offsets

This repository extends `bart_star()` from the [countSTAR](https://github.com/bking124/countSTAR) package to support **model offsets** (exposures) for count-valued BART regression. **Regression weights** are later work and are documented separately in `README_weight.md`. The motivating application is national density and distribution modelling for Canadian birds, but the extension applies to any STAR model with a known, mechanistic exposure term.

## Motivation

`bart_star()` fits a **S**imultaneous **T**ransformation **A**nd **R**ounding (STAR) model with a Bayesian Additive Regression Trees (BART) latent mean via `dbarts`. It suits ecological count data because it captures strong nonlinearities and covariate interactions (e.g. thresholds in temperature, elevation, and proximity to water) that GLMs/GAMs miss. It also returns calibrated posterior uncertainty for predictions and marginal effects. Boosted regression trees (BRTs), which ecologists have used previously, can only provide that uncertainty through costly bootstrapping.

**Offsets.** Each point count has a known offset $e_i > 0$, e.g. the estimated proportion of birds present that are detected, or the survey effort. Under a log link, the commonly used model is $\text{count}_i \sim \text{Poisson}\big(e_i \exp(\eta_i)\big), \eta_i = \mu(x_i)$, so $\mu(x_i)$ is a log density, not a log expected count. The relationship with $e_i$ is mechanistic and multiplicative, so it enters as an offset rather than as a covariate with an estimated effect.

Offsets convert the distribution of counts into a distribution of absolute population density. This lets us compare areas by relative abundance and also estimate population size, a pressing problem for ecologists. In STAR, we don't use Poisson model and $g(\cdot)$ need not be $\log$, so the offset cannot be absorbed by shifting the linear predictor. Replacing $y$ by $y/e$ is also incorrect, even for $g = \log$, as shown below.

## Background: the STAR model

STAR defines a count-valued model in two parts: (i) A continuous latent variable $y_i^\ast$ with a Gaussian regression model (ii) A transformation-and-rounding link from $y_i^\ast$ to the observed count $y_i$.

$$y_i = h(y_i^\ast) = \lfloor y_i^\ast\rfloor, \qquad z_i = g(y_i^\ast), \qquad z_i = \mu(x_i) + \varepsilon_i, \quad \varepsilon_i \sim N(0, \sigma^2)$$

Here $h$ is the floor operator, capped at `y_max`, and $g$ is a known or estimated increasing transformation. In `bart_star()`, $\mu(\cdot)$ is a BART sum-of-trees function. Because $g$ is monotone,

$$\Pr(y_i = j \mid x_i) = \Pr\big(g(a_j) \le z_i < g(a_{j+1})\big), \qquad a_j = j, \quad a_0 = -\infty, \quad a_{y_{\max}+1} = \infty$$

These cut points are given by `a_j()` in `helper_functions.R`. The interval endpoints $z_\text{lower, i} = g(a_{y_i})$ and $z_\text{upper, i} = g(a_{y_i+1})$ drive the Gibbs sampler:

1. **Sample $z^\ast$:** draw $z_i \mid y_i$ from $N(\mu(x_i), \sigma^2)$ truncated to $[g(a_{y_i}), g(a_{y_i+1}))$.
2. **Sample $(\mu, \sigma)$:** update the BART fit (`dbarts`) with response $z^\ast$.
3. **(Optional) update the transformation**: if $g$ is being learned rather than fixed.
4. **Predict:** repeat, draw $\tilde z \sim N(\mu, \sigma^2)$. Use $g^{-1}$ and rounding to generate posterior predictive draws $\tilde y = h(g^{-1}(\tilde z))$.

## Model with offsets

Let $e_i > 0$ be the known offset for observation $i$. The rounding step is unchanged, so $y_i$ stays count-valued. The transformation now acts on the offset-adjusted latent variable:

$$y_i = h(y_i^\ast) = \lfloor y_i^\ast\rfloor, \qquad z_i = g\left(\frac{y_i^\ast}{e_i}\right), \qquad z_i = \mu(x_i) + \varepsilon_i, \quad \varepsilon_i \sim N(0, \sigma^2)$$

Here $y_i^\ast/e_i$ is the latent count per unit exposure (density), and $\mu(\cdot)$ is modelled on that scale.

### Likelihood

Since $g$ is monotone and $e_i > 0$,

$$y_i = j \iff a_j \le y_i^\ast < a_{j+1} \iff \frac{a_j}{e_i} \le \frac{y_i^\ast}{e_i} < \frac{a_{j+1}}{e_i} \iff g\left(\frac{a_j}{e_i}\right) \le z_i < g\left(\frac{a_{j+1}}{e_i}\right)$$

Therefore

$$\Pr(y_i = j \mid \mu_i, \sigma) = \Phi\left(\frac{g(a_{j+1}/e_i) - \mu_i}{\sigma}\right) - \Phi\left(\frac{g(a_j/e_i) - \mu_i}{\sigma}\right)$$

with $\pm\infty$ endpoints preserved ($-\infty/e_i = -\infty$, $\infty/e_i = \infty$). The log-likelihood computed by `logLikeRcpp` / `logLikePointRcpp` is

$$\ell(\mu, \sigma, g) = \sum_{i=1}^n \log[\Phi(\frac{g(a_{y_i+1}/e_i) - \mu_i}{\sigma}) - \Phi(\frac{g(a_{y_i}/e_i) - \mu_i}{\sigma})]$$

**Why not $y/e$.** The interval for $y_i = j$ is $[j/e_i, (j+1)/e_i)$, with width $1/e_i$. Passing $y/e$ as data would instead give $[j/e_i, j/e_i + 1)$, which has the wrong upper endpoint whenever $e_i \neq 1$.

**Special case $g = \log$.** The model becomes $\log y_i^\ast = \mu(x_i) + \log e_i + \varepsilon_i$. This recovers the familiar additive log-offset of Poisson/negative-binomial regression.

### Gibbs sampler with offsets

The offset changes only the cut points, from $a_j$ to $a_j / e_i$. The sampler structure is unchanged:

| Step | With offsets |
|---|---|
| Block 1: sample $z$ | $z_i \sim N(\mu_i, \sigma^2)$ truncated to $[g(a_{y_i}/e_i),\, g(a_{y_i+1}/e_i))$ |
| Block 2: sample $(\mu, \sigma)$ | unchanged (`dbarts` on the $z$ scale) |
| Block 3: sample Box-Cox $\lambda$ | slice sampler on $\ell(\lambda) + \log \pi(\lambda)$, with $g_\lambda(a_{y_i}/e_i)$ and $g_\lambda(a_{y_i+1}/e_i)$; $\pi(\lambda) = N(1/2, 1)$ truncated to $[0, 3]$ |
| Pointwise log-likelihood / WAIC | uses the offset-adjusted bounds |
| Test log-predictive score | uses $g(a_{y_j}/e_j^{\text{test}})$ and $g(a_{y_j+1}/e_j^{\text{test}})$ |

The scaled cut points $a_{y_i}/e_i$ and $a_{y_i+1}/e_i$ are computed **once**, before the MCMC loop. For a fixed $g$, the bounds $g(\cdot)$ are also computed once. For Box-Cox, only $g_\lambda$ is re-applied to these stored cut points at each iteration, as in the original sampler.

### Prediction

The offset is re-introduced after inverting $g$ and before rounding:

$$\tilde z_i \sim N(\mu_i, \sigma^2), \qquad \tilde y_i = h\big(e_\text{test}\cdot g^{-1}(\tilde z_i)\big)$$

Setting $e^{\text{test}}$ to a survey's exposure gives predicted counts for that survey.

### Conditional expectation

$$\mathbb{E}(y_i \mid \mu_i, \sigma, e_i) = \sum_{j=0}^{J_i - 1} j [\Phi(\frac{g(a_{j+1}/e_i) - \mu_i}{\sigma}) - \Phi(\frac{g(a_j/e_i) - \mu_i}{\sigma})]$$

The truncation point is

$$J_i = \min\{\lceil h\big(e_i \, g^{-1}(q_{0.9999})\big)\rceil,\ \lceil 2 \max_k (y_k/e_k) \cdot e_i \rceil\}$$

where $q_{0.9999}$ is the 0.9999 quantile of $N(\mu_i, \sigma^2)$. The cap $2\max_k(y_k/e_k)$ is stated on the density scale and converted to counts at exposure $e_i$; for unit offsets it reduces to $2\max(y)$. `expectation_gRcpp()` takes a cut-point grid $g(a_j)$, $j = 0, 1, \dots$, shared by all observations. With offsets the grid $g(a_j/e_i)$ depends on $e_i$. Observations are therefore grouped by distinct offset value $e$, and `expectation_gRcpp()` is called once per group with the grid `g(a_j(0:J)/e)`. With constant offsets this is a single call, identical to the original code.

## Transformation $g$ and normalization of the offsets

### Normalization

This step is to address the issue with `g_cdf(y)` without modifying the original helper function. For the transformations estimated from the data (`np`, `pois`, `neg-bin`), `g_cdf(y)` is fitted to the raw counts, with knots at $t = 1, 2, \dots$, but the sampler evaluates it at the offset-adjusted values $y^\ast/e$. Without normalization, where these values fall relative to the knots depends on the arbitrary units or range of $e$ (e.g. minutes vs. hours), which can push most of them into the linear extrapolation region and change the fitted model. Rescaling the offsets so that a typical observation has relative exposure 1 keeps $g$ evaluated where it was estimated and makes the fit invariant to the units of $e$. For the Box-Cox family (`identity`, `log`, `sqrt`, `box-cox`), a rescaling of $e$ only shifts and scales $z$, which $(\mu, \sigma)$ absorb, so the normalization has no effect there. Internally, the offsets are replaced by **relative exposures**

$$\tilde e_i = \frac{e_i}{\bar e}, \qquad \tilde e^{\text{test}}_j = \frac{e^{\text{test}}_j}{\bar e}, \qquad \bar e = \exp\left(\frac{1}{n} \sum_{i=1}^n \log e_i\right)$$

where $\bar e$ is the geometric mean of the training offsets. The test offsets are divided by the **same** $\bar e$ (which is part of the fitted model), so inputs and outputs stay in the original units. The normalization is a reparametrization of $g$:

$$z_i = g_0(\frac{y_i^\ast}{\tilde e_i}) = g(\frac{y_i^\ast}{e_i}) \quad \text{with} \quad g(t) = g_0(\bar e t)$$

It has three consequences:

1. **Invariance to the units of $e$.** Rescaling $e \to ce$ for $c > 0$ (e.g. minutes to hours) leaves $\tilde e$ unchanged, so the fitted model is identical.
2. **No change for unit offsets.** If $e_i \equiv 1$, then $\bar e = 1$ and $\tilde e = e$.
3. **Natural scale for an estimated $g$.** An observation at the typical exposure ($e_i = \bar e$, i.e. $\tilde e_i = 1$) has exactly the original cut points $a_j$. A $g$ estimated from the counts is therefore evaluated on the scale where it was estimated.

### Case 1: known, fixed $g$ (`identity`, `log`, `sqrt`)

The Box-Cox family is closed under rescaling up to an affine map. For $\lambda > 0$,

$$g_\lambda(t/c) = \frac{(t/c)^\lambda - 1}{\lambda} = c^{-\lambda} g_\lambda(t) + \frac{c^{-\lambda} - 1}{\lambda}$$

and for $\lambda = 0$, $\log(t/c) = \log t - \log c$. An affine map $z \mapsto a z + b$ with $a > 0$ is absorbed by $(\mu, \sigma) \mapsto (a\mu + b,\, a\sigma)$, so the likelihood does not depend on the choice of $\bar e$. The normalization affects only the prior scale of $\mu$ and $\sigma$.

### Case 2: $g$ estimated once from the data (`np`, `pois`, `neg-bin`)

$g$ is estimated before the sampler by the unmodified `g_cdf(y, distribution)`:

$$g(j) = \Phi^{-1}\big(\hat F_y(j - 1)\big), \qquad j \in \{\text{unique positive } y_i\}$$

This is interpolated by a monotone Hermite spline (`splinefun(method = "monoH.FC")`, linear outside the knot range), with

$$\hat F^{\text{np}}_y(t) = \frac{n}{n+1}\widehat{\text{ecdf}}_y(t), \qquad \hat F_y(t)^{\text{pois}} = F_{\text{Pois}}(t; \hat\mu_y), \qquad \hat F_y(t)^{\text{neg-bin}} = F_{\text{NB}}(t; \hat r = \frac{\hat\mu_y^2}{\hat\sigma_y^2 - \hat\mu_y}, \hat p = \frac{\hat\mu_y}{\hat\sigma_y^2})$$

where $\hat\mu_y$ and $\hat\sigma_y$ are the sample mean and SD of $y$. Combined with the normalization, $g$ is calibrated to the pooled count distribution at the reference exposure $\bar e$. It is then evaluated at the relative cut points $a_j/\tilde e_i$ and held fixed for the rest of the sampler, as in Case 1.

### Case 3: $g$ sampled every iteration (`box-cox`)

$\lambda$ is updated by the univariate slice sampler with target

$$\pi(\lambda \mid \mu, \sigma, y, e) \propto \exp\{\ell(\lambda)\} \pi(\lambda), \qquad \ell(\lambda) = \sum_i \log[\Phi(\frac{g_\lambda(a_{y_i+1}/\tilde e_i) - \mu_i}{\sigma}) - \Phi(\frac{g_\lambda(a_{y_i}/\tilde e_i) - \mu_i}{\sigma})]$$

The scaled cut points are stored once, and only $g_\lambda$ is re-applied after each update to refresh $z_{\text{lower}}$ and $z_{\text{upper}}$. By the affine argument in Case 1, $\ell(\lambda)$ does not depend on $\bar e$ once $(\mu, \sigma)$ are rescaled accordingly.

### `ispline` is not supported with offsets

The I-spline sampler locates cut points by exact matching (`match(a_y, t_g)`) against an integer grid, which assumes $a_j = j$ is shared across observations. Under an offset, $a_j/e_i$ is continuous and observation-specific. `transformation = "ispline"` therefore stops with an error when any offset differs from 1. With unit offsets, it is dispatched to the original I-spline sampler unchanged.

## Initialization

Block 1 draws $z_i$ by inverse CDF:

$$z_i = \mu_i + \sigma\Phi^{-1}\big(G_{l,i} + u_i\,(G_{u,i} - G_{l,i})\big), \qquad G_{l,i} = \Phi(\tfrac{z_{\text{lower},i} - \mu_i}{\sigma}),\quad G_{u,i} = \Phi(\tfrac{z_{\text{upper},i} - \mu_i}{\sigma})$$

with the result clamped to $[z_{\text{lower},i}, z_{\text{upper},i}]$. If the current $\mu_i$ lies many $\sigma$ away from its truncation interval, $G_{l,i}$ and $G_{u,i}$ are numerically both 0 or both 1. The draw then sticks at an interval endpoint instead of following the truncated normal. With offsets, the intervals $[g(a_{y_i}/\tilde e_i), g(a_{y_i+1}/\tilde e_i))$ live on the offset-adjusted scale. When the offsets vary, their locations shift from observation to observation by an amount driven by $\tilde e_i$ alone. The starting values must therefore be computed on the same $y/\tilde e$ scale, so that the first $\mu$ and $\sigma$ are close to these intervals. `sigest` deserves particular care. It is used only once, but it fixes the prior for $\sigma$ (via `sigquant`) for the entire run, so a mis-scaled value does not wash out.

The code sets `offsets_vary = length(unique(offsets)) > 1`, evaluated on the supplied offsets before normalization, since the offsets can be different across individuals or constant for all $i$.

- **Constant offsets** (including the default $e_i \equiv 1$). Then $\tilde e_i \equiv 1$, the offset-adjusted scale coincides with the count scale, and the original initialization is kept. With the defaults the output is identical to the original `bart_star()`.
- **Different offsets.** The initial latent values are placed strictly inside their truncation intervals. For `box-cox`, `sigest` is computed on the offset-adjusted scale rather than by the pilot MCMC, which works on the raw counts.

| Quantity | Constant offsets | Different offsets |
|---|---|---|
| Latent $z$ | $z_i^{(0)} = g((y_i + \lvert\xi_i\rvert)/\tilde e_i)$, $\xi_i \sim N(0, 1)$ | $z_i^{(0)} = g((y_i + U_i)/\tilde e_i)$, $U_i \sim \text{Unif}(0, 1)$ |
| `sigest` (all except `box-cox`) | $\mathrm{sd}{g((y_i + 1)/\tilde e_i)}$ | $\mathrm{sd}{g((y_i + 1)/\tilde e_i)}$ |
| `sigest` (`box-cox`) | original pilot MCMC (mean-only STAR model) | $\mathrm{sd}{g_{\lambda^{(0)}}((y_i + 1)/\tilde e_i)}$, $\lambda^{(0)} \sim \text{Unif}(0, 1)$ (the initial $\lambda$) |
| Grid for $g^{-1}$ (`np`, `pois`, `neg-bin`) | 250 equally spaced points on $\big[0, \min\{2\max_i(y_i/\tilde e_i), y_{\max}/\min_i \tilde e_i\}]$ plus 250 quantiles of the unique values of $(y_i + 1)/\tilde e_i$, $y_i < y_{\max}$ | same |

**Latent start inside the interval.** Since $U_i \in (0, 1)$,

$$\frac{y_i}{\tilde e_i} < \frac{y_i + U_i}{\tilde e_i} < \frac{y_i + 1}{\tilde e_i} \Longrightarrow g\left(\frac{a_{y_i}}{\tilde e_i}\right) \le z_i^{(0)} < g\left(\frac{a_{y_i+1}}{\tilde e_i}\right)$$

so every initial $z_i$ is consistent with its observed count. For $y_i = 0$ the lower endpoint is $-\infty$. For $y_i = y_{\max}$ the upper endpoint is $+\infty$. The half-normal start $y_i + \lvert\xi_i\rvert$ exceeds $y_i + 1$ with probability $2\{1 - \Phi(1)\} \approx 0.32$, placing those starts outside their intervals.

**Box-Cox `sigest`.** The pilot MCMC fits a mean-only model to the raw counts. When the offsets vary across individuals, the raw counts carry extra spread from exposure differences that the offsets already account for. The pilot therefore overstates the residual SD on the offset-adjusted scale. Computing $\mathrm{sd}\{g_\lambda((y+1)/\tilde e)\}$ uses the same rule as the other transformations. The grid for $g^{-1}$ spans the density scale $y/\tilde e$, the domain of $g$'s argument. The approximate inverse `g_inv_approx(g, t_grid)` is then used unchanged.

## Implementation

All changes are in **`bart_star_offset.R`**, which defines `bart_star()` with two additional arguments. No helper function is added or modified. `a_j`, `g_bc`, `g_inv_bc`, `g_cdf`, `g_inv_approx`, `round_floor`, `rtruncnormRcpp`, `logLikeRcpp`, `logLikePointRcpp`, `expectation_gRcpp` and `uni.slice` are all called unchanged, with offset-scaled arguments where needed.

### Interface

```r
bart_star(y, X, X_test = NULL, y_test = NULL,
          offsets      = rep(1, length(y)),     # training offsets e_i > 0
          offsets_test = rep(1, NROW(X_test)),  # test offsets (1 = density)
          transformation = 'np', ...)           # all other arguments as before
```

- `offsets` and `offsets_test`take the offset values one per observation. A single number will be used for every observation, e.g. `offset_test = 1` means $e_\text{test} = 1$ for all test points.
- Both must be finite and strictly positive, and have lengths `length(y)` and `nrow(X_test)` respectively. Otherwise an error is raised.
- The returned list has the same elements as the original. Count-scale outputs (`post.pred`, `fitted.values`, `post.fitted.values`) are at the training exposures `offsets`. The test outputs (`post.pred.test`, `post.fitted.values.test`, `post.log.pred.test`) are at `offsets_test`. `post.mu.test` is on the latent $z$ scale.

### Code changes (all marked `# OFFSETS:` in the `bart_star_offset.R` code)

| Location | Change |
|---|---|
| Argument checks | validate and recycle `offsets` / `offsets_test` |
| Normalization | `offsets_vary = length(unique(offsets)) > 1` (before normalization); `e_bar = exp(mean(log(offsets)))`; `offsets = offsets/e_bar`; `offsets_test = offsets_test/e_bar` |
| `ispline` dispatch | error if offsets differ from 1; otherwise drop the offset arguments and call the original sampler |
| Grid for `g_inv_approx()` | built from `y/offsets` and `(y + 1)/offsets` |
| Initial `z_star` | different offsets: `g((y + runif(n))/offsets)`; constant offsets: `g((y + abs(rnorm(n)))/offsets)` |
| `sigest` | `sd(g((y + 1)/offsets))`; for `box-cox`, the pilot MCMC is used only when the offsets are constant |
| Cut points | `a_y = a_j(y, y_max)/offsets`; `a_yp1 = a_j(y + 1, y_max)/offsets`; these feed Block 1, Block 3 (Box-Cox) and the pointwise log-likelihood |
| Predictive draws | `round_floor(g_inv(z_tilde) * offsets, y_max)`, and likewise with `offsets_test` |
| Conditional expectation | `Jmax` computed at exposure $e_i$ and capped at `ceiling(2*max(y/offsets)*offsets)`; one `expectation_gRcpp()` call per distinct offset with grid `g(a_j(0:J)/e)` (training and test) |
| Test log-predictive score | `g(a_j(y_test)/offsets_test)`, `g(a_j(y_test + 1)/offsets_test)` |

### Example

```r
library(countSTAR)
source("bart_star_offset.R")

# y: counts, X: covariates, e: offsets (e.g. detection proportion or effort)
fit = bart_star(y = y, X = X, offsets = e,
                X_test = X_new, offsets_test = 1,   # density predictions
                transformation = 'log', save_y_hat = TRUE)

fit$fitted.values                        # E(y_i | x_i, e_i): expected counts at the observed exposures
colMeans(fit$post.fitted.values.test)    # E(y | x_new, e = 1): expected density
```

### Notes

- **Computational cost:** Only the conditional expectation scales with the number of distinct offset values, through one `expectation_gRcpp()` call each. Sampling, prediction and the log-likelihood add no per-iteration cost beyond the original, because the scaled cut points are precomputed. Default `offsets_test = 1` gives a single call for the test set.
- **Not yet supported:** `transformation = "ispline"` with non-unit offsets; regression weights (see `README_weight.md`).
