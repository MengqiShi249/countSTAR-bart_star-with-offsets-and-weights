# countSTAR: `bart_star` with offsets and weights
This repository extends `bart_star()` from the ['countSTAR'](https://github.com/bking124/countSTAR) package to support **model offsets** and, eventually **regression weights** (later work), for count-valued BART regression. The motivating application is national density and distribution modelling for Canadian birds, but the extension is general to any STAR model with a known, mechanistic exposure term.

## Motivation

`bart_star()` fits a **S**imultaneous **T**ransformation **A**nd **R**ounding (STAR) model with a Bayesian Additive Regression Trees (BART) latent mean, via `dbarts`. It is attractive for ecological count data because it captures strong nonlinearities and covariate interactions (e.g. thresholds in temperature, elevation, and proximity to water) that GLMs/GAMs miss, while still returning calibrated posterior uncertainty for predictions and marginal effects, for which boosted regression trees (BRTs), an approach ecologists previously used, could only achieve at large computational cost via bootstrapping.

Two features are needed to make `bart_star()` usable for this application:

1. **Offsets**. Each point count has an associated offset term $e_i>0$ (e.g. the estimated proportion of birds present that are actually detected). Under a log-link, a commonly used model is $\text{count}_i\sim \text{Poisson}\big(\exp(\eta_i)\, e_i\big)$ so that $\eta_i = \mu(x_i)$ represents log density, not log expected count. This relationship is mechanistic and multiplicative, so it belongs as an offset rather than as a covariate with an estimated effect. STAR's transformation $g(\cdot)$ is not restricted to $\log$ and used different model, however, so the offset cannot simply be absorbed by dividing $y$ by $e$. Also, the strategy of dividing $y$ by $e$ does not work even we only used log-transformation. The purpose of these offsets is to help the models convert the distribution of counts to a distribution of absolute population density in an area. This way, we can not only assess areas that have a relatively large number of birds (i.e., relatively high counts), but we can also estimate the population size of a species (a pressing problem for ecologists).

2. **Weights**. See separate README_weight.MD file.

 
## Background: unmodified STAR generative model
STAR defines a count-valued model by (1) a continuous latent variable/proxy $y_i^\ast$ with a Gaussian regression model, and (2) a transformation-and-rounding link connecting $y_i^\ast$ to the observed count $y_i$:

$$y_i = \lfloor y_i^\ast \rfloor,\qquad 
z_i = g(y_i^\ast),\qquad 
z_i = \mu(x_i) + \varepsilon_i, \quad \varepsilon_i\sim N(0,\sigma^2)$$

where $g(\cdot)$ is a known or estimated monotone transformation and, in `bart_star()`, $\mu(x_i)$ is a BART sum-of-trees function. Because $g$ is monotone, event $\{y_i = j\}$ is equivalent to $y_i^\ast$ falling in $[j, j+1)$, i.e.

$$\Pr(y_i = j\mid x_i) = \Pr\big(z_i\in [g(a_j), g(a_{j + 1}))\big),\qquad a_j = j \ (a_0 = -\infty, \ a_{y_{\max} + 1} = \infty)$$

This is the `a_j()` / `round_floor()` in `helper_functions.R`, and the resulting interval endpoints $z_\text{lower} = g(a_j)$, $z_\text{upper} = g(a_{j + 1})$ derive the Gibbs sampler in `bart_star()`:

1. **Sample $z^\ast$**: draw $z_i\mid y_i$ from $N(\mu(x_i),\sigma^2)$ truncated to $[z_{\text{lower},i}, z_{\text{upper},i}]$.
2. **Sample $(\mu,\sigma)$**: update the BART fit by calling `dbarts` with response $z^\ast$.
3. **(Optional) update the transformation**: if $g$ is being learned rather than fixed.
4. Repeat, use $g^{-1}$ and rounding to generate posterior predictive draws of $y$.


## Algorithm design: incorporating offsets
Let $e_i>0$ be the known offset / exposure for observation $i$, with $e_i = 1$ at location where "exposure-free" (density) prediction is wanted. 

**Key idea: take offset through $g(\cdot)$, not through $y$.** Now keep $y_i = \lfloor y_i^\ast \rfloor$ exactly as before (so $y$ stays count-valued), but change the transformation step to act on the offset-adjusted continuous latent variable / proxy:

$$y_i = \lfloor y_i^\ast\rfloor,\qquad 
z_i = g\ (\frac{y_i^\ast}{e_i}),\qquad
z_i = \mu(x_i) + \varepsilon_i$$

This models $\mu(x_i)$ on the multiplicative, offset-adjusted scale (density), while $y_i$ remains the observed raw count. Since $g$ is still monotone, $\{y_i = j\}$ is equivalent to $y_i^\ast/e_i\in [j/e_i,(j + 1)/e_i)$, giving the modified likelihood

$$\Pr(y_i = j\mid x_i) = \Pr(y_i^\ast\in [j,\, j + 1)) = \Pr(\frac{y_i^\ast}{e_i}\in \Big[\frac{j}{e_i},\, \frac{j + 1}{e_i}\Big)) = \Pr(z_i\in \Big[g\big(\tfrac{j}{e_i}\big), g\big(\tfrac{j + 1}{e_i}\big)\Big)$$

Note the interval is $[g(j/e_i),\, g((j + 1)/e_i))$ instead of $g(j/e_i + 1)$, which is the incorrect endpoint implied by the rejected $y/e$ approach above.

### Required code changes
The change is mainly to cut points, not to the sampler structure itself:
 - Replace $a_j = j$ with the offset-scaled cut points $a_j(e_i) = j/e_i$ and symmetrically $a_{j + 1}(e_i) = (j + 1)/e_i$, computed per-observation since $e_i$ varies across $i$.
 - `z_lower`, `z_upper` become $g(a_j / e)$, $g(a_{j + 1} / e)$.
 - Block 1 (truncated-normal draw of $z^\ast$) and the point wise log-likelihood / WAIC computation use these offset-adjusted bounds; no other part of the Gibbs sampler changes, since $g$, $\mu(\cdot)$ and $\sigma$ are all still defined on the $z$ scale.
 - **Prediction** must re-introduce the test-set offset $e_\text{test}$ after inverting $g$,
$$\tilde y = \Big\lfloor g^{-1}(\tilde z)\cdot e_\text{test}\, \Big\rfloor$$

### Transformation $g$
The offset enters $g$ as $g(y^\ast/e_i)$, and there are three cases on how $g$ itself is obtained. Whatever $g$ turns out to be, the pointwise contribution of observation $i$ is always STAR interval probability, with offset in the cut points $a_j(e_i) = j/e_i$,

$$\Pr(y_i = j|x_i) = \Phi(\frac{g(a_{j + 1}(e_i)) - \mu_i}{\sigma}) - \Phi(\frac{g(a_{j}(e_i)) - \mu_i}{\sigma})$$

so the log-likelihood (the `logLikePointRcpp` function in `rounded_source.cpp`) is 

$$l = \sum_{i = 1}^{n}\log [\Phi(\frac{g(a_{j + 1}(e_i)) - \mu_i}{\sigma}) - \Phi(\frac{g(a_{j}(e_i)) - \mu_i}{\sigma})]$$

1. **Known, fixed $g$** (`identity`, `log`, `sqrt`, or `box-cox` with a user-supplied $\lambda$). $g$ never changes across iterations, so the offset-adjusted cut points $z_\text{lower} = g(a_j/e)$, $z_\text{upper} = g(a_{j + 1}/e)$ can be computed once, before the sampling iteration, exactly as `z_lower` / `z_upper` are currently computed once in `bart_star()`. No per-iteration cost is added.
  
2. **$g$, estimated once from the data, then held fixed** (`np`, `pois`, `neg-bin`). Here $g$ is fit before the sampler starts (e.g. via `g_cdf()` or moment matching), so the offset must be folded into that one-time estimation step: $g$ should be fit to the offset-adjusted values $y_i/e_i$, since it is meant to approximate the distribution of $y^\ast$ itself. Once fit, $g$ is treated as known for the rest of the algorithm, and cut points are computed once, as in Case 1. Here the estimator is

 $$\hat{g}(\frac{y_i}{e_i}) = \Phi^{-1}(\hat{F}_Y(y_i - 1)), \quad i = 1,...,n, y_i\neq 0$$

 with $\hat{F}_Y$ could be one of 

 $$\hat{F}^\text{np}_Y(t) = \frac{n}{n + 1}\hat{\text{ecdf}}(y)(t),\quad \hat{F}^\text{pois}_Y(t) = F_\text{Pois}(t; \hat{\mu}_y),\quad \hat{F}^\text{neg-bin}_Y(t) = F_\text{NB}(t; \hat{r} = \frac{\hat{\mu}_y^2}{\hat{\sigma}_y^2 - \hat{\mu}_y}, \hat{p} = \frac{\hat{\mu}_y}{\hat{\sigma}^2_y})$$

 where $\hat{\mu}_y$, $\hat{\sigma}_y$ are the mean, SD of the $y$.
  
3. **$g$ sampled at every iteration (fully Bayesian)** (`box-cox` with $\lambda$ updated by slice sampling). Here $g$ changes every iteration, so $z_\text{lower}$ and $z_\text{upper}$ must be **re-computed at every iteration** from the current draw of $g$ (`g(a_j/e)`, `g(a_{j + 1}/e)`), exactly as `bart_star()` already does for `box-cox` in Block 3. Without an offset, $a_j = j$ depends only on the observed counts, so $g$ can be evaluated once on a small grid and looked up per observations; with an offset, $a_j(e_i) = j/e_i$ depends on the pair $(j, e_i)$. Also, any likelihood used to update $g$ itself need offset-adjusted arguments `g_bc(a_y/e, lambda)`, `g_bc(a_yp1/e,lambda)`. Then the log-likelihood gives update, the full conditional posterior for $\lambda$ (given the current draws of $(\mu, \sigma)$),

$$\pi(\lambda|\mu, \sigma, y, e) \propto \exp(l(\lambda))\cdot \pi(\lambda), \quad \pi(\lambda) = N(\frac{1}{2}, 1)\text{ truncated to }[0, 3]$$

and the $l(\lambda)$ is the same expression but the $g$ is specified as $g_\text{bc}$. As a temporary remark here, the `bart_star()` with offset feature has not been extended to `ispline` (*save for next-step exploration*), one reason is ispline does not work well enough for original `bart_star()` thus not recommended to use during application; another technical reason is its sampler locates cut points via `match(a_y, t_g)` against an integer grid `t_g`, which assumes $a_j = j$ is shared across observations, but under an offset, $a_j(e_i) = j/e_i$ is continuous and observation-specific, so `match()` function would fail.

### Implementation notes

Codes for the function change is in **`bart_star_offset.R`** file, and everything else in original `bart_star()`'s Gibbs sampler (`a_j`, `g_bc`, `g_inv_bc`, `round_floor`, `g_inv_approx`, `rtruncnormRcpp`, `logLikeRcpp`, `logLikePointRcpp`) is used unmodified, called with offset scaled arguments ($a_j(e_i) = j/e_i$ in place of $$a_j = j).

- `set_offset(y, offset)` (new function): attaches `offset` as an attribute on `y`, so `bart_star(y,..., offset = TRUE)` can recover it later without passing the vector again explicitly.
- `g_cdf_offset(y, offset, distribution)`: offset-adjusted replacement for `g_cdf()`. Relocates each height $\Phi^{-1}(\hat{F}_y(y_i - 1))$ to $t_0 = y_i/e_i$ instead of $t_0 = y_i$. Because the height depends on $y_i$ but the location depends on $y_i/e_i$, e.g. a larger count with a large $e_i$ can land at a smaller $t_0$ than a smaller count with $e_i = 1$, so the pooled pairs ($t_0$, height) are no longer guaranteed increasing in $t_0$, even though $g$ must be monotone. So **isotonic regression** is applied first to enforce this before a monotone Hermite spline is fit. Out-of-range queries are clamped rather than extrapolated.
- `expectation_g_offset(g, mu, sigma, offset, Jmax)`: offset-adjusted replacement for `expectation_gRcpp()`. The original assumes one cut-point grid shared across observations, valid only when $a_j$ does not depend on $i$. With an offset $a_j(e_i) = j/e_i$ is observations-specific, so cut points are built as an $n\times (J_\max + 1)$ matrix and $\mathbb{E}(Y_i)$ is computed via the tail-sum identity $\sum_{j = 0}^{J_{\max, i}}\Pr(Y_i>j)$ row-wise. Used only when `offset` is non-constant (which is more common in application), the `expectation_gRcpp()` still handles the constant (including all 1's) case.
- `bart_star(offset)`: adds `offset`/`offset_test` arguments. Cut points $a_j(e_i)$ are computed once and reused in initialisation, the Box-Cox slice sampler and the log-likelihood. `z_star` and `sigest` are initialised from `y/offset`. For `np/pois/neg-bin`, $g$ is fit via `g_cdf_offset()` in place of `g_cdf()`. Predictive draws re-introduce the offset after inverting $g$: $\tilde{y} = \lfloor g^{-1}(\tilde{z}\cdot e)\rfloor$. `transformation = ispline` is not supported for current offset version.
