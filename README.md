# countSTAR: `bart_star` with offsets and weights
This repository extends `bart_star()` from the ['countSTAR'](https://github.com/bking124/countSTAR) package to support **model offsets** and, eventually **regression weights**, for count-valued BART regression. The motivating application is national density and distribution modelling for Canadian birds, but the extension is general to any STAR model with a known, mechanistic exposure term.

## Motivation

`bart_star()` fits a **S**imultaneous **T**ransformation **A**nd **R**ounding (STAR) model with a Bayesian Additive Regression Trees (BART) latent mean, via `dbarts`. It is attractive for ecological count data because it captures strong nonlinearities and covariate interactions (e.g. thresholds in temperature, elevation, and proximity to water) that GLMs/GAMs miss, while still returning calibrated posterior uncertainty for predictions and marginal effects, for which boosted regression trees (BRTs), the previous approach, could only achieve at large computational cost via bootstrapping.

Two features are needed to make `bart_star()` usable for this application:

1. **Offsets**. Each point count has an associated offset term $e_i>0$ (e.g. the estimated promotion of birds present that are actually detected). Under a log-link, the natural model is
$$\text{count}_i\sim \text{Poisson}\big(\exp(\eta_i)\, e_i\big)$$
so that $\eta_i = \mu(x_i)$ represents *log density*, not log expected count. Fitting with the offset fixed at $e_i = 1$ at prediction times recovers estimated density/abundance directly. This relationship is mechanistic and multiplicative, so it belongs as an offset rather than as a covariate with an estimated effect. STAR's transformation $g(\cdot)$ is not restricted to $\log$, however, so the offset cannot simply be absorbed by dividing $y$ by $e$. The purpose of these offsets is to help the models convert the distribution of counts to a distribution of absolute population density in an area. This way, we can not only assess areas that have a relatively large number of birds (i.e., relatively high counts), but we can also estimate the population size of a species (a pressing problem for ecologists).

2. **Weights**. Sampling locations are not uniformly distributed across Canada (some regions are easier to access), and the previous BRT-based approach corrected for this by thinning/bootstrap resampling. Since one benefit of moving to BART is avoiding bootstrap, this bias instead needs to be handled with likelihood (sample) weights, this bias instead needs to be handled with likelihood (sample) weights, analogous to survey weighting in a GLM, i.e. raising each observation's likelihood contribution to a power $1/w_i$ (or $w_i$ based on later calculation). This is **not** the same as the heteroskedasticity weighting (`weights` in `dbarts::dbarts()`, which rescales the error variance to $\sigma^2/w_i$) natively supported by `dbarts`, so it requires additional modification beyond simply forwarding a `weights` argument. Because it is more involved and lower priority for the application, it is treated as a follow-on feature once offsets are implemented. Details derivation will be discussed later.  
## Background: unmodified STAR generative model
STAR defines a count-valued model by (1) a continuous latent variable/proxy $y_i^\ast$ with a Gaussian regression model, and (2) a transformation-and-rounding link connecting $y_i^\ast$ to the observed count $y_i$:

$$y_i = \lfloor y_i^\ast \rfloor,\qquad 
z_i = g(y_i^\ast),\qquad 
z_i = \mu(x_i) + \varepsilon_i, \quad \varepsilon_i\sim N(0,\sigma^2)$$

where $g(\cdot)$ is a known or estimated monotone transformation and, in `bart_star()`, $\mu(x_i)$ is a BART sum-of-trees function. Because $g$ is monotone, event $\{y_i = j\}$ is equivalent to $y_i^\ast$ falling in $[j, j+1)$, i.e.

$$\Pr(y_i = j\mid x_i) = \Pr\big(z_i\in [g(a_j), g(a_{j + 1}))\big),\qquad a_j = j \ (a_0 = \infty, \ a_{y_{\max} + 1} = \infty)$$

This is the `a_j()` / `round_floor()` in `helper_functions.R`, and the resulting interval endpoints $z_\text{lower} = g(a_j)$, $z_\text{upper} = g(a_{j + 1})$ derive the Gibbs sampler in `bart_star()`:

1. **Sample $z^\ast$**: draw $z_i\mid y_i$ from $N(\mu(x_i),\sigma^2)$ truncated to $[z_{\text{lower},i}, z_{\text{upper},i}]$.
2. **Sample $(\mu,\sigma)$**: update the BART fit by calling `dbarts` with response $z^\ast$.
3. **(Optional) update the transformation**: if $g$ is being learned rather than fixed.
4. Repeat, use $g^{-1}$ and rounding to generate posterior predictive draws of $y$.


## Algorithm design: incorporating offsets
Let $e_i>0$ be the known offset / exposure for observation $i$, with $e_i = 1$ at location where "exposure-free" (density) prediction is wanted. Two candidate approaches were considered and discussed.

**Rejected approach: divide $y$ by $e$.** Passing $y_i/e_i$ as the response and fitting the usual STAR model is not correct in general, for two reasons: (i) it requires $y/e$ to be treated as if it were itself a rounded count, breaking the integer-valued check in `bart_star()`; and (ii) it implicitly integrates over the interval $[j/e,\ j/e + 1)$ rather than correct interval, since flooring $(y/e)^\ast$ is not the same operation as flooring $y^\ast$ and then dividing by $e$.

**Feasible approach: take offset through $g(\cdot)$, not through $y$.** Now keep $y_i = \lfloor y_i^\ast \rfloor$ exactly as before (so $y$ stays count-valued), but change the transformation step to act on the offset-adjusted continuous latent variable / proxy:

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
The offset enters $g$ as $g(y^\ast/e_i)$, and there are three cases on how $g$ itself is obtained:
1. **Known, fixed $g$** (`identity`, `log`, `sqrt`, or `box-cox` with a user-supplied $\lambda$). $g$ never changes across iterations, so the offset-adjusted cut points $z_\text{lower} = g(a_j/e)$, $z_\text{upper} = g(a_{j + 1}/e)$ can be computed once, before the sampling iteration, exactly as `z_lower` / `z_upper` are currently computed once in `bart_star()`. No per-iteration cost is added.
2. **$g$, estimated once from the data, then held fixed** (`np`, `pois`, `neg-bin`). Here $g$ is fit before the sampler starts (e.g. via `g_cdf()` or moment matching), so the offset must be folded into that one-time estimation step: $g$ should be fit to the offset-adjusted values $y_i/e_i$, since it is meant to approximate the distribution of $y^\ast$ itself. Once fit, $g$ is treated as known for the rest of the algorithm, and cut points are computed once, as in Case 1.
3. **$g$ sampled at every iteration (fully Bayesian)** (`box-cox` with $\lambda$ updated by slice sampling, `ispline`, `bnp`). Here $g$ changes every iteration, so $z_\text{lower}$ and $z_\text{upper}$ must be **re-computed at every iteration** from the current draw of $g$ (`g(a_j/e)`, `g(a_{j + 1}/e)`), exactly as `bart_star()` already does for `box-cox` in Block 3. This case is genuinely more expensive with offsets than without: without an offset, $a_j = j$ depends only on the observed counts, so $g$ can be evaluated once on a small grid and looked up per observations; with an offset, $a_j(e_i) = j/e_i$ depends on the pair $(j, e_i)$. Also, any likelihood used to update $g$ itself need offset-adjusted arguments `g_bc(a_y/e, lambda)`, `g_bc(a_yp1/e,lambda)`.

### Implementation notes
- New argument `offset` (length- $n$ vector, $e_i > 0$; default `rep(1, n)`)
- A corresponding `offset_test` is needed for `X_test` prediction (default `1`)

## Weights (later work)
In R, `dbarts`'s `weights` argument implements heteroskedasticity weighting, not the sample weighting needed in this project. I start with some derivation to show why simple forwarding `weights` to `darts::darts()` is not the desired model.

### Why `weights` cannot simply be passed to `dbarts()`
The posterior needs to target the **sample-weighted likelihood**, in which observation $i$'s contribution is raised to the power of its weight $w_i>0$ (or $1/w_i>0$, depends on computation):

$$L_w(\theta) = \prod_{i = 1}^{n}p(y_i\mid x_i, \theta)^{w_i},\quad \ell_w(\theta) = \sum_{i = 1}^n w_i\log p(y_i\mid x_i,\theta)$$

Under the STAR/BART framework, the model assumes $z_i = \mu(x_i) + \varepsilon_i$, $\varepsilon_i\sim N(0,\sigma^2)$, $\mu_i:=\mu(x_i)$. The desired weighted log-density contribution is 
$$l_w(\theta) = \sum_{i=1}^n w_i\log \phi(z_i; \mu_i,\sigma^2) = -(\sum_i w_i)\log \sigma - \frac{\sum_i w_i}{2}\log(2\pi) - \frac{1}{2\sigma^2}\sum_i w_i(z_i - \mu_i)^2$$

What `darts(weights = w)` actually computes, under its assumed model $\varepsilon_i\sim N(0, \sigma^2/w_i)$, and sum over all $I$
$$\sum_{i = 1}^n \log \phi(z_i;\mu_i,\sigma^2/w_i) = -n\log \sigma + \frac{1}{2}\sum_i \log w_i - \frac{n}{2}\log(2\pi) - \frac{1}{2\sigma^2}\sum_i w_i(z_i - \mu_i)^2$$

The problem happens to the coefficient on $\log\sigma$ in desired function: the coefficient is $n$ in `dbarts` but it is $\sum_{i} w_i$ in desired likelihood. *This might needs modification on `dbarts` function especially on the weight part or other methods to re-calculate from original `dbarts` (save for later discussion)*.
