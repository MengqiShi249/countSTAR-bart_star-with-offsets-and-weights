# countSTAR: sample weights for `bart_star()`

This file describes the second planned extension to `bart_star()`: **sample weights**, which correct for non-uniform sampling effort across locations (e.g. some regions of Canada are easier to survey than others). It is a design document only. No code is implemented yet, which leaves to later work, after the offset extension (`README_offset.md`) is finalized.

## Motivation

Sampling locations are not a random sample, so each observation $i$ carries a known sample weight $w_i > 0$ that reflects how over- or under-represented its location is. A larger $w_i$ means the observation represents a larger population share.

A standard approach is the Bayesian **pseudo-posterior** of [Savitsky & Toth (2016)](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-10/issue-1/Bayesian-estimation-under-informative-sampling/10.1214/16-EJS1153.pdf). It raises each unit's likelihood contribution to the power of its weight, after normalizing the weights to sum to $n$:

$$L_{\tilde w}(\theta) = \prod_{i=1}^{n} p_i(\theta)^{\tilde w_i}, \qquad \tilde w_i = \frac{n\thinspace w_i}{\sum_j w_j}, \qquad \sum_i \tilde w_i = n$$

Without normalization, $\sum_i w_i$ (often of the order of the population size) acts as the effective sample size, and the pseudo-posterior is far too concentrated.

## Why weighting STAR is not immediate

Under STAR, $`p_i(\theta) = \Phi\big((z_{\text{upper},i} - \mu_i)/\sigma\big) - \Phi\big((z_{\text{lower},i} - \mu_i)/\sigma\big)`$. The sampler never evaluates this directly. It augments with latent $z_i \sim N(\mu_i, \sigma^2)$ truncated to $`[z_{\text{lower},i}, z_{\text{upper},i})`$, and updates $(\mu, \sigma)$ as if $z$ were observed. This works for $w \equiv 1$ because integrating the latent density over the interval returns $p_i$. With weights, powers and integrals do not commute:

$$\left(\int_{z_{\text{lower},i}}^{z_{\text{upper},i}} \phi(z; \mu_i, \sigma^2)\thinspace dz\right)^{\tilde w_i} \neq \int_{z_{\text{lower},i}}^{z_{\text{upper},i}} \phi(z; \mu_i, \sigma^2)^{\tilde w_i}\thinspace dz$$

so weighting on the latent scale does not reproduce $L_{\tilde w}$ exactly.

### Why weights cannot simply be passed to `dbarts()`

Even on the latent scale, the weighted density is not the model `dbarts(weights = w)` assumes. With $\mu_i := \mu(x_i)$, the desired weighted log-density is

$$\ell^{z}_{w}(\theta) = \sum_{i=1}^{n} w_i \log \phi(z_i; \mu_i, \sigma^2) = -\Big(\sum_i w_i\Big)\log\sigma - \frac{\sum_i w_i}{2}\log(2\pi) - \frac{1}{2\sigma^2}\sum_i w_i (z_i - \mu_i)^2$$

`dbarts(weights = w)` instead assumes $\varepsilon_i \sim N(0, \sigma^2/w_i)$, whose log-likelihood summed over all $i$ is

$$\sum_{i=1}^{n} \log \phi(z_i; \mu_i, \sigma^2/w_i) = -n\log\sigma + \frac{1}{2}\sum_i \log w_i - \frac{n}{2}\log(2\pi) - \frac{1}{2\sigma^2}\sum_i w_i (z_i - \mu_i)^2$$

The quadratic terms are identical, so the $\mu$ update of `dbarts` already matches. The coefficients of $\log\sigma$ differ, however, $\sum_i w_i$ in the desired log-density versus $n$ in `dbarts`, so the $\sigma$ update does not match for arbitrary weights. Normalizing to $\tilde w$ (with $\sum_i \tilde w_i = n$) removes this mismatch; see the relation to the pseudo-likelihood below.

## Proposed approach: precision weights on the latent scale

**Model.** Give each latent error a weighted variance:

$$z_i = \mu(x_i) + \varepsilon_i, \qquad \varepsilon_i \sim N(0, \sigma^2/\tilde w_i), \qquad y_i = j \iff z_i \in [z_{\text{lower},i},\ z_{\text{upper},i})$$

This is exactly the model `dbarts(weights = w_tilde)` fits on $z$ ($y \mid x \sim N(f(x), \sigma^2/w)$; [dbarts docs](https://rdrr.io/cran/dbarts/man/bart.html)).

**Gibbs sampler.** The joint target is $`\pi(\mu)\thinspace\pi(\sigma)\prod_i \phi(z_i; \mu_i, \sigma^2/\tilde w_i)\thinspace 1\lbrace z_i \in [z_{\text{lower},i}, z_{\text{upper},i})\rbrace`$, with full conditionals:

- **Block 1:** $z_i \mid y_i, \mu_i, \sigma \sim N(\mu_i, \sigma^2/\tilde w_i)$ truncated to $`[z_{\text{lower},i}, z_{\text{upper},i})`$.
- **Block 2:** the $(\mu, \sigma)$ update of `dbarts(weights = w_tilde)`, unmodified.

Keeping Block 1 at variance $\sigma^2$ while passing weights to `dbarts` would not be a Gibbs sampler for any joint distribution.

**Exact target.** Integrating out $z$, the sampler targets the proper likelihood

$$L(\theta) = \prod_{i=1}^{n}\left[\Phi\left(\frac{\sqrt{\tilde w_i}\thinspace(z_{\text{upper},i} - \mu_i)}{\sigma}\right) - \Phi\left(\frac{\sqrt{\tilde w_i}\thinspace(z_{\text{lower},i} - \mu_i)}{\sigma}\right)\right]$$

i.e. a STAR model with heteroscedastic latent errors.

**Relation to the pseudo-likelihood.** For each $i$,

$$\phi(z; \mu, \sigma^2)^{w} = w^{-1/2}\thinspace(2\pi\sigma^2)^{(1-w)/2}\thinspace\phi(z; \mu, \sigma^2/w)$$

Taking the product over $i$, the extra factor is proportional to $\sigma^{\thinspace n - \sum_i \tilde w_i} = 1$. So, **because the weights are normalized**, the weighted latent density $\prod_i \phi(z_i;\mu_i,\sigma^2)^{\tilde w_i}$ and the `dbarts` model agree up to a constant, and the `dbarts` update of $\sigma$ is correct. On the $y$ scale, $L(\theta)$ coincides with $L_{\tilde w}(\theta)$ (up to a constant) when the intervals are narrow relative to $\sigma$, i.e. for large counts, where both reduce to weighted least squares on $z$. They differ for small counts and zeros, which matters for bird data. Weights therefore act as **precision weights**: higher-weight sites are modelled as less noisy on the latent scale. This is an approximation to the pseudo-posterior, not the pseudo-posterior itself.

**Code changes.** Current idea might require very little new code, because the helpers below already accept a per-observation `sigma` vector:

| Location | Change |
|---|---|
| Arguments | new `weights` (default 1), checked as finite and $> 0$, normalized to `w_tilde = n*w/sum(w)` |
| `dbarts()` | `weights = w_tilde` |
| Block 1 (`rtruncnormRcpp`) | `sigma = params$sigma/sqrt(w_tilde)` |
| Block 3, Box–Cox $\lambda$ (`logLikeRcpp`) | `sigma = params$sigma/sqrt(w_tilde)`, so that $\lambda$ targets the same joint distribution |
| Pointwise log-likelihood / WAIC (`logLikePointRcpp`) | `sigma = params$sigma/sqrt(w_tilde)` |
| Prediction and `fitted.values` | `params$sigma` (i.e. $w = 1$) for population-level predictions, since the weights describe the sampling design, not outcome precision; `params$sigma/sqrt(w_tilde)` only for posterior predictive checks on the training data |

## Transformations and offsets

- **`box-cox`:** handled by the $\lambda$ row in the code-changes table.
- **`np`, `pois`, `neg-bin`:** $g$ is fitted once, before the sampler. A weighted fit can be approximated without changing `g_cdf()` by replicating the values passed to it, e.g. `g_cdf(rep(v, round(k * w_tilde)), transformation)` with $k$ large. This gives a weighted empirical CDF for `np` and weighted moments for `pois` and `neg-bin`, and $k$ does not matter here because the CDF and moments are normalized. Here $v = y$, or $v = M y/e$ for `np` with offsets.
- **`ispline`:** not considered.
- **Offsets:** offsets and weights act on different parts of the model. Offsets move the cut points to $a_j/e_i$; weights scale the latent variance to $\sigma^2/\tilde w_i$. They combine directly: Block 1 draws $N(\mu_i, \sigma^2/\tilde w_i)$ truncated to $[g(a_{y_i}/e_i),\ g(a_{y_i+1}/e_i))$.

## References

- Savitsky, T. D., & Toth, D. (2016). Bayesian estimation under informative sampling. *Electronic Journal of Statistics*, 10(1), 1677–1708.
