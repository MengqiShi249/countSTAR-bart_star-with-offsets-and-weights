# countSTAR: sample weights for `bart_star`

This file describes the second planned extension to `bart_star()`: **sample weights**, which correct for non-uniform sampling effort across locations (e.g. some regions of Canada are easier to survey than others). It is a design document only. No code is implemented yet; that is later work, after the offset extension is finalized.

## Motivation

Sampling locations are not a random sample, so each observation $i$ carries a known sample weight $w_i > 0$ that reflects how over- or under-represented its location is (a larger $w_i$ means the observation represents a larger population share).

A standard approach in the literature is the Bayesian **pseudo-posterior** of [Savitsky & Toth (2016)](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-10/issue-1/Bayesian-estimation-under-informative-sampling/10.1214/16-EJS1153.pdf): each unit's likelihood contribution is raised to the power of its weight,

$$L_w(\theta) = \prod_{i=1}^{n} p(y_i \mid x_i, \theta)^{w_i}, \qquad \ell_w(\theta) = \sum_{i=1}^{n} w_i \log p(y_i \mid x_i, \theta).$$

## Weighted STAR log-likelihood

Under STAR, $p(y_i \mid x_i, \theta)$ is the interval probability $`\Phi\big(\frac{z_{\text{upper},i} - \mu_i}{\sigma}\big) - \Phi\big(\frac{z_{\text{lower},i} - \mu_i}{\sigma}\big)`$, so the target on the $y$ scale is

$$\ell_w(\theta) = \sum_{i=1}^{n} w_i \log\left[\Phi\left(\frac{z_{\text{upper},i} - \mu_i}{\sigma}\right) - \Phi\left(\frac{z_{\text{lower},i} - \mu_i}{\sigma}\right)\right].$$

The sampler in `bart_star()` does not evaluate this directly. It draws $z_i^\ast \mid y_i, \mu_i, \sigma \sim N(\mu_i, \sigma^2)$ truncated to $`[z_{\text{lower},i}, z_{\text{upper},i})`$, then updates $(\mu, \sigma)$ from $\phi(z_i^\ast; \mu_i, \sigma^2)$ as if $z^\ast$ were observed. Weighting the density on the $z$ scale gives

$$\ell^{z}_w(\theta) = \sum_i w_i \log \phi(z_i; \mu_i, \sigma^2) = -\Big(\sum_i w_i\Big)\log\sigma - \frac{\sum_i w_i}{2}\log(2\pi) - \frac{1}{2\sigma^2}\sum_i w_i (z_i - \mu_i)^2,$$

whereas the model assumed by `dbarts(weights = w)`, $\varepsilon_i \sim N(0, \sigma^2/w_i)$, gives

$$\sum_{i=1}^{n} \log \phi(z_i; \mu_i, \sigma^2/w_i) = -n\log\sigma + \frac{1}{2}\sum_i \log w_i - \frac{n}{2}\log(2\pi) - \frac{1}{2\sigma^2}\sum_i w_i (z_i - \mu_i)^2.$$

The quadratic terms match, but the coefficients of $\log\sigma$ differ: $\sum_i w_i$ versus $n$.

In addition, $`\ell^{z}_w(\theta)`$ on the $z$ scale is not the same as $`\ell_w(\theta)`$ on the $y$ scale. When $w \equiv 1$, integrating the latent density over the interval recovers the $y$-scale likelihood exactly:

```math
\int_{z_{\text{lower},i}}^{z_{\text{upper},i}} \phi(z; \mu_i, \sigma^2)\, dz = \Phi\Big(\frac{z_{\text{upper},i} - \mu_i}{\sigma}\Big) - \Phi\Big(\frac{z_{\text{lower},i} - \mu_i}{\sigma}\Big)
```

Raising a density to a power does not commute with integration, however:

$$\left(\int_{z_{\text{lower},i}}^{z_{\text{upper},i}} \phi(z; \mu_i, \sigma^2)\thinspace dz\right)^{w_i} \neq \int_{z_{\text{lower},i}}^{z_{\text{upper},i}} \phi(z; \mu_i, \sigma^2)^{w_i}\thinspace dz.$$

So once $w \not\equiv 1$, the $z$-scale target is not guaranteed to match $\ell_w(\theta)$. The consequences differ for $\mu$ and $\sigma$:

- **$\mu \mid z^\ast, \sigma$:** with $\sigma$ fixed, $`\ell^{z}_w`$ depends on the data only through $`\sum_i w_i (z_i^\ast - \mu_i)^2`$, which `dbarts(weights = w)` already uses. The $\mu$ update needs nothing beyond `dbarts`.
- **$\sigma \mid z^\ast, \mu$:** the $\log\sigma$ coefficient differs from that of `dbarts` (see Additional thoughts, point 1).

## Proposed solution

One option is to sample $\sigma$ directly from $\ell_w(\theta)$ rather than on the latent scale, as `bart_star()` does for the Box-Cox $\lambda$ in Block 3:

$$\pi(\sigma \mid y, \mu, w) \propto \exp\lbrace\ell_w(\sigma)\rbrace\thinspace \pi(\sigma),$$

evaluated with `logLikeRcpp` / `logLikePointRcpp`, with each term weighted by $w_i$ and $\pi(\sigma)$ the usual prior. This should cost no more than the $\lambda$ update. The $\mu$ update via `dbarts` would stay unmodified. The idea is not to redesign `dbarts`'s internal steps, only to change how $\mu$ and $\sigma$ are updated. See Additional thoughts, point 3, for a caveat.

On $g$, the same distinction applies to any transformation fitted from $z^\ast$. The Box-Cox $\lambda$ is already updated from the exact interval likelihood, so it only needs per-observation weights $w_i$, as for $\sigma$ above. The one-time CDF-based estimators (`np`, `pois`, `neg-bin`) are fitted before the sampler starts, from the raw counts rather than $z^\ast$, so this issue does not affect them. Their estimation should still use $w_i$, e.g. through a weighted empirical CDF. *This can be left for later work. How sample weights combine with offsets is also an interesting question for later.*

**Some remarks & connection with offsets:**

**1. Normalize the weights.** Savitsky & Toth (2016) scale the weights to sum to the sample size:

$$\tilde w_i = \frac{n\thinspace w_i}{\sum_j w_j}, \qquad \sum_i \tilde w_i = n.$$

Without this, $\sum_i w_i$ (often of the order of the population size) acts as the effective sample size and the pseudo-posterior is far too concentrated. With $\tilde w$, the two $\log\sigma$ coefficients above coincide ($\sum_i \tilde w_i = n$). $`\ell^{z}_{\tilde w}`$ and the `dbarts` log-likelihood then differ only by the constant $\frac{1}{2}\sum_i \log \tilde w_i$, so **the `dbarts` update of $\sigma$ is already correct on the $z$ scale**. This also mirrors the geometric-mean normalization in the offsets.

**2. Block 1 must use the weighted variance.** The joint $z$-scale target $\prod_i \phi(z_i; \mu_i, \sigma^2)^{\tilde w_i}\thinspace 1\lbrace z_i \in [z_{\text{lower},i}, z_{\text{upper},i})\rbrace$ has full conditional

$$z_i \mid y_i, \mu_i, \sigma \sim N(\mu_i, \sigma^2/\tilde w_i) \text{ truncated to } [z_{\text{lower},i}, z_{\text{upper},i}).$$

Keeping Block 1 at variance $\sigma^2$ while passing `weights` to `dbarts` does not give a Gibbs sampler for any joint distribution. With the weighted variance, integrating out $z$ shows that the sampler **exactly** targets

$$L(\theta) = \prod_{i=1}^{n}\left[\Phi\left(\frac{\sqrt{\tilde w_i}\thinspace(z_{\text{upper},i} - \mu_i)}{\sigma}\right) - \Phi\left(\frac{\sqrt{\tilde w_i}\thinspace(z_{\text{lower},i} - \mu_i)}{\sigma}\right)\right],$$

i.e. a STAR model with heteroscedastic errors $\varepsilon_i \sim N(0, \sigma^2/\tilde w_i)$. This is a proper likelihood and needs almost no new code: `dbarts(weights = w_tilde)`, and `sigma = params$sigma/sqrt(w_tilde)` in `rtruncnormRcpp`, `logLikeRcpp` and `logLikePointRcpp`. All of these already accept a per-observation `sigma` vector. It coincides with the pseudo-likelihood $\prod_i p_i^{\tilde w_i}$ when the intervals are narrow relative to $\sigma$ (large counts, where both reduce to weighted least squares on $z$). It differs for small counts and zeros, which matters for bird data.

**3. Caveat on the proposed $\sigma$ step.** Sampling $\lambda$ from the interval likelihood is valid because it is a partially collapsed Gibbs step: $\lambda \mid \mu, \sigma, y$ (with $z$ integrated out) and $z \mid \lambda, \mu, \sigma, y$ are conditionals of the *same* joint distribution. With non-integer weights, $\prod_i p_i^{w_i}$ has no latent - $z$ representation whose $\mu$-conditional is the `dbarts` update. Combining the `dbarts` $\mu$ step ($z$ - scale target) with a $y$ - scale $\sigma$ step therefore mixes two targets, and the chain is not guaranteed to converge to either. In practice it would also require overriding the $\sigma$ that `dbarts` draws internally in `sampler$run()`.

**4. Exact benchmark with no new code.** For integer weights, $p_i^{w_i}$ is the likelihood of $w_i$ independent copies of observation $i$. Rounding $\tilde w_i$ to integers and replicating rows, then running the **unmodified** `bart_star()`, gives the exact pseudo-posterior for the rounded weights. This is a useful reference for judging the precision-weight approximation in point 2.

**5. Uncertainty.** Even with normalized weights, pseudo-posterior credible intervals are not design-calibrated and tend to under-cover under informative or clustered sampling. [Williams & Savitsky (2021)](https://doi.org/10.1111/insr.12376) propose a fast post-processing (sandwich-type) correction of the MCMC draws (I have not yet delved into this part).

**6. Transformation and offsets.** For `np`, a weighted ECDF $`\hat F_w(t) = \sum_i \tilde w_i 1\{ y_i \le t\} / \sum_i \tilde w_i`$ can be approximated without changing `g_cdf()` by passing replicated counts, `g_cdf(rep(y, round(k * w_tilde)))`. For `pois` and `neg-bin`, use weighted moments. Offsets and weights act on different parts of the model: offsets move the cut points to $a_j/\tilde e_i$, while weights scale the latent variance to $\sigma^2/\tilde w_i$. They combine directly: Block 1 draws $N(\mu_i, \sigma^2/\tilde w_i)$ truncated to $`[g(a_{y_i}/\tilde e_i),\, g(a_{y_i+1}/\tilde e_i))`$.

## References

- Savitsky, T. D., & Toth, D. (2016). Bayesian estimation under informative sampling. *Electronic Journal of Statistics*, 10(1), 1677–1708.
- Williams, M. R., & Savitsky, T. D. (2021). Uncertainty estimation for pseudo-Bayesian inference under complex sampling. *International Statistical Review*, 89(1), 72–107.
