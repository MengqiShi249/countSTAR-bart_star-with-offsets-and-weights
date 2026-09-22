# countSTAR: sample weights for `bart_star` (later work)
This file elaborates on the second planned extension to `bart_star()`: **sample weights**, needed to correct for non-uniform sampling effort across locations (e.g. some regions of Canada are easier to survey than others). This is a design document only, no code is implemented or proposed yet, which should be later work after finalizing offset work.

## Motivation
Sampling locations are not a random sample, so each observation $I$ carries a known sample weight $w_i>0$ reflecting how over/under-represented its location is (larger $w_i$ = more representative of a larger population share). 

A standard approach in the literature is Bayesian **pseudo-posterior** by [Savitsky & Toth (2016)](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-10/issue-1/Bayesian-estimation-under-informative-sampling/10.1214/16-EJS1153.pdf): each unit's contribution to the likelihood is raised to the power of its own weight.

$$L_w(\theta) = \prod_{i =1}^{n}p(y_i\mid x_i, \theta)^{w_i},\quad \ell_w(\theta) = \sum_{i = 1}^{n}w_i\log p(y_i\mid x_i,\theta)$$

## Weighted STAR log-likelihood
Under STAR, $p(y_i|x_i, \theta)$ is the interval probability $\Phi(\frac{z_\text{upper, i}-\mu_i}{\sigma}) - \Phi(\frac{z_\text{lower, i}-\mu_i}{\sigma})$, so our target on $y$ scale is

$$\ell_w(\theta) = \sum_{i = 1}^{n}w_i \log [\Phi(\frac{z_\text{upper, i}-\mu_i}{\sigma}) - \Phi(\frac{z_\text{lower, i}-\mu_i}{\sigma})]$$

Worth noting that `bart_star()`'s sampler does not evaluate above formula directly: it uses $z_{I}^\ast\mid y_i, \mu_i,\sigma\sim N(\mu_i,\sigma^2)$ truncated to $[z_\text{lower, i}, z_\text{upper, I}]$, then updates $(\mu, \sigma)$ from density $\phi(z_i^\ast; \mu_i, \sigma^2)$ as if $z^\ast$ were observed. Weighting the density on $z$ scale gives 

$$\ell^{z}_w(\theta) = \sum_i w_i\log \phi(z_i;\mu_i, \sigma^2) = -(\sum_i w_i)\log \sigma - \frac{\sum_i w_i}{2}\log(2\pi) - \frac{1}{2\sigma^2}\sum_{i}w_i(z_i - \mu_i)^2$$

while `dbart(weights = w)`'s assumed model $\varepsilon_i\sim N(0, \sigma^2/w_i)$ sums to 

$$\sum_{i = 1}^{n}\log \phi(z_i; \mu_i,\sigma^2/w_i) = -n\log \sigma + \frac{1}{2}\log(w_i) - \frac{n}{2}\log(2\pi) - \frac{1}{2\sigma^2}\sum_i w_i(z_i - \mu_i)^2$$

Comparing two equations in this part shows: the quadratic term matches, but the $\log\sigma$ coefficient is $\sum_i w_i$ and $n$ separately. 

Also, another thing needs to be clarified is that $\ell_{w}^z (\theta)$ on $z$-scale is not equal to $\ell(\theta)$ on $y$-scale. Integrating the truncated-normal gives $\ell_w^{z}(\theta)$ on $y$-scale exactly when $w\equiv 1$, since $\int_{z_\text{lower, i}}^{z_\text{upper, i}}\phi(z; \mu,\sigma^2)dz = \Phi(\frac{z_\text{upper, i}-\mu_i}{\sigma}) - \Phi(\frac{z_\text{lower, i}-\mu_i}{\sigma})$. But raising density to a power does not commute with integrating it. So $\ell_w^{z}(\theta)$ on $z$-scale is not guaranteed to target $\ell_w(\theta)$ on $y$-scale once $w\not\equiv 1$.

$$(\int_{z_\text{lower, I}}^{z_\text{upper, I}}\phi(z; \mu,\sigma^2)dz)^{w_i}\neq \int_{z_\text{lower, I}}^{z_\text{upper, I}}\phi(z; \mu,\sigma^2)^{w_i}dz$$

Therefore, we can discuss how conditional distribution matters differently for $\mu$ and $\sigma$ separately:

- $\mu\mid z^\ast, \sigma$: when fixing $\sigma$, $\ell_{w}^{z}(\theta)$ depends on data through $\sum_i w_i(z_i^\ast - \mu_i)^2$ which is already computed by `darts(weight = w)`. So using latent scale for $\mu$ needs nothing beyond what `dbarts` does.
- $\sigma\mid z^\ast, \mu$: the term in $\ell_{w}^{z}(\theta)$ is not exactly the same as required in $\ell_w(\theta)$.

## Proposed solution
I considered if we sample $\sigma$ directly from $\ell_w(\theta)$, not from the latent scale (how `bart_star()` handles Box-Cox parameter $\lambda$ in Block 3), which is updated by sampling

$$\pi(\sigma|y, \mu, w)\propto \exp(l_w(\sigma))\cdot \pi(\sigma)$$

which is evaluated via `logLikeRcpp/logLikePointRcpp` with each term weighted by $w_i$ and $\pi(\sigma)$ is the usual prior. This should be no more expensive $\lambda$ update (from my assumption). $\mu$'s update via `dbarts` can be kept unmodified, since it introduces no extra approximation. My idea is not to re-design `dbarts`'s internal step, and only modify how we update the density of $\mu$ and $\sigma$.

On $g$, the same distinction may also appear on any transformation fit from $z^\ast$. Box-Cox $\lambda$ is already updated from exact interval likelihood, so it needs only $w_i$ weighting for per observation, the idea of how I treat $\sigma$ above. The one-time CDF-based estimator (`np/pois/neg-bin`) are fit before the sampler starts, directly from the raw count rather than $z^\ast$, so they should be unaffected by this issue, but their estimation should still incorporate $w_i$ (such as a weighted empirical CDF). *This can be left for later work, also whether this sample weights can be incorporated with offsets, how this combination will work, will be interesting to discover later.*

## References
$\bullet$ Savitsky, T. D., & Toth, D. (2016). Bayesian estimation under informative sampling. Electronic Journal of Statistics, 10(1), 1677–1708.
