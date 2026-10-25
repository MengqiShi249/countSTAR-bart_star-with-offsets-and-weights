# countSTAR: `bart_star()` with offsets

This version of `bart_star()` (file `bart_star_offset.R`) adds known **offsets** (exposures) $e_i > 0$ to the BART-STAR model for counts, for example survey effort or detection proportion in point-count data. No countSTAR helper function is added or modified. All changes are to the inputs of the existing helpers, mainly the cut points.

## Model

Original STAR model with a BART mean:

$$y_i = \lfloor y_i^\ast \rfloor, \qquad z_i = g(y_i^\ast), \qquad z_i = \mu(x_i) + \varepsilon_i, \qquad \varepsilon_i \sim N(0, \sigma^2)$$

The observation $y_i = j$ corresponds to $y_i^\ast \in [a_j, a_{j+1})$, with $a_j = j$, $a_0 = -\infty$ and $a_{y_{\max}+1} = \infty$.

**With offsets**, $g$ is applied to $y^\ast/e$ and nothing else changes:

$$y_i = \lfloor y_i^\ast \rfloor, \qquad z_i = g(y_i^\ast/e_i), \qquad z_i = \mu(x_i) + \varepsilon_i, \qquad \varepsilon_i \sim N(0, \sigma^2)$$

Since $e_i > 0$ and $g$ is increasing,

$$y_i = j \iff \frac{y_i^\ast}{e_i} \in [\frac{a_j}{e_i}, \frac{a_{j+1}}{e_i}) \iff g(\frac{a_j}{e_i}) \le z_i < g(\frac{a_{j+1}}{e_i})$$

with $\pm\infty/e_i = \pm\infty$. The cut points $[j, j+1)$ become $[j/e_i, (j+1)/e_i)$; the upper endpoint is $(j+1)/e_i$, not $j/e_i + 1$. The likelihood is

$$\ell(\mu, \sigma) = \sum_{i=1}^n \log[\Phi(\frac{g(a_{y_i+1}/e_i) - \mu_i}{\sigma}) - \Phi(\frac{g(a_{y_i}/e_i) - \mu_i}{\sigma})]$$

For $g = \log$ this is the usual log-offset: $\log y_i^\ast = \log e_i + \mu(x_i) + \varepsilon_i$.

In the code, `a_y = a_j(y, y_max)/offsets`, `a_yp1 = a_j(y + 1, y_max)/offsets`, `z_lower = g(a_y)` and `z_upper = g(a_yp1)`. These are used in the $z$ sampling step (`rtruncnormRcpp`), in the $\lambda$ update for `'box-cox'`, and in the pointwise log-likelihood, so also in the WAIC. The BART step (`dbarts`) is unchanged.

## Transformation

**Box-Cox family (`'identity'`, `'log'`, `'sqrt'`, `'box-cox'`).** $g = g_\lambda$ is applied to $y^\ast/e$. For `'box-cox'`, $\lambda$ is sampled as in the original; its likelihood uses the intervals $[a_{y_i}/e_i, a_{y_i+1}/e_i)$.

**`'np'`: what is estimated.** As in the original, $g = \Phi^{-1}\circ F$. Here $F$ is the marginal CDF of the latent rate $y^\ast/e$, the quantity $g$ transforms. $F$ is estimated by the empirical CDF of the observed $y/e$, taken strictly below each observed value:

$$g(t) = \Phi^{-1}\left(\frac{1}{n+1}\sum_{i=1}^{n} \mathbb{1}\left(\frac{y_i}{e_i} \lt t\right)\right), \qquad t \in \left\lbrace \frac{y_i}{e_i} : \frac{y_i}{e_i} \gt 0 \right\rbrace$$

dropping infinite values and interpolating with the same monotone spline as `g_cdf()`. `g_cdf(y)` uses $\hat F_y(t-1)$. For counts this equals $\frac{1}{n}\sum_{i=1}^{n} \mathbb{1}(y_i \lt t)$, because $t-1$ is the next-smaller possible count. So both use the empirical CDF strictly below each observed value, of $y$ in the original and of $y/e$ here. With $e \equiv 1$ they are identical.

- **Why $y/e$.** The latent rate is known only to lie in $[y_i/e_i, (y_i+1)/e_i)$, so the natural choices are its lower endpoint $y/e$, upper endpoint $(y+1)/e$ and midpoint $(y+0.5)/e$. We use $y/e$ for two reasons. First, with $e \equiv 1$ it gives exactly the original g_cdf(y). Second, its empirical CDF jumps at every knot, so $g$ is strictly increasing and every observation has positive probability. The other two can be flat between knots and give an observation zero probability.
- **Why not `g_cdf(y/offsets)`.** Its $\hat F(t-1)$ step is the empirical CDF of $y/e + 1$, so the upper endpoint becomes $j/e + 1$ instead of $(j+1)/e$, and the results depend on the units of $e$. The estimator above does not: replacing $e$ by $ce$ gives $g_{ce}(t) = g_e(ct)$, so all results are unchanged.

**`'pois'`, `'neg-bin'`.** These are unchanged: $g$ is the moment-matched Poisson or Negative Binomial CDF of the counts $y$. The offsets enter only through the cut points, initialization, grid and predictions.

**`'ispline'`** is not available with offsets. The function stops if any `offsets` or `offsets_test` differs from 1; otherwise it calls the original `bart_star_ispline()`.

## Predictions, conditional expectation, scores

**Posterior predictive draws.** At each saved iteration, $\tilde z_i \sim N(\mu_i, \sigma^2)$ and

$$\tilde y_i = \lfloor g^{-1}(\tilde z_i) \cdot e_i \rfloor$$

computed as `round_floor(g_inv(z_tilde) * offsets, y_max)`. Test draws (`post.pred.test`) use `offsets_test`. As in the original, $g^{-1}$ is exact for the Box-Cox family and a grid approximation (`g_inv_approx()`) for `'np'`, `'pois'` and `'neg-bin'`.

**Conditional expectation.** The original `expectation_gRcpp()` is used unchanged. Only its inputs change, with the cut points divided by $e_i$:

$$\mathbb{E}(y_i \mid x_i, e_i) = \sum_{j=0}^{J_i-1} j[\Phi(\frac{g(a_{j+1}/e_i) - \mu(x_i)}{\sigma}) - \Phi(\frac{g(a_j/e_i) - \mu(x_i)}{\sigma})]$$

Here $J_i$ is `ceiling(round_floor(g_inv(qnorm(0.9999, mu, sigma)) * offsets, y_max))`, the 0.9999 quantile of $y_i$ at exposure $e_i$, capped at $2\max(y)$ as in the original; mass above the cap is not counted. The cut points $a_j/e_i$ differ between observations, so `expectation_gRcpp()` is called once per observation. `fitted.values` and `post.fitted.values` are at `offsets`, and `post.fitted.values.test` is at `offsets_test`. With `offsets_test = 1`, the latter is the expected count per unit exposure. With $e \equiv 1$ this is the original computation.

**Log-predictive scores** for the test data use the intervals `[a_j(y_test)/offsets_test, a_j(y_test + 1)/offsets_test)`.

## Initialization

The original rules, applied to the scale $y/e$:

| Quantity | Code |
|---|---|
| Initial `z_star` | `g((y + abs(rnorm(n)))/offsets)` |
| `sigest` (when `NULL`) | `sd(g((y + 1)/offsets))` |
| `sigest`, `'box-cox'` | the original pilot MCMC if all offsets equal 1; otherwise the rule above at the initial $\lambda$, because the pilot MCMC does not take offsets |
| Grid for `g_inv_approx()` | 250 equally spaced points on $[0, \min\{2\max(y/e),\ y_{\max}/\min(e)\}]$, plus 250 quantiles of the unique $(y+1)/e$ over $y < y_{\max}$ |

## Usage

```r
library(countSTAR)
source("bart_star_offset.R")
environment(bart_star) <- asNamespace("countSTAR")   # bart_star() uses countSTAR internals

fit = bart_star(y = y, X = X, offsets = e,             # e: known exposures
                X_test = X, offsets_test = 1,
                transformation = 'np', save_y_hat = TRUE)

fit$post.pred                          # predictive draws of y at the observed exposures e
fit$fitted.values                      # E(y | x, e) at the observed exposures e
colMeans(fit$post.fitted.values.test)  # E(y | x, e = 1): expected count per unit exposure
```

- `offsets`: `n x 1` vector or scalar, $e > 0$; the default is 1. `offsets_test`: `n_test x 1` vector or scalar; the default is 1. Scalars are recycled, and wrong lengths or values $\le 0$ stop with an error.
- `post.pred`, `post.log.like.point` and the fitted values are at `offsets`. `post.pred.test`, `post.fitted.values.test` and `post.log.pred.test` are at `offsets_test`.
- `post.mu.test` and `post.sigma` are on the scale of $z = g(y^\ast/e)$. With `'np'` they do not depend on the units of $e$. With the other transformations, they do. For example, with `'log'`, using $ce$ instead of $e$ is the same model with $\mu$ shifted by $-\log c$.

## Code changes (relative to `bart_star()` in `STAR_Bayesian.R`)

| Location | Change |
|---|---|
| Arguments | `offsets = 1, offsets_test = 1`, added last |
| Checks | recycle scalar `offsets`; length and positivity |
| `'ispline'` | stop if offsets differ from 1; drop `offsets`, `offsets_test` before calling `bart_star_ispline()` |
| `'np'` transformation | three lines computing $g$ above, in place of `g_cdf(y)`; `'pois'`, `'neg-bin'` unchanged |
| Grid for `g_inv_approx()` | built from `y/offsets` and `(y + 1)/offsets` |
| Initial `z_star`, `sigest` | divided by `offsets`; `'box-cox'` pilot only if `all(offsets == 1)` |
| Test-set checks | recycle scalar `offsets_test`; length and positivity |
| Cut points | `a_y`, `a_yp1` divided by `offsets` |
| Conditional expectation | `Jmax` uses `g_inv(.) * offsets`; `expectation_gRcpp()` called per observation with `g(a_j(.)/offsets[i])` (training) and `g(a_j(.)/offsets_test[i])` (test) |
| Predictive draws | `g_inv(.) * offsets` (training), `g_inv(.) * offsets_test` (test), inside `round_floor()` |
| Test log-scores | `a_j(.)/offsets_test` |

Everything else is identical to the original: the BART update, the $\lambda$ slice sampler, the log-likelihood, the WAIC and the returned list.
