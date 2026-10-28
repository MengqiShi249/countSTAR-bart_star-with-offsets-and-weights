# `y_max` test: `bart_star()` with offsets at a finite upper bound

All results are produced by **`ymax_test.R`**, which sources:

- `bart_star_offset.R`: `bart_star()` and `g_cdf()` with offsets.

This test is to understand whether `bart_star()` runs properly when the upper bound $y_{\max} < \infty$ is reached often because the offsets are large.

## Test design

**Data.** $n = 500$ training and $n_{\text{test}} = 200$ test observations, $p = 5$, $x_{ij} \sim \text{Unif}(0, 1)$, offsets $e_{i} \sim \text{Unif}(2, 20)$. The counts are $y_{i} = \min(y_{i}^{0}, y_{\max})$ with $y_{\max} = 10$, where $y_{i}^{0} \sim \text{NB}$ with mean $e_{i} r(x_{i})$ and size 3, and $r(x) = f(x)/20$, with $f$ Friedman's function. **30.2%** of training counts and **34.5%** of test counts equal $y_{\max}$ exactly.

**Fits.** `bart_star(..., y_max = 10, offsets = e, offsets_test = e_test, save_y_hat = TRUE)` for 7 transformations (`ispline` does not take offsets). MCMC settings: `nburn = 500`, `nsave = 500`. There is one data set and one chain per fit.

## Checks

- **Part 1** (all 7 transformations): all outputs finite; predictive draws in $[0, y_{\max}]$; share of predictive draws at $y_{\max}$ vs. observed; fitted values vs. predictive means, which should agree because both estimate $\mathbb{E}(y \mid x)$.
- **Part 2** (`log` fit): whether the gap between predictive mean and fitted value equals the top term $y_{\max} P(y = y_{\max})$.
  1. Per observation, from the returned output.
  2. Exactly, on the test set. With $g = \log$ and $\Phi$ the standard normal CDF, for $j < y_{\max}$

$$P(y = j) = \Phi\left(\frac{\log((j+1)/e) - \mu}{\sigma}\right) - \Phi\left(\frac{\log(j/e) - \mu}{\sigma}\right), \qquad P(y = y_{\max}) = 1 - \Phi\left(\frac{\log(y_{\max}/e) - \mu}{\sigma}\right)$$

  with $\mu$ and $\sigma$ from `post.mu.test` and `post.sigma`.

## Results

### Part 1: all transformations

**Table 1** (observed share at $y_{\max}$: 0.302 training, 0.345 test).

| $g$ | All finite | Pred. range | Pred. share at $y_{\max}$ (train / test) | Mean fitted (train / test) | Mean pred. (train / test) | WAIC |
|---|---|---|---|---|---|---|
| identity | TRUE | 0–10 | 0.346 / 0.345 | 2.30 / 2.38 | 5.76 / 5.82 | 2078 |
| log | TRUE | 0–10 | 0.331 / 0.332 | 2.43 / 2.46 | 5.74 / 5.78 | 1980 |
| sqrt | TRUE | 0–10 | 0.328 / 0.326 | 2.48 / 2.55 | 5.76 / 5.82 | 1973 |
| box-cox | TRUE | 0–10 | 0.328 / 0.328 | 2.47 / 2.52 | 5.74 / 5.80 | 1968 |
| np | TRUE | 0–10 | 0.326 / 0.320 | 2.53 / 2.61 | 5.79 / 5.81 | 2009 |
| pois | TRUE | 0–10 | 0.326 / 0.321 | 2.52 / 2.58 | 5.77 / 5.79 | 2029 |
| neg-bin | TRUE | 0–10 | 0.325 / 0.323 | 2.50 / 2.57 | 5.75 / 5.80 | 1979 |

**Interpretation**

- **The bound is handled correctly in sampling, the likelihood and prediction.** All outputs are finite, and the predictive draws stay in $[0, 10]$ and reach $y_{\max}$ 32–35% of the time (observed 30.2% / 34.5%). The top category is the interval $[y_{\max}/e, \infty)$, which matches how the data were capped.
- **The fitted values are too low.** They average about 2.5, against about 5.8 for the predictive draws, and the largest is 3.1–5.0 for every transformation, although 30% of the data equal 10.

### Part 2: the conditional expectation omits $y_{\max}$ (`log`)

**Table 2: training observations most likely to be at $y_{\max}$.**

| $y$ | $P(y = y_{\max})$ | Pred. mean | Fitted | Gap | $y_{\max} P(y = y_{\max})$ |
|---|---|---|---|---|---|
| 10 | 0.896 | 9.59 | 0.77 | 8.82 | 8.96 |
| 10 | 0.882 | 9.56 | 0.83 | 8.72 | 8.82 |
| 10 | 0.866 | 9.48 | 0.92 | 8.56 | 8.66 |
| 10 | 0.858 | 9.54 | 1.03 | 8.51 | 8.58 |

Over all 500 observations, cor(gap, $y_{\max} P(y = y_{\max})$) = **0.9987**, and the mean absolute difference is 0.097.

**Exact check on the test set.** Computing $P(y = j)$ exactly from `post.mu.test` and `post.sigma`, the returned fitted value equals $\sum_{j<y_{\max}} j P(y=j)$ to within 0.0014 in all 100,000 draw × observation pairs, and adding $y_{\max} P(y = y_{\max})$ raises the mean from 2.46 to 5.78, matching the predictive mean.

**Interpretation**

- **The top term is missing.** For a capped count, $\mathbb{E}(y) = \sum_{j=0}^{y_{\max}-1} j P(y=j) + y_{\max} P(y = y_{\max})$. The returned fitted values equal only the first sum (Exact check), so the gap is exactly $y_{\max} P(y = y_{\max})$. Observations that are almost certainly 10 get a fitted value of about 1 (Table 2), and adding the term back recovers the predictive mean (5.778 vs. 5.783).
- **Cause in the code.** `Jmax = ceiling(round_floor(g_inv(.) * offsets, y_max))` is at most $y_{\max}$, and `expectation_gRcpp()` sums $j = 0, \dots, J_{\max} - 1$, so $j = y_{\max}$ is never included. The cut points `a_j(1:(Jmaxmax + 1))` also omit `y_max`, so the top cell ends at $(y_{\max}+1)/e$ instead of $\infty$. The test log-predictive score, `a_j(y_test + 1)`, has the same omission.
- **What is not affected.** The expectation is computed after each MCMC update and is not used by the sampler. The $z$ sampling, BART, $\sigma$ and $\lambda$ updates, the log-likelihood and WAIC, and the predictive draws (`post.pred`, `post.pred.test`) all use the top interval $[y_{\max}/e, \infty)$ are correct. With $y_{\max} = \infty$ (default), the omitted mass is negligible.

**Conclusion.** With large offsets and about a third of the counts at the bound, `bart_star()` with offsets works properly for all 7 transformations. The one issue, present without offsets as well, is that `fitted.values`, `post.fitted.values` and `post.fitted.values.test` understate $\mathbb{E}(y \mid x)$ by $y_{\max} P(y = y_{\max})$, and `post.log.pred.test` uses $[y_{\max}, y_{\max}+1)$ instead of $[y_{\max}, \infty)$ at $y = y_{\max}$.
