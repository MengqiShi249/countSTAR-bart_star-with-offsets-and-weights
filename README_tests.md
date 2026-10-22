# Tests for `bart_star()` with offsets

`test.R` fits the offset version of `bart_star_offset.R` with 7 transformations on simulated offset count data (friedman or heterogeneous, 3 offset designs $\times$ `y_max` $=\infty, 30$) to check two things:

1. **Conditional expectations.** `post.fitted.values` and `post.fitted.values.test`, computed by `expect_y()`, must agree with the posterior predictive draws `post.pred` and `post.pred.test`, which serve as the benchmark.
2. **`y_max`.** With a known upper bound, no predictive draw or fitted value may exceed `y_max`, with or without offsets.

These are not performance comparisons: there is no WAIC and no comparison with the true mean.
- `test.R`: runs both tests and prints Tables 1 and 2; set `scenario` at the top 
- `helper_sim.R`: simulators with offsets 
- `bart_star_offset.R`: function under test (current version) 

## Setup

**Data.** Negative-binomial counts with mean proportional to the exposure, $`\mathbb E(y_i \mid x_i, e_i) = e_i\,\lambda(x_i)`$, with $`x_i \sim U(0,1)^5`$:

| Scenario | $`\lambda(x)`$ | NB size |
|---|---|---|
| `friedman` | $`\exp\{\log 1.5 + \log 5\cdot\mathrm{std}(f(x))\}`$, Friedman $f$ | 2 |
| `heterogeneous` | $`\exp\{\log 1.5 + \log 3\cdot\mathrm{std}(x_1)\}`$ | 0.7 if $`x_2 \le 0.5`$, else 8 |

Each dataset has 150 training and 60 test cases (data seed 1). For finite `y_max`, the counts are truncated with `pmin(y, y_max)`.

**Designs.** Three offset structures × `y_max` $\in \{\infty, 30\}$ give 6 designs per scenario:

| Offsets | Purpose |
|---|---|
| $`e = 1`$ | unit-offset path |
| $`e \sim U(0.5, 5)`$ | continuous, all distinct |
| $`e \in \{0.5, 1, 2, 5\}`$ | few distinct values (e.g. fixed survey durations) |

The test offsets are generated the same way and passed as `offsets_test`. All transformations except `ispline` are used: `identity`, `log`, `sqrt`, `box-cox`, `np`, `pois`, `neg-bin`. MCMC settings are `n.trees = 50`, `nburn = 200`, `nsave = 500`, with `set.seed(10)` before each fit. In total there are 2 × 6 × 7 = 84 fits.

## Test 1: conditional expectation vs. predictive draws

At each saved draw $s$, the fitted value $`F_i^{(s)} = \mathbb E(y_i \mid \theta^{(s)}, e_i)`$ and the predictive draw $`\tilde y_i^{(s)}`$ use the same parameters $`\theta^{(s)}`$. So $`d_i^{(s)} = \tilde y_i^{(s)} - F_i^{(s)}`$ has mean 0 given $`\theta^{(s)}`$ if `expect_y()` is correct. The draws are independent across $s$ and $i$ given the chain, so the check does not depend on convergence.

| Column | Definition | Expected if correct |
|---|---|---|
| `fitted`, `pred` | mean of $`F_i^{(s)}`$ and of $`\tilde y_i^{(s)}`$ over $s, i$ | equal |
| `diff_pct` | $`100(\texttt{fitted}/\texttt{pred} - 1)`$ | within about $\pm 2\%$ (Monte Carlo error) |
| `z` | $`\sum_{s,i} d_i^{(s)} / \sqrt{\sum_{s,i}(d_i^{(s)} - \bar d_i)^2}`$ (pooled) | $`\approx N(0,1)`$: $`\lvert z\rvert < 2`$ in ~95% of rows |
| `max_abs_zi` | $`\max_i \lvert\bar d_i\rvert/(s_i/\sqrt{500})`$ (per observation) | 95% reference ≈ 3.6 (train), 3.3 (test) |

**Results.** Range of `diff_pct` and max $`\lvert z\rvert`$ over the three offset designs and both sets. Bold marks $`\lvert z\rvert > 2`$.

| $g$ | `friedman`, $`y_{\max}=\infty`$ | `friedman`, $`y_{\max}=30`$ | `heterogeneous`, $`y_{\max}=\infty`$ | `heterogeneous`, $`y_{\max}=30`$ |
|---|---|---|---|---|
| `identity` | −0.6 to 0.7%, 1.96 | −0.8 to 0.4%, 1.83 | −0.4 to 0.4%, 1.31 | −0.9 to 0.5%, 1.61 |
| `log` | −7.4 to 0.6%, **4.57** | −0.7 to 0.7%, 1.82 | −5.9 to −2.3%, **3.67** | −1.0 to 0.7%, 1.82 |
| `sqrt` | −0.7 to 0.4%, 1.32 | −0.2 to 0.3%, 0.91 | −0.3 to 0.5%, 1.23 | 0.0 to 0.8%, 1.55 |
| `box-cox` | −2.9 to 0.3%, **2.35** | −0.8 to 0.3%, 1.23 | −0.5 to 0.9%, 1.24 | −0.7 to 0.6%, 1.51 |
| `np` | −1.8 to 1.7%, 1.63 | −1.9 to 0.5%, **2.70** | −0.7 to 1.2%, 1.61 | −0.5 to 1.5%, **2.05** |
| `pois` | −0.8 to 0.2%, 1.49 | −0.5 to 0.4%, 1.62 | −0.3 to 0.7%, 1.08 | −0.2 to 0.6%, 1.74 |
| `neg-bin` | −1.2 to 0.9%, 1.64 | −0.5 to 0.3%, 1.52 | −0.5 to 0.6%, 1.37 | −0.3 to 0.6%, 1.63 |

**Interpretation.**

1. **Offsets are handled correctly.** Leave aside `log` and `box-cox` at $`y_{\max} = \infty`$ (point 3). In the other 144 rows, $`\lvert\texttt{diff\_pct}\rvert \le 1.9\%`$ and $`\lvert z\rvert \le 2.70`$. Only 2 rows have $`\lvert z\rvert > 2`$, fewer than the ~7 expected by chance. Agreement is the same for unit, continuous and discrete offsets, and for the training set (`offsets`) and the test set (`offsets_test`).
2. **Finite `y_max` is exact on the upper side.** With $`y_{\max} = 30`$, every transformation agrees, including `log`. The tail sum includes $`\Pr(y_i \ge y_{\max})`$. The previous `expectation_gRcpp()` sum dropped the top category and was biased by −20% to −77% in these designs. As a consistency check, `heterogeneous`/`identity` with $e = 1$ gives identical rows for $`y_{\max} = \infty`$ and $30$, because the bound never binds there (Table 2).
3. **Heavy-tailed $g$ with $`y_{\max} = \infty`$: truncation, not offsets.** `log` is 2.3–7.4% low in 10 of 12 rows, and `box-cox` (`friedman`) is up to 2.9% low in 3 rows. The sum stops at the 0.9999 quantile (capped at $`2\max_k(y_k/e_k)\,e_i`$) and misses the upper tail. For log-normal $`y^\ast`$ the missing share of the mean is $`1 - \Phi(3.72 - \sigma)`$, e.g. 4.3% at $\sigma = 2$. This is the original cutoff rule: the gap also appears at $e = 1$ and vanishes with finite `y_max`.
4. **Large `max_abs_zi` reflects skewed draws, not error.** The largest values are 8.24, 7.08 and 5.54, all in `friedman`, $`e \in \{0.5,1,2,5\}`$, $`y_{\max} = 30`$, training set. Their pooled $`\lvert z\rvert`$ is below 0.5. They likely come from observations whose draws are almost all at `y_max`, so the per-observation SD is close to 0 and $z_i$ is inflated.
5. 
## Test 2: `y_max`

Computed for $`y_{\max} = 30`$ only.

| Column | Definition | Expected |
|---|---|---|
| `max_pred`, `max_pred_test`, `max_fitted` | maxima of the predictive draws and fitted values | $`\le y_{\max}`$ |
| `obs_at_ymax` | share of training counts equal to $`y_{\max}`$ | data |
| `pred_at_ymax`, `pred_q025`, `pred_q975` | posterior mean and 95% interval of the predicted share at $`y_{\max}`$ | interval contains `obs_at_ymax` |
| `loglik_finite` | `post.log.like.point` and `post.log.pred.test` all finite | `TRUE` |

**Results** (summary over the 7 transformations):

| Scenario | Offsets | `obs_at_ymax` | `pred_at_ymax` | Interval contains obs | Maxima $`\le 30`$ | `loglik_finite` |
|---|---|---|---|---|---|---|
| `friedman` | $e = 1$ | 0.047 | 0.007–0.050 | 4/7 (not `identity`, `sqrt`, `pois`) | 7/7 | 7/7 |
| `friedman` | $`U(0.5,5)`$ | 0.080 | 0.085–0.110 | 7/7 | 7/7 | 6/7 (`identity`) |
| `friedman` | $`\{0.5,1,2,5\}`$ | 0.080 | 0.071–0.090 | 7/7 | 7/7 | 7/7 |
| `heterogeneous` | $e = 1$ | 0 | 0–0.004 | 7/7 | 7/7 (bound not binding) | 7/7 |
| `heterogeneous` | $`U(0.5,5)`$ | 0.060 | 0.056–0.073 | 7/7 | 7/7 | 7/7 |
| `heterogeneous` | $`\{0.5,1,2,5\}`$ | 0.013 | 0.028–0.033 | 7/7 | 7/7 | 7/7 |

**Interpretation.**

1. **The bound is respected in all 42 fits.** Predictive draws are scaled by the exposure before rounding and capping, $`h(e_i\,g^{-1}(\tilde z_i))`$, and `expect_y()` sums only up to $`J_i \le y_{\max}`$.
2. **The mass at the bound is reproduced in 39 of 42 fits.** The 3 misses are light-tailed $g$ (`identity`, `sqrt`, `pois`) in `friedman` with $e = 1$, which underpredict the pile-up at 30. This is model fit (the choice of $g$), not the handling of `y_max`. With 150 counts, the observed shares are coarse; e.g. 0.013 is 2 counts.
3. **One non-finite log-likelihood remains** (`friedman`, `identity`, $`U(0.5,5)`$). This is unchanged by the `y_max` fix to the test log-predictive score, so it is not caused by the test intervals at the bound. It most likely comes from underflow in the unchanged helper `logLikePointRcpp()`: $`\log\{\Phi(b) - \Phi(a)\}`$ becomes $`-\infty`$ when the observation lies far into the tail of $`N(\mu_i, \sigma^2)`$. A light-tailed $g$ makes this more likely. It is not related to offsets or `expect_y()`.

## Conclusions

- `expect_y()` matches the predictive draws for every offset design, for both training and test sets and all seven transformations. With finite `y_max` it is exact on the upper side, which fixes the old underestimation at the bound.
- The only systematic gap is the inherited truncation for heavy-tailed $g$ (`log`, `box-cox` near $`\lambda = 0`$) with $`y_{\max} = \infty`$: up to about 7%. In that case `colMeans(post.pred)` is the safer estimate of $`\mathbb E(y)`$.
- `y_max` is respected by all draws and fitted values, with and without offsets.

## Reproducing

Set `scenario = "friedman"` or `"heterogeneous"` and the source paths in `test.R`, then run `source("test.R")`. Seeds are fixed, so the tables reproduce with the same countSTAR and dbarts versions.
