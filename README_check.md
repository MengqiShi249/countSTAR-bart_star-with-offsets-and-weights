# Function check

`function check.R` checks each part of `bart_star_offset.R` that changed with offsets, against data simulated from the model. `_code` is the output of the component being tested; `_true` and `_sim` come from the simulated data. In every table the code agrees with the model up to simulation or estimation error, including at $y_{\max}$. Sections 4 and 5 check the conditional expectation without the for-loop: its values and its run time.

## 1. `g_cdf()`

**`'pois'`, `'neg-bin'`.** 

Tested: $g$ from `g_cdf()` fitted to $y_i \sim$ Pois($3e_i$) or NB(mean $3e_i$, size 2), $n = 20000$; with $z = g(y^*/e) \sim N(0,1)$, $\Phi(g(j+1)) - \Phi(g(j))$ is $P(y = j)$ at unit exposure. 

Against: the true Pois(3) and NB(mean 3, size 2) probabilities.

| $j$ | pois_code | pois_true | nb_code | nb_true |
|---|---|---|---|---|
| 0 | 0.0499 | 0.0498 | 0.1586 | 0.1600 |
| 1 | 0.1495 | 0.1494 | 0.1925 | 0.1920 |
| 2 | 0.2242 | 0.2240 | 0.1740 | 0.1728 |
| 3 | 0.2240 | 0.2240 | 0.1392 | 0.1382 |
| 4 | 0.1679 | 0.1680 | 0.1043 | 0.1037 |
| 5 | 0.1007 | 0.1008 | 0.0748 | 0.0746 |
| 6 | 0.0503 | 0.0504 | 0.0522 | 0.0523 |
| 7 | 0.0216 | 0.0216 | 0.0356 | 0.0358 |
| 8 | 0.0081 | 0.0081 | 0.0239 | 0.0242 |

**`'np'`.** 

Tested: $\Phi(g(t))$, with $g$ from `g_cdf()` fitted to $n = 5000$ NB counts. 

Against: $P(y/e < t)$ from $10^6$ new draws.

| $t$ | np_code | np_true |
|---|---|---|
| 0.5 | 0.099 | 0.108 |
| 1 | 0.197 | 0.202 |
| 2 | 0.424 | 0.419 |
| 3 | 0.595 | 0.595 |
| 5 | 0.828 | 0.824 |
| 8 | 0.953 | 0.956 |

## 2. Cut points

Tested: $P(y = j)$ from `logLikePointRcpp()` with the cut points $[g(a_j/e), g(a_{j+1}/e))$, for fixed $g = \log$, $\mu = 0.2$, $\sigma = 0.8$, $e = 8$, $y_{\max} = 10$. 

Against: frequencies of $y = \min(\lfloor e g^{-1}(z) \rfloor, y_{\max})$, $z \sim N(\mu, \sigma^2)$, in $10^5$ draws.

| $j$ | P_code | P_sim |
|---|---|---|
| 0 | 0.0022 | 0.0026 |
| 1 | 0.0215 | 0.0219 |
| 2 | 0.0463 | 0.0453 |
| 3 | 0.0622 | 0.0608 |
| 4 | 0.0690 | 0.0691 |
| 5 | 0.0699 | 0.0698 |
| 6 | 0.0673 | 0.0665 |
| 7 | 0.0629 | 0.0641 |
| 8 | 0.0578 | 0.0580 |
| 9 | 0.0525 | 0.0528 |
| 10 | 0.4885 | 0.4891 |

## 3. Expectation and test log-score

These are computed inside `bart_star()`, so a short fit is run ($y_{\max} = 10$) and one saved draw ($\mu$, $\sigma$, $\lambda$) is used. Each row is one simulated test point (only show 6 out of 100 generated points). `e` is its offset, and $y_\text{test}$ is the observed count

Tested: `post.fitted.values.test` (E_code) and `exp(post.log.pred.test)` (P_code), i.e. $E(y\mid \mu, \sigma, e)$ and $P(y = y_\text{test}\mid \mu,\sigma, e)$ from `bart_star_offset` code. 

Against: the same quantities computed from the $10^5$ counts simulated from the same $\mu$, $\sigma$, $\lambda$, i.e. their mean (E_sim) and the frequency of $y_\text{test}$ (P_sim).

**`'log'`**

| $e$ | $y_{\text{test}}$ | E_code | E_sim | P_code | P_sim |
|---|---|---|---|---|---|
| 6.03 | 2 | 1.846 | 1.841 | 0.3449 | 0.3411 |
| 7.56 | 5 | 1.236 | 1.241 | 0.0028 | 0.0030 |
| 2.82 | 1 | 0.012 | 0.012 | 0.0120 | 0.0122 |
| 19.14 | 10 | 7.716 | 7.706 | 0.3627 | 0.3605 |
| 19.15 | 10 | 5.821 | 5.813 | 0.1169 | 0.1160 |
| 19.42 | 10 | 7.426 | 7.433 | 0.3131 | 0.3131 |

**`'box-cox'`**

| $e$ | $y_{\text{test}}$ | E_code | E_sim | P_code | P_sim |
|---|---|---|---|---|---|
| 6.03 | 2 | 3.180 | 3.171 | 0.2663 | 0.2664 |
| 7.56 | 5 | 2.895 | 2.888 | 0.0751 | 0.0736 |
| 2.82 | 1 | 0.108 | 0.107 | 0.1020 | 0.1010 |
| 19.14 | 10 | 8.216 | 8.225 | 0.4829 | 0.4859 |
| 19.15 | 10 | 7.283 | 7.295 | 0.3168 | 0.3191 |
| 19.42 | 10 | 7.482 | 7.472 | 0.3488 | 0.3472 |

## 4. Expectation without the for-loop

For the Box-Cox family, the single call to `expectation_gRcpp()` is E_code in section 3. For the CDF-based transformations, the long vector is checked here with `'np'`, on the data and test points of section 3.

Tested: `post.fitted.values.test` (E_code), for the first saved draw ($\mu$, $\sigma$).

Against: `expectation_gRcpp()` called for one test point at a time with its cut points $g(a_j/e)$, over all $j = 0, \dots, y_{\max}$ (E_loop).

| $e$ | E_code | E_loop |
|---|---|---|
| 6.03 | 2.2595 | 2.2595 |
| 7.56 | 2.1327 | 2.1327 |
| 2.82 | 0.3616 | 0.3617 |
| 19.14 | 6.7010 | 6.7010 |
| 19.15 | 6.3547 | 6.3547 |
| 19.42 | 7.6417 | 7.6417 |

E_code sums $j$ only between the 0.0001 and 0.9999 quantiles of $y$, which is the difference in the third row.

## 5. Run time

Total run time in seconds of one `bart_star()` fit with `'np'` and offsets (`nburn = nsave = 500`), for larger counts: $y_i \sim$ Pois( $10 e_i \exp(x_{i1} - 1.5)$ ), $e_i \sim \text{Unif}(2, 20)$, means from about 4 to 121. `no_y_hat` is `save_y_hat = FALSE`. `long_vector` is `save_y_hat = TRUE`. `for_loop` is `no_y_hat` plus `nsave` times one pass of the earlier loop, with one call to `expectation_gRcpp()` per observation.

| $n$ | no_y_hat | long_vector | for_loop |
|---|---|---|---|
| 1000 | 2.1 | 9.3 | 26.1 |
| 10000 | 19.6 | 95.1 | 343.6 |

Without the fitted values (`no_y_hat`), one iteration takes about 0.002 s ($n = 1000$) and 0.020 s ($n = 10000$) on average. With the long vector, the fitted values add about 0.014 s and 0.15 s per saved iteration, against 0.048 s and 0.65 s with the for-loop. The total run time is about 4.4 and 4.9 times that of `no_y_hat`, against about 12 and 18 times with the for-loop.
