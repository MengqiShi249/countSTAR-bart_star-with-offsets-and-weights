# Simulation results: `bart_star()` with offsets

All results and figures are produced by **`simulation_full.R`**, which sources:

- `bart_star_offset.R`: `bart_star()` with the new `offsets` / `offsets_test` arguments, and `g_cdf()` with a new `offsets` argument. No other `countSTAR` helper function is modified. The `g_cdf()` change is a correctness fix: the CDF-based transformations (`np`, `pois`, `neg-bin`) are now estimated on the $y/e$ scale on which they are applied. It is not guaranteed to improve the fit.
- `helper_sim.R`: the three data-generating functions.

`simulation_full.R` fits every scenario $\times$ transformation pair, prints the results tables, and draws the fitted-vs-truth and PPC plots. Figures are stored in `(i) data_offset/<transformation>/` and `(ii) data_no/<transformation>/`, with identical file names across subfolders (`fr` / `im` / `ht` = friedman / imbalanced / heterogeneous).

## Simulation design

**Data.** $n = 250$, $p = 6$, $x_{ij} \sim \text{Unif}(0, 1)$, and $Y_{i} \sim \text{NB}$ with mean $m_{i} = e_{i}\exp\{\eta(x_{i})\}$ and size $r$, so $\text{Var}(Y_{i}) = m_{i} + m_{i}^{2}/r$. Here $s(\cdot)$ standardises to mean 0 and SD 1 (R's `scale()`).

| Scenario | $\eta(x)$ | $r$ | Zeros (case i) | Tests |
|---|---|---|---|---|
| `friedman` | $\log 1.5 + \log 5 \cdot s(f(x))$, with $f(x) = 10\sin(\pi x_{1}x_{2}) + 20(x_{3}-0.5)^{2} + 10x_{4} + 5x_{5}$ | 2 | 22.8% | nonlinearity, interactions |
| `imbalanced` | $\log 0.2 + \log 20 \cdot \text{logit}^{-1}\{15(x_{1}-0.7)\}$ | 0.7 | 54.0% | zero-heavy counts |
| `heterogeneous` | $\log 1.5 + \log 3 \cdot s(x_{1})$ | 0.7 if $x_{2} \le 0.5$, else 8 | 24.8% | group-varying overdispersion |

**Cases.**

- **Case (i)**, exposure heterogeneity: $e_{i} \sim \text{Unif}(0.5, 5)$, seed 1. Does modelling the offset help? (7 transformations; `ispline` is skipped.)
- **Case (ii)**, no heterogeneity: $e_{i} \equiv 1$, seed 2. Does `offset = TRUE` reduce to `offset = FALSE`? (All 8 transformations.)

In every cell, `bart_star()` is fit twice on the **same** data: `offset = FALSE` uses the default `offsets = 1` (the original model), and `offset = TRUE` uses `offsets` $= e_{i}$. MCMC settings: `nburn = 500`, `nsave = 1000`, `nskip = 0`, `n.trees = 75`, `save_y_hat = TRUE`. There is one data set and one chain per fit.

## Metrics and plots

- **WAIC** (lower is better), from the pointwise log-likelihood of the observed counts. Both fits in a pair use the same counts, so their WAICs are comparable.
- **RMSE to truth** (case i): $[n^{-1}\sum_{i}(\hat{y_i} - m_{i})^{2}]^{1/2}$, where $\hat{y_i}$ is the posterior mean of $\mathbb{E}(y_{i} \mid x_{i})$ (at exposure $e_{i}$ for `offset = TRUE`).
- **Agreement** (case ii): RMSE and correlation between the two fits' $\hat{y}_{i}$.
- **Fitted vs. truth**: $\hat{y_i}$ with 95% HPD intervals, against the true mean ($m_{i}$ in case i, $\exp\{\eta(x_{i})\}$ in case ii), with a 1:1 line.
- **PPC, proportion of zeros**: histogram over posterior predictive draws, with the observed proportion as a blue line.

## Results

### Case (i): data generated with offsets

**Table 1: WAIC** ($\Delta = \text{TRUE} - \text{FALSE}$).

| $g$ | friedman FALSE | friedman TRUE | friedman $\Delta$ | imbalanced FALSE | imbalanced TRUE | imbalanced $\Delta$ | heterogeneous FALSE | heterogeneous TRUE | heterogeneous $\Delta$ |
|---|---|---|---|---|---|---|---|---|---|
| identity | 1629.7 | 1787.6 | +157.8 | 962.1 | 921.2 | −40.8 | 1549.3 | 1501.5 | −47.8 |
| log | 1399.9 | 1339.8 | −60.2 | 811.1 | 788.5 | −22.6 | 1297.1 | 1250.2 | −46.9 |
| sqrt | 1551.5 | 1432.9 | −118.6 | 850.1 | 825.0 | −25.1 | 1347.9 | 1310.6 | −37.3 |
| box-cox | 1410.4 | 1343.8 | −66.6 | 816.3 | 791.1 | −25.2 | 1300.2 | 1252.9 | −47.3 |
| np | 1376.0 | 1312.3 | −63.8 | 792.3 | 806.0 | +13.7 | 1274.8 | 1247.4 | −27.5 |
| pois | 1670.3 | 1541.0 | −129.3 | 873.9 | 881.8 | +7.9 | 1402.1 | 1370.9 | −31.3 |
| neg-bin | 1439.6 | 1380.3 | −59.3 | 821.4 | 861.5 | +40.1 | 1309.4 | 1303.0 | −6.3 |

**Table 2: RMSE to truth** ($\Delta\%$ = relative change, TRUE vs. FALSE).

| $g$ | friedman FALSE | friedman TRUE | friedman $\Delta\%$ | imbalanced FALSE | imbalanced TRUE | imbalanced $\Delta\%$ | heterogeneous FALSE | heterogeneous TRUE | heterogeneous $\Delta\%$ |
|---|---|---|---|---|---|---|---|---|---|
| identity | 18.81 | 15.77 | −16.2 | 2.679 | 2.295 | −14.3 | 6.533 | 6.222 | −4.8 |
| log | 15.92 | 12.07 | −24.2 | 2.571 | 2.088 | −18.8 | 5.187 | 3.907 | −24.7 |
| sqrt | 17.86 | 14.49 | −18.9 | 2.538 | 2.125 | −16.3 | 5.440 | 4.406 | −19.0 |
| box-cox | 17.03 | 12.41 | −27.1 | 2.574 | 2.107 | −18.1 | 5.252 | 3.834 | −27.0 |
| np | 15.16 | 12.36 | −18.5 | 2.509 | 2.207 | −12.1 | 5.190 | 3.817 | −26.4 |
| pois | 16.63 | 14.62 | −12.1 | 2.561 | 2.056 | −19.7 | 5.827 | 4.979 | −14.5 |
| neg-bin | 17.64 | 14.39 | −18.4 | 2.569 | 2.060 | −19.8 | 5.363 | 4.229 | −21.1 |

*Fitted vs. truth* (case i; one of `box-cox`, `log`, `np` per scenario, with its RMSE change):

| | `offset = FALSE` | `offset = TRUE` |
|---|---|---|
| friedman, `box-cox` (−27.1%) | <img src="(i) data_offset/box-cox/fr-fitted-FALSE.png" width="400"> | <img src="(i) data_offset/box-cox/fr-fitted-TRUE.png" width="400"> |
| imbalanced, `log` (−18.8%) | <img src="(i) data_offset/log/im-fitted-FALSE.png" width="400"> | <img src="(i) data_offset/log/im-fitted-TRUE.png" width="400"> |
| heterogeneous, `np` (−26.4%) | <img src="(i) data_offset/np/ht-fitted-FALSE.png" width="400"> | <img src="(i) data_offset/np/ht-fitted-TRUE.png" width="400"> |

Without the offset, the fitted means scatter widely around the 1:1 line, and in imbalanced the large true means (above about 10) are under-fitted, mostly at 3–8. With the offset, the fitted means concentrate along the line, the large means in imbalanced move towards it, and the intervals narrow for small and moderate means.

**Interpretation**

- **The offset improves the fitted means everywhere.** RMSE to truth falls in all 21 cells (−4.8% to −27.1%; −12.1% to −27.1% excluding `identity`). WAIC falls in 17 of the 21 cells.
- **Largest gains.** `box-cox`, `log` and `neg-bin` reduce RMSE by 18–27% in every scenario, consistent with the multiplicative exposure in the data, which `log` represents exactly. `np` matches them in heterogeneous (−26.4%).
- **Transformation matters.** The choice of $g$ affects fit as the offset does. `log`, `box-cox` and `np` give the best WAIC in every scenario, while `identity` and `pois` are the weakest. In the zero-heavy scenario, the CDF-based transformations (`np`, `pois`, `neg-bin`) have a higher WAIC with the offset (+13.7, +7.9, +40.1), although their RMSE to truth falls by 12–20%. In practice, `log`, `box-cox` or `np` might be safer defaults.

*PPC, proportion of zeros* (case i, friedman; blue line = observed 0.228):

| | `offset = FALSE` | `offset = TRUE` |
|---|---|---|
| `np` | <img src="(i) data_offset/np/fr-PPC-FALSE.png" width="400"> | <img src="(i) data_offset/np/fr-PPC-TRUE.png" width="400"> |
| `log` | <img src="(i) data_offset/log/fr-PPC-FALSE.png" width="400"> | <img src="(i) data_offset/log/fr-PPC-TRUE.png" width="400"> |
| `box-cox` | <img src="(i) data_offset/box-cox/fr-PPC-FALSE.png" width="400"> | <img src="(i) data_offset/box-cox/fr-PPC-TRUE.png" width="400"> |

Zero calibration is driven mainly by $g$:

- `log`, `box-cox`: calibrated in all three scenarios, under both fits.
- `np`: without the offset, friedman zeros are under-predicted (draws centred near 0.16 vs. observed 0.228). With the offset, they are well calibrated.
- `sqrt`, `pois`: zeros are over-predicted in friedman and heterogeneous, under both fits.
- `neg-bin`: calibrated without the offset. With the offset, zeros are over-predicted in friedman (draws centred near 0.30 vs. observed 0.228) and heterogeneous (near 0.32 vs. 0.248).
- `identity`: the worst calibrated.

### Case (ii): `offset = TRUE` with $e \equiv 1$

With $e_{i} \equiv 1$, `offset = TRUE` passes the same offsets as the default, so the two fits run the same code and differ only in the random-number stream. Case (ii) is therefore a pair of independent MCMC runs of the same model on the same data.

**Table 3: WAIC** ($\Delta = \text{TRUE} - \text{FALSE}$).

| $g$ | friedman FALSE | friedman TRUE | friedman $\Delta$ | imbalanced FALSE | imbalanced TRUE | imbalanced $\Delta$ | heterogeneous FALSE | heterogeneous TRUE | heterogeneous $\Delta$ |
|---|---|---|---|---|---|---|---|---|---|
| identity | 1179.7 | 1186.6 | +6.9 | 685.8 | 687.1 | +1.3 | 989.8 | 994.3 | +4.6 |
| log | 1014.7 | 1015.8 | +1.2 | 645.4 | 645.4 | 0.0 | 948.4 | 950.1 | +1.7 |
| sqrt | 1042.1 | 1030.2 | −11.9 | 650.7 | 649.0 | −1.7 | 942.5 | 945.9 | +3.4 |
| box-cox | 1005.7 | 999.7 | −6.1 | 656.4 | 651.6 | −4.7 | 946.0 | 941.4 | −4.6 |
| np | 984.6 | 986.9 | +2.3 | 638.9 | 640.0 | +1.1 | 936.1 | 936.2 | +0.2 |
| pois | 1081.9 | 1081.2 | −0.7 | 655.6 | 652.6 | −2.9 | 947.4 | 948.4 | +1.0 |
| neg-bin | 1005.9 | 1009.0 | +3.1 | 645.6 | 645.3 | −0.2 | 938.5 | 938.5 | 0.0 |
| ispline | 1015.4 | 1009.6 | −5.8 | 660.7 | 650.3 | −10.4 | 946.6 | 945.0 | −1.6 |

**Table 4: Agreement between the two fits' fitted values** (RMSE / correlation).

| $g$ | friedman | imbalanced | heterogeneous |
|---|---|---|---|
| identity | 0.779 / 0.9968 | 0.093 / 0.9984 | 0.144 / 0.9980 |
| log | 0.589 / 0.9978 | 0.077 / 0.9989 | 0.123 / 0.9985 |
| sqrt | 0.445 / 0.9987 | 0.093 / 0.9983 | 0.107 / 0.9987 |
| box-cox | 0.488 / 0.9984 | 0.076 / 0.9989 | 0.129 / 0.9982 |
| np | 0.419 / 0.9989 | 0.074 / 0.9990 | 0.109 / 0.9988 |
| pois | 0.509 / 0.9983 | 0.083 / 0.9986 | 0.132 / 0.9982 |
| neg-bin | 0.574 / 0.9976 | 0.091 / 0.9984 | 0.114 / 0.9986 |
| ispline | 1.475 / 0.9936 | 0.361 / 0.9959 | 0.845 / 0.9986 |

*Fitted vs. truth* (case ii, `np`):

| | `offset = FALSE` | `offset = TRUE` |
|---|---|---|
| friedman | <img src="(ii) data_no/np/fr-fitted-FALSE.png" width="400"> | <img src="(ii) data_no/np/fr-fitted-TRUE.png" width="400"> |
| heterogeneous | <img src="(ii) data_no/np/ht-fitted-FALSE.png" width="400"> | <img src="(ii) data_no/np/ht-fitted-TRUE.png" width="400"> |

**Interpretation**

- **`offset = TRUE` reduces to the original model.** The correlation is $\ge 0.9936$ in all 24 cells, $\lvert\Delta\text{WAIC}\rvert \le 11.9$ ($\le 1.6\%$ of WAIC), and the paired panels are visually indistinguishable. The fitted means are stable across runs (correlation $\ge 0.9967$ for all transformations except `ispline`).
