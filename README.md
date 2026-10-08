# Simulation results: `bart_star()` with offsets

All results and figures below are produced by **`simulation_full.R`**, which sources:

- `bart_star_offset.R`: `bart_star()` with the new `offsets` / `offsets_test` arguments. No `countSTAR` helper function is modified.
- `helper_sim.R`: the three data-generating functions.

`simulation_full.R` fits every scenario $\times$ transformation pair, prints the results tables, and draws the fitted-vs-truth and PPC plots. Figures are stored in `(i) data_offsets/<transformation>/` and `(ii) data_no/<transformation>/`, with identical file names across subfolders (`fr` / `im` / `ht` = friedman / imbalanced / heterogeneous). All transformations are included to test how robust the offset extension is, and how the choice of transformation affects it.

## Model

With known exposures $e_{i} > 0$, the offset STAR model is

$$
y_{i} = h(y_{i}^{\ast}), \qquad g\left(y_{i}^{\ast}/e_{i}\right) = \mu(x_{i}) + \varepsilon_{i}, \qquad \varepsilon_{i} \overset{iid}{\sim} N(0,\sigma^{2})
$$

where $\mu(\cdot)$ is a BART ensemble and $h$ is the floor operator: $y_{i} = j$ if and only if $a_{j} \le y_{i}^{\ast} < a_{j+1}$, with $a_{0} = -\infty$ and $a_{j} = j$ for $j \ge 1$. Writing $z_{i} = g(y_{i}^{\ast}/e_{i})$,

$$
y_{i} = j \iff z_{i} \in \big[g(a_{j}/e_{i}),\ g(a_{j+1}/e_{i}) \big)
$$

so the offset enters **only through the observation-specific cut points** $a_{j}/e_{i}$. For $g = \log$ this is the GLM log-offset, $\log y_{i}^{\ast} = \log e_{i} + \mu(x_{i}) + \varepsilon_{i}$. With $e_{i} \equiv 1$ the code performs exactly the computations of the original `bart_star()`.

How each transformation is handled with offsets:

| Type | Transformations | $g$ with offsets |
|---|---|---|
| Fixed | `identity`, `log`, `sqrt` | Box–Cox $g_{\lambda}$ applied to the density $y^{\ast}/e$ ($\lambda = 1, 0, 1/2$) |
| Sampled | `box-cox` | $g_{\lambda}(y^{\ast}/e)$, with $\lambda$ slice-sampled each iteration (prior $N(1/2, 1)$ truncated to $[0, 3]$). The pilot MCMC used to calibrate the $\sigma$ prior runs only when $e \equiv 1$; with offsets, the $\sigma$ prior is calibrated from $\text{sd}\{g_{\lambda_{0}}((y+1)/e)\}$ at the random initial $\lambda_{0}$ |
| Plug-in CDF | `np` | empirical CDF of the observed densities $y_{i}/e_{i}$ |
| Plug-in CDF | `pois`, `neg-bin` | $g(t) = g_{y}(\bar{e}\, t)$, where $g_{y}$ is the moment-matched count CDF and $\bar{e}$ is the geometric mean of the $e_{i}$ |
| — | `ispline` | not available with non-unit offsets; run in case (ii) only |

## Simulation design

**Data.** $n = 250$, $p = 6$, $x_{ij} \sim \text{Unif}(0, 1)$, and $Y_{i} \sim \text{NB}$ with mean $m_{i} = e_{i}\exp\{\eta(x_{i})\}$ and size $r$, so $\text{Var}(Y_{i}) = m_{i} + m_{i}^{2}/r$. Here $\eta(x)$ is the log mean per unit exposure. Smaller $r$ means stronger overdispersion, and $r \to \infty$ gives a Poisson. Below, $s(\cdot)$ standardises to sample mean 0 and SD 1 (R's `scale()`, computed over the $n$ simulated observations), and $\text{logit}^{-1}$ is the logistic function.

| Scenario | $\eta(x)$ | $r$ | Zeros (case i) | Tests |
|---|---|---|---|---|
| `friedman` | $\log 1.5 + \log 5 \cdot s(f(x))$, with $f(x) = 10\sin(\pi x_{1}x_{2}) + 20(x_{3}-0.5)^{2} + 10x_{4} + 5x_{5}$ | 2 | 22.8% | nonlinearity, interactions |
| `imbalanced` | $\log 0.2 + \log 20 \cdot \text{logit}^{-1}\{15(x_{1}-0.7)\}$ | 0.7 | 54.0% | zero-heavy counts |
| `heterogeneous` | $\log 1.5 + \log 3 \cdot s(x_{1})$ | 0.7 if $x_{2} \le 0.5$, else 8 | 24.8% | group-varying overdispersion |

Negative binomial counts with a log-exposure offset are a standard model for overdispersed counts with known effort. The data are not generated from a STAR model, so the simulation tests `bart_star()` under realistic misspecification.

**Cases.**

- **Case (i)**, real exposure heterogeneity: $e_{i} \sim \text{Unif}(0.5, 5)$, seed 1. Does modelling the offset help? (7 transformations; `ispline` is skipped.)
- **Case (ii)**, no heterogeneity: $e_{i} \equiv 1$, seed 2. Does `offset = TRUE` reduce to `offset = FALSE`? (All 8 transformations.)

In every cell `bart_star()` is fit twice on the **same** data. In the tables and plots, `offset = FALSE` means the default `offsets = 1` (the original model) and `offset = TRUE` means `offsets` $= e_{i}$. Each data set is seeded inside its generator, so the same two data sets per scenario are used for every transformation. MCMC settings: `nburn = 500`, `nsave = 1000`, `nskip = 0`, `n.trees = 75`, `save_y_hat = TRUE`. There is one data set and one chain per fit.

## Metrics and plots

- **WAIC** (lower is better), from the pointwise log-likelihood of the observed counts. Both fits in a pair use the same counts, so their WAICs are comparable.
- **RMSE to truth** (case i): $\big[n^{-1}\sum_{i=1}^{n}(\hat{y_i} - m_{i})^{2}\big]^{1/2}$, where $\hat{y_i}$ is the posterior mean of $\mathbb{E}(y_{i} \mid x_{i})$ under the fitted model (at exposure $e_{i}$ for `offset = TRUE`).
- **Correlation** (case ii): RMSE and correlation between the two fits' $\hat{y}_{i}$.
- **Fitted vs. truth**: $\hat{y_i}$ with 95% HPD intervals for $\mathbb{E}(y_{i})$, plotted against the true mean ($m_{i}$ in case i, $\exp\{\eta(x_{i})\}$ in case ii, labelled "Data Values"), with a 1:1 line.
- **PPC, proportion of zeros**: histogram of the zero proportion across posterior predictive draws, with the observed proportion as a blue line.

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
| pois | NaN | 1563.5 | NaN | 871.9 | 854.1 | −17.9 | 1400.9 | 1367.4 | −33.4 |
| neg-bin | 1439.7 | 1356.2 | −83.5 | 825.3 | 812.9 | −12.5 | 1307.0 | 1269.9 | −37.0 |

**Table 2: RMSE to truth** ($\Delta\%$ = relative change, TRUE vs. FALSE).

| $g$ | friedman FALSE | friedman TRUE | friedman $\Delta\%$ | imbalanced FALSE | imbalanced TRUE | imbalanced $\Delta\%$ | heterogeneous FALSE | heterogeneous TRUE | heterogeneous $\Delta\%$ |
|---|---|---|---|---|---|---|---|---|---|
| identity | 18.81 | 15.77 | −16.2 | 2.679 | 2.295 | −14.3 | 6.533 | 6.222 | −4.8 |
| log | 15.92 | 12.09 | −24.1 | 2.571 | 2.098 | −18.4 | 5.187 | 3.967 | −23.5 |
| sqrt | 17.86 | 14.49 | −18.9 | 2.539 | 2.126 | −16.3 | 5.440 | 4.406 | −19.0 |
| box-cox | 17.03 | 12.44 | −26.9 | 2.574 | 2.113 | −17.9 | 5.252 | 3.856 | −26.6 |
| np | 15.16 | 12.36 | −18.5 | 2.509 | 2.200 | −12.3 | 5.190 | 3.856 | −25.7 |
| pois | 16.32 | 14.47 | −11.3 | 2.557 | 2.098 | −18.0 | 5.963 | 5.118 | −14.2 |
| neg-bin | 17.62 | 13.80 | −21.7 | 2.561 | 2.093 | −18.3 | 5.332 | 4.081 | −23.5 |

*Fitted vs. truth* (case i, `box-cox`):

| | `offset = FALSE` | `offset = TRUE` |
|---|---|---|
| friedman | <img src="(i) data_offsets/box-cox/fr-fitted-FALSE.png" width="400"> | <img src="(i) data_offsets/box-cox/fr-fitted-TRUE.png" width="400"> |
| imbalanced | <img src="(i) data_offsets/box-cox/im-fitted-FALSE.png" width="400"> | <img src="(i) data_offsets/box-cox/im-fitted-TRUE.png" width="400"> |
| heterogeneous | <img src="(i) data_offsets/box-cox/ht-fitted-FALSE.png" width="400"> | <img src="(i) data_offsets/box-cox/ht-fitted-TRUE.png" width="400"> |

Without the offset, the fitted means scatter widely around the 1:1 line, and large means are under-estimated (most clearly in imbalanced and heterogeneous). With the offset, the points concentrate along the 1:1 line and the intervals narrow for small and moderate means. The largest means remain under-estimated in imbalanced and in heterogeneous (to a lesser extent).

**Interpretation**

- **The offset helps consistently.** RMSE to truth falls in all 21 cells (−4.8% to −26.9%; −11.3% to −26.9% excluding `identity`), and WAIC falls in 18 of the 20 finite cells.
- **Size of the gains.**
  - RMSE: `log` and `box-cox` (which nests `log`) reduce RMSE by 17.9–26.9% in every scenario. This is consistent with the multiplicative exposure in the data, which `log` represents exactly. `neg-bin` is close behind (−18.3% to −23.5%), and `np` is comparable in heterogeneous (−25.7%). In imbalanced, `log`, `neg-bin`, `pois` and `box-cox` all lie within −17.9% to −18.4%.
  - WAIC: the largest drops in friedman are `sqrt` (−118.6) and `neg-bin` (−83.5), which fit relatively poorly without the offset.
- **Exceptions.**
  - `identity` / friedman: the mean improves (−16.2%) but WAIC worsens by 158. A likely cause is that a constant-$\sigma$ Gaussian on the density $y/e$ cannot represent the NB variance $\text{Var}(Y/e) = \lambda/e + \lambda^{2}/r$ (with $\lambda = m/e$), which changes with both the rate and the exposure. This is most damaging for friedman's wide range of rates (0.04–56).
  - `np` / imbalanced: $\Delta\text{WAIC} = +13.7$, of the same order as the largest Monte Carlo difference in case (ii) ($\lvert \Delta\text{WAIC} \rvert \le 11.9$). RMSE still improves (−12.3%). More generally, `np` has the smallest WAIC gain in imbalanced and heterogeneous, and the best `offset = FALSE` WAIC in all three scenarios. Its $g$ is a fixed plug-in empirical CDF of $y_{i}/e_{i}$, so exposure noise enters the estimate of $g$ itself, and all zeros (54% here) collapse to $y/e = 0$. Without the offset, this flexible $g$ already absorbs part of the exposure variation.
  - `pois` / friedman / `offset = FALSE` returns `WAIC = NaN`. This is the original code, so it is unrelated to the offset. The friedman counts are heavily overdispersed, but the moment-matched Poisson CDF has a light tail and equals 1 numerically for the largest counts. Those points of $g$ are dropped, and $g$ is extrapolated linearly beyond them, so the largest counts are mapped far into the tail of the latent normal. There, $\Phi(\text{upper}) - \Phi(\text{lower})$ can underflow to 0 in some draws. The resulting $-\infty$ pointwise log-likelihood makes $p_{\text{WAIC}}$, and hence WAIC is NaN. This is a numerical fragility of pois with heavy-tailed counts, and it should be flagged in applications.
- **Ranking of transformations** (`offset = TRUE`).
  - By WAIC, `log`, `box-cox` and `np` are the top three in every scenario, followed by `neg-bin`, `sqrt`, `pois` and `identity`, in that order.
  - By RMSE, the best is `log` (friedman), `neg-bin` (imbalanced) and `np` ≈ `box-cox` (heterogeneous). `identity` is worst everywhere. `pois` is second-worst in heterogeneous but competitive in imbalanced, where `log`, `box-cox`, `pois` and `neg-bin` differ by less than 1% and `np` is second-worst (2.200).

*PPC, proportion of zeros* (case i, friedman; blue line = observed 0.228):

| | `offset = FALSE` | `offset = TRUE` |
|---|---|---|
| `np` | <img src="(i) data_offsets/np/fr-PPC-FALSE.png" width="400"> | <img src="(i) data_offsets/np/fr-PPC-TRUE.png" width="400"> |
| `pois` | <img src="(i) data_offsets/pois/fr-PPC-FALSE.png" width="400"> | <img src="(i) data_offsets/pois/fr-PPC-TRUE.png" width="400"> |
| `box-cox` | <img src="(i) data_offsets/box-cox/fr-PPC-FALSE.png" width="400"> | <img src="(i) data_offsets/box-cox/fr-PPC-TRUE.png" width="400"> |

Zero calibration is driven mainly by $g$. The offset changes it materially only for `np` in friedman.

- **`log`, `box-cox`**: calibrated in all three scenarios, under both fits.
- **`np`**: without the offset, friedman zeros are under-predicted (draws centred near 0.16 vs. observed 0.228). With the offset they are well calibrated. Imbalanced and heterogeneous are calibrated under both fits.
- **`neg-bin`**: calibrated or close. With the offset, the observed proportion sits in the lower tail in friedman and heterogeneous (draws centred near 0.27–0.29).
- **`sqrt`, `pois`**: zeros are over-predicted in friedman and heterogeneous (draws centred near 0.32–0.37 vs. observed 0.23–0.25), under both fits.
- **`identity`**: the worst calibrated. Draws are centred near 0.41–0.43 in friedman and heterogeneous, and near 0.62 vs. 0.54 in imbalanced.
- **imbalanced** (54% zeros): the observed proportion lies within the predictive distribution for every transformation except `identity`.

### Case (ii): `offset = TRUE` with $e \equiv 1$

With $e_{i} \equiv 1$, both fits execute the same computations:

- `np`: the scaling constant is $M = 1$.
- `pois` / `neg-bin`: $\bar{e} = 1$.
- `box-cox`: the same pilot-MCMC initialisation is used.

The two fits therefore differ only in the random-number stream. Case (ii) is a pair of independent MCMC runs of the original `bart_star()` on the same data, which makes it a direct check of its Monte Carlo stability.

**Table 3: WAIC** ($\Delta = \text{TRUE} - \text{FALSE}$).

| $g$ | friedman FALSE | friedman TRUE | friedman $\Delta$ | imbalanced FALSE | imbalanced TRUE | imbalanced $\Delta$ | heterogeneous FALSE | heterogeneous TRUE | heterogeneous $\Delta$ |
|---|---|---|---|---|---|---|---|---|---|
| identity | 1179.7 | 1186.6 | +6.9 | 685.8 | 687.1 | +1.3 | 989.8 | 994.3 | +4.6 |
| log | 1014.7 | 1015.8 | +1.2 | 645.4 | 645.4 | 0.0 | 948.4 | 950.1 | +1.7 |
| sqrt | 1042.1 | 1030.2 | −11.9 | 650.7 | 649.0 | −1.7 | 942.5 | 945.9 | +3.4 |
| box-cox | 1005.7 | 999.7 | −6.1 | 656.4 | 651.6 | −4.7 | 946.0 | 941.4 | −4.6 |
| np | 984.6 | 986.9 | +2.3 | 638.9 | 640.0 | +1.1 | 936.1 | 936.2 | +0.2 |
| pois | 1082.3 | 1071.2 | −11.0 | 652.2 | 656.9 | +4.7 | 951.2 | 950.7 | −0.5 |
| neg-bin | 1004.5 | 1006.5 | +2.0 | 644.0 | 643.7 | −0.3 | 939.3 | 938.9 | −0.4 |
| ispline | 1015.4 | 1009.6 | −5.8 | 660.7 | 650.3 | −10.4 | 946.6 | 945.0 | −1.6 |

**Table 4: Agreement between the two fits' fitted values** (RMSE / correlation).

| $g$ | friedman | imbalanced | heterogeneous |
|---|---|---|---|
| identity | 0.779 / 0.9968 | 0.093 / 0.9984 | 0.144 / 0.9980 |
| log | 0.589 / 0.9978 | 0.077 / 0.9989 | 0.123 / 0.9985 |
| sqrt | 0.445 / 0.9987 | 0.093 / 0.9983 | 0.107 / 0.9987 |
| box-cox | 0.488 / 0.9984 | 0.076 / 0.9989 | 0.129 / 0.9982 |
| np | 0.419 / 0.9989 | 0.074 / 0.9990 | 0.109 / 0.9988 |
| pois | 0.624 / 0.9979 | 0.088 / 0.9984 | 0.106 / 0.9988 |
| neg-bin | 0.504 / 0.9982 | 0.093 / 0.9983 | 0.102 / 0.9989 |
| ispline | 1.475 / 0.9936 | 0.361 / 0.9959 | 0.845 / 0.9986 |

*Fitted vs. truth* (case ii, `np`):

| | `offset = FALSE` | `offset = TRUE` |
|---|---|---|
| friedman | <img src="(ii) data_no/np/fr-fitted-FALSE.png" width="400"> | <img src="(ii) data_no/np/fr-fitted-TRUE.png" width="400"> |
| heterogeneous | <img src="(ii) data_no/np/ht-fitted-FALSE.png" width="400"> | <img src="(ii) data_no/np/ht-fitted-TRUE.png" width="400"> |

**Interpretation**

- **`offset = TRUE` reduces to the original model.** The correlation is $\ge 0.9936$ in all 24 cells and $\lvert \Delta\text{WAIC} \rvert \le 11.9$ ($\le 1.6\%$ of WAIC). The paired panels are visually indistinguishable. The visible patterns (shrinkage of the largest means, wide intervals) appear in both fits, so they are properties of the model and data rather than of the offset.
- **Is the original `bart_star()` stable?**
  - For the **posterior mean fit, yes**. For the fixed and plug-in transformations, the correlation is $\ge 0.9968$. The RMSE between runs is $\le 0.15$ outside friedman and $\le 0.78$ in friedman, where the true means range up to $\approx 192$ (all but one are below 50).
  - **WAIC carries Monte Carlo error of up to $\approx 12$ points**, even with 1000 saved draws. Case (i) differences of that order (e.g. `neg-bin` / imbalanced −12.5, `np` / imbalanced +13.7) should not be over-interpreted.
  - `ispline` is the least stable: its RMSE between runs (0.36–1.48) is 2–6 times the largest among the other transformations, consistent with slower mixing of its sampled transformation. `box-cox`, which also samples its transformation, is as stable as the fixed ones (RMSE 0.08–0.49, $\lvert \Delta\text{WAIC} \rvert$ 4.6–6.1). Thus, we don't recommend using this transformation.
- **Transformation choice for the original model** ($e \equiv 1$, `offset = FALSE` WAIC). `np` is best in all three scenarios (984.6 / 638.9 / 936.1), `neg-bin` is second in all three, and `identity` is worst in all three (1179.7 / 685.8 / 989.8). However, `np`'s margin over `neg-bin` (19.9 / 5.2 / 3.2) exceeds the Monte Carlo error only in friedman.

## Summary

- **The offset works.** With real exposure heterogeneity, `offset = TRUE` reduces RMSE to truth in all 21 cells (by up to 26.9%) and lowers WAIC in 18 of 20 comparable cells. The two WAIC reversals are `identity` / friedman (misspecified mean–variance relation) and `np` / imbalanced (+13.7, of the order of the Monte Carlo error).
- **Transformation choice.** `log`, `box-cox` and `np` perform best by WAIC, and `log` / `box-cox` show the largest and most consistent RMSE gains from the offset. `identity` is consistently worst. `pois` is weak under overdispersion and numerically fragile with heavy-tailed counts (NaN WAIC). `log`, `box-cox` and `np` are therefore the natural default candidates in applications.
- **Zero calibration is driven by $g$.** `log`, `box-cox`, `np` (with the offset) and `neg-bin` reproduce the observed zero proportion. `identity`, `sqrt` and `pois` over-predict zeros in friedman and heterogeneous.
- **No-op check.** With $e \equiv 1$, `offset = TRUE` reproduces `offset = FALSE` up to Monte Carlo error. This also shows that the original `bart_star()` gives stable fitted means across runs.

## Limitations

- There is one data set per scenario and one chain per fit, so no between-replicate uncertainty is reported. WAIC differences below $\approx 12$, and the `ispline` results, need replicates or longer chains to be conclusive.
- For `box-cox` in case (i), the two fits also differ in how the $\sigma$ prior is calibrated (pilot MCMC vs. initial $\lambda_{0}$), so that comparison is not purely the effect of the offset.
- Runtime increased relative to an earlier test (`nburn = 300`, `nsave = 300`), as expected with 2.5 times as many iterations, but it remains acceptable.
