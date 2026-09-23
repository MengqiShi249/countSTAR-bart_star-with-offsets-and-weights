# Simulation results: `bart_star(offset)`

Figures below are in `Results/(i)data_offset/<transformation>/` and `Results/(ii)data_no/<transformation>/`, where `<transformation>` is `log` or `box-cox` and `(i)`/`(ii)` match the offset regimes below; file names are identical across the `log`/`box-cox` subfolders (`fr`/`im`/`ht` = friedman/imbalanced/heterogeneous scenarios). Code files are

- `bart_star_offset.R`: offset-adjusted function `bart_star(offset = TRUE/FALSE)` and modified helper functions `set_offset`, `g_cdf_offset`, `expectation_g_offset`.
- `helper_sim.R`: data generation helper functions
- `simulation.R`: main simulation process, results and plots

## Simulation settings

Each of 3 data-generating scenarios (see "Data generation") is run under **two offset cases**, and for **two transformations**, `log` and `box-cox`. In every combination, `bart_star(offset = FALSE)` and `bart_star(offset = TRUE)` are fit to the *same* simulated data and compared:

- **(i)**: data simulated with a heterogeneous offset $e_i \sim \text{Unif}(0.5, 5)$, testing whether modeling a real offset helps.
- **(ii)**: data simulated with $e_i = 1$ for all $i$ (no true heterogeneity), testing whether `offset = TRUE` reduces to `offset = FALSE`.

`box-cox` needs no extra configuration to include: $\lambda$ is initialized $\sim\text{Unif}(0,1)$ and slice-sampled every iteration, exactly like `log` and `identity` are handled as fixed transformations elsewhere in the sampler.

Other settings shared across all fits: $n = 250$, $p = 6$, `nburn` = 300, `nsave` = 300, `nskip` = 0, `n.trees` = 75. These are reduced settings but sufficient to compare `offset = FALSE` vs. `offset = TRUE` on the same data. The posterior intervals might be wider than a full run would give (small `nsave`).

## Data generation

Three scenarios are considered for simulation, which aims to test how offsets work for different data type: 

- **`friedman`**: Friedman's nonlinear benchmark function ($\sin$, quadratic, linear terms) + NB noise, which shows **nonlinearity and covariate interaction**.
- **`imbalanced`**: logistic (smooth threshold) transition in $x_1$ from a low to a high baseline rate, strong NB overdispersion, which shows **zero-heavy, imbalanced counts** (54% zeros).
- **`heterogeneous`**: linear mean in $x_1$, NB dispersion that differs by group in $x_2$, which shows **heterogeneous noise across subpopulations**.

## Metrics and plots

- **WAIC**: in-sample predictive fit (lower is better).
- **RMSE to truth**: RMSE between posterior mean fitted values and the true offset-adjusted mean `Ey_obs` $=e_i\cdot E[Y_i]$ (regime (i) only).
- **prop. zeros**: proportion of zeros in the observed data (context for the PPC plots below).
- **Consistency metrics**: WAIC for `offset=FALSE` vs. `TRUE`, and the RMSE/correlation between their fitted values, mainly for case (ii).

Two plot types are used:
- **Fitted-vs-data scatterplots** plot posterior mean fitted values against the true mean Ey, with 95% posterior intervals and a 1:1 reference line, for `offset = FALSE` vs. `TRUE`. Also, the numerical metrics can convey the same message from the plots.
- **PPC: proportion of zeros**: histogram of the proportion of zeros across posterior predictive draws, with the observed proportion overlaid (blue line); checks whether the model reproduces the data's zero-inflation. Shown below for regime (i), `log` transformation, `offset = FALSE` vs. `TRUE`.

## Results and interpretation

### Case (i): does including offsets help?

| scenario | transformation | offset | WAIC | RMSE to truth | prop. zeros |
|---|---|---|---|---|---|
| friedman | log | FALSE | 1398.92 | 16.573 | 0.228 |
| friedman | log | TRUE | 1342.89 | 10.798 | 0.228 |
| friedman | box-cox | FALSE | 1403.73 | 16.519 | 0.228 |
| friedman | box-cox | TRUE | 1347.68 | 11.786 | 0.228 |
| imbalanced | log | FALSE | 811.31 | 2.531 | 0.540 |
| imbalanced | log | TRUE | 790.21 | 2.038 | 0.540 |
| imbalanced | box-cox | FALSE | 814.52 | 2.584 | 0.540 |
| imbalanced | box-cox | TRUE | 788.12 | 2.108 | 0.540 |
| heterogeneous | log | FALSE | 1299.92 | 5.225 | 0.248 |
| heterogeneous | log | TRUE | 1249.46 | 4.147 | 0.248 |
| heterogeneous | box-cox | FALSE | 1293.85 | 5.210 | 0.248 |
| heterogeneous | box-cox | TRUE | 1249.15 | 4.057 | 0.248 |

**`offset = TRUE` wins on both metrics, in every scenario, under both transformations.** RMSE-to-truth is clearly lower, and WAIC drops by 21–56 points in every row. The key point: the offset benefit holds for both `g` is log transformation or estimated via Box-Cox, confirming the offset correction (entering through cut points $a_j(e_i)=j/e_i$, independent of which `g` is used) works as designed regardless of the transformation. The fitted value points for `bart_star(offset = TRUE)` are more closer to data value (red line in the fitted value plots, not shown in this file)

*PPC zero-proportion histograms* (`log`, regime (i)):

| | `offset = FALSE` | `offset = TRUE` |
|---|---|---|
| **friedman** | ![](Results/(i)data_offset/log/fr-PPC-FALSE.png) | ![](Results/(i)data_offset/log/fr-PPC-TRUE.png) |
| **imbalanced** | ![](Results/(i)data_offset/log/im-PPC-FALSE.png) | ![](Results/(i)data_offset/log/im-PPC-TRUE.png) |
| **heterogeneous** | ![](Results/(i)data_offset/log/ht-PPC-FALSE.png) | ![](Results/(i)data_offset/log/ht-PPC-TRUE.png) |

In every scenario, the observed proportion of zeros (blue line) falls within the bulk of the posterior predictive distribution under **both** models, including the zero-heavy `imbalanced` case (54% zeros). So the offset correction improves the conditional mean (table above) without hurting zero calibration. The benefit of `offset = TRUE` is specific to correctly locating the offset-adjusted mean, not to modeling excess zeros.

### Case (ii): does `offset = TRUE` reduce to original when $e=1$?

| scenario | transformation | WAIC (FALSE) | WAIC (TRUE) | RMSE between fits | cor(fitted) |
|---|---|---|---|---|---|
| friedman | log | 1014.04 | 1011.28 | 0.805 | 0.9960 |
| friedman | box-cox | 996.28 | 1013.06 | 1.115 | 0.9930 |
| imbalanced | log | 647.15 | 645.19 | 0.134 | 0.9969 |
| imbalanced | box-cox | 651.77 | 648.09 | 0.159 | 0.9951 |
| heterogeneous | log | 946.51 | 945.73 | 0.226 | 0.9950 |
| heterogeneous | box-cox | 942.38 | 939.12 | 0.212 | 0.9952 |

**The answer is yes, for both transformations**, though `box-cox` is a little noisier than `log`. WAIC differs very small for `log` in every scenario, and fitted-value correlation is $\geq$ 0.995. For `box-cox`, correlation stays $\geq$ 0.993, but the friedman WAIC gap is larger and its `RMSE_between` is the largest in either table. This agrees with our expectation for `offset = TRUE` should reduce to original `bart_star()` function and the variability comes from randomness of data generation itself, not the offset idea.

## Summary

- Modeling a real offset (`offset = TRUE`) reduces both WAIC and RMSE-to-truth substantially across all three scenarios (nonlinear mean, zero-heavy counts, heterogeneous dispersion), and this holds for both `log` and `box-cox`, confirming the benefit comes from the offset correction itself, not from a particular transformation.
- When no real offset heterogeneity exists ($e\equiv1$), `offset = TRUE` closely reproduces `offset = FALSE` for both transformations.
- Zero-inflation calibration (PPC) is comparable with and without the offset correction, indicating the gain from `offset = TRUE` is specifically in locating the offset-adjusted conditional mean, not in modeling excess zeros.
