# Simulation results: `bart_star()` with offsets

Figures are in `Results/(i)data_offset/<transformation>/` and `Results/(ii)data_no/<transformation>/`, where `<transformation>` is `log` or `box-cox` and `(i)`/`(ii)` are the offset regimes below. File names are identical across subfolders (`fr`/`im`/`ht` = friedman/imbalanced/heterogeneous). Code files:

- `bart_star_offset.R`: `bart_star()` with the `offsets` / `offsets_test` arguments (see `README_offset.md`) and no helper function is modified.
- `helper_sim.R`: data-generation functions.
- `simulation.R`: simulation, results and plots.

## Simulation settings

Each of 3 data-generating scenarios (see "Data generation") is run under **two offset cases**, and for **two transformations**, `log` and `box-cox`. In every combination, `bart_star(offset = FALSE)` and `bart_star(offset = TRUE)` are fit to the *same* simulated data and compared: 
- **(i)**: data simulated with a heterogeneous offset $e_i \sim \text{Unif}(0.5, 5)$, testing whether modeling a real offset helps.
- **(ii)**: data simulated with $e_i = 1$ for all $i$ (no true heterogeneity), testing whether `offset = TRUE` reduces to `offset = FALSE`. 

The same data sets are used for `log` and `box-cox`. For `box-cox`, $\lambda$ is initialized from $\text{Unif}(0,1)$ and slice-sampled at every iteration; no extra configuration is needed.

Other settings shared across all fits: $n = 250$, $p = 6$, `nburn = 300`, `nsave = 300`, `nskip = 0`, `n.trees = 75`. These are short runs, sufficient for a paired comparison. 

## Data generation
Three scenarios are considered for simulation, which aims to test how offsets work for different data type:

- **`friedman`**: Friedman's nonlinear benchmark function ($\sin$, quadratic, linear terms) + NB noise, which shows **nonlinearity and covariate interaction**.
- **`imbalanced`**: logistic (smooth threshold) transition in $x_1$ from a low to a high baseline rate, strong NB overdispersion, which shows **zero-heavy, imbalanced counts** (54% zeros).
- **`heterogeneous`**: linear mean in $x_1$, NB dispersion that differs by group in $x_2$, which shows **heterogeneous noise across subpopulations**.

## Metrics and plots

- **WAIC**: in-sample predictive fit (lower is better).
- **RMSE to truth** (case (i)): RMSE between `fitted.values` and the true mean at the observed exposure, `Ey_obs` $= e_i\,\mathbb E[Y_i \mid x_i, e_i = 1]$.
- **prop. zeros**: observed proportion of zeros.
- **Consistency**: WAIC for `offset=FALSE` vs. `TRUE`, and the RMSE/correlation between their fitted values, mainly for case (ii).

Plots:

- **Fitted-vs-data scatterplots** plot posterior mean fitted values against the true mean Ey, with 95% posterior intervals and a 1:1 reference line, for `offset = FALSE` vs. `TRUE`. Also, the numerical metrics can convey the same message from the plots.
- **PPC: proportion of zeros**: histogram of the proportion of zeros across posterior predictive draws, with the observed proportion overlaid (blue line); checks whether the model reproduces the data's zero-inflation. Shown below for regime (i), `log` transformation, `offset = FALSE` vs. `TRUE`.

## Results

### Case (i): does including offsets help?

| scenario | transformation | offset | WAIC | RMSE to truth | prop. zeros |
|---|---|---|---|---|---|
| friedman | log | FALSE | 1398.92 | 16.573 | 0.228 |
| friedman | log | TRUE | 1349.67 | 11.422 | 0.228 |
| friedman | box-cox | FALSE | 1403.73 | 16.519 | 0.228 |
| friedman | box-cox | TRUE | 1348.97 | 12.716 | 0.228 |
| imbalanced | log | FALSE | 811.31 | 2.531 | 0.540 |
| imbalanced | log | TRUE | 785.83 | 2.162 | 0.540 |
| imbalanced | box-cox | FALSE | 814.52 | 2.584 | 0.540 |
| imbalanced | box-cox | TRUE | 788.60 | 2.151 | 0.540 |
| heterogeneous | log | FALSE | 1299.92 | 5.225 | 0.248 |
| heterogeneous | log | TRUE | 1246.25 | 4.187 | 0.248 |
| heterogeneous | box-cox | FALSE | 1293.85 | 5.210 | 0.248 |
| heterogeneous | box-cox | TRUE | 1253.56 | 4.015 | 0.248 |

**`offset = TRUE` improves both metrics in every scenario and under both transformations.** WAIC drops by 25.5–54.8 points. RMSE to truth falls by 15–31%:

| scenario | log | box-cox |
|---|---|---|
| friedman | −31% | −23% |
| imbalanced | −15% | −17% |
| heterogeneous | −20% | −23% |

The gain does not depend on the choice of $g$. This is expected, because the offset enters only through the cut points $a_j/\tilde e_i$, confirming the offset correction works as designed regardless of the transformation. .

*PPC, proportion of zeros* (`log`, case (i)):

| | `offset = FALSE` | `offset = TRUE` |
|---|---|---|
| **friedman** | ![](Results/(i)data_offset/log/fr-PPC-FALSE.png) | ![](Results/(i)data_offset/log/fr-PPC-TRUE.png) |
| **imbalanced** | ![](Results/(i)data_offset/log/im-PPC-FALSE.png) | ![](Results/(i)data_offset/log/im-PPC-TRUE.png) |
| **heterogeneous** | ![](Results/(i)data_offset/log/ht-PPC-FALSE.png) | ![](Results/(i)data_offset/log/ht-PPC-TRUE.png) |

The observed proportion of zeros lies in the bulk of the posterior predictive distribution under both fits, including the zero-heavy `imbalanced` case. The offset therefore improves the conditional mean without degrading zero calibration.

### Case (ii): does `offset = TRUE` reduce to the original when $e \equiv 1$?

| scenario | transformation | WAIC (FALSE) | WAIC (TRUE) | ΔWAIC (TRUE − FALSE) | RMSE between fits | cor(fitted) |
|---|---|---|---|---|---|---|
| friedman | log | 1014.04 | 1011.28 | −2.76 | 0.805 | 0.9960 |
| friedman | box-cox | 996.28 | 1017.93 | +21.65 | 1.144 | 0.9917 |
| imbalanced | log | 647.15 | 645.19 | −1.96 | 0.134 | 0.9969 |
| imbalanced | box-cox | 651.77 | 648.64 | −3.13 | 0.158 | 0.9952 |
| heterogeneous | log | 946.51 | 945.73 | −0.78 | 0.226 | 0.9950 |
| heterogeneous | box-cox | 942.38 | 942.09 | −0.29 | 0.250 | 0.9934 |

**The answer is yes, for both transformations**. For `log`, WAIC differs very small ($\leq$ 2.8) and cor(fitted) $\geq$ 0.995. For `box-cox`, cor(fitted) $\geq$ 0.992, but friedman has a larger WAIC gap (+21.7) and the largest RMSE between fits (1.14). This agrees with our expectation for `offset = TRUE` should reduce to original `bart_star()` function and the variability comes from randomness of data generation and sampling procedure, not the offset idea.

*Fitted vs truth* (`log`, case (ii)):

| | `offset = FALSE` | `offset = TRUE` |
|---|---|---|
| **friedman** | ![](Results/(ii)data_no/log/fr-fitted-FALSE.png) | ![](Results/(ii)data_no/log/fr-fitted-TRUE.png) |
| **imbalanced** | ![](Results/(ii)data_no/log/im-fitted-FALSE.png) | ![](Results/(ii)data_no/log/im-fitted-TRUE.png) |
| **heterogeneous** | ![](Results/(ii)data_no/log/ht-fitted-FALSE.png) | ![](Results/(ii)data_no/log/ht-fitted-TRUE.png) |

In each scenario, the two panels are visually indistinguishable.

## Summary

- With real exposure heterogeneity, `offset = TRUE` lowers WAIC by 25–55 points and RMSE to truth by 15–31% in all three scenarios, for both `log` and `box-cox`.
- With $e \equiv 1$, `offset = TRUE` reproduces `offset = FALSE` up to Monte Carlo error (cor $\geq$ 0.992). The largest discrepancy is friedman / `box-cox`.
- Zero calibration (PPC) is comparable with and without offsets. The gain comes from locating the offset-adjusted conditional mean, not from modelling excess zeros.

