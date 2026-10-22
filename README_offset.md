# countSTAR: `bart_star()` with offsets

This repository extends `bart_star()` from [countSTAR](https://github.com/bking124/countSTAR) to support **model offsets** (exposures) for count-valued BART regression. **Regression weights** are documented separately in `README_weight.md`. The motivating application is national density and distribution modelling for Canadian birds, but the extension applies to any STAR model with a known, mechanistic exposure term. The new version `bart_star()` with offsets does not require any modification to `countSTAR` helper functions.

## Motivation

`bart_star()` fits a **S**imultaneous **T**ransformation **A**nd **R**ounding (STAR) model with a BART latent mean via `dbarts`. It captures nonlinearities and interactions (e.g. thresholds in temperature, elevation, distance to water) that GLMs/GAMs miss, and returns calibrated posterior uncertainty without the bootstrapping that boosted regression trees require.

**Offsets.** Each point count has a known offset $e_i > 0$ (e.g. detection proportion or survey effort). The usual log-link model is $\text{count}_i \sim \text{Poisson}\big(e_i \exp(\eta_i)\big)$, $\eta_i = \mu(x_i)$, so $\mu(x_i)$ is a log density rather than a log expected count. Because $e_i$ acts multiplicatively and mechanistically, it enters as an offset, not as a covariate. Offsets turn a model for counts into a model for absolute density, which allows population-size estimation. In STAR, $g$ need not be $\log$, so the offset cannot be absorbed into the linear predictor. Replacing $y$ by $y/e$ as the response is also incorrect, even for $g = \log$ (see below).

## Background: the STAR model

$$y_i = h(y_i^\ast) = \lfloor y_i^\ast\rfloor, \qquad z_i = g(y_i^\ast), \qquad z_i = \mu(x_i) + \varepsilon_i, \quad \varepsilon_i \sim N(0, \sigma^2)$$

Here $h$ is the floor (`round_floor()`: negatives mapped to 0, capped at `y_max`), $g$ is increasing, and $\mu(\cdot)$ is a BART sum of trees. Since $g$ is monotone,

$$\Pr(y_i = j \mid x_i) = \Pr\big(g(a_j) \le z_i < g(a_{j+1})\big), \qquad a_j = j, \quad a_0 = -\infty, \quad a_{y_{\max}+1} = \infty$$

with $a_j$ given by `a_j()`. The Gibbs sampler:

1. **Sample $z$:** draw $z_i \mid y_i$ from $N(\mu(x_i), \sigma^2)$ truncated to $[g(a_{y_i}), g(a_{y_i+1}))$.
2. **Sample $(\mu, \sigma)$:** update the BART fit (`dbarts`) with response $z$.
3. **(Optional) update the transformation**, if $g$ is being learned rather than fixed.
4. **Predict:** draw $\tilde z \sim N(\mu, \sigma^2)$ and use $g^{-1}$ and rounding to generate posterior predictive draws $\tilde y = h(g^{-1}(\tilde z))$.

## Model with offsets

The rounding is unchanged. The transformation acts on the latent count per unit exposure:

$$y_i = \lfloor y_i^\ast\rfloor, \qquad z_i = g\left(\frac{y_i^\ast}{e_i}\right) = \mu(x_i) + \varepsilon_i, \quad \varepsilon_i \sim N(0, \sigma^2)$$

Here $\mu$ and $\sigma$ are on the scale of the transformed density $g(y^\ast/e)$. No rescaling of the offsets or parameters is involved, so the priors are unchanged and the likelihood changes only through the cut points. Also, the regression is only for latent $z$ and covariate $x$, and $y/e$ is never used as response.

### Likelihood

For $e_i > 0$ and monotone $g$,

$$y_i = j \iff a_j \le y_i^\ast < a_{j+1} \iff g\left(\frac{a_j}{e_i}\right) \le z_i < g\left(\frac{a_{j+1}}{e_i}\right)$$

$$\ell(\mu, \sigma, g) = \sum_{i=1}^n \log\left[\Phi\left(\frac{g(a_{y_i+1}/e_i) - \mu_i}{\sigma}\right) - \Phi\left(\frac{g(a_{y_i}/e_i) - \mu_i}{\sigma}\right)\right]$$

with $\pm\infty/e_i = \pm\infty$. This is what `logLikeRcpp` / `logLikePointRcpp` compute on the training data.

**Why not $y/e$.** For $j \ge 1$, the interval for $y_i = j$ on the density scale is $[j/e_i, (j+1)/e_i)$, of width $1/e_i$. Using $y/e$ as the response gives $[j/e_i, j/e_i + 1)$, which is wrong whenever $e_i \neq 1$.

**Special case $g = \log$.** $\log y_i^\ast = \mu(x_i) + \log e_i + \varepsilon_i$ is the familiar additive log-offset.

### Gibbs sampler with offsets

Only the cut points change, from $a_j$ to $a_j/e_i$:

| Step | With offsets |
|---|---|
| Block 1: sample $z$ | $N(\mu_i, \sigma^2)$ truncated to $[g(a_{y_i}/e_i),\thinspace g(a_{y_i+1}/e_i))$ |
| Block 2: sample $(\mu, \sigma)$ | unchanged |
| Block 3: Box–Cox $\lambda$ | slice sampler on $\ell(\lambda) + \log \pi(\lambda)$ with $g_\lambda(a_{y_i}/e_i)$, $g_\lambda(a_{y_i+1}/e_i)$; $\pi(\lambda) = N(1/2, 1)$ truncated to $[0, 3]$ |
| Pointwise log-likelihood / WAIC | offset-adjusted bounds |
| Test log-predictive score | $g(a_{y_j}/e_j^{\text{test}})$, $g(a_{y_j+1}/e_j^{\text{test}})$ |

The scaled cut points are computed once, before the MCMC loop. For Box–Cox, only $g_\lambda$ is re-applied at each iteration.

### Prediction

Predictions take separate test offsets $e^{\text{test}}$ (`offsets_test`, a scalar or one value per row of `X_test`). At each saved draw, for test point $j$,

$$\tilde z_j \sim N\big(\mu(x_j^{\text{test}}), \sigma^2\big), \qquad \tilde y_j = \big\lfloor g^{-1}(\tilde z_j)\cdot e_j^{\text{test}} \big\rfloor.$$

Here $g^{-1}(\tilde z_j)$ is a draw of the latent density $y^\ast/e$ at $x_j^{\text{test}}$. Multiplying by $e_j^{\text{test}}$ gives the latent count at the test exposure, and the floor rounds it exactly as in the model $y = \lfloor y^\ast\rfloor$. As everywhere in the code, $\lfloor\cdot\rfloor$ is `round_floor()`: the floor, with negatives mapped to 0 and values capped at `y_max`.

$g^{-1}$ is exact for the Box–Cox family and the grid approximation `g_inv_approx()` for `np`, `pois` and `neg-bin`. Setting $e^{\text{test}}$ to a survey's exposure gives predicted counts for that survey. With $e^{\text{test}} = 1$, $\tilde y$ is the **rounded count at unit exposure**. This is not the continuous density $y^\ast/e$. See [Interpreting outputs](#interpreting-outputs).

### Conditional expectation

For $1 \le j \le y_{\max}$, $y_i \ge j \iff y_i^\ast \ge a_j \iff z_i \ge g(a_j/e_i)$ (the cap at $y_{\max}$ and the mapping of negatives to 0 do not affect these events), and $\Pr(y_i \ge j) = 0$ for $j > y_{\max}$. By the tail-sum formula for a non-negative integer variable,

$$\mathbb{E}(y_i \mid \mu_i, \sigma, e_i) = \sum_{j=1}^{y_{\max}} \Pr(y_i \ge j) = \sum_{j=1}^{y_{\max}} \left[1 - \Phi\left(\frac{g(a_j/e_i) - \mu_i}{\sigma}\right)\right].$$

Only the counts between the extreme quantiles of $y_i$ need to be summed, since $\Pr(y_i \ge j) \approx 1$ below the lower one and $\approx 0$ above the upper one:

$$\mathbb{E}(y_i \mid \mu_i, \sigma, e_i) \approx L_i + \sum_{j=L_i+1}^{J_i} \left[1 - \Phi\left(\frac{g(a_j/e_i) - \mu_i}{\sigma}\right)\right],$$

```math
J_i = \min\{h(e_i\, g^{-1}(q_{0.9999})),\ \lceil 2 \max_k (y_k/e_k)\, e_i \rceil\}, \qquad L_i = \min\{h(e_i\, g^{-1}(q_{0.0001})),\ J_i\}
```

where $q_p$ is the $p$ quantile of $N(\mu_i, \sigma^2)$. Since $h(e_i\thinspace g^{-1}(\cdot))$ is non-decreasing, $h(e_i\thinspace g^{-1}(q_p))$ is the $p$ quantile of $y_i \mid \mu_i, \sigma, e_i$. The cap is stated on the density scale and reduces to the original $2\max(y)$ for unit offsets.

- **Error.** $`\Pr(y_i \ge j) \ge 0.9999`$ for $`j \le L_i`$, so the lower truncation error is at most $`10^{-4} L_i`$. The upper truncation omits $`\sum_{j > J_i} \Pr(y_i \ge j) = \mathbb{E}\max(y_i - J_i, 0)`$. This is zero when $`J_i = y_{\max}`$ and negligible for light-tailed $g$. For heavy-tailed $g$ (`log`, `box-cox` near $`\lambda = 0`$) with $`y_{\max} = \infty`$, it can be a few percent of the mean, from both the 0.9999 quantile and the cap (see `Test.md`). For `np`, `pois` and `neg-bin`, $J_i$ is also limited by the top of the $g^{-1}$ grid times $e_i$ (see Initialization), which never exceeds the cap. The sum is exact when $`L_i = 0`$ and $`J_i = y_{\max}`$.
- **Relation to the original.** The original computes $`\sum_{j=0}^{J_i-1} j \Pr(y_i = j)`$ with `expectation_gRcpp()`. It drops every value at or above $J_i$, so its error is $`J_i \Pr(y_i \ge J_i) + \mathbb{E}\max(y_i - J_i, 0)`$. The tail sum equals $`\mathbb{E}\min(y_i, J_i)`$: values above $J_i$ count as $J_i$ instead of 0, so its error is only $`\mathbb{E}\max(y_i - J_i, 0)`$. That error is zero at a finite, binding $`y_{\max}`$, where the original loses the mass at the bound. Starting at $L_i$ also reduces the cost from $J_i$ to $`J_i - L_i`$ terms.
- **Offsets.** Each observation is evaluated on its own cut points $a_j/e_i$, so all observations (training at `offsets`, test at `offsets_test`) are handled in one vectorized call, with no common grid and no grouping by offset. The computation is done by the local function `expect_y()`.

## Transformation $g$ on the density scale

The offsets enter the likelihood only through the cut points $a_j/e_i$, and $\mu$ and $\sigma$ are never rescaled. The transformations differ only in how $g$ is defined on the density scale $y^\ast/e$.

### Case 1: fixed $g$ (`identity`, `log`, `sqrt`)

$g = g_\lambda$ ($\lambda = 1, 0, 1/2$) is applied directly to $y^\ast/e$, so $\mu$ and $\sigma$ are on the scale of $g_\lambda(y^\ast/e)$ in the units of $e$, and the priors are those of the original `bart_star()` on that scale. Changing the units of $e$ changes $\mu$ as it should. For example, for $g = \log$, $e \to ce$ shifts $\mu$ by $-\log c$, with identical count-scale predictions.

### Case 2: $g$ estimated once (`np`, `pois`, `neg-bin`)

`g_cdf(v, distribution)` (unmodified) estimates a transformation from a vector of non-negative values $v$ (the counts $y$ in the original `bart_star()`):

```math
g_v(t) = \Phi^{-1}\big(\hat F_v(t - 1)\big), \qquad t \in \{\text{unique positive } v_i\}
```

It drops knots where this is non-finite and interpolates with `splinefun(method = "monoH.FC")`, which extrapolates linearly. The estimated CDFs are

$$\hat F^{\text{np}}_v(t) = \tfrac{n}{n+1}\widehat{\text{ecdf}}_v(t), \qquad \hat F^{\text{pois}}_v(t) = F_{\text{Pois}}(t; \hat\mu_v), \qquad \hat F^{\text{nb}}_v(t) = F_{\text{NB}}\Big(t;\ \hat r = \tfrac{\hat\mu_v^2}{\hat\sigma_v^2 - \hat\mu_v},\ \hat p = \tfrac{\hat\mu_v}{\hat\sigma_v^2}\Big),$$

where $\hat\mu_v$ and $\hat\sigma_v$ are the sample mean and SD of $v$. For `neg-bin` with underdispersed data ($\hat\mu_v \ge \hat\sigma_v^2$), the code warns and uses $\hat\sigma_v = 1.1\sqrt{\hat\mu_v}$. For integer counts, $\hat F_v(t-1)$ is the estimated probability of a value strictly below $t$.

**How the offset enters the CDF.** As in [Kowal & Wu (2023)](https://onlinelibrary.wiley.com/doi/10.1111/biom.13617), $g$ is chosen from a marginal working model $`z \sim F_Z = \Phi`$. This only selects $g$; the likelihood is valid for any increasing $g$. Without offsets, $`y \le t \iff y^\ast < t+1 \iff z < g(t+1)`$. With offsets, $`z = g(D)`$ with $`D = y^\ast/e`$, so $`y \le t \iff D < (t+1)/e`$. Hence

```math
F_Y(t) = F_Z\big(g(t+1)\big) \;\Rightarrow\; g(t) = F_Z^{-1}\circ F_Y(t-1), \qquad
F_{Y\mid e}(t) = F_Z\big(g(\tfrac{t+1}{e})\big) \;\Rightarrow\; g(\tfrac{t}{e}) = F_Z^{-1}\circ F_{Y\mid e}(t-1).
```

The offset enters only $F_Y$ (the count CDF at exposure $e$) and the evaluation point $t/e$, which is the cut point $`a_t/e_i`$; $F_Z$ is unchanged. Since $`F_{Y\mid e}(t-1) = \Pr(D < t/e) = F_D(t/e)`$, this defines one function $`g = F_Z^{-1}\circ F_D`$ at every exposure. The target is the CDF of the density $`y^\ast/e`$, not of the pooled counts. For `np`, observation $i$ estimates $`F_D(s) = \Pr(y_i < s\,e_i)`$ by $`\mathbf 1(y_i < s\,e_i) = \mathbf 1(y_i/e_i < s)`$. The average, $`U(s)`$ below, satisfies $`F_D(s) \le \mathbb E[U(s)\mid e] \le \frac{1}{n}\sum_i F_D(s + 1/e_i)`$ and is exact for constant offsets. `pois` and `neg-bin` cannot satisfy the relation at every exposure (e.g. $`\text{Pois}(e\lambda)`$ at $e = 1, 2$ needs $`e^{-2\lambda}(1+2\lambda) = e^{-\lambda}`$, which fails except at $`\lambda \approx 1.26`$), so they use a reference exposure.

**`np`: empirical CDF of the observed densities.** `g_cdf()` assumes integer input: its knot at $t$ is $`\Phi^{-1}(\hat F(t-1))`$, and "$`\le t-1`$" means "$`< t`$" only because counts are at least 1 apart. The densities $`d_i = y_i/e_i`$ (rounded to 12 significant digits) can be closer than 1, so `g_cdf(d)` would omit densities in $`(d_k - 1, d_k)`$ and drop every knot with $`d_k < 1`$. The code therefore stretches the densities: with $\delta$ the smallest gap between distinct $d_i$, $`M = 1`$ if $`\delta \ge 1`$ and $`M = 2/\delta`$ otherwise, and

```math
g(s) = g_M(M s), \qquad g_M = \texttt{g\_cdf}(M d, \text{"np"})
```

Since $`M\delta \ge 1`$, $`M d_k - 1 \in [M d_{k-1}, M d_k)`$, so at every distinct $d_k$

```math
g(d_k) = \Phi^{-1}\Big(\tfrac{n}{n+1} U(d_k)\Big), \qquad U(s) = \tfrac{1}{n}\textstyle\sum_i \mathbf 1(d_i < s)
```

the plug-in estimate derived above. Rescaling by $M > 0$ preserves the ECDF and the monotone interpolation, so $M$ only restores the meaning of "$`t-1`$". When $`M \ne 1`$, $`M\delta = 2`$ leaves a margin against floating-point error. With $`e \equiv 1`$, $`M = 1`$ and $g$ is the original `g_cdf(y, "np")`.

**`pois`, `neg-bin`: count CDF at a reference exposure.** `g_cdf(y, transformation)` returns a count-scale transformation $`g_y(t) = \Phi^{-1}(F_{\hat\theta}(t-1))`$ with knots at integer counts, but the sampler evaluates $g$ at densities $`y^\ast/e`$ and cut points $`a_j/e_i`$. Using $g_y$ directly would fix the reference exposure at $`e = 1`$ and make $g$ depend on the units of $e$. A Poisson or NB CDF cannot be fitted to the non-integer $`y/e`$. Poisson and NB families with mean proportional to $e$ do not give one $g$ at every exposure (see the derivation above). The count CDF is therefore attached to one reference exposure, $`F_{Y \mid \bar e} = F_{\hat\theta}`$, which gives

$$g(s) = g_y(\bar e\, s), \qquad \bar e = \exp\Big(\tfrac{1}{n}\textstyle\sum_i \log e_i\Big),$$

so observation $i$ enters as $`g_y(\bar e\, y_i^\ast/e_i)`$, its count rescaled to exposure $`\bar e`$. Unlike $M$ in `np`, $`\bar e`$ is a modelling choice: it fixes where the count CDF sits on the density scale. The geometric mean (i) is exact for constant offsets ($`e \equiv c \Rightarrow \bar e = c`$; $`e \equiv 1`$ gives the original $g_y$), (ii) does not depend on units ($`e \to ce \Rightarrow \bar e \to c\bar e`$, so $z_i$ is unchanged), and (iii) is centred where the offset acts: $`\sum_i \log(\bar e/e_i) = 0`$, it minimizes $`\sum_i \log^2(\bar e/e_i)`$, and it treats halving and doubling symmetrically. Since $`\hat\theta`$ is matched to the pooled counts, whose mean corresponds roughly to the arithmetic-mean exposure, the mapping is exact only for constant offsets. As with ignoring covariates in the original, this affects only the choice of $g$, not the validity of the likelihood. `np` needs no $`\bar e`$ because it estimates $`F_D`$ directly.

### Case 3: $g$ sampled every iteration (`box-cox`)

$\lambda$ is updated by the slice sampler with target

$$\pi(\lambda \mid \mu, \sigma, y, e) \propto \exp\lbrace \ell(\lambda)\rbrace \thinspace \pi(\lambda), \qquad \ell(\lambda) = \sum_i \log\left[\Phi\left(\frac{g_\lambda(a_{y_i+1}/e_i) - \mu_i}{\sigma}\right) - \Phi\left(\frac{g_\lambda(a_{y_i}/e_i) - \mu_i}{\sigma}\right)\right]$$

The scaled cut points are stored once and only $g_\lambda$ is re-applied after each update.

### `ispline` is not supported with offsets

The I-spline sampler matches cut points exactly (`match(a_y, t_g)`) on an integer grid, which is incompatible with continuous, observation-specific cut points $a_j/e_i$. The code stops with an error if any training or test offset differs from 1. With unit offsets, the call is passed to the original I-spline sampler.

## Initialization

Block 1 samples the truncated normal by inverse CDF. Let $G_{l,i}$ and $G_{u,i}$ be the $N(\mu_i, \sigma^2)$ CDF at the interval endpoints, i.e. the interval on the probability scale:

$$G_{l,i} = \Phi\left(\frac{z_{\text{lower},i} - \mu_i}{\sigma}\right), \qquad G_{u,i} = \Phi\left(\frac{z_{\text{upper},i} - \mu_i}{\sigma}\right), \qquad z_i = \mu_i + \sigma\thinspace \Phi^{-1}\big(G_{l,i} + u_i (G_{u,i} - G_{l,i})\big),$$

with $u_i \sim \text{Unif}(0, 1)$ and the result clamped to $`[z_{\text{lower},i}, z_{\text{upper},i}]`$ (`G_lower_i`, `G_upper_i` in `rtruncnormRcpp()`). If $\mu_i$ lies many $\sigma$ from its interval, $G_{l,i} = G_{u,i} \in \lbrace 0, 1\rbrace$ numerically and the draw collapses to an endpoint, which may be infinite (e.g. $-\infty$ for $y_i = 0$). The starting values therefore use the original rules, evaluated on the density scale $y/e$ where the intervals live:

| Quantity | Rule |
|---|---|
| Latent $z^{(0)}$ | $g((y_i + \lvert\xi_i\rvert)/e_i)$, $\xi_i \sim N(0, 1)$ |
| `sigest` (not `box-cox`) | $\mathrm{sd}\lbrace g((y_i + 1)/e_i)\rbrace$ |
| `sigest` (`box-cox`) | original pilot MCMC if all $e_i = 1$; otherwise $\mathrm{sd}\lbrace g_{\lambda^{(0)}}((y_i + 1)/e_i)\rbrace$, $\lambda^{(0)} \sim \text{Unif}(0, 1)$ |
| Grid for $g^{-1}$ (`np`, `pois`, `neg-bin`) | 250 equally spaced points on $[0, \min\lbrace 2\max_i(y_i/e_i),\ y_{\max}/\min_i e_i\rbrace]$ plus 250 quantiles of unique $(y_i + 1)/e_i$, $y_i < y_{\max}$ |

The `sigest` rules apply only when `sigest = NULL` (the default). `sigest` matters most: it is computed once but fixes the $\sigma$ prior (via `sigquant`) for the whole run. The `box-cox` pilot MCMC fits a mean-only model to the raw counts and cannot take offsets, so with offsets it is replaced by the same rule as the other transformations. Since the half-normal start is the original rule applied on the scale $y/e$, it sits in the same relative position within each interval as in the original sampler. With $e \equiv 1$, every rule reduces to the original code.

## Implementation

All changes are in **`bart_star_offset.R`**. No `countSTAR` helper is added or modified: `a_j`, `g_bc`, `g_inv_bc`, `g_cdf`, `g_inv_approx`, `round_floor`, `rtruncnormRcpp`, `logLikeRcpp`, `logLikePointRcpp`, `uni.slice`, `genMCMC_star` and `bart_star_ispline` are called unchanged. `expectation_gRcpp` is no longer called; the conditional expectation is computed by the local function `expect_y()` (see [Conditional expectation](#conditional-expectation)).

### Interface

```r
bart_star(y, X, X_test = NULL, y_test = NULL,
          offsets      = 1,                     # training offsets e_i > 0 (scalar or length(y))
          offsets_test = 1,                     # test offsets (scalar or nrow(X_test))
          transformation = 'np', ...)           # all other arguments as before
```

- A scalar is recycled, otherwise one value per observation.
- Values must be finite and strictly positive, with lengths `length(y)` and `nrow(X_test)`. Otherwise an error is raised. `offsets_test` is validated only when `X_test` is supplied and is otherwise ignored, apart from the `ispline` check.
- The returned list has the same elements as the original. `post.pred`, `post.log.like.point`, `fitted.values` and `post.fitted.values` are at `offsets`. `post.pred.test`, `post.fitted.values.test` and `post.log.pred.test` are at `offsets_test`.

### Interpreting outputs

- **Latent scale.** `post.mu.test` and `post.sigma` are on the scale of $g(y^\ast/e)$. For `identity`, `log`, `sqrt` and `box-cox`, this is $g_\lambda$ of the density in the units of $e$. For `np`, it is the transformation estimated from the empirical CDF of $y/e$. For `pois` and `neg-bin`, it is $g(t) = g_y(\bar e\thinspace t)$.
- **Continuous density.** For $g = \log$, $y^\ast/e \mid x, \mu, \sigma$ is log-normal, so $\mathbb E(y^\ast/e \mid x, \mu, \sigma) = \exp(\mu(x) + \sigma^2/2)$.
- **Counts.** Count-scale outputs are at the supplied exposures (see Prediction).

### Code changes (marked `# OFFSETS:`)

| Location | Change |
|---|---|
| Argument checks | validate and recycle `offsets` / `offsets_test` |
| `ispline` dispatch | error if any `offsets` or `offsets_test` $\neq 1$; otherwise drop the offset arguments and call the original sampler |
| `g` for `np` | `d = signif(y/offsets, 12)`; `M = 1` if the smallest gap of `unique(d)` is $\ge 1$, else `2/gap`; `g_M = g_cdf(M*d, 'np')`; `g = function(t) g_M(M*t)` |
| `g` for `pois`, `neg-bin` | `e_bar = exp(mean(log(offsets)))`; `g_y = g_cdf(y, transformation)`; `g = function(t) g_y(e_bar*t)` |
| Grid for `g_inv_approx()` | built from `y/offsets` and `(y + 1)/offsets` |
| Initial `z_star` | `g((y + abs(rnorm(n)))/offsets)` |
| `sigest` | `sd(g((y + 1)/offsets))`; pilot MCMC for `box-cox` only if `all(offsets == 1)` |
| Cut points | `a_y = a_j(y, y_max)/offsets`, `a_yp1 = a_j(y + 1, y_max)/offsets` (Blocks 1, 3, pointwise log-likelihood) |
| Predictive draws | `round_floor(g_inv(z_tilde) * offsets, y_max)`; likewise with `offsets_test` |
| Conditional expectation | `expect_y(mu, e)`: tail sum `lo + sum_{j = lo+1}^{hi} Pr(y >= j)` with `Pr(y >= j) = pnorm(g(j/e), mu, sigma, lower.tail = FALSE)`; `lo`, `hi` the 0.0001 and 0.9999 quantiles of $y_i$ at exposure $e_i$, `hi` capped at `ceiling(2*max(y/offsets)*e)`; one vectorized call for all observations (training at `offsets`, test at `offsets_test`), replacing `expectation_gRcpp()` |
| Test log-predictive score | `g(a_j(y_test, y_max)/offsets_test)`, `g(a_j(y_test + 1, y_max)/offsets_test)`; the original omits `y_max` here, so a test count at a finite `y_max` was scored on $`[y_{\max}, y_{\max}+1)`$ instead of $`[y_{\max}, \infty)`$ |

### Example

```r
library(countSTAR)
source("bart_star_offset.R")

fit = bart_star(y = y, X = X, offsets = e,
                X_test = X_new, offsets_test = 1,
                transformation = 'log', save_y_hat = TRUE)

fit$fitted.values                        # E(y_i | x_i, e_i): expected counts at observed exposures
colMeans(fit$post.fitted.values.test)    # E(y | x_new, e = 1): expected count at unit exposure
```

### Notes

- **Checks.** With `offsets = 1`, all outputs other than the conditional expectations (`fitted.values`, `post.fitted.values`, `post.fitted.values.test`) are `identical()` to the original `bart_star()` (same seed) for all seven transformations. `expect_y()` draws no random numbers, so the random stream is unchanged. The one other exception is `post.log.pred.test` when `y_max` is finite and a test count equals `y_max`, because the test score now uses the same `y_max` intervals as the training likelihood. The conditional expectations differ from the original only through the summation described in [Conditional expectation](#conditional-expectation) (`ispline` with unit offsets calls the original sampler and is unchanged). With offsets, `np`, `pois` and `neg-bin` results are unchanged by $e \to ce$ by construction (up to the $g^{-1}$ grid), and `log` results are identical except that $\mu$ shifts by $-\log c$.
- **Cost.** Sampling, prediction and the log-likelihood add no per-iteration cost. The conditional expectation costs $\sum_i (J_i - L_i)$ normal CDF evaluations per saved draw, independent of the number of distinct offsets. Runtime in the simulation study (`README.md`) was acceptable.
- **Not supported:** `ispline` with non-unit offsets; regression weights (`README_weight.md`).
