library(countSTAR)
library(dbarts)
env = new.env(parent = asNamespace("countSTAR"))
sys.source("/Users/shimengqi/Desktop/Courses/0-research/BART-STAR/offset/bart_star_offset.R", envir = env)
a_j = countSTAR:::a_j
round_floor = countSTAR:::round_floor
g_bc = countSTAR:::g_bc; g_inv_bc = countSTAR:::g_inv_bc
logLikePointRcpp = countSTAR:::logLikePointRcpp
set.seed(123)

# 1. g_cdf(). Data: y_i ~ Pois(3 e_i) or NB(mean 3 e_i, size 2).
# pois/neg-bin: with z ~ N(0, 1), P(y = j) at unit exposure is Phi(g(j + 1)) - Phi(g(j)).
n = 20000; e = runif(n, 0.5, 5); j = 0:8
g_pois = env$g_cdf(rpois(n, 3*e), 'pois', e)
g_nb = env$g_cdf(rnbinom(n, size = 2, mu = 3*e), 'neg-bin', e)
print(data.frame(j, pois_code = diff(c(0, pnorm(g_pois(j + 1)))), pois_true = dpois(j, 3),
                 nb_code = diff(c(0, pnorm(g_nb(j + 1)))), nb_true = dnbinom(j, size = 2, mu = 3)), digits = 3)

# np: Phi(g(t)) against P(y/e < t) from a large new sample.
n = 5000; e = runif(n, 0.5, 5); s = c(0.5, 1, 2, 3, 5, 8)
g_np = env$g_cdf(rnbinom(n, size = 2, mu = 3*e), 'np', e)
e = runif(1e6, 0.5, 5); rate = rnbinom(1e6, size = 2, mu = 3*e)/e
print(data.frame(t = s, np_code = pnorm(g_np(s)), np_true = sapply(s, function(u) mean(rate < u))), digits = 3)

# 2. Cut points, with g = log and y_max = 10: P(y = j) from [g(a_j/e), g(a_{j+1}/e)) against
# the frequencies of y = min(floor(e g^{-1}(z)), y_max), z ~ N(mu, sigma^2).
y_max = 10; mu = 0.2; sigma = 0.8; e = 8; j = 0:y_max
lo = g_bc(a_j(j, y_max)/e, 0); up = g_bc(a_j(j + 1, y_max)/e, 0)
y_sim = round_floor(exp(rnorm(1e5, mu, sigma))*e, y_max)
print(data.frame(j, P_code = exp(logLikePointRcpp(lo, up, rep(mu, y_max + 1), rep(sigma, y_max + 1))),
                 P_sim = tabulate(y_sim + 1, y_max + 1)/1e5), digits = 3)

# 3. Expectation and test log-score from bart_star() (y_max = 10), for 'log' and 'box-cox'.
# For each test point, simulate y from the first saved draw (mu, sigma, lambda): the mean of y should
# match post.fitted.values.test, and the frequency of y_test should match exp(post.log.pred.test).
n = 300; X = matrix(runif(n*5), n); X_test = matrix(runif(500), 100)
e = runif(n, 2, 20); e_test = runif(100, 2, 20)
y = pmin(rpois(n, e*exp(X[,1] - 1.5)), y_max); y_test = pmin(rpois(100, e_test*exp(X_test[,1] - 1.5)), y_max)
i = c(which(y_test < y_max)[1:3], which(y_test == y_max)[1:3])

for(tr in c('log', 'box-cox')){
  fit = env$bart_star(y = y, X = X, X_test = X_test, y_test = y_test, transformation = tr, y_max = y_max,
                      offsets = e, offsets_test = e_test, nsave = 200, nburn = 200, save_y_hat = TRUE, verbose = FALSE)
  lambda = if(tr == 'log') 0 else fit$post.lambda[1]
  y_sim = sapply(i, function(k) round_floor(g_inv_bc(rnorm(1e5, fit$post.mu.test[1, k], fit$post.sigma[1]), lambda)*e_test[k], y_max))
  cat("\n", tr, "\n")
  print(data.frame(e = e_test[i], y_test = y_test[i],
                   E_code = fit$post.fitted.values.test[1, i], E_sim = colMeans(y_sim),
                   P_code = exp(fit$post.log.pred.test[1, i]), P_sim = colMeans(y_sim == rep(y_test[i], each = 1e5))), digits = 3)
}
