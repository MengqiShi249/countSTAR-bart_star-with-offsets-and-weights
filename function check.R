library(countSTAR)
library(dbarts)
# Updated package files (bart_star() and g_cdf() with offsets); all other functions are from countSTAR:
path = "(replace with your folder path)"
env = new.env(parent = asNamespace("countSTAR"))
sys.source(paste0(path, "helper_functions.R"), envir = env)
sys.source(paste0(path, "STAR_Bayesian.R"), envir = env)
a_j = countSTAR:::a_j
round_floor = countSTAR:::round_floor
g_bc = countSTAR:::g_bc; g_inv_bc = countSTAR:::g_inv_bc
logLikePointRcpp = countSTAR:::logLikePointRcpp
expectation_gRcpp = countSTAR:::expectation_gRcpp
set.seed(123)

# 1. g_cdf(). Data: y_i ~ Pois(3 e_i) or NB(mean 3 e_i, size 2).
# pois/neg-bin: with z ~ N(0, 1), P(y = j) at unit exposure is Phi(g(j + 1)) - Phi(g(j)).
n = 20000; e = runif(n, 0.5, 5); j = 0:8
g_pois = env$g_cdf(rpois(n, 3*e), 'pois', e)
g_nb = env$g_cdf(rnbinom(n, size = 2, mu = 3*e), 'neg-bin', e)
print(data.frame(j, pois_code = diff(c(0, pnorm(g_pois(j + 1)))), pois_true = dpois(j, 3),
                 nb_code = diff(c(0, pnorm(g_nb(j + 1)))), nb_true = dnbinom(j, size = 2, mu = 3)), digits = 3)

# np with offsets: Phi(g(t)) against P(y/e < t) from a large new sample.
n = 5000; e = runif(n, 0.5, 5); s = c(0.5, 1, 2, 3, 5, 8)
y = rnbinom(n, size = 2, mu = 3*e)
g_np = env$g_cdf(y, 'np', e)
# np without offsets: largest difference between g from g_cdf() and from the original countSTAR::g_cdf().
t = seq(0.5, max(y) + 1, by = 0.01)
print(max(abs(env$g_cdf(y, 'np')(t) - countSTAR::g_cdf(y, 'np')(t))))
e = runif(1e6, 0.5, 5); rate = rnbinom(1e6, size = 2, mu = 3*e)/e
print(data.frame(t = s, np_code = pnorm(g_np(s)), np_true = sapply(s, function(u) mean(rate < u))), digits = 3)

# 2. Cut points, with g = log and y_max = 10: P(y = j) from [g(a_j/e), g(a_{j+1}/e)) against
# the frequencies of y = min(floor(e g^{-1}(z)), y_max), z ~ N(mu, sigma^2).
y_max = 10; mu = 0.2; sigma = 0.8; e = 8; j = 0:y_max
lo = g_bc(a_j(j, y_max)/e, 0); up = g_bc(a_j(j + 1, y_max)/e, 0)
y_sim = round_floor(exp(rnorm(1e5, mu, sigma))*e, y_max)
print(data.frame(j, P_code = exp(logLikePointRcpp(lo, up, rep(mu, y_max + 1), rep(sigma, y_max + 1))),
                 P_sim = tabulate(y_sim + 1, y_max + 1)/1e5), digits = 3)

# 3. Expectation (long vector) and test log-score from bart_star() (y_max = 10), for 'log' and 'box-cox'.
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

# 4. Expectation with offsets, computed over one long vector instead of a for-loop, for 'np' and 'box-cox'.
# Same data as in 3. For each test point and the first saved draw (mu, sigma, lambda), post.fitted.values.test
# should match expectation_gRcpp() called for that point alone with its cut points a_j/e.
for(tr in c('np', 'box-cox')){
  fit = env$bart_star(y = y, X = X, X_test = X_test, transformation = tr, y_max = y_max,
                      offsets = e, offsets_test = e_test, nsave = 200, nburn = 200, verbose = FALSE)
  g = if(tr == 'np') env$g_cdf(y, 'np', e) else function(t) g_bc(t, fit$post.lambda[1])
  E_loop = sapply(i, function(k) expectation_gRcpp(g(a_j(0:y_max, y_max)/e_test[k]), g(a_j(1:(y_max + 1), y_max)/e_test[k]),
                                                   fit$post.mu.test[1, k], fit$post.sigma[1], y_max + 1))
  cat("\n", tr, "\n")
  print(data.frame(e = e_test[i], E_code = fit$post.fitted.values.test[1, i], E_loop), digits = 4)
}

# 5. Run time in seconds for 'np' and 'box-cox' with offsets (nburn = nsave = 500): without the fitted values,
# with the fitted values (long vector), and with the for-loop. The for-loop time is the time
# without the fitted values plus nsave times one pass of the loop over the n observations.
time = NULL
for(n in c(1000, 10000)){
  X = matrix(runif(n*5), n); e = runif(n, 2, 20); y = rpois(n, 10*e*exp(X[,1] - 1.5))
  for(tr in c('np', 'box-cox')){
    t_no = system.time(env$bart_star(y = y, X = X, transformation = tr, offsets = e,
                                     nsave = 500, nburn = 500, verbose = FALSE))[[3]]
    t_vec = system.time(env$bart_star(y = y, X = X, transformation = tr, offsets = e,
                                      nsave = 500, nburn = 500, save_y_hat = TRUE, verbose = FALSE))[[3]]
    # One draw (mu, sigma, lambda) at the training points, for the loop:
    fit = env$bart_star(y = y, X = X, X_test = X, transformation = tr, offsets = e, offsets_test = e,
                        nsave = 1, nburn = 100, verbose = FALSE)
    g = if(tr == 'np') env$g_cdf(y, 'np', e) else function(t) g_bc(t, fit$post.lambda[1])
    J = 2*max(y)
    t_loop = system.time(for(k in 1:n) expectation_gRcpp(g(a_j(0:J)/e[k]), g(a_j(1:(J + 1))/e[k]),
                                                         fit$post.mu.test[1, k], fit$post.sigma, J))[[3]]
    time = rbind(time, data.frame(n, transformation = tr, no_y_hat = t_no, long_vector = t_vec, for_loop = t_no + 500*t_loop))
  }
}
print(time, digits = 3)
