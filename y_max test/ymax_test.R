library(countSTAR); library(dbarts)
# Source the offsets version (bart_star and g_cdf) on top of the countSTAR internals:
env = new.env(parent = asNamespace("countSTAR"))
sys.source("/Users/shimengqi/Desktop/Courses/0-research/BART-STAR/offset/bart_star_offset.R", envir = env)
bart_star = env$bart_star

# Simulate data: y = min(y0, y_max), y0 ~ NB(mean e*r(x), size = 3), r(x) = Friedman(x)/20
set.seed(123)
n = 500; n_test = 200; p = 5; y_max = 10
X = matrix(runif(n*p), n, p); X_test = matrix(runif(n_test*p), n_test, p)
r_fun = function(X) (10*sin(pi*X[,1]*X[,2]) + 20*(X[,3] - 0.5)^2 + 10*X[,4] + 5*X[,5])/20
e = runif(n, 2, 20); e_test = runif(n_test, 2, 20)
y = pmin(rnbinom(n, size = 3, mu = e*r_fun(X)), y_max)
y_test = pmin(rnbinom(n_test, size = 3, mu = e_test*r_fun(X_test)), y_max)
cat("Proportion of training counts at y_max:", mean(y == y_max), "\n\n")

#----------------------------------------------------------------------------
# Part 1: fit for each transformation and check the output
#----------------------------------------------------------------------------
fits = list(); out = NULL
for(transformation in c('identity', 'log', 'sqrt', 'box-cox', 'np', 'pois', 'neg-bin')){
  fit = bart_star(y = y, X = X, X_test = X_test, y_test = y_test,
                  transformation = transformation, y_max = y_max,
                  offsets = e, offsets_test = e_test,
                  nsave = 500, nburn = 500, save_y_hat = TRUE, verbose = FALSE)
  fits[[transformation]] = fit
  out = rbind(out, data.frame(
    transformation = transformation,
    # all outputs finite:
    all_finite = all(sapply(fit[c('post.pred', 'post.fitted.values', 'post.log.like.point',
                                  'post.pred.test', 'post.fitted.values.test',
                                  'post.log.pred.test', 'WAIC')], function(x) all(is.finite(x)))),
    # predictive draws within [0, y_max], and how often they reach y_max (vs. observed):
    pred_range = paste(range(c(fit$post.pred, fit$post.pred.test)), collapse = '-'),
    at_ymax_obs = mean(y == y_max),
    at_ymax_pred = mean(fit$post.pred == y_max),
    at_ymax_obs_test = mean(y_test == y_max),
    at_ymax_pred_test = mean(fit$post.pred.test == y_max),
    # fitted values vs. predictive means (training and test):
    max_fitted = max(fit$fitted.values),
    mean_fitted = mean(fit$fitted.values),
    mean_pred = mean(fit$post.pred),
    mean_fitted_test = mean(fit$post.fitted.values.test),
    mean_pred_test = mean(fit$post.pred.test),
    WAIC = fit$WAIC))
}
print(out, digits = 3)

#----------------------------------------------------------------------------
# Part 2: direct check that the conditional expectation omits the y_max term
# (uses the 'log' fit from Part 1)
#----------------------------------------------------------------------------
fit = fits[['log']]

# (1) Per observation, from the returned output only (any transformation):
#     predictive mean - fitted value  vs.  y_max * P(y = y_max)
pred_mean = colMeans(fit$post.pred)
P_top     = colMeans(fit$post.pred == y_max)
gap       = pred_mean - fit$fitted.values
tab = data.frame(y = y, P_top = P_top, pred_mean = pred_mean, fitted = fit$fitted.values,
                 gap = gap, ymax_P_top = y_max*P_top)
cat("\n(1) Training observations with the largest P(y = y_max):\n")
print(head(tab[order(-P_top), ], 8), digits = 3, row.names = FALSE)
cat("\nAll", n, "observations: cor(gap, y_max*P_top) =", round(cor(gap, y_max*P_top), 4),
    "; mean |gap - y_max*P_top| =", round(mean(abs(gap - y_max*P_top)), 3), "\n")
plot(y_max*P_top, gap, xlab = 'y_max * P(y = y_max)', ylab = 'predictive mean - fitted value',
     main = 'Gap vs. omitted y_max term (training, log)'); abline(0, 1, col = 'red', lwd = 2)

# (2) Exact check on the test set (log: g(t) = log(t), using post.mu.test and post.sigma):
#     the returned value is sum_{j=0}^{y_max-1} j*P(y = j), so adding y_max*P(y = y_max)
#     gives the full expectation sum_{j=0}^{y_max} j*P(y = j).
mu = fit$post.mu.test; sg = fit$post.sigma; E = matrix(e_test, nrow(mu), n_test, byrow = TRUE)
P_j = function(j) pnorm(log((j + 1)/E), mu, sg) - pnorm(log(j/E), mu, sg)  # P(y = j), j < y_max
E_without_top = Reduce(`+`, lapply(1:(y_max - 1), function(j) j*P_j(j)))
P_top_exact   = 1 - pnorm(log(y_max/E), mu, sg)                              # P(y >= y_max)
cat("\n(2) Test set, exact (all", length(mu), "draws x observations):\n")
cat("    max |returned - sum_{j<y_max} j*P(y=j)|        =",
    signif(max(abs(fit$post.fitted.values.test - E_without_top)), 3), "\n")
cat("    mean returned fitted value                     =", round(mean(fit$post.fitted.values.test), 3), "\n")
cat("    mean full expectation (+ y_max*P(y = y_max))   =", round(mean(E_without_top + y_max*P_top_exact), 3), "\n")
cat("    mean of predictive draws                       =", round(mean(fit$post.pred.test), 3), "\n")