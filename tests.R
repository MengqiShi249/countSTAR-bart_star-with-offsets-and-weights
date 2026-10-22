#----------------------------------------------------------------------------
# Correctness tests for bart_star() with offsets (bart_star_offset.R).
#
# These are quick checks that the code does what it is meant to do in every
# case we expect to use. They are not a performance comparison.
#
#   T1  Conditional expectations: post.fitted.values vs. the posterior
#       predictive draws, training and test sets, all transformations,
#       with/without offsets, with/without y_max.
#   T2  y_max: support of the predictive draws, mass at y_max, finite
#       pointwise log-likelihood / WAIC / log-predictive scores.
#   T3  offsets = 1 reproduces countSTAR::bart_star() exactly (same seed),
#       all outputs, with/without y_max.
#   T4  Invariance to the units of the offsets.
#   T5  Argument checks and the 'ispline' dispatch.
#
# Each check prints [PASS]/[FAIL] with its statistics as it runs; a summary
# table and the list of failures are printed at the end. Nothing is saved.
#
# Usage: set code_dir to the folder holding bart_star_offset.R and
# helper_sim.R, then source() this file or run it with Rscript.
#----------------------------------------------------------------------------
library(countSTAR)
code_dir = "."   # e.g. "/Users/shimengqi/Desktop/Courses/0-research/BART-STAR/offset"
source(file.path(code_dir, "bart_star_offset.R"))   # masks countSTAR::bart_star
source(file.path(code_dir, "helper_sim.R"))

transformations = c('identity', 'log', 'sqrt', 'box-cox', 'np', 'pois', 'neg-bin')
n = 150; n_test = 60; p = 5
mcmc = list(n.trees = 50, nburn = 200, nsave = 500, nskip = 0, verbose = FALSE)

results = data.frame()
record = function(test, case, pass, detail = ''){
  results <<- rbind(results, data.frame(test = test, case = case,
                                        result = ifelse(pass, 'PASS', 'FAIL'), detail = detail))
  cat(sprintf('[%s] %-4s %-38s %s\n', ifelse(pass, 'PASS', 'FAIL'), test, case, detail))
}
fit_star = function(...) do.call(bart_star, c(list(...), mcmc))

# Data: 'heterogeneous' scenario, training + test set from one draw
make_data = function(offset_range, y_max = Inf, seed = 1){
  dat = simulate_heterogeneous_offset(n = n + n_test, p = p, offset_range = offset_range, seed = seed)
  y = pmin(dat$y, y_max)
  itr = 1:n; ite = n + 1:n_test
  list(y = y[itr], X = dat$X[itr, ], e = dat$offset[itr],
       y_test = y[ite], X_test = dat$X[ite, ], e_test = dat$offset[ite])
}

# Fitted values vs. predictive draws. For draw s, d_s = y_tilde_s - E(y | theta_s):
# given the chain, the d_s are independent with mean 0, so the per-observation
# z_i = mean_s(d_si)/(sd_s(d_si)/sqrt(S)) is approx. N(0, 1), and so is the
# aggregate z over all draws and observations.
check_expectation = function(pred, fitted){
  d = pred - fitted
  S = nrow(d)
  sd_i = apply(d, 2, sd); ok = sd_i > 0
  z_i = colMeans(d)[ok]/(sd_i[ok]/sqrt(S))
  z_all = sum(d)/sqrt(sum(sweep(d, 2, colMeans(d))^2))
  list(rel_bias = mean(fitted)/mean(pred) - 1, z_all = z_all,
       max_z = max(abs(z_i)), n_z = length(z_i))
}
summ_expect = function(ce) sprintf('rel.bias = %+.2f%%, z_all = %+.2f, max|z_i| = %.2f',
                                   100*ce$rel_bias, ce$z_all, ce$max_z)
# Bonferroni threshold for max|z_i| at family-wise level 0.01, and |z_all| < 4
pass_expect = function(ce) is.finite(ce$z_all) && abs(ce$z_all) < 4 &&
  ce$max_z < qnorm(1 - 0.005/ce$n_z)

#----------------------------------------------------------------------------
# T1 + T2: every transformation x {e = 1, e ~ U(0.5, 5)} x {y_max = Inf, y_max = 6}
#----------------------------------------------------------------------------
designs = list(
  list(lab = 'e = 1,      y_max = Inf', range = c(1, 1),   y_max = Inf),
  list(lab = 'e ~ U(0.5,5), y_max = Inf', range = c(0.5, 5), y_max = Inf),
  list(lab = 'e = 1,      y_max = 6',   range = c(1, 1),   y_max = 6),
  list(lab = 'e ~ U(0.5,5), y_max = 6',   range = c(0.5, 5), y_max = 6)
)
for(des in designs){
  dat = make_data(des$range, des$y_max)
  cat(sprintf('\n=== %s: %.0f%% zeros, %.0f%% at y_max ===\n', des$lab,
              100*mean(dat$y == 0), 100*mean(dat$y == des$y_max)))
  for(tr in transformations){
    set.seed(10)
    fit = fit_star(y = dat$y, X = dat$X, offsets = dat$e,
                   X_test = dat$X_test, y_test = dat$y_test, offsets_test = dat$e_test,
                   transformation = tr, y_max = des$y_max, save_y_hat = TRUE)
    case = paste0(tr, ' | ', des$lab)
    
    # T1: conditional expectations (training and test sets)
    ce_tr = check_expectation(fit$post.pred, fit$post.fitted.values)
    ce_te = check_expectation(fit$post.pred.test, fit$post.fitted.values.test)
    record('T1', paste(case, '(train)'), pass_expect(ce_tr), summ_expect(ce_tr))
    record('T1', paste(case, '(test)'),  pass_expect(ce_te), summ_expect(ce_te))
    
    # T2: support, finiteness, mass at y_max
    supp = all(fit$post.pred >= 0, fit$post.pred <= des$y_max,
               fit$post.pred.test >= 0, fit$post.pred.test <= des$y_max,
               fit$post.pred == round(fit$post.pred),
               fit$post.fitted.values <= des$y_max, fit$post.fitted.values.test <= des$y_max)
    fin = all(is.finite(fit$post.log.like.point), is.finite(fit$WAIC),
              is.finite(fit$post.log.pred.test))
    detail = sprintf('WAIC = %.1f', fit$WAIC)
    if(is.finite(des$y_max)){
      # observed share of y = y_max vs. its posterior predictive distribution
      ppc = apply(fit$post.pred, 1, function(x) mean(x == des$y_max))
      obs = mean(dat$y == des$y_max)
      q = quantile(ppc, c(0.005, 0.995))
      detail = sprintf('%s; Pr(y = y_max): obs %.3f, PPC 99%% [%.3f, %.3f]', detail, obs, q[1], q[2])
      supp = supp && obs >= q[1] && obs <= q[2]
    }
    record('T2', case, supp && fin, detail)
  }
}

#----------------------------------------------------------------------------
# T3: offsets = 1 reproduces the original countSTAR::bart_star() (same seed)
#----------------------------------------------------------------------------
cat('\n=== T3: offsets = 1 vs countSTAR::bart_star() ===\n')
dat = make_data(c(1, 1))
dat6 = make_data(c(1, 1), y_max = 6)
for(tr in transformations){
  for(ym in c(Inf, 6)){
    dd = if(is.finite(ym)) dat6 else dat
    args = c(list(y = dd$y, X = dd$X, X_test = dd$X_test, y_test = dd$y_test,
                  transformation = tr, y_max = ym, save_y_hat = TRUE), mcmc)
    set.seed(20); f0 = do.call(countSTAR::bart_star, args)
    set.seed(20); f1 = do.call(bart_star, args)                                 # default offsets
    set.seed(20); f2 = do.call(bart_star, c(args, list(offsets = rep(1, n))))   # explicit ones
    keys = c('post.pred', 'post.sigma', 'post.log.like.point', 'WAIC', 'p_waic',
             'post.pred.test', 'post.fitted.values.test', 'post.mu.test', 'post.log.pred.test',
             'fitted.values', 'post.fitted.values', 'post.lambda')
    same_0 = sapply(keys, function(k) identical(f0[[k]], f1[[k]]))   # default offsets vs countSTAR
    same_1 = sapply(keys, function(k) identical(f1[[k]], f2[[k]]))   # default vs explicit ones
    detail = if(all(same_0, same_1)) 'all outputs identical' else
      paste('differ:', paste(unique(keys[!(same_0 & same_1)]), collapse = ', '))
    record('T3', paste0(tr, ' | y_max = ', ym), all(same_0, same_1), detail)
  }
}

#----------------------------------------------------------------------------
# T4: units of the offsets (e -> c*e). Count-scale outputs unchanged for the
# CDF transformations (up to the g^{-1} grid) and for log (mu shifts by -log c)
#----------------------------------------------------------------------------
cat('\n=== T4: invariance to e -> 10 e ===\n')
dat = make_data(c(0.5, 5))
for(tr in c('log', 'np', 'pois', 'neg-bin')){
  set.seed(30); fa = fit_star(y = dat$y, X = dat$X, offsets = dat$e, transformation = tr, save_y_hat = TRUE)
  set.seed(30); fb = fit_star(y = dat$y, X = dat$X, offsets = 10*dat$e, transformation = tr, save_y_hat = TRUE)
  d_fit = max(abs(fa$fitted.values - fb$fitted.values))
  d_waic = abs(fa$WAIC - fb$WAIC)
  ok = d_fit < 0.05*max(fa$fitted.values) && d_waic < 1
  record('T4', tr, ok, sprintf('max|diff fitted| = %.3g, |diff WAIC| = %.3g', d_fit, d_waic))
}

#----------------------------------------------------------------------------
# T5: argument checks and ispline dispatch
#----------------------------------------------------------------------------
cat('\n=== T5: argument checks ===\n')
dat = make_data(c(0.5, 5))
expect_error = function(expr) inherits(try(expr, silent = TRUE), 'try-error')
record('T5', 'offsets wrong length', expect_error(fit_star(y = dat$y, X = dat$X, offsets = dat$e[-1])))
record('T5', 'offsets <= 0',         expect_error(fit_star(y = dat$y, X = dat$X, offsets = replace(dat$e, 1, 0))))
record('T5', 'offsets NA',           expect_error(fit_star(y = dat$y, X = dat$X, offsets = replace(dat$e, 1, NA))))
record('T5', 'offsets_test wrong length',
       expect_error(fit_star(y = dat$y, X = dat$X, X_test = dat$X_test, offsets_test = dat$e_test[-1])))
record('T5', 'y > y_max',            expect_error(fit_star(y = dat$y, X = dat$X, y_max = max(dat$y) - 1)))
record('T5', 'ispline with offsets', expect_error(fit_star(y = dat$y, X = dat$X, offsets = dat$e, transformation = 'ispline')))
ok_isp = !expect_error(fit_star(y = dat$y, X = dat$X, transformation = 'ispline'))
record('T5', 'ispline without offsets runs', ok_isp)

#----------------------------------------------------------------------------
cat('\n=== Summary ===\n')
print(table(results$test, results$result))
fails = results[results$result == 'FAIL', ]
if(nrow(fails) > 0){ cat('\nFailures:\n'); print(fails, row.names = FALSE) }