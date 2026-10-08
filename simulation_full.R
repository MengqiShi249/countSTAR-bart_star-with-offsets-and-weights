library(countSTAR)
source("/Users/shimengqi/Desktop/Courses/0-research/BART-STAR/offset/bart_star_offset.R")
source("/Users/shimengqi/Desktop/Courses/0-research/BART-STAR/offset/helper_sim.R")

set.seed(123)

n = 250; p = 6
nburn = 500; nsave = 1000; n.trees = 75

scenarios = list(
  friedman      = function(...) simulate_nb_friedman_offset(n = n, p = p, r_nb = 2, ...),
  imbalanced    = function(...) simulate_imbalanced_offset(n = n, p = p, ...),
  heterogeneous = function(...) simulate_heterogeneous_offset(n = n, p = p, ...)
)
# Each data set is generated with its own seed, so the results for a given
# transformation do not depend on the order of this list.
transformations = c('identity', 'log', 'sqrt', 'box-cox', 'np', 'pois', 'neg-bin', 'ispline')

# plot_fitted(), same scale for straightforward comparison
plot_fitted_shared = function(y, post_y, xlim, ylim, main){
  ci = coda::HPDinterval(coda::as.mcmc(post_y), prob = 0.95)
  y_hat = colMeans(post_y)
  plot(y, y_hat, type = 'n', xlim = xlim, ylim = ylim,
       xlab = 'Data Values', ylab = 'Fitted Values', main = main)
  for(i in seq_along(y)) lines(rep(y[i], 2), ci[i,], col = adjustcolor('gray', 0.4), lwd = 3)
  points(y, y_hat, pch = 1, col = adjustcolor('black', 0.6))
  lines(xlim, xlim, lwd = 2, lty = 2, col = 'red')
}
shared_lim = function(y1, post_y1, y2, post_y2, alpha = 0.05){
  ci1 = coda::HPDinterval(coda::as.mcmc(post_y1), prob = 1 - alpha)
  ci2 = coda::HPDinterval(coda::as.mcmc(post_y2), prob = 1 - alpha)
  range(ci1, ci2, y1, y2)
}

# Shared-x-axis proportion-of-zeros PPC histogram, for comparison.
ppc_zeros_shared = function(post_pred, obs_y, xlim, main){
  prop0 = apply(post_pred, 1, function(x) mean(x == 0))
  hist(prop0, xlim = xlim, main = main, xlab = '')
  abline(v = mean(obs_y == 0), lwd = 4, col = 'blue')
}

# Fit the new bart_star() without and with the offsets:
#   'FALSE' = default offsets (e_i = 1, i.e. the original bart_star())
#   'TRUE'  = offsets = the simulated exposures e_i
fit_pair = function(y, X, offset, transformation){
  fit_false = bart_star(y = y, X = X, transformation = transformation,
                        n.trees = n.trees, nburn = nburn, nsave = nsave, nskip = 0,
                        save_y_hat = TRUE, verbose = FALSE)
  fit_true  = bart_star(y = y, X = X, offsets = offset, transformation = transformation,
                        n.trees = n.trees, nburn = nburn, nsave = nsave, nskip = 0,
                        save_y_hat = TRUE, verbose = FALSE)
  list(`FALSE` = fit_false, `TRUE` = fit_true)
}

results_i  = data.frame() # (i) offset should help
results_ii = data.frame() # (ii) offset should be a no-op
zoom_lim = list(friedman = c(0, 80))

for(transformation in transformations){
  for(name in names(scenarios)){
    gen = scenarios[[name]]
    tag = paste0(name, ' [', transformation, ']')
    
    #--------------------------------------------------------------------
    # (i) Data WITH offset heterogeneity
    #--------------------------------------------------------------------
    if(transformation == 'ispline'){
      cat('\n===', tag, ': (i) skipped, ispline is not available with offsets ===\n')
    } else {
      cat('\n===', tag, ': (i) data WITH offset ===\n')
      dat_i = gen(offset_range = c(0.5, 5), seed = 1)
      fits_i = fit_pair(dat_i$y, dat_i$X, dat_i$offset, transformation)
      
      results_i = rbind(results_i, data.frame(
        scenario = name,
        transformation = transformation,
        offset = c('FALSE', 'TRUE'),
        WAIC = c(fits_i[['FALSE']]$WAIC, fits_i[['TRUE']]$WAIC),
        RMSE_to_truth = c(sqrt(mean((fits_i[['FALSE']]$fitted.values - dat_i$Ey_obs)^2)),
                          sqrt(mean((fits_i[['TRUE']]$fitted.values  - dat_i$Ey_obs)^2))),
        prop_zero_obs = mean(dat_i$y == 0)
      ))
      
      lim_i = shared_lim(dat_i$Ey_obs, fits_i[['FALSE']]$post.fitted.values,
                         dat_i$Ey_obs, fits_i[['TRUE']]$post.fitted.values)
      if(!is.null(zoom_lim[[name]])) lim_i = zoom_lim[[name]]
      plot_fitted_shared(dat_i$Ey_obs, fits_i[['FALSE']]$post.fitted.values, lim_i, lim_i,
                         main = paste0(tag, ' (i): bart_star(offset = FALSE)'))
      plot_fitted_shared(dat_i$Ey_obs, fits_i[['TRUE']]$post.fitted.values, lim_i, lim_i,
                         main = paste0(tag, ' (i): bart_star(offset = TRUE)'))
      
      hist_lim_i = range(apply(fits_i[['FALSE']]$post.pred, 1, function(x) mean(x == 0)),
                         apply(fits_i[['TRUE']]$post.pred, 1, function(x) mean(x == 0)),
                         mean(dat_i$y == 0))
      ppc_zeros_shared(fits_i[['FALSE']]$post.pred, dat_i$y, hist_lim_i,
                       main = paste0(tag, ' (i) offset=FALSE: PPC prop. zeros'))
      ppc_zeros_shared(fits_i[['TRUE']]$post.pred, dat_i$y, hist_lim_i,
                       main = paste0(tag, ' (i) offset=TRUE: PPC prop. zeros'))
      
    } 
    
    #--------------------------------------------------------------------
    # (ii) Data WITHOUT offset (e = 1 for all)
    #--------------------------------------------------------------------
    cat('\n===', tag, ': (ii) data WITHOUT offset ===\n')
    dat_ii = gen(offset = rep(1, n), seed = 2)
    fits_ii = fit_pair(dat_ii$y, dat_ii$X, dat_ii$offset, transformation)
    
    results_ii = rbind(results_ii, data.frame(
      scenario = name,
      transformation = transformation,
      WAIC_FALSE = fits_ii[['FALSE']]$WAIC, WAIC_TRUE = fits_ii[['TRUE']]$WAIC,
      RMSE_between = sqrt(mean((fits_ii[['FALSE']]$fitted.values - fits_ii[['TRUE']]$fitted.values)^2)),
      cor_fitted = cor(fits_ii[['FALSE']]$fitted.values, fits_ii[['TRUE']]$fitted.values)
    ))
    
    lim_ii = shared_lim(dat_ii$Ey, fits_ii[['FALSE']]$post.fitted.values,
                        dat_ii$Ey, fits_ii[['TRUE']]$post.fitted.values)
    if(!is.null(zoom_lim[[name]])) lim_ii = zoom_lim[[name]]
    plot_fitted_shared(dat_ii$Ey, fits_ii[['FALSE']]$post.fitted.values, lim_ii, lim_ii,
                       main = paste0(tag, ' (ii): bart_star(offset = FALSE)'))
    plot_fitted_shared(dat_ii$Ey, fits_ii[['TRUE']]$post.fitted.values, lim_ii, lim_ii,
                       main = paste0(tag, ' (ii): bart_star(offset = TRUE)'))
  }
}

cat('\n--- (i) offset = TRUE should help: WAIC / RMSE to truth ---\n')
print(results_i, row.names = FALSE)

cat('\n--- (ii) offset = TRUE should be a no-op when e = 1 ---\n')
print(results_ii, row.names = FALSE)

# Compact comparison across transformations, case (i):
# change in WAIC and relative change in RMSE to truth from adding the offsets
if(nrow(results_i) > 0){
  summary_i = do.call(rbind, lapply(split(results_i, list(results_i$transformation, results_i$scenario), drop = TRUE), function(r){
    data.frame(scenario = r$scenario[1], transformation = r$transformation[1],
               dWAIC = r$WAIC[r$offset == 'TRUE'] - r$WAIC[r$offset == 'FALSE'],
               dRMSE_pct = 100*(r$RMSE_to_truth[r$offset == 'TRUE']/r$RMSE_to_truth[r$offset == 'FALSE'] - 1))
  }))
  summary_i = summary_i[order(summary_i$scenario, match(summary_i$transformation, transformations)), ]
  cat('\n--- (i) effect of offsets by transformation (TRUE - FALSE) ---\n')
  print(summary_i, row.names = FALSE, digits = 3)
}
