#' MCMC Algorithm for BART-STAR
#'
#' Run the MCMC algorithm for a BART model for count-valued responses using STAR.
#' The transformation can be known (e.g., log or sqrt) or unknown
#' (Box-Cox or estimated nonparametrically) for greater flexibility.
#'
#' @param y \code{n x 1} vector of observed counts
#' @param X \code{n x p} matrix of predictors
#' @param X_test \code{n_test x p} matrix of predictors for test data
#' @param y_test \code{n_test x 1} vector of the test data responses (used for
#' computing log-predictive scores)
#' @param offsets \code{n x 1} vector (or scalar) of known offsets (exposures)
#' \code{e_i > 0} for the training data; default is 1 (no offsets). See Details.
#' @param offsets_test \code{n_test x 1} vector (or scalar) of known offsets for
#' the test data; default is 1, i.e., predictions at unit exposure. Ignored if
#' \code{X_test = NULL}
#' @param transformation transformation to use for the latent process; must be one of
#' \itemize{
#' \item "identity" (identity transformation)
#' \item "log" (log transformation)
#' \item "sqrt" (square root transformation)
#' \item "np" (nonparametric transformation estimated from empirical CDF)
#' \item "pois" (transformation for moment-matched marginal Poisson CDF)
#' \item "neg-bin" (transformation for moment-matched marginal Negative Binomial CDF)
#' \item "box-cox" (box-cox transformation with learned parameter)
#' \item "ispline" (transformation is modeled as unknown, monotone function
#' using I-splines)
#' }
#' @param y_max a fixed and known upper bound for all observations; default is \code{Inf}
#' @param n.trees number of trees to use in BART; default is 200
#' @param sigest positive numeric estimate of the residual standard deviation (see ?bart)
#' @param sigdf  degrees of freedom for error variance prior (see ?bart)
#' @param sigquant quantile of the error variance prior that the rough estimate (sigest)
#' is placed at. The closer the quantile is to 1, the more aggressive the fit will be (see ?bart)
#' @param k the number of prior standard deviations E(Y|x) = f(x) is away from +/- 0.5.
#' The response is internally scaled to range from -0.5 to 0.5.
#' The bigger k is, the more conservative the fitting will be (see ?bart)
#' @param power power parameter for tree prior (see ?bart)
#' @param base  base parameter for tree prior (see ?bart)
#' @param nsave number of MCMC iterations to save
#' @param nburn number of MCMC iterations to discard
#' @param nskip number of MCMC iterations to skip between saving iterations,
#' i.e., save every (nskip + 1)th draw
#' @param save_y_hat logical; if TRUE, compute and save the posterior draws of
#' the expected counts, E(y), which may be slow to compute
#' @param verbose logical; if TRUE, print time remaining
#'
#' @return a list with the following elements:
#' \itemize{
#' \item \code{post.pred}: draws from the posterior predictive distribution of \code{y}
#' \item \code{post.sigma}: draws from the posterior distribution of \code{sigma}
#' \item \code{post.log.like.point}: draws of the log-likelihood for each of the \code{n} observations
#' \item \code{WAIC}: Widely-Applicable/Watanabe-Akaike Information Criterion
#' \item \code{p_waic}: Effective number of parameters based on WAIC
#' \item \code{post.pred.test}: draws from the posterior predictive distribution at the test points \code{X_test}
#' (\code{NULL} if \code{X_test} is not given)
#' \item \code{post.fitted.values.test}: posterior draws of the conditional mean at the test points \code{X_test}
#' (\code{NULL} if \code{X_test} is not given)
#' \item \code{post.mu.test}: draws of the conditional mean of z_star at the test points \code{X_test}
#' (\code{NULL} if \code{X_test} is not given)
#' \item \code{post.log.pred.test}: draws of the log-predictive distribution for each of the \code{n_test} test cases
#' (\code{NULL} if \code{X_test} is not given)
#' \item \code{fitted.values}: the posterior mean of the conditional expectation of the counts \code{y}
#' (\code{NULL} if \code{save_y_hat=FALSE})
#' \item \code{post.fitted.values}: posterior draws of the conditional mean of the counts \code{y}
#' (\code{NULL} if \code{save_y_hat=FALSE})
#' }
#' In the case of \code{transformation="ispline"}, the list also contains
#' \itemize{
#' \item \code{post.g}: draws from the posterior distribution of the transformation \code{g}
#' \item \code{post.sigma.gamma}: draws from the posterior distribution of \code{sigma.gamma},
#' the prior standard deviation of the transformation g() coefficients
#' }
#' If \code{transformation="box-cox"}, then the list also contains
#' \itemize{
#' \item \code{post.lambda}: draws from the posterior distribution of \code{lambda}
#' }
#'
#' @details STAR defines a count-valued probability model by
#' (1) specifying a Gaussian model for continuous *latent* data and
#' (2) connecting the latent data to the observed data via a
#' *transformation and rounding* operation. Here, the model in (1)
#' is a Bayesian additive regression tree (BART) model.
#'
#' Posterior and predictive inference is obtained via a Gibbs sampler
#' that combines (i) a latent data augmentation step (like in probit regression)
#' and (ii) an existing sampler for a continuous data model.
#'
#' There are several options for the transformation. First, the transformation
#' can belong to the *Box-Cox* family, which includes the known transformations
#' 'identity', 'log', and 'sqrt', as well as a version in which the Box-Cox parameter
#' is inferred within the MCMC sampler ('box-cox'). Second, the transformation
#' can be estimated (before model fitting) using the empirical distribution of the
#' data \code{y}. Options in this case include the empirical cumulative
#' distribution function (CDF), which is fully nonparametric ('np'), or the parametric
#' alternatives based on Poisson ('pois') or Negative-Binomial ('neg-bin')
#' distributions. For the parametric distributions, the parameters of the distribution
#' are estimated using moments (means and variances) of \code{y}. Third, the transformation can be
#' modeled as an unknown, monotone function using I-splines ('ispline'). The
#' Robust Adaptive Metropolis (RAM) sampler is used for drawing the parameter
#' of the transformation function.
#'
#' \strong{Offsets.} With known exposures \code{e_i > 0} (\code{offsets}), the
#' transformation acts on the latent count per unit exposure:
#' \code{y_i = floor(y_i*)} and \code{g(y_i*/e_i) = mu(x_i) + eps_i}, so \code{mu}
#' and \code{sigma} are on the scale of the transformed density \code{g(y*/e)}.
#' The offsets enter only through the observation-specific cut points
#' \code{a_j/e_i}; for \code{transformation = 'log'} this is the usual
#' log-offset. For 'np', 'pois' and 'neg-bin', the transformation is estimated
#' on the density scale: 'np' uses the empirical CDF of \code{y/offsets};
#' 'pois' and 'neg-bin' use the moment-matched count CDF at the reference
#' exposure given by the geometric mean of \code{offsets}. 'ispline' is not
#' available with non-unit offsets. All count-scale outputs (\code{post.pred},
#' \code{post.fitted.values}, ...) are at the supplied exposures: \code{offsets}
#' for the training data and \code{offsets_test} for the test data. With
#' \code{offsets = 1}, the computations are those of the original sampler.
#'
#' @examples
#' \donttest{
#' # Simulate data with count-valued response y:
#' sim_dat = simulate_nb_friedman(n = 100, p = 5)
#' y = sim_dat$y; X = sim_dat$X
#'
#' if (requireNamespace("dbarts", quietly = TRUE)) {
#'   # BART-STAR with log-transformation:
#'   fit_log = bart_star(y = y, X = X, transformation = 'log',
#'                       save_y_hat = TRUE, nburn=1000, nskip=0)
#'
#'   # Fitted values
#'   plot_fitted(y = sim_dat$Ey,
#'               post_y = fit_log$post.fitted.values,
#'               main = 'Fitted Values: BART-STAR-log')
#'
#'   # WAIC for BART-STAR-log:
#'   fit_log$WAIC
#'
#'   # MCMC diagnostics:
#'   plot(as.ts(fit_log$post.fitted.values[,1:10]))
#'
#'   # Posterior predictive check:
#'   hist(apply(fit_log$post.pred, 1,
#'              function(x) mean(x==0)), main = 'Proportion of Zeros', xlab='');
#'   abline(v = mean(y==0), lwd=4, col ='blue')
#'
#'   # BART-STAR with nonparametric transformation:
#'   fit = bart_star(y = y, X = X,
#'                        transformation = 'np', save_y_hat = TRUE)
#'
#'   # Fitted values
#'   plot_fitted(y = sim_dat$Ey,
#'               post_y = fit$post.fitted.values,
#'               main = 'Fitted Values: BART-STAR-np')
#'
#'   # WAIC for BART-STAR-np:
#'   fit$WAIC
#'
#'   # MCMC diagnostics:
#'   plot(as.ts(fit$post.fitted.values[,1:10]))
#'
#'   # Posterior predictive check:
#'   hist(apply(fit$post.pred, 1,
#'              function(x) mean(x==0)), main = 'Proportion of Zeros', xlab='');
#'   abline(v = mean(y==0), lwd=4, col ='blue')
#'
#'# BART-STAR with known offsets (exposures) e_i:
#' e = runif(length(y), min = 0.5, max = 5)
#' y_e = rnbinom(length(y), size = 1, mu = e*sim_dat$Ey)  # counts proportional to exposure
#' fit_off = bart_star(y = y_e, X = X, offsets = e,
#'                    X_test = X, offsets_test = 1,
#'                    transformation = 'np', save_y_hat = TRUE)
#'
#'   # Expected counts at the observed exposures:
#'   plot_fitted(y = e*sim_dat$Ey,
#'               post_y = fit_off$post.fitted.values,
#'               main = 'Fitted Values: BART-STAR-np with offsets')
#'
#'   # Expected counts at unit exposure (density per unit effort):
#'   plot_fitted(y = sim_dat$Ey,
#'               post_y = fit_off$post.fitted.values.test,
#'               main = 'Fitted Values at e = 1')
#' }
#' }
#'
#' @export
bart_star = function(y,
                     X,
                     X_test = NULL, y_test = NULL,
                     offsets = 1,
                     offsets_test = 1,
                     transformation = 'np',
                     y_max = Inf,
                     n.trees = 200,
                     sigest = NULL, sigdf = 3, sigquant = 0.90, k = 2.0, power = 2.0, base = 0.95,
                     nsave = 1000,
                     nburn = 1000,
                     nskip = 0,
                     save_y_hat = FALSE,
                     verbose = TRUE){
  
  # Library required here:
  if (!requireNamespace("dbarts", quietly = TRUE)) stop("Package \"dbarts\" must be installed to use this function.", call. = FALSE)
  #----------------------------------------------------------------------------
  # To report the full computing time:
  if(verbose) timer00 = proc.time()[3]
  
  # Check: currently implemented for nonnegative integers
  if(any(y < 0) || any(y != floor(y)))
    stop('y must be nonnegative counts')
  
  # Check: y_max must be a true upper bound
  if(any(y > y_max))
    stop('y must not exceed y_max')
  
  # OFFSETS: checks (scalars are recycled)
  if(length(offsets) == 1) offsets = rep(offsets, length(y))
  if(length(offsets) != length(y)) stop('offsets must have length equal to length(y)')
  if(any(!is.finite(offsets)) || any(offsets <= 0)) stop('offsets must be finite and strictly positive')
  
  #If transformation is ispline, call the appropriate function and return the results
  if(transformation=="ispline"){
    # OFFSETS: the I-spline sampler assumes integer cut points a_j = j
    if(any(offsets != 1) || any(offsets_test != 1))
      stop("transformation = 'ispline' is not available with offsets")
    .args = as.list(match.call())[-1]
    .args[['transformation']] <- NULL
    .args[['offsets']] <- NULL; .args[['offsets_test']] <- NULL
    return(do.call(countSTAR:::bart_star_ispline, .args))
  }
  
  # Check: does the transformation make sense?
  transformation = tolower(transformation);
  if(!is.element(transformation, c("identity", "log", "sqrt", "np", "pois", "neg-bin", "box-cox")))
    stop("The transformation must be one of 'identity', 'log', 'sqrt', 'np', 'pois', 'neg-bin', or 'box-cox'")
  
  # Assign a family for the transformation: Box-Cox or CDF?
  transform_family = ifelse(
    test = is.element(transformation, c("identity", "log", "sqrt", "box-cox")),
    yes = 'bc', no = 'cdf'
  )
  
  # Length of the response vector:
  n = length(y)
  
  # Number of predictors:
  p = ncol(X)
  
  # Include a test dataset:
  include_test = !is.null(X_test)
  if(include_test) {
    n_test = nrow(X_test) # Size of test dataset
    
    # OFFSETS: checks for the test offsets (scalars are recycled)
    if(length(offsets_test) == 1) offsets_test = rep(offsets_test, n_test)
    if(length(offsets_test) != n_test) stop('offsets_test must have length equal to nrow(X_test)')
    if(any(!is.finite(offsets_test)) || any(offsets_test <= 0)) stop('offsets_test must be finite and strictly positive')
  }
  
  # Define the transformation:
  if(transform_family == 'bc'){
    # Lambda value for each Box-Cox argument:
    if(transformation == 'identity') lambda = 1
    if(transformation == 'log') lambda = 0
    if(transformation == 'sqrt') lambda = 1/2
    if(transformation == 'box-cox') lambda = runif(n = 1) # random init on (0,1)
    
    # Transformation function:
    g = function(t) g_bc(t,lambda = lambda)
    
    # Inverse transformation function:
    g_inv = function(s) g_inv_bc(s,lambda = lambda)
  }
  
  if(transform_family == 'cdf'){
    
    if(transformation == 'np'){
      # OFFSETS: estimate g on the density scale, from the empirical CDF of the
      # observed densities d = y/e, with the unmodified g_cdf(). g_cdf() sets
      # g(t) = qnorm(F(t - 1)) at the data points, where 't - 1' means 'strictly
      # below t' for integer counts. Passing M*d with M*(min gap of d) >= 1 (= 2 when rescaled) keeps
      # this meaning for non-integer densities; g(t) = g_M(M*t) undoes the scaling.
      # M = 1 (and g = g_cdf(y)) whenever the densities are >= 1 apart, e.g. offsets = 1.
      d = signif(y/offsets, 12)
      d_unique = sort(unique(d))
      d_gap = if(length(d_unique) > 1) min(diff(d_unique)) else 1
      M = if(d_gap >= 1) 1 else 2/d_gap
      g_M = g_cdf(y = M*d, distribution = 'np')
      g = function(t) g_M(M*t)
    } else {
      # OFFSETS: 'pois'/'neg-bin' are count distributions, so g_cdf() is estimated
      # from the counts y and mapped to the density scale y*/e with the reference
      # exposure e_bar (geometric mean of the offsets): g(t) = g_y(e_bar*t).
      # e_bar = 1 (and g = g_y) when offsets = 1.
      e_bar = exp(mean(log(offsets)))
      g_y = g_cdf(y = y, distribution = transformation)
      g = function(t) g_y(e_bar*t)
    }
    
    # Define the grid for approximations using equally-spaced + quantile points
    # OFFSETS: on the offset-adjusted scale y/e, the domain of g's argument
    t_grid = sort(unique(round(c(
      seq(0, min(2*max(y/offsets), y_max/min(offsets)), length.out = 250),
      quantile(unique((y[y < y_max] + 1)/offsets[y < y_max]), seq(0, 1, length.out = 250))), 8)))
    
    # Inverse transformation function:
    g_inv = g_inv_approx(g = g, t_grid = t_grid)
    
    # No Box-Cox transformation:
    lambda = NULL
  }
  
  # Random initialization for z_star (OFFSETS: original rule on the scale y*/e):
  z_star = g((y + abs(rnorm(n = n)))/offsets)
  
  # Now initialize the model: BART!
  
  # Initialize the dbarts() object:
  control = dbarts::dbartsControl(n.chains = 1, n.burn = 0, n.samples = 1,
                                  n.trees = n.trees)
  # Initialize the standard deviation:
  if(is.null(sigest)){
    # OFFSETS: the pilot MCMC models the raw counts and cannot take offsets,
    # so it is used only when there are none
    if(transformation == 'box-cox' && all(offsets == 1)){
      # Lambda is unknown, so use pilot MCMC with mean-only model to identify sigma estimate:
      fit0 = genMCMC_star(y = y,
                          sample_params = countSTAR:::sample_params_mean,
                          init_params = countSTAR:::init_params_mean,
                          transformation = 'box-cox', nburn = 1000, nsave = 100, verbose = FALSE)
      sigest = median(fit0$post.sigma)
    } else {
      # Transformation is known, so simply transform the raw data and estimate the SD
      # (OFFSETS: on the scale y/e; for 'box-cox' with offsets, at the initial lambda):
      sigest = sd(g((y + 1)/offsets), na.rm=TRUE)
    }
  }
  
  # Initialize the sampling object, which includes the prior specs:
  sampler = dbarts::dbarts(z_star ~ X, test = X_test,
                           control = control,
                           tree.prior = cgm(power, base),
                           node.prior = normal(k),
                           resid.prior = chisq(sigdf, sigquant),
                           sigma = sigest)
  samp = sampler$run(updateState = TRUE)
  
  # Initialize and store the parameters:
  params = list(mu = samp$train,
                sigma = samp$sigma)
  
  # Lower and upper intervals (OFFSETS: cut points a_j/e_i):
  a_y = a_j(y, y_max = y_max)/offsets; a_yp1 = a_j(y + 1, y_max = y_max)/offsets
  z_lower = g(a_y); z_upper = g(a_yp1)
  
  # Store MCMC output:
  if(save_y_hat)  post.fitted.values = array(NA, c(nsave, n)) else post.fitted.values = NULL
  post.pred = array(NA, c(nsave, n))
  post.mu = array(NA, c(nsave, n))
  post.sigma = numeric(nsave)
  post.log.like.point = array(NA, c(nsave, n)) # Pointwise log-likelihood
  # Test data: fitted values and posterior predictive distribution
  if(include_test){
    post.pred.test = post.fitted.values.test = post.mu.test = array(NA, c(nsave, n_test))
    if(!is.null(y_test)) {post.log.pred.test = array(NA, c(nsave, n_test))} else post.log.pred.test = NULL
  } else {
    post.pred.test = post.fitted.values.test = post.mu.test = post.log.pred.test = NULL
  }
  if(transformation == 'box-cox') {
    post.lambda = numeric(nsave)
  } else post.lambda = NULL
  
  # Conditional expectation E(y | mu, sigma, e) (OFFSETS): tail sum lo + sum_{j=lo+1}^{hi} Pr(y >= j),
  # with lo, hi the 0.0001 and 0.9999 quantiles of y at exposure e (see README_offset.md).
  expect_y = function(mu, e){
    hi = ceiling(round_floor(g_inv(qnorm(0.9999, mean = mu, sd = params$sigma))*e, y_max = y_max))
    hi = pmin(hi, ceiling(2*max(y/offsets)*e)) # cap, as in the original (computing time)
    lo = pmin(round_floor(g_inv(qnorm(0.0001, mean = mu, sd = params$sigma))*e, y_max = y_max), hi)
    len = hi - lo; id = rep(seq_along(mu), len); j = sequence(len, from = lo + 1)
    p_ge = pnorm(g(j/e[id]), mean = mu[id], sd = params$sigma, lower.tail = FALSE)
    cs = c(0, cumsum(p_ge)); ends = cumsum(len)
    lo + cs[ends + 1] - cs[ends - len + 1]
  }
  
  # Total number of MCMC simulations:
  nstot = nburn+(nskip+1)*(nsave)
  skipcount = 0; isave = 0 # For counting
  
  # Run the MCMC:
  if(verbose) timer0 = proc.time()[3] # for estimating the time remaining
  for(nsi in 1:nstot){
    
    #----------------------------------------------------------------------------
    # Block 1: sample the z_star
    z_star = countSTAR:::rtruncnormRcpp(y_lower = z_lower,
                                        y_upper = z_upper,
                                        mu = params$mu,
                                        sigma = rep(params$sigma, n),
                                        u_rand = runif(n = n))
    #----------------------------------------------------------------------------
    # Block 2: sample the conditional mean mu (+ any corresponding parameters)
    #   and the conditional SD sigma
    sampler$setResponse(z_star)
    samp = sampler$run(updateState = TRUE)
    params$mu = samp$train; params$sigma = samp$sigma
    #----------------------------------------------------------------------------
    # Block 3: sample lambda (transformation); a_y, a_yp1 already include the offsets
    if(transformation == 'box-cox'){
      lambda = countSTAR:::uni.slice(x0 = lambda,
                                     g = function(l_bc){
                                       countSTAR:::logLikeRcpp(g_a_j = g_bc(a_y, l_bc),
                                                               g_a_jp1 = g_bc(a_yp1, l_bc),
                                                               mu = params$mu,
                                                               sigma = rep(params$sigma, n)) +
                                         # This is the prior on lambda, truncated to [0, 3]
                                         dnorm(l_bc, mean = 1/2, sd = 1, log = TRUE)
                                     },
                                     w = 1/2, m = 50, lower = 0, upper = 3)
      
      # Update the transformation and inverse transformation function:
      g = function(t) g_bc(t, lambda = lambda)
      g_inv = function(s) g_inv_bc(s, lambda = lambda)
      
      # Update the lower and upper limits:
      z_lower = g(a_y); z_upper = g(a_yp1)
    }
    #----------------------------------------------------------------------------
    # Store the MCMC:
    if(nsi > nburn){
      
      # Increment the skip counter:
      skipcount = skipcount + 1
      
      # Save the iteration:
      if(skipcount > nskip){
        # Increment the save index
        isave = isave + 1
        
        # Posterior predictive distribution (OFFSETS: y = h(e * g^{-1}(z))):
        post.pred[isave,] = round_floor(g_inv(rnorm(n = n, mean = params$mu, sd = params$sigma))*offsets, y_max=y_max)
        
        # Conditional expectation (OFFSETS: at exposure e_i; see expect_y() above):
        if(save_y_hat) post.fitted.values[isave,] = expect_y(mu = params$mu, e = offsets)
        
        if(include_test){
          # Conditional of the z_star at test points (useful for predictive distribution later)
          post.mu.test[isave,] = samp$test
          
          # Posterior predictive distribution at test points (OFFSETS: at offsets_test):
          post.pred.test[isave,] = round_floor(g_inv(rnorm(n = n_test, mean = samp$test, sd = params$sigma))*offsets_test, y_max=y_max)
          # Conditional expectation at test points (OFFSETS: at offsets_test):
          post.fitted.values.test[isave,] = expect_y(mu = samp$test, e = offsets_test)
          
          # Test points for log-predictive score (OFFSETS: cut points a_j/e_i; y_max as in training):
          if(!is.null(y_test))
            post.log.pred.test[isave,] = countSTAR:::logLikePointRcpp(g_a_j = g(a_j(y_test, y_max = y_max)/offsets_test),
                                                                      g_a_jp1 = g(a_j(y_test + 1, y_max = y_max)/offsets_test),
                                                                      mu = samp$test,
                                                                      sigma = rep(params$sigma, n_test))
        }
        
        # Nonlinear parameter of Box-Cox transformation:
        if(transformation == 'box-cox') post.lambda[isave] = lambda
        
        # SD parameter:
        post.sigma[isave] = params$sigma
        
        # Conditional mean parameter:
        post.mu[isave,] = params$mu
        
        # Pointwise Log-likelihood (z_lower, z_upper already include the offsets):
        post.log.like.point[isave, ] = countSTAR:::logLikePointRcpp(g_a_j = z_lower,
                                                                    g_a_jp1 = z_upper,
                                                                    mu = params$mu,
                                                                    sigma = rep(params$sigma, n))
        
        # And reset the skip counter:
        skipcount = 0
      }
      
    }
    if(verbose) countSTAR:::computeTimeRemaining(nsi, timer0, nstot)
  }
  
  # Summarize computing time:
  if(verbose){
    tfinal = proc.time()[3] - timer00
    if(tfinal > 60){
      print(paste('Total time:', round(tfinal/60,1), 'minutes'))
    } else print(paste('Total time:', round(tfinal), 'seconds'))
  }
  
  # Compute WAIC:
  lppd = sum(log(colMeans(exp(post.log.like.point))))
  p_waic = sum(apply(post.log.like.point, 2, function(x) sd(x)^2))
  WAIC = -2*(lppd - p_waic)
  
  #Compute fitted values if necessary
  if(save_y_hat) fitted.values = colMeans(post.fitted.values) else fitted.values=NULL
  
  # Return a named list:
  result = list(post.pred = post.pred,  post.sigma = post.sigma, post.log.like.point = post.log.like.point,
                WAIC = WAIC, p_waic = p_waic,
                post.pred.test = post.pred.test, post.fitted.values.test = post.fitted.values.test,
                post.mu.test = post.mu.test, post.log.pred.test = post.log.pred.test,
                fitted.values = fitted.values, post.fitted.values = post.fitted.values)
  
  #Add lambda samples for box-cox transformation
  if(transformation=="box-cox"){
    result = c(result, list(post.lambda = post.lambda))
  }
  return(result)
}
