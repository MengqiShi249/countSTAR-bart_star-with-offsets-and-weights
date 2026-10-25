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
#' @param offsets \code{n x 1} vector (or scalar) of known offsets (exposures) \code{e > 0};
#' the model is \code{y = floor(y*)} with \code{g(y*/e)} modeled by BART, so the cut points
#' \code{[j, j+1)} become \code{[j/e, (j+1)/e)}. Default is 1 (no offsets).
#' 'ispline' is not available with offsets
#' @param offsets_test \code{n_test x 1} vector (or scalar) of known offsets for the test data;
#' default is 1
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
#' }
#' }
#'
#' @export
bart_star = function(y,
                     X,
                     X_test = NULL, y_test = NULL,
                     transformation = 'np',
                     y_max = Inf,
                     n.trees = 200,
                     sigest = NULL, sigdf = 3, sigquant = 0.90, k = 2.0, power = 2.0, base = 0.95,
                     nsave = 1000,
                     nburn = 1000,
                     nskip = 0,
                     save_y_hat = FALSE,
                     verbose = TRUE,
                     offsets = 1, offsets_test = 1){
  
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
  
  # Check: offsets must be positive (a scalar is recycled)
  if(length(offsets) == 1) offsets = rep(offsets, length(y))
  if(length(offsets) != length(y) || any(offsets <= 0))
    stop('offsets must be positive with length equal to length(y)')
  
  #If transformation is ispline, call the appropriate function and return the results
  if(transformation=="ispline"){
    if(any(offsets != 1) || any(offsets_test != 1))
      stop("transformation = 'ispline' is not available with offsets")
    .args = as.list(match.call())[-1]
    .args[['transformation']] <- NULL
    .args[['offsets']] <- NULL; .args[['offsets_test']] <- NULL
    return(do.call(bart_star_ispline, .args))
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
    
    # Transformation function (offsets: for 'np', g(t) = qnorm(n/(n+1)*mean(y/e < t)) at the
    # observed t = y/e > 0, which equals g_cdf(y, 'np') when e = 1, since F_y(t - 1) = mean(y < t)):
    if(transformation == 'np'){
      d = sort(unique(y/offsets)); t0 = d[d != 0]
      g0 = qnorm(n/(n+1)*ecdf(y/offsets)(c(-Inf, d)[match(t0, d)]))
      g = splinefun(t0[is.finite(g0)], g0[is.finite(g0)], method = 'monoH.FC')
    } else g = g_cdf(y = y, distribution = transformation)
    
    # Define the grid for approximations using equally-spaced + quantile points:
    t_grid = sort(unique(round(c(
      seq(0, min(2*max(y/offsets), y_max/min(offsets)), length.out = 250),
      quantile(unique((y[y < y_max] + 1)/offsets[y < y_max]), seq(0, 1, length.out = 250))), 8)))
    
    # Inverse transformation function:
    g_inv = g_inv_approx(g = g, t_grid = t_grid)
    
    # No Box-Cox transformation:
    lambda = NULL
  }
  
  # Random initialization for z_star (offsets: on the y/e scale):
  z_star = g((y + abs(rnorm(n = n)))/offsets)
  
  # Now initialize the model: BART!
  
  # Include a test dataset:
  include_test = !is.null(X_test)
  if(include_test){
    n_test = nrow(X_test) # Size of test dataset
    
    # Check: test offsets must be positive (a scalar is recycled)
    if(length(offsets_test) == 1) offsets_test = rep(offsets_test, n_test)
    if(length(offsets_test) != n_test || any(offsets_test <= 0))
      stop('offsets_test must be positive with length equal to nrow(X_test)')
  }
  
  # Initialize the dbarts() object:
  control = dbarts::dbartsControl(n.chains = 1, n.burn = 0, n.samples = 1,
                                  n.trees = n.trees)
  # Initialize the standard deviation:
  if(is.null(sigest)){
    if(transformation == 'box-cox' && all(offsets == 1)){ # (the pilot MCMC does not take offsets)
      # Lambda is unknown, so use pilot MCMC with mean-only model to identify sigma estimate:
      fit0 = genMCMC_star(y = y,
                          sample_params = sample_params_mean,
                          init_params = init_params_mean,
                          transformation = 'box-cox', nburn = 1000, nsave = 100, verbose = FALSE)
      sigest = median(fit0$post.sigma)
    } else {
      # Transformation is known, so simply transform the raw data and estimate the SD
      # (offsets: on the y/e scale; for 'box-cox' with offsets, at the initial lambda):
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
  
  # Lower and upper intervals (offsets: [j/e, (j+1)/e)):
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
  
  # Total number of MCMC simulations:
  nstot = nburn+(nskip+1)*(nsave)
  skipcount = 0; isave = 0 # For counting
  
  # Run the MCMC:
  if(verbose) timer0 = proc.time()[3] # for estimating the time remaining
  for(nsi in 1:nstot){
    
    #----------------------------------------------------------------------------
    # Block 1: sample the z_star
    z_star = rtruncnormRcpp(y_lower = z_lower,
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
    # Block 3: sample lambda (transformation)
    if(transformation == 'box-cox'){
      lambda = uni.slice(x0 = lambda,
                         g = function(l_bc){
                           logLikeRcpp(g_a_j = g_bc(a_y, l_bc),
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
        
        # Posterior predictive distribution (offsets: y = floor(e * g^{-1}(z))):
        post.pred[isave,] = round_floor(g_inv(rnorm(n = n, mean = params$mu, sd = params$sigma))*offsets, y_max=y_max)
        
        # Conditional expectation (offsets: at e_i, with cut points a_j/e_i):
        if(save_y_hat){
          Jmax = ceiling(round_floor(g_inv(
            qnorm(0.9999, mean = params$mu, sd = params$sigma))*offsets, y_max=y_max))
          Jmax[Jmax > 2*max(y)] = 2*max(y) # To avoid excessive computation times, cap at 2*max(y)
          Jmaxmax = max(Jmax)
          for(i in 1:n) # offsets: the cut points a_j/e_i differ by observation
            post.fitted.values[isave, i] = expectation_gRcpp(g_a_j = g(a_j(0:Jmaxmax)/offsets[i]),
                                                             g_a_jp1 = g(a_j(1:(Jmaxmax + 1))/offsets[i]),
                                                             mu = params$mu[i], sigma = params$sigma,
                                                             Jmax = Jmax[i])
        }
        
        if(include_test){
          # Conditional of the z_star at test points (useful for predictive distribution later)
          post.mu.test[isave,] = samp$test
          
          # Posterior predictive distribution at test points (offsets: at offsets_test):
          post.pred.test[isave,] = round_floor(g_inv(rnorm(n = n_test, mean = samp$test, sd = params$sigma))*offsets_test, y_max=y_max)
          # Conditional expectation at test points (offsets: at offsets_test, with cut points a_j/e_i):
          Jmax = ceiling(round_floor(g_inv(
            qnorm(0.9999, mean = samp$test, sd = params$sigma))*offsets_test, y_max=y_max))
          Jmax[Jmax > 2*max(y)] = 2*max(y) # To avoid excessive computation times, cap at 2*max(y)
          Jmaxmax = max(Jmax)
          for(i in 1:n_test) # offsets: the cut points a_j/e_i differ by observation
            post.fitted.values.test[isave, i] = expectation_gRcpp(g_a_j = g(a_j(0:Jmaxmax)/offsets_test[i]),
                                                                  g_a_jp1 = g(a_j(1:(Jmaxmax + 1))/offsets_test[i]),
                                                                  mu = samp$test[i], sigma = params$sigma,
                                                                  Jmax = Jmax[i])
          
          # Test points for log-predictive score (offsets: cut points [j/e, (j+1)/e)):
          if(!is.null(y_test))
            post.log.pred.test[isave,] = logLikePointRcpp(g_a_j = g(a_j(y_test)/offsets_test),
                                                          g_a_jp1 = g(a_j(y_test + 1)/offsets_test),
                                                          mu = samp$test,
                                                          sigma = rep(params$sigma, n_test))
        }
        
        # Nonlinear parameter of Box-Cox transformation:
        if(transformation == 'box-cox') post.lambda[isave] = lambda
        
        # SD parameter:
        post.sigma[isave] = params$sigma
        
        # Conditional mean parameter:
        post.mu[isave,] = params$mu
        
        # Pointwise Log-likelihood:
        post.log.like.point[isave, ] = logLikePointRcpp(g_a_j = z_lower,
                                                        g_a_jp1 = z_upper,
                                                        mu = params$mu,
                                                        sigma = rep(params$sigma, n))
        
        # And reset the skip counter:
        skipcount = 0
      }
      
    }
    if(verbose) computeTimeRemaining(nsi, timer0, nstot)
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
