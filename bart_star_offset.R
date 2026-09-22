#----------------------------------------------------------------------------
#' Attach a known offset/exposure to a count vector
#'
#' Stores \code{offset} as an attribute on \code{y}, so it can be used later
#' via \code{bart_star(y = y, ..., offset = TRUE)} rather than passing the
#' vector again explicitly.
#'
#' @param y \code{n x 1} vector of counts
#' @param offset \code{n x 1} vector (or scalar) of offsets/exposures, e_i > 0
#' @return \code{y} with an \code{"offset"} attribute attached
#' @export
set_offset = function(y, offset){
  if(length(offset) == 1) offset = rep(offset, length(y))
  if(length(offset) != length(y)) stop('offset must have length equal to length(y)')
  if(any(offset <= 0)) stop('offset must be strictly positive')
  attr(y, 'offset') = offset
  y
}
#----------------------------------------------------------------------------
#' CDF-based transformation, adjusted for a known offset/exposure term
#'
#' Offset-adjusted version of \code{g_cdf()}, needed because \code{g_cdf()}
#' assumes \code{a_j = j} is the same for every observation:
#' \itemize{
#' \item \strong{Relocated support.} Each height \code{qnorm(F_Y(j-1))}
#' (pooled CDF of the raw counts, as in \code{g_cdf()}) is placed at
#' \code{t0 = j/e_i} instead of \code{t0 = j}, via \code{F_W(j/e_i) = F_Y(j-1)}.
#' \item \strong{Monotonicity.} Heterogeneous \code{e_i} can make the pooled
#' \code{(t0, height)} pairs locally non-monotone in \code{t0} even though
#' each pair is exact; isotonic regression restores monotonicity, as
#' required by \code{method = 'monoH.FC'}.
#' \item \strong{Bounded extrapolation.} Out-of-range queries are clamped
#' to the boundary rather than extrapolated, since heterogeneous offsets can
#' push the unbounded cubic-Hermite extrapolation to extreme values.
#' }
#'
#' @param y \code{n x 1} vector of observed counts
#' @param offset \code{n x 1} vector of offsets/exposures, e_i > 0
#' @param distribution "np" (empirical CDF), "pois", or "neg-bin"
#' @return A smooth monotone function for evaluation at offset-adjusted
#' arguments, e.g. \code{g(a_j(y)/offset)}.
#'
#' @export
g_cdf_offset = function(y, offset = rep(1, length(y)), distribution = "np") {
  
  distribution = tolower(distribution);
  if(!is.element(distribution, c("np", "pois", "neg-bin")))
    stop("The distribution must be one of 'np', 'pois', or 'neg-bin'")
  
  n = length(y)
  mu_y = mean(y); sigma_y = sd(y)
  
  if(distribution == 'np') {
    F_y = function(t) n/(n+1)*ecdf(y)(t)
  }
  if(distribution == 'pois'){
    F_y = function(t) ppois(t, lambda = mu_y)
  }
  if(distribution == 'neg-bin') {
    if(mu_y >= sigma_y^2){
      warning("'neg-bin' not recommended for underdispersed data")
      sigma_y = 1.1*sqrt(abs(mu_y))
    }
    F_y = function(t) pnbinom(t, size = mu_y^2/(sigma_y^2 - mu_y), prob = mu_y/sigma_y^2)
  }
  
  keep = which(y != 0)
  
  # Location y/e (offset scale); height still qnorm(F_y(y-1)) at the raw count.
  t0 = y[keep] / offset[keep]
  g0 = qnorm(F_y(y[keep] - 1))
  
  t0 = t0[which(is.finite(g0))]; g0 = g0[which(is.finite(g0))]
  
  # Collapse (near-)duplicate locations, averaging colliding heights.
  ord = order(t0)
  t0 = round(t0[ord], 8); g0 = g0[ord]
  g0 = ave(g0, t0)
  dup = duplicated(t0)
  t0 = t0[!dup]; g0 = g0[!dup]
  
  # Enforce monotonicity in t0-order (heterogeneous e_i can break it locally).
  g0 = isoreg(g0)$yf
  
  # Clamp finite out-of-range queries (flat extrapolation); +-Inf pass through untouched.
  g_spline = splinefun(t0, g0, method = 'monoH.FC')
  t0_min = min(t0); t0_max = max(t0)
  function(t){
    out = numeric(length(t))
    finite = is.finite(t)
    out[!finite] = t[!finite]
    out[finite]  = g_spline(pmin(pmax(t[finite], t0_min), t0_max))
    out
  }
}
#----------------------------------------------------------------------------
#' Posterior conditional expectation of y, adjusted for a known offset
#'
#' Offset-adjusted replacement for \code{expectation_gRcpp()}, which assumes a
#' single cut-point grid \code{g(a_j(0:Jmax))} shared across observations
#' (only valid when \code{a_j} doesn't depend on \code{i}). With an offset,
#' cut points \code{a_j(e_i) = j/e_i} are observation-specific, so they're
#' built as an \code{n x (Jmax+1)} matrix. Uses the tail-sum identity
#' \code{E[Y_i] = sum_{j=0}^{Jmax_i} Pr(Y_i > j)}.
#'
#' @param g the current transformation function
#' @param mu,sigma \code{n x 1} vectors of conditional mean/SD (z-scale)
#' @param offset \code{n x 1} vector of offsets/exposures, e_i > 0
#' @param Jmax \code{n x 1} vector of per-observation truncation points
#' @return length-\code{n} vector of \code{E[y_i | mu_i, sigma_i, e_i]}.
#'
#' @export
expectation_g_offset = function(g, mu, sigma, offset, Jmax){
  
  # dbarts can return an n x 1 matrix rather than a vector; strip dim so
  # downstream comparisons/matrix ops don't fail.
  mu = as.vector(mu); sigma = as.vector(sigma)
  offset = as.vector(offset); Jmax = as.vector(Jmax)
  
  n = length(mu)
  Jmaxmax = max(Jmax)
  
  jp1 = matrix(1:(Jmaxmax + 1), nrow = n, ncol = Jmaxmax + 1, byrow = TRUE)
  a_over_e = jp1 / offset
  g_a_over_e = matrix(g(as.vector(a_over_e)), nrow = n, ncol = Jmaxmax + 1)
  
  tail_prob = 1 - pnorm(g_a_over_e, mean = mu, sd = sigma)
  tail_prob[!(jp1 <= (Jmax + 1))] = 0
  
  rowSums(tail_prob)
}
#----------------------------------------------------------------------------
#' Posterior and predictive inference for BART-STAR, with an optional offset
#'
#' \code{bart_star()}, extended with an optional known offset/exposure
#' \eqn{e_i > 0}: fits \eqn{y_i = \lfloor y^*_i \rfloor},
#' \eqn{z_i = g(y^*_i / e_i)}, \eqn{z_i = \mu(x_i) + \varepsilon_i}. Leaving
#' \code{offset = FALSE} (the default) reproduces the original model.
#'
#' @param y \code{n x 1} vector of counts (nonnegative integers)
#' @param X \code{n x p} matrix of predictors
#' @param X_test \code{n_test x p} matrix of predictors for test data
#' @param y_test \code{n_test x 1} vector of test-set counts (optional)
#' @param offset \code{FALSE} (default): no offset. \code{TRUE}: use the
#' offset attribute attached to \code{y} via \code{set_offset(y, e)}. Or an
#' \code{n x 1} vector (or scalar) of offsets/exposures \code{e_i > 0},
#' supplied directly.
#' @param offset_test \code{n_test x 1} vector (or scalar) of test offsets;
#' use \code{1} for exposure-free (density) predictions; ignored unless an
#' offset is in use
#' @param transformation one of 'identity', 'log', 'sqrt', 'np', 'pois',
#' 'neg-bin', 'box-cox', 'ispline' ('ispline' does not support \code{offset})
#' @param y_max a fixed and known upper bound for \code{y} (and \code{y_test})
#' @param n.trees number of trees in the BART sum-of-trees model
#' @param sigest,sigdf,sigquant,k,power,base BART/\code{dbarts} hyperparameters
#' @param nsave number of MCMC iterations to save
#' @param nburn number of MCMC iterations to discard as burn-in
#' @param nskip number of MCMC iterations to skip between saved samples
#' @param save_y_hat logical; store posterior draws of the conditional
#' expectation of \code{y}?
#' @param verbose logical; print progress?
#'
#' @return a list as in the original \code{bart_star()}, plus \code{offset}
#' and \code{offset_test} when an offset is used.
#'
#' @details
#' Changes relative to the original \code{bart_star()}, all gated on
#' \code{offset} being supplied:
#' \itemize{
#' \item Cut points \code{a_j = j} become \code{a_j(e_i) = j/e_i}, computed
#' once as \code{a_y}/\code{a_yp1} and reused throughout (initialization,
#' the box-cox slice sampler, pointwise log-likelihoods); no per-iteration
#' cost added except for 'box-cox', which already recomputed these every
#' iteration.
#' \item \code{z_star} and \code{sigest} are initialized from \code{y/offset}.
#' \item Predictive draws re-introduce the offset after inverting \code{g}:
#' \code{y_tilde = floor(g_inv(z_tilde) * e)}, for training and test draws.
#' \item For \code{c('np','pois','neg-bin')}, \code{g} is estimated via
#' \code{g_cdf_offset()} instead of \code{g_cdf()}.
#' \item \code{expectation_g_offset()} replaces \code{expectation_gRcpp()}
#' whenever the offset is non-constant (the original, faster routine is
#' still used when it's constant, including all-1s).
#' \item Numerical guarantee: \code{g_cdf_offset()}'s flat CDF regions can
#' give some observation's \code{z_lower}/\code{z_upper} a zero or
#' (floating-point noise) negative-width gap, which can push
#' \code{rtruncnormRcpp()}'s \code{qnorm()} outside \code{[0,1]} and produce
#' a \code{NaN} draw that corrupts the whole joint \code{dbarts} fit.
#' \code{z_upper} is floored at \code{z_lower + 1e-6} wherever computed;
#' pointwise log-likelihoods are additionally floored at \code{-745} so an
#' exact zero-width gap can't turn WAIC into \code{NaN}.
#' }
#' \code{offset = FALSE} runs the same code as the original \code{bart_star()}
#' (including its box-cox pilot-MCMC \code{sigest}); \code{offset = TRUE} or
#' a supplied vector (e.g. \code{rep(1, n)}) instead routes through the code
#' above with the given \code{e_i}, which is close to but not always
#' bit-for-bit identical to the \code{offset = FALSE} result (exactly
#' identical for 'identity'/'log'/'sqrt'; very close for 'box-cox', from the
#' simpler \code{sigest} anchor used above; close for
#' 'np'/'pois'/'neg-bin', from \code{g_cdf_offset()}'s boundary clamping).
#'
#' \code{transformation = 'ispline'} is not supported with an offset: its
#' sampler looks up cut points on an integer grid (exact \code{match()}),
#' which assumes \code{a_j = j}; under an offset \code{a_j(e_i) = j/e_i} is
#' continuous, so the grid and lookup would need to be rebuilt as in
#' \code{g_cdf_offset()}/\code{g_inv_approx()} (nearest-point or
#' interpolated lookup on a dense grid) throughout its sampler -- not
#' attempted here.
#'
#' @export
bart_star = function(y,
                     X,
                     X_test = NULL, y_test = NULL,
                     offset = FALSE, offset_test = NULL,
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
  
  # NEW: resolve offset. FALSE (default) or NULL: no offset. TRUE: pull the
  # offset vector attached to y via set_offset(). A numeric vector: used directly.
  if(isTRUE(offset)){
    offset = attr(y, 'offset')
    if(is.null(offset)) stop("offset = TRUE requires y to carry an offset attribute; set it first with set_offset(y, e).")
  }
  has_offset = !is.null(offset) && !isFALSE(offset)
  
  #If transformation is ispline, call the appropriate function and return the results
  if(transformation=="ispline"){
    if(has_offset) stop("transformation = 'ispline' does not support offset; see @details.")
    .args = as.list(match.call())[-1]
    .args[['transformation']] <- NULL
    .args[['offset']] <- NULL; .args[['offset_test']] <- NULL
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
  
  # NEW: offset checks. FALSE/NULL (default) reproduces bart_star() unmodified.
  if(has_offset){
    if(length(offset) == 1) offset = rep(offset, n)
    if(length(offset) != n) stop('offset must have length equal to length(y)')
    if(any(offset <= 0)) stop('offset must be strictly positive')
  } else offset = rep(1, n)
  
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
    
    # CHANGED: g_cdf_offset() in place of g_cdf() (see its documentation).
    g = g_cdf_offset(y = y, offset = offset, distribution = transformation)
    
    # CHANGED: grid spans y/offset (density scale), the domain of g's argument.
    w = y / offset
    t_grid = sort(unique(round(c(
      seq(0, min(2*max(w), y_max), length.out = 250),
      quantile(unique(w[y < y_max] + 1/max(offset)), seq(0, 1, length.out = 250))), 8)))
    
    # Inverse transformation function:
    g_inv = g_inv_approx(g = g, t_grid = t_grid)
    
    # No Box-Cox transformation:
    lambda = NULL
  }
  
  # CHANGED: initialize z_star on the offset-adjusted scale (reduces to the
  # original when offset = 1).
  z_star = g((y + abs(rnorm(n = n))) / offset)
  
  # Now initialize the model: BART!
  
  # Include a test dataset:
  include_test = !is.null(X_test)
  if(include_test) n_test = nrow(X_test) # Size of test dataset
  
  # NEW: offset_test checks
  if(include_test){
    if(has_offset){
      if(is.null(offset_test)) offset_test = rep(1, n_test)
      if(length(offset_test) == 1) offset_test = rep(offset_test, n_test)
      if(length(offset_test) != n_test) stop('offset_test must have length equal to nrow(X_test)')
      if(any(offset_test <= 0)) stop('offset_test must be strictly positive')
    } else offset_test = rep(1, n_test)
  }
  
  # Initialize the dbarts() object:
  control = dbarts::dbartsControl(n.chains = 1, n.burn = 0, n.samples = 1,
                                  n.trees = n.trees)
  # Initialize the standard deviation:
  if(is.null(sigest)){
    if(transformation == 'box-cox' && !has_offset){
      # Original pilot MCMC (no offset support -- see @details):
      fit0 = genMCMC_star(y = y,
                          sample_params = countSTAR:::sample_params_mean,
                          init_params = countSTAR:::init_params_mean,
                          transformation = 'box-cox', nburn = 1000, nsave = 100, verbose = FALSE)
      sigest = median(fit0$post.sigma)
    } else {
      # CHANGED: offset-adjusted anchor (reduces to the original sd(g(y+1))
      # when offset = 1); used for box-cox too when an offset is supplied.
      sigest = sd(g((y + 1)/offset), na.rm=TRUE)
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
  
  # CHANGED: cut points offset-adjusted (a_j(e_i) = j/e_i); reduces to the
  # original a_j(y), a_j(y+1) when offset = 1. Computed once; box-cox reuses
  # these every iteration.
  a_y = a_j(y, y_max = y_max) / offset; a_yp1 = a_j(y + 1, y_max = y_max) / offset
  z_lower = g(a_y); z_upper = g(a_yp1)
  # NEW: enforce a strictly positive gap -- see @details.
  if(has_offset) z_upper = pmax(z_upper, z_lower + 1e-6)
  
  # Store MCMC output:
  if(save_y_hat) post.fitted.values = array(NA, c(nsave, n)) else post.fitted.values = NULL
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
    # Block 1: sample the z_star (unchanged; bounds already offset-adjusted)
    z_star = countSTAR:::rtruncnormRcpp(y_lower = z_lower,
                                        y_upper = z_upper,
                                        mu = params$mu,
                                        sigma = rep(params$sigma, n),
                                        u_rand = runif(n = n))
    #----------------------------------------------------------------------------
    # Block 2: sample the conditional mean mu (+ any corresponding parameters)
    #   and the conditional SD sigma -- unchanged
    sampler$setResponse(z_star)
    samp = sampler$run(updateState = TRUE)
    params$mu = samp$train; params$sigma = samp$sigma
    #----------------------------------------------------------------------------
    # Block 3: sample lambda (transformation). Unchanged: a_y/a_yp1 already
    # carry the offset, so the likelihood is automatically offset-adjusted.
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
      if(has_offset) z_upper = pmax(z_upper, z_lower + 1e-6)
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
        
        # CHANGED: re-introduce the offset AFTER inverting g (reduces to the
        # original when offset = 1):
        post.pred[isave,] = round_floor(g_inv(rnorm(n = n, mean = params$mu, sd = params$sigma)) * offset, y_max=y_max)
        
        # Conditional expectation:
        if(save_y_hat){
          Jmax = ceiling(round_floor(g_inv(
            qnorm(0.9999, mean = params$mu, sd = params$sigma)) * offset, y_max=y_max))
          Jmax[Jmax > 2*max(y)] = 2*max(y) # To avoid excessive computation times, cap at 2*max(y)
          
          # CHANGED: expectation_gRcpp() assumes a shared cut-point grid,
          # invalid once offset is non-constant.
          if(all(offset == 1)){
            Jmaxmax = max(Jmax)
            post.fitted.values[isave,] = countSTAR:::expectation_gRcpp(g_a_j = g(a_j(0:Jmaxmax)),
                                                                       g_a_jp1 = g(a_j(1:(Jmaxmax + 1))),
                                                                       mu = params$mu, sigma = rep(params$sigma, n),
                                                                       Jmax = Jmax)
          } else {
            post.fitted.values[isave,] = expectation_g_offset(g = g,
                                                              mu = params$mu, sigma = rep(params$sigma, n),
                                                              offset = offset, Jmax = Jmax)
          }
        }
        
        if(include_test){
          # Conditional of the z_star at test points (unchanged)
          post.mu.test[isave,] = samp$test
          
          # CHANGED: re-introduce offset_test AFTER inverting g:
          post.pred.test[isave,] = round_floor(g_inv(rnorm(n = n_test, mean = samp$test, sd = params$sigma)) * offset_test, y_max=y_max)
          
          # Conditional expectation at test points:
          Jmax = ceiling(round_floor(g_inv(
            qnorm(0.9999, mean = samp$test, sd = params$sigma)) * offset_test, y_max=y_max))
          Jmax[Jmax > 2*max(y)] = 2*max(y)
          
          if(all(offset_test == 1)){
            Jmaxmax = max(Jmax)
            post.fitted.values.test[isave,] = countSTAR:::expectation_gRcpp(g_a_j = g(a_j(0:Jmaxmax)),
                                                                            g_a_jp1 = g(a_j(1:(Jmaxmax + 1))),
                                                                            mu = samp$test, sigma = rep(params$sigma, n_test),
                                                                            Jmax = Jmax)
          } else {
            post.fitted.values.test[isave,] = expectation_g_offset(g = g,
                                                                   mu = samp$test, sigma = rep(params$sigma, n_test),
                                                                   offset = offset_test, Jmax = Jmax)
          }
          
          # CHANGED: offset-adjusted cut points for the test log-predictive
          # score, own gap enforcement + floor (see @details):
          if(!is.null(y_test)){
            g_a_j_test  = g(a_j(y_test, y_max = y_max)/offset_test)
            g_a_jp1_test = g(a_j(y_test + 1, y_max = y_max)/offset_test)
            if(has_offset) g_a_jp1_test = pmax(g_a_jp1_test, g_a_j_test + 1e-6)
            post.log.pred.test[isave,] = countSTAR:::logLikePointRcpp(g_a_j = g_a_j_test,
                                                                      g_a_jp1 = g_a_jp1_test,
                                                                      mu = samp$test,
                                                                      sigma = rep(params$sigma, n_test))
            if(has_offset) post.log.pred.test[isave,] = pmax(post.log.pred.test[isave,], -745)
          }
        }
        
        # Nonlinear parameter of Box-Cox transformation:
        if(transformation == 'box-cox') post.lambda[isave] = lambda
        
        # SD parameter:
        post.sigma[isave] = params$sigma
        
        # Conditional mean parameter:
        post.mu[isave,] = params$mu
        
        # Pointwise Log-likelihood (unchanged; z_lower/z_upper already
        # offset-adjusted). NEW: floor at -745 when offset is used (see @details).
        post.log.like.point[isave, ] = countSTAR:::logLikePointRcpp(g_a_j = z_lower,
                                                                    g_a_jp1 = z_upper,
                                                                    mu = params$mu,
                                                                    sigma = rep(params$sigma, n))
        if(has_offset) post.log.like.point[isave, ] = pmax(post.log.like.point[isave, ], -745)
        
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
  
  # NEW: return offset/offset_test when used.
  if(has_offset) result = c(result, list(offset = offset, offset_test = offset_test))
  
  #Add lambda samples for box-cox transformation
  if(transformation=="box-cox"){
    result = c(result, list(post.lambda = post.lambda))
  }
  return(result)
}
