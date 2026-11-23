#----------------------------------------------------------------------------
#' Simulate offset count data from Friedman's nonlinear function
#'
#' Nonlinear covariate effect, NB counts, multiplicative offset (baseline
#' scenario; see \code{simulate_nb_friedman()}).
#'
#' @param n,p,r_nb,b_int,b_sig,sigma_true as in \code{simulate_nb_friedman()}
#' @param offset_range \code{c(min, max)} for a \code{Uniform} offset
#' @param offset optional offset vector/scalar (overrides \code{offset_range})
#' @param seed optional random seed
#' @return list(y, X, Ey, Ey_obs, offset)
#' @export
simulate_nb_friedman_offset = function(n = 100, p = 10, r_nb = 1,
                                       b_int = log(1.5), b_sig = log(5.0),
                                       sigma_true = sqrt(2*log(1.0)),
                                       offset_range = c(0.5, 5), offset = NULL,
                                       seed = NULL){
  if(!is.null(seed)) set.seed(seed)
  if(p < 5) stop('p >= 5 required')
  
  f = function(x) 10*sin(pi*x[,1]*x[,2]) + 20*(x[,3]-.5)^2 + 10*x[,4] + 5*x[,5]
  X = matrix(runif(n*p), nrow = n, ncol = p)
  
  if(is.null(offset)) offset = runif(n, offset_range[1], offset_range[2])
  if(length(offset) == 1) offset = rep(offset, n)
  if(any(offset <= 0)) stop('offset must be strictly positive')
  
  z_star = b_int + b_sig*scale(f(X)) + sigma_true*rnorm(n)
  y = rnbinom(n, size = r_nb, prob = r_nb/(r_nb + offset*exp(z_star)))
  
  Ey = as.numeric(exp(b_int + b_sig*scale(f(X)))*exp(sigma_true^2/2))
  list(y = y, X = X, Ey = Ey, Ey_obs = as.numeric(offset*Ey), offset = offset)
}

#----------------------------------------------------------------------------
#' Simulate imbalanced (sparse, zero-heavy) offset count data
#'
#' A rare-species-like scenario: a low baseline rate everywhere, with a
#' smooth (logistic) rise to a much higher mean once \code{x[,1]} exceeds a
#' threshold, plus strong NB overdispersion. Produces mostly zeros with
#' occasional large counts -- imbalanced counts as in sparse species
#' detection data -- while keeping the true mean continuous in \code{x[,1]}
#' (a hard step function would make \code{Ey} take only two values).
#'
#' @param n,p number of observations/predictors (only \code{x[,1]} matters)
#' @param r_nb NB dispersion (small = heavy overdispersion)
#' @param hotspot_thresh midpoint of the transition on \code{x[,1]}
#' @param steepness how sharply the transition happens (larger = sharper)
#' @param b_lo,b_hi log-mean well below/above the transition
#' @param offset_range,offset,seed as in \code{simulate_nb_friedman_offset()}
#' @return list(y, X, Ey, Ey_obs, offset)
#' @export
simulate_imbalanced_offset = function(n = 100, p = 5, r_nb = 0.7,
                                      hotspot_thresh = 0.7, steepness = 15,
                                      b_lo = log(0.2), b_hi = log(4),
                                      offset_range = c(0.5, 5), offset = NULL,
                                      seed = NULL){
  if(!is.null(seed)) set.seed(seed)
  if(p < 1) stop('p >= 1 required')
  
  X = matrix(runif(n*p), nrow = n, ncol = p)
  
  if(is.null(offset)) offset = runif(n, offset_range[1], offset_range[2])
  if(length(offset) == 1) offset = rep(offset, n)
  if(any(offset <= 0)) stop('offset must be strictly positive')
  
  # Smooth (logistic) transition instead of a hard step, so Ey is continuous:
  w = plogis(steepness*(X[,1] - hotspot_thresh))
  log_mean = b_lo + (b_hi - b_lo)*w
  y = rnbinom(n, size = r_nb, prob = r_nb/(r_nb + offset*exp(log_mean)))
  
  Ey = as.numeric(exp(log_mean))
  list(y = y, X = X, Ey = Ey, Ey_obs = as.numeric(offset*Ey), offset = offset)
}

#----------------------------------------------------------------------------
#' Simulate offset count data with heterogeneous (group-varying) dispersion
#'
#' A simple linear mean in one covariate, with NB dispersion that differs
#' across two groups defined by a second covariate -- e.g. habitat types
#' with different detectability/noise, on top of the usual survey-effort
#' offset.
#'
#' @param n,p number of observations/predictors (\code{x[,1]} drives the
#' mean, \code{x[,2]} defines the group)
#' @param b_int,b_sig intercept/slope for the linear mean in \code{x[,1]}
#' @param r_lo,r_hi NB dispersion in the noisy/clean group
#' @param group_thresh threshold on \code{x[,2]} defining the group
#' @param offset_range,offset,seed as in \code{simulate_nb_friedman_offset()}
#' @return list(y, X, Ey, Ey_obs, offset)
#' @export
simulate_heterogeneous_offset = function(n = 100, p = 5,
                                         b_int = log(1.5), b_sig = log(3),
                                         r_lo = 0.7, r_hi = 8,
                                         group_thresh = 0.5,
                                         offset_range = c(0.5, 5), offset = NULL,
                                         seed = NULL){
  if(!is.null(seed)) set.seed(seed)
  if(p < 2) stop('p >= 2 required')
  
  X = matrix(runif(n*p), nrow = n, ncol = p)
  group = X[,2] > group_thresh
  r_nb = ifelse(group, r_hi, r_lo)
  
  if(is.null(offset)) offset = runif(n, offset_range[1], offset_range[2])
  if(length(offset) == 1) offset = rep(offset, n)
  if(any(offset <= 0)) stop('offset must be strictly positive')
  
  log_mean = b_int + b_sig*scale(X[,1])
  y = rnbinom(n, size = r_nb, prob = r_nb/(r_nb + offset*exp(log_mean)))
  
  Ey = as.numeric(exp(log_mean))
  list(y = y, X = X, Ey = Ey, Ey_obs = as.numeric(offset*Ey), offset = offset)
}
