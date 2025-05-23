#' Conditional Density Estimation in the \code{\{nadir\}} Package
#'
#' The following learners are available for conditional density estimation:
#' \itemize{
#'  \item \code{lnr_lm_density}
#'  \item \code{lnr_glm_density}
#'  \item \code{lnr_homoscedastic_density}
#' }
#'
#' There are a few important things to know about conditional density
#' estimation in the \code{nadir} package.
#'
#' Firstly, conditional density
#' learners must produce prediction functions that predict _densities_
#' at the new outcome values given the new covariates.
#'
#' Secondly, the implemented density estimators come in two flavors:
#' those with a strong assumption (that of conditional normality), and those
#' with much weaker assumptions.  The strong assumption is encoded
#' into learners like \code{lnr_lm_density} and \code{lnr_glm_density}
#' and says "after we model the predicted mean given covariates, we expect
#' the remaining errors to be normally distributed." The
#' more flexible learners produced by \code{lnr_homoskedastic_density}
#' are similar in spirit, except they fit a \code{stats::density} kernel
#' bandwidth smoother to the error distribution (after predicting the
#' conditional expected mean).
#'
#' A subpoint to the above point that's worth calling attention to is that
#' \code{lnr_homoskedastic_density} is a learner factory. That is to say,
#' given a \code{mean_lnr}, \code{lnr_homoskedastic_density} produces a
#' conditional density learner that uses that \code{mean_lnr}.
#'
#' Work is ongoing on implementing a \code{lnr_heteroskedastic_density}
#' learner that allows for predicting higher or lower variance in the
#' conditional density given covariates.
#'
#' Conditional density learners should be combined with the negative log loss
#' function when using \code{super_learner()} or using \code{compare_learners}.
#' Refer to the 2003 Dudoit and van der Laan paper for a starting place on the
#' appropriate loss functions to use for different types of outcomes.
#' <https://biostats.bepress.com/ucbbiostat/paper130/>
#'
#' @seealso learners binary_learners
#' @rdname density_learners
#' @name density_learners
#' @importFrom stats density dnorm model.frame model.matrix.default predict
#'   residuals var as.formula approx
#' @keywords density_learners
NULL


#' Conditional Normal Density Estimation Given Mean Predictors
#'
#' This is the simplest possible density estimator that is
#' entertainable.  It fits a \code{lm} model to the data, and
#' uses the variance of the residuals to parameterize a
#' model of the data as \eqn{\mathcal N(y | \beta x, \sigma^2)}.
#'
#' @returns a closure (function) that produces density estimates
#' at the \code{newdata} given according to the fit model.
#'
#' @inheritParams lnr_lm
#' @export
lnr_lm_density <- function(data, formula, weights = NULL, ...) {
  model_args <- list(data = data, formula = formula)
  if (! is.null(weights) & is.numeric(weights) & length(weights) == nrow(data)) {
    model_args$weights <- weights
  }
  model <- do.call(stats::lm, args = c(model_args, list(...)))
  residual_variance <- var(residuals(model))
  residual_sd <- sqrt(residual_variance)
  y_variable <- as.character(formula)[[2]]

  return(function(newdata) {
    predictions <- predict(model, newdata = newdata)
    dnorm(
      x = newdata[[y_variable]],
      mean = predictions,
      sd = residual_sd
    )
  })
}

attr(lnr_lm_density, 'sl_lnr_name') <- 'lm_density'
attr(lnr_lm_density, 'sl_lnr_type') <- 'density'


#' Conditional Normal Density Estimation Given Mean Predictors — with GLMs
#'
#' This is a step up from the \code{lnr_lm_density} in that it uses
#' a \code{glm} for the conditional mean model.
#' Note that this allows for specification of \code{glm} features
#' like \code{family = ...} in the \code{,..} arguments, and
#' that's the main advantage over the \code{lnr_lm_density}.
#' Also note that this still differs from using \code{lnr_homoskedastic_density}
#' with \code{mean_lnr = lnr_glm} because \code{lnr_homoscedastic_density}
#' uses \code{stats::density} to do kernel bandwidth smoothing
#' on the error distribution of the mean predictions..
#'
#' @returns a closure (function) that produces density estimates
#' at the \code{newdata} given according to the fit model.
#'
#' @inheritParams lnr_lm
#' @export
lnr_glm_density <- function(data, formula, weights, ...) {
  model_args <- list(data = data, formula = formula)
  if (! is.null(weights) & is.numeric(weights) & length(weights) == nrow(data)) {
    model_args$weights <- weights
  }
  model <- do.call(stats::glm, args = c(model_args, list(...)))
  residual_variance <- var(residuals(model))
  residual_sd <- sqrt(residual_variance)
  y_variable <- as.character(formula)[[2]]

  return(function(newdata) {
    predictions <- predict(model, newdata = newdata)
    dnorm(
      x = newdata[[y_variable]],
      mean = predictions,
      sd = residual_sd
    )
  })
}

attr(lnr_glm_density, 'sl_lnr_name') <- 'glm_density'
attr(lnr_glm_density, 'sl_lnr_type') <- 'density'


#' Conditional Density Estimation with Homoskedasticity Assumption
#'
#' This function accepting an \code{mean_lnr}, which it then trains on the data
#' and formula given. Then \code{stats::density} is fit to the error (difference
#' between observed outcome and the \code{mean_lnr} predictions).
#'
#' This returns a function that takes in \code{newdata} and produces density
#' estimates according to the estimated \code{stats::density} fit the error
#' from the \code{newdata} observed outcome and the prediction from the \code{mean_lnr}.
#'
#' That is to say, this follows the following procedure (assuming \eqn{Y} as the outcome
#' and \eqn{X} as a matrix of predictors):
#'
#' \deqn{\texttt{obtain } \hat{\mathbb E}(Y | X) \quad \mathtt{using \quad mean\_learner}}
#' \deqn{\texttt{fit } \hat{f} \gets \mathtt{density}(Y - \hat{\mathbb E}(Y | X))}
#' \deqn{\mathtt{return \quad  function(newdata) \{ } \hat{f}(\mathtt{newdata\$Y} -
#'   \hat{\mathbb E}[Y | \mathtt{newdata\$X}]) \} }
#'
#' @param mean_lnr should be a suitable \code{learner} (see \code{?learners}) that can take in
#' the \code{data} and \code{formula} given.
#' @returns A predictor function that takes in \code{newdata} and produces density
#' estimates
#'
#' @export
#' @examples
#' \dontrun{
#' # fit a conditional density model with mean model as a randomForest
#' fit_density_lnr <- lnr_homoskedastic_density(
#'   data = mtcars,
#'   formula = mpg ~ hp,
#'   mean_lnr = lnr_rf)
#'
#' # and what we should get back should be predicted densities at the
#' # observed mpg given the covariates hp
#' fit_density_lnr(mtcars)
#' }
#' @inheritParams lnr_lm
#' @param mean_lnr A learner (function) passed in to be trained on the data with
#'   the given formula and then used to predict conditional means for provided
#'   \code{newdata}.
#' @param mean_lnr_args Extra arguments to be passed to the \code{mean_lnr}
#' @param density_args Extra arguments to be passed to the kernel
#'   density smoother \code{stats::density}, especially things like
#'   \code{bw} for specifying the smoothing bandwidth. See \code{?stats::density}.
lnr_homoskedastic_density <- function(
    data,
    formula,
    mean_lnr,
    mean_lnr_args = NULL,
    density_args = NULL,
    weights = NULL) {

  # fit the mean_lnr — this is the conditional mean model
  mean_lnr_args <- c(list(data = data, formula = formula), mean_lnr_args)
  if (is.numeric(weights) & length(weights) == nrow(data)) {
    mean_lnr_args$weights <- weights
  }
  mean_predictor <- do.call(
    mean_lnr,
    args = mean_lnr_args)

  # determine the y_variable from the regression formula
  y_variable <- as.character(formula)[[2]]
  y_values <- data[[y_variable]]

  # calculate error in true Ys from the predictions from mean_lnr
  errors <- y_values - mean_predictor(data)

  # fit a kernel density
  density_model <- do.call(
    stats::density,
    args = c(list(errors), density_args))

  predictor <- function(newdata) {
    mean_predictions <- mean_predictor(newdata)
    errors_in_newdata_predictions <- newdata[[y_variable]] - mean_predictions
    predicted_densities <- approx(x = density_model$x, y = density_model$y, xout = errors_in_newdata_predictions, rule = 2)$y
    return(predicted_densities)
  }
  return(predictor)
}

attr(lnr_homoskedastic_density, 'sl_lnr_name') <- 'homoskedastic_density'
attr(lnr_homoskedastic_density, 'sl_lnr_type') <- 'density'


#' Conditional Density Estimation with Heteroskedasticity
#'
#'
#' TODO: The following code has a bug / statistical issue.
#' =======================================================
#'
#' I think there are bugs with this because performing a basic
#' test that if we fix the conditioning set (X) and integrate, integrating
#' a conditional probability density with X fixed should yield 1.
#'
#' In numerical tests, when the variance is scaled for, integrating conditional
#' densities seems to yield integration values exceeding 1 (sometimes by a lot).
#' I am pretty sure this poses a problem for optimizing negative log likelihood loss.
#'
#' Said numerical tests are displayed in the `Density-Estimation` article.
#' @export
#' @returns a closure (function) that produces density estimates
#' at the \code{newdata} given according to the fit model.
#' @inheritParams lnr_homoskedastic_density
#' @param var_lnr A learner (function) passed in to be trained on the squared
#'   error from the \code{mean_lnr} on the given data and then used to predict
#'   the expected variance for the density distribution of the outcome centered
#'   around the predicted conditional mean in the output.
#' @param var_lnr_args Extra arguments to be passed to the \code{var_lnr}
lnr_heteroskedastic_density <- function(data, formula,
                                       mean_lnr, var_lnr,
                                       mean_lnr_args = NULL,
                                       var_lnr_args = NULL,
                                       density_args = NULL) {

  # fit the mean_lnr
  mean_predictor <- do.call(
    mean_lnr,
    args = c(list(data, formula), mean_lnr_args))

  # determine the y_variable from the regression formula
  y_variable <- as.character(formula)[[2]]
  y_values <- data[[y_variable]]
  index_of_y_variable <- which(colnames(data) == y_variable)[[1]]

  # calculate error in true Ys from the predictions from mean_lnr
  errors <- y_values - mean_predictor(data)

  # calculate squared errors from the conditional mean predictor model
  errors_squared <- errors^2
  data$.errors_squared <- errors_squared
  var_formula <- as.formula(
    paste0(".errors_squared ~ ", as.character(formula)[[3]]))

  # train a predictor for the squared error
  var_predictor <- do.call(
    var_lnr,
    args = c(list(
      data = data[,-index_of_y_variable], # y needs to be not included here — this is predicting the squared error from the y ~ x model, so including both y and x makes it completely determined
      formula = var_formula),
      var_lnr_args))

  # fit density model
  density_model <- do.call(stats::density, args = c(list(errors), density_args))

  min_obs_error_squared <- 2 * min(data$.errors_squared)

  predictor <- function(newdata) {
    mean_predictions <- mean_predictor(newdata)
    errors <- newdata[[y_variable]] - mean_predictions
    var_predictions <- var_predictor(newdata)
    var_predictions[var_predictions < 0] <- min_obs_error_squared # should this be .Machine$double.eps ?
    sd_predictions <- sqrt(var_predictions)
    predicted_densities <- approx(density_model$x, density_model$y, errors / sd_predictions, rule = 2)$y / sd_predictions
    return(predicted_densities)
  }
  return(predictor)
}

attr(lnr_heteroskedastic_density, 'sl_lnr_name') <- 'homoskedastic_density'
attr(lnr_heteroskedastic_density, 'sl_lnr_type') <- 'density'


