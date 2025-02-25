#' Conditional Normal Density Estimation Given Mean Predictors
#'
#' This is the simplest possible density estimator that is
#' entertainable.  It fits a \code{lm} model to the data, and
#' uses the variance of the residuals to parameterize a
#' model of the data as \eqn{\mathcal N(y | \beta x, \sigma^2)}.
#'
#' @return a closure (function) that produces density estimates
#' at the \code{newdata} given according to the fit model.
#'
#' @export
lnr_lm_density <- function(data, regression_formula, ...) {
  model <- lm(data = data, formula = regression_formula, ...)
  residual_variance <- var(residuals(model))
  residual_sd <- sqrt(residual_variance)
  y_variable <- as.character(regression_formula)[[2]]

  return(function(newdata) {
    predictions <- predict(model, newdata = newdata)
    dnorm(
      x = newdata[[y_variable]],
      mean = predictions,
      sd = residual_sd
    )
  })
}

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
#' the \code{data} and \code{regression_formula} given.
#' @return A predictor function that takes in \code{newdata} and produces density
#' estimates
#'
#' @export
#' @examples
#' \dontrun{
#' # fit a conditional density model with mean model as a randomForest
#' fit_density_lnr <- lnr_homoskedastic_density(
#'   data = mtcars,
#'   regression_formula = mpg ~ hp,
#'   mean_lnr = lnr_rf)
#'
#' # and what we should get back should be predicted densities at the
#' # observed mpg given the covariates hp
#' fit_density_lnr(mtcars)
#' }
lnr_homoskedastic_density <- function(
    data, regression_formula, mean_lnr,
    extra_mean_lnr_args = NULL,
    extra_density_args = NULL) {

  # fit the mean_lnr
  mean_predictor <- do.call(
    mean_lnr,
    args = c(list(data, regression_formula), extra_mean_lnr_args))

  # determine the y_variable from the regression formula
  y_variable <- as.character(regression_formula)[[2]]
  y_values <- data[[y_variable]]

  # calculate error in true Ys from the predictions from mean_lnr
  errors <- y_values - mean_predictor(data)

  # fit a kernel density
  density_model <- do.call(
    stats::density,
    args = c(list(errors), extra_density_args))

  predictor <- function(newdata) {
    mean_predictions <- mean_predictor(newdata)
    errors <- newdata[[y_variable]] - mean_predictions
    predicted_densities <- approx(density_model$x, density_model$y, errors, rule = 2)$y
    return(predicted_densities)
  }
  return(predictor)
}

#' Conditional Density Estimation with Heteroskedasticity
#'
#'
lnr_heteroskedastic_density <- function(data, regression_formula,
                                       mean_lnr, var_lnr,
                                       mean_lnr_extra_arguments = NULL,
                                       var_lnr_extra_arguments = NULL) {

  mean_predictor <-
    do.call(what = mean_lnr,
            args = c(list(data, regression_formula), mean_lnr_extra_arguments))
  y_variable <- as.character(regression_formula)[[2]]
  y_values <- data[[y_variable]]
  errors <- y_values - mean_predictor(data)
  errors_squared <- errors^2
  var_training_data <- data
  var_training_data$.errors_squared <- errors_squared
  var_formula <- as.formula(
    paste0(".errors_squared ~ ", as.character(regression_formula)[[3]]))
  var_training_data[[y_variable]] <- NULL # so that regressions on ~ . aren't singular.
  var_predictor <- do.call(
    var_lnr,
    args = c(list(
      data = var_training_data,
      regression_formula = var_formula),
      var_lnr_extra_arguments))
  density_model <- stats::density(errors)
  min_obs_error <- 2 * min(var_training_data$.errors_squared)

  predictor <- function(newdata) {
    mean_predictions <- mean_predictor(newdata)
    errors <- newdata[[y_variable]] - mean_predictions
    var_predictions <- var_predictor(newdata)
    var_predictions[var_predictions < 0] <- min_obs_error
    sd_predictions <- sqrt(var_predictions)
    predicted_densities <- approx(density_model$x, density_model$y, errors / sd_predictions, rule = 2)$y / sd_predictions
    return(predicted_densities)
  }
  return(predictor)
}
