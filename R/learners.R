#' @export
lnr_mean <- function(data, regression_formula) {
  y_mean <- mean(data[[as.character(regression_formula[[2]])]])
  return(function(newdata) {
    rep(y_mean, nrow(newdata))
  })
}

#' @export
#' @importFrom ranger ranger
lnr_ranger <- function(data, regression_formula) {
  model <- ranger::ranger(data = data, formula = regression_formula)
  return(function(newdata) {
    predict(model, data = newdata)$predictions
  })
}

#' @export
#' @importFrom stats lm model.matrix predict
lnr_glmnet <- function(data, regression_formula, lambda = .2) {
  # glmnet takes Y and X separately, so we shall pull them out from the
  # data based on the regression_formula
  #
  # TODO: if there's a way to get the model.matrix without fitting an extra
  # lm, that would be great.
  yvar <- as.character(regression_formula[[2]])
  xdata <- model.matrix(lm(formula = regression_formula, data = data))
  model <- glmnet::glmnet(y = data[[yvar]], x = xdata, lambda = lambda)
  return(function(newdata) {
    xdata = model.matrix(lm(formula = regression_formula, data = newdata))
    as.vector(predict(model, newx = xdata, type = 'response'))
  })
}

#' @export
#' @importFrom randomForest randomForest
lnr_rf <- function(data, regression_formula, ...) {
  model <- randomForest::randomForest(formula = regression_formula, data = data, ...)
  return(function(newdata) {
    predict(model, newdata = newdata, type = 'response')
  })
}

#' @export
#' @importFrom stats lm
lnr_lm <- function(data, regression_formula, ...) {
  model <- stats::lm(formula = regression_formula, data = data, ...)

  return(function(newdata) {
    predict(model, newdata = newdata, type = 'response')
  })
}

#' @export
#' @importFrom stats glm
lnr_glm <- function(data, regression_formula, ...) {
  model <- stats::glm(formula = regression_formula, data = data, ...)

  return(function(newdata) {
    predict(model, newdata = newdata, type = 'response')
  })
}

#' @export
#' @importFrom mgcv gam
lnr_gam <- function(data, regression_formula, ...) {
  model <- mgcv::gam(formula = regression_formula, data = data, ...)

  return(function(newdata) {
    predict(model, newdata = newdata, type = 'response')
  })
}

#' @export
#' @importFrom lme4 lmer
lnr_lmer <- function(data, regression_formula, ...) {
  model <- lme4::lmer(formula = regression_formula, data = data, ...)

  return(function(newdata) {
    predict(model, newdata = newdata, type = 'response')
  })
}

#' @export
#' @importFrom lme4 glmer
lnr_glmer <- function(data, regression_formula, ...) {
  model <- lme4::glmer(formula = regression_formula, data = data, ...)

  return(function(newdata) {
    predict(model, newdata = newdata, type = 'response')
  })
}


#' Learners in the {nadir} Package
#'
#' The following learners are available:
#'
#'   * `lnr_mean`
#'   * `lnr_gam`
#'   * `lnr_glm`
#'   * `lnr_glmer`
#'   * `lnr_glmnet`
#'   * `lnr_lm`
#'   * `lnr_lmer`
#'   * `lnr_ranger`
#'   * `lnr_rf`
#'   * `lnr_xgboost`
#'
#' `lnr_mean` is generally provided only for benchmarking purposes to compare
#' other learners against to ensure correct specification of learners, since any
#' prediction algorithm should (in theory) out-perform just using the mean of
#' the outcome for all predictions.
#'
#' If you'd like to build a new learner, we recommend reading the
#' source code of several of the learners provided with `{nadir}` to
#' get a sense of how they should be specified.
#'
#' A learner, as `{nadir}` understands them, is a function which
#' takes in `data`, a `regression_formula`, possibly `...`, and
#' returns a function that predicts on its input `newdata`.
#'
#' A simple example is reproduced here for ease of reference:
#'
#' @examples
#' \dontrun{
#'  lnr_glm <- function(data, regression_formula, ...) {
#'   model <- stats::glm(formula = regression_formula, data = data, ...)
#'
#'   return(function(newdata) {
#'     predict(model, newdata = newdata, type = 'response')
#'   })
#'   }
#' }
#'
#' @rdname learners
#' @name learners
#' @keywords learners
NULL
