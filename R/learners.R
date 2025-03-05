#' @export
lnr_mean <- function(data, formula) {
  y_mean <- mean(data[[as.character(formula[[2]])]])
  mean_predict <- function(newdata) {
    rep(y_mean, nrow(newdata))
  }
  return(mean_predict)
}

#' @export
#' @importFrom ranger ranger
lnr_ranger <- function(data, formula, ...) {
  model <- ranger::ranger(data = data, formula = formula, ...)
  ranger_predict <- function(newdata) {
    predict(model, data = newdata)$predictions
  }
  return(ranger_predict)
}

#' glmnet Learner
#'
#' glmnet predictions will by default, if lambda is unspecified, return a matrix
#' of predictions for varied lambda values, hence the need to explicitly handle
#' the lambda argument in building glmnet learners.
#'
#' @export
#' @importFrom stats lm model.matrix predict
lnr_glmnet <- function(data, formula, lambda = .2, ...) {
  # glmnet takes Y and X separately, so we shall pull them out from the
  # data based on the formula
  yvar <- as.character(formula[[2]])
  xdata <- model.matrix.default(formula, data = data)
  model <- glmnet::glmnet(y = data[[yvar]], x = xdata, lambda = lambda, ...)
  return(function(newdata) {
    xdata = model.matrix.default(formula, data = newdata)
    as.vector(glmnet::predict.glmnet(model, newx = xdata, type = 'response'))
  })
}

#' @export
#' @importFrom randomForest randomForest
lnr_rf <- function(data, formula, ...) {
  y_variable <- as.character(formula)[[2]]
  y <- data[[y_variable]]
  index_of_yvar <- which(colnames(data) == y_variable)[[1]]
  xdata <- data[,-index_of_yvar]
  model <- randomForest::randomForest(x = xdata, y = y, formula = formula, ...)
  return(function(newdata) {
    randomForest:::predict.randomForest(object = model, newdata = newdata, type = 'response')
  })
}

#' @export
#' @importFrom stats lm
lnr_lm <- function(data, formula, ...) {
  model <- stats::lm(formula = formula, data = data, ...)

  predict_from_trained_lm <- function(newdata) {
    predict(model, newdata = newdata, type = 'response')
  }
  return(predict_from_trained_lm)
}

#' @export
#' @importFrom earth earth
lnr_earth <- function(data, formula, ...) {
  xdata <- model.matrix.default(formula, data)
  y <- data[[as.character(formula)[[2]]]]
  fit_earth_model <- earth::earth(x = xdata, y = y)

  predict_from_earth <- function(newdata) {
    newdata_mat <- model.matrix.default(formula, newdata)
    as.vector(earth:::predict.earth(fit_earth_model, newdata = newdata_mat, type = 'response'))
  }
}

#' @export
#' @importFrom stats glm
lnr_glm <- function(data, formula, ...) {
  model <- stats::glm(formula = formula, data = data, ...)

  return(function(newdata) {
    predict(model, newdata = newdata, type = 'response')
  })
}

#' @export
#' @importFrom mgcv gam
lnr_gam <- function(data, formula, ...) {
  model <- mgcv::gam(formula = formula, data = data, ...)

  return(function(newdata) {
    as.vector(predict(model, newdata = newdata, type = 'response'))
  })
}

#' @export
#' @importFrom lme4 lmer
lnr_lmer <- function(data, formula, ...) {
  model <- lme4::lmer(formula = formula, data = data, ...)

  return(function(newdata) {
    predict(model, newdata = newdata, type = 'response')
  })
}

#' @export
#' @importFrom lme4 glmer
lnr_glmer <- function(data, formula, ...) {
  model <- lme4::glmer(formula = formula, data = data, ...)

  return(function(newdata) {
    predict(model, newdata = newdata, type = 'response')
  })
}

#' @export
#' @importFrom xgboost xgboost
lnr_xgboost <- function(data, formula, nrounds = 1000, verbose = 0, ...) {
  xdata <- model.matrix.default(formula, data)
  yvar <- as.character(formula)[[2]]
  y <- data[[yvar]]

  model <- xgboost::xgboost(data = xdata, label = y, nrounds = nrounds, verbose = verbose, ...)

  return(function(newdata) {
    newdata_mat <- model.matrix.default(formula, newdata)
    predict(model, newdata = newdata_mat)
  })
}

#' Learners in the \code{\{nadir\}} Package
#'
#' The following learners are available for continuous outcomes:
#'
#' \itemize{
#'  \item \code{lnr_mean}
#'  \item \code{lnr_earth}
#'  \item \code{lnr_gam}
#'  \item \code{lnr_glm}
#'  \item \code{lnr_glmer}
#'  \item \code{lnr_glmnet}
#'  \item \code{lnr_lm}
#'  \item \code{lnr_lmer}
#'  \item \code{lnr_ranger}
#'  \item \code{lnr_rf}
#'  \item \code{lnr_xgboost}
#' }
#'
#' See \code{?density_learners} to learn more about using conditional density
#' estimation in \code{nadir}.
#'
#' \code{lnr_mean} is generally provided only for benchmarking purposes to compare
#' other learners against to ensure correct specification of learners, since any
#' prediction algorithm should (in theory) out-perform just using the mean of
#' the outcome for all predictions.
#'
#' If you'd like to build a new learner, we recommend reading the
#' source code of several of the learners provided with \code{\{nadir\}} to
#' get a sense of how they should be specified.
#'
#' A learner, as \code{\{nadir\}} understands them, is a function which
#' takes in `data`, a `formula`, possibly `...`, and
#' returns a function that predicts on its input `newdata`.
#'
#' A simple example is reproduced here for ease of reference:
#'
#' @examples
#' \dontrun{
#'  lnr_glm <- function(data, formula, ...) {
#'   model <- stats::glm(formula = formula, data = data, ...)
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

