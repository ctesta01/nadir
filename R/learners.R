#' Mean Learner
#'
#' This is a very naive/simple learner that simply predicts the mean of the
#' outcome for every row of input \code{newdata}.  This is primarily
#' useful for benchmarking and confirming that other learners are
#' performing better than \code{lnr_mean}. Additionally, it may be the case
#' that some learners are over-fitting the data, and giving some weight to
#' \code{lnr_mean} helps to reduce over-fitting in \code{super_learner()}.
#'
#' @inheritParams lnr_lm
#' @seealso learners
#'
#' @export
lnr_mean <- function(data, formula) {
  y_mean <- mean(data[[as.character(formula[[2]])]])
  mean_predict <- function(newdata) {
    rep(y_mean, nrow(newdata))
  }
  return(mean_predict)
}


#' ranger Learner
#'
#' A wrapper for \code{ranger::ranger()} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
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
#' A wrapper for \code{glmnet::glmnet()} for use in \code{nadir::super_learner()}.
#'
#' glmnet predictions will by default, if lambda is unspecified, return a matrix
#' of predictions for varied lambda values, hence the need to explicitly handle
#' the lambda argument in building glmnet learners.
#'
#' @inheritParams lnr_lm
#' @param lambda The multiplier parameter for the penalty; see \code{?glmnet::glmnet}
#' @seealso learners
#' @export
#' @importFrom stats lm model.matrix
#' @importFrom glmnet glmnet predict.glmnet
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

#' randomForest Learner
#'
#' A wrapper for \code{randomForest::randomForest()} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @export
#' @importFrom randomForest randomForest
lnr_rf <- function(data, formula, ...) {
  y_variable <- as.character(formula)[[2]]
  y <- data[[y_variable]]
  index_of_yvar <- which(colnames(data) == y_variable)[[1]]
  # xdata <- data[,-index_of_yvar]
  xdata <- model.frame(formula, data)
  index_of_yvar_in_model_frame <- which(colnames(xdata) == y_variable)
  xdata <- xdata[,-index_of_yvar_in_model_frame,drop=FALSE]
  model <- randomForest::randomForest(x = xdata, y = y, formula = formula, ...)
  return(function(newdata) {
    if (y_variable %in% colnames(newdata)) {
      index_of_yvar <- which(colnames(newdata) == y_variable)[[1]]
      newdata <- newdata[, -index_of_yvar, drop=FALSE]
    }
    predict(object = model, newdata = newdata, type = 'response')
  })
}

#' Linear Model Learner
#'
#' A wrapper for \code{lm()} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @param data A dataframe to train a learner / learners on.
#' @param formula A regression formula to use inside this learner.
#' @param ... Any extra arguments that should be passed to the internal model
#'   for model fitting purposes.
#' @export
#' @importFrom stats lm
lnr_lm <- function(data, formula, ...) {
  model <- stats::lm(formula = formula, data = data, ...)

  predict_from_trained_lm <- function(newdata) {
    predict(model, newdata = newdata, type = 'response')
  }
  return(predict_from_trained_lm)
}

#' Earth Learner
#'
#' A wrapper for \code{earth::earth()} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @export
#' @importFrom earth earth
lnr_earth <- function(data, formula,  ...) {
  xdata <- model.frame(formula, data)
  y_variable <- as.character(formula)[[2]]
  if (y_variable %in% colnames(xdata)) {
  index_of_yvar_in_xdata <- which(colnames(xdata) == y_variable)
  xdata <- xdata[,-index_of_yvar_in_xdata,drop=FALSE]
  }
  index_of_yvar_in_data <- which(colnames(data) == y_variable)
  y <- data[[index_of_yvar_in_data]]
  fit_earth_model <- earth::earth(x = xdata, y = y)

  predict_from_earth <- function(newdata) {
    if (y_variable %in% colnames(newdata)) {
      index_of_yvar_in_newdata <- which(colnames(newdata) == y_variable)
      newdata <- newdata[,-index_of_yvar_in_newdata,drop=FALSE]
    }
    as.vector(predict(fit_earth_model, newdata = newdata, type = 'response'))
  }
  return(predict_from_earth)
}

#' GLM Learner
#'
#' A wrapper for \code{stats::glm()} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @export
#' @importFrom stats glm
lnr_glm <- function(data, formula, ...) {
  model <- stats::glm(formula = formula, data = data, ...)

  return(function(newdata) {
    predict(model, newdata = newdata, type = 'response')
  })
}

#' Generalized Additive Model Learner
#'
#' A wrapper for \code{mgcv::gam()} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @export
#' @importFrom mgcv gam
lnr_gam <- function(data, formula, ...) {
  model <- mgcv::gam(formula = formula, data = data, ...)

  return(function(newdata) {
    as.vector(predict(model, newdata = newdata, type = 'response'))
  })
}

#' Random/Mixed-Effects (\code{lme4::lmer}) Learner
#'
#' A wrapper for \code{lme4::lmer} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @export
#' @importFrom lme4 lmer
lnr_lmer <- function(data, formula, ...) {
  model <- lme4::lmer(formula = formula, data = data, ...)

  return(function(newdata) {
    predict(model, newdata = newdata, type = 'response')
  })
}

#' Generalized Linear Mixed-Effects (\code{lme4::glmer}) Learner
#'
#' A wrapper for \code{lme4::glmer()} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @export
#' @importFrom lme4 glmer
lnr_glmer <- function(data, formula, ...) {
  model <- lme4::glmer(formula = formula, data = data, ...)

  return(function(newdata) {
    predict(model, newdata = newdata, type = 'response')
  })
}

#' XGBoost Learner
#'
#' A wrapper for \code{xgboost::xgboost()} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @param nrounds The max number of boosting iterations
#' @param verbose If verbose is \code{> 0} then \code{xgboost::xgboost()} will print out messages
#'   about its fitting process. See \code{?xgboost::xgboost}
#' @export
#' @importFrom xgboost xgboost
lnr_xgboost <-
  function(data,
           formula,
           nrounds = 1000,
           verbose = 0,
           ...) {

  xdata <- model.matrix.default(formula, data)
  yvar <- as.character(formula)[[2]]
  y <- data[[yvar]]

  model <-
    xgboost::xgboost(
      data = xdata,
      label = y,
      nrounds = nrounds,
      verbose = verbose,
      ...
    )

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



attr(lnr_mean, 'sl_lnr_name') <- 'mean'
attr(lnr_earth, 'sl_lnr_name') <- 'earth'
attr(lnr_gam, 'sl_lnr_name') <- 'gam'
attr(lnr_glm, 'sl_lnr_name') <- 'glm'
attr(lnr_glmer, 'sl_lnr_name') <- 'glmer'
attr(lnr_glmnet, 'sl_lnr_name') <- 'glmnet'
attr(lnr_lm, 'sl_lnr_name') <- 'lm'
attr(lnr_lmer, 'sl_lnr_name') <- 'lmer'
attr(lnr_ranger, 'sl_lnr_name') <- 'ranger'
attr(lnr_rf, 'sl_lnr_name') <- 'rf'
attr(lnr_xgboost, 'sl_lnr_name') <- 'xgboost'


attr(lnr_mean, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_earth, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_gam, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_glm, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_glmer, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_glmnet, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_lm, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_lmer, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_ranger, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_rf, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_xgboost, 'sl_lnr_type') <- c('continuous', 'binary')

