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
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#'
#' @export
lnr_mean <- function(data, formula, weights = NULL) {
  y_mean <- mean(data[[as.character(formula[[2]])]])
  mean_predict <- function(newdata) {
    rep(y_mean, nrow(newdata))
  }
  return(mean_predict)
}
attr(lnr_mean, 'sl_lnr_name') <- 'mean'
attr(lnr_mean, 'sl_lnr_type') <- c('continuous', 'binary')



#' ranger Learner
#'
#' A wrapper for \code{ranger::ranger()} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @export
#' @importFrom ranger ranger
lnr_ranger <- function(data, formula, weights = NULL, ...) {
  model <- ranger::ranger(data = data, case.weights = weights, formula = formula, ...)
  ranger_predict <- function(newdata) {
    predict(model, data = newdata)$predictions
  }
  return(ranger_predict)
}
attr(lnr_ranger, 'sl_lnr_name') <- 'ranger'
attr(lnr_ranger, 'sl_lnr_type') <- c('continuous', 'binary')


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
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @importFrom stats lm model.matrix
#' @importFrom glmnet glmnet predict.glmnet
lnr_glmnet <- function(data, formula, weights = NULL, lambda = .2, ...) {
  # glmnet takes Y and X separately, so we shall pull them out from the
  # data based on the formula
  yvar <- as.character(formula[[2]])
  formula_without_lhs <- formula
  formula_without_lhs[2] <- NULL
  xdata <- model.matrix.default(formula_without_lhs, data = data)

  # it is quite important that a single value of lambda be passed (not multiple),
  # otherwise we are fitting multiple models instead of just one, and the returned
  # values may be multiple predictions over a grid of lambda values.
  if (length(lambda) > 1) {
    warning("lambda must be a single (length=1) numeric value, not a vector of length > 1.")
  }

  model <- glmnet::glmnet(y = data[[yvar]], x = xdata, lambda = lambda, weights = weights, ...)
  return(function(newdata) {
    if (yvar %in% colnames(newdata)) {
      newdata[[yvar]] <- NULL
    }
    # if the y-variable (lhs) appears in the formula given to model.matrix.default, then
    # errors will be thrown if the same y-variable doesn't appear in the newdata.
    #
    # that would be bad, since we expect newdata should be allowed to only contain predictors
    # without the response variable. hence we make a copy of the formula without the lhs to
    # use for constructing the model matrix for prediction purposes.
    formula_without_lhs <- formula
    formula_without_lhs[2] <- NULL
    xdata = model.matrix.default(formula_without_lhs, data = newdata)

    # return the prediction results as a vector
    # (normally they come out as a matrix, which makes more sense with multiple values of lambda)
    as.vector(glmnet::predict.glmnet(model, newx = xdata, type = 'response'))
  })
}
attr(lnr_glmnet, 'sl_lnr_name') <- 'glmnet'
attr(lnr_glmnet, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_glmnet, 'outcome_type_dependent_args') <- list(
  'binary' = list(family = binomial(link = 'logit')))

#' randomForest Learner
#'
#' A wrapper for \code{randomForest::randomForest()} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @export
#' @importFrom randomForest randomForest
lnr_rf <- function(data, formula, weights = NULL, ...) {
  y_variable <- as.character(formula)[[2]]
  y <- data[[y_variable]]
  index_of_yvar <- which(colnames(data) == y_variable)[[1]]
  xdata <- model.frame(formula, data)
  index_of_yvar_in_model_frame <- which(colnames(xdata) == y_variable)
  xdata <- xdata[,-index_of_yvar_in_model_frame,drop=FALSE]
  model <- randomForest::randomForest(x = xdata, y = y, formula = formula, weights = weights, ...)
  return(function(newdata) {
    # make sure the y_variable doesn't appear in the set of predictors
    if (y_variable %in% colnames(newdata)) {
      index_of_yvar <- which(colnames(newdata) == y_variable)[[1]]
      newdata <- newdata[, -index_of_yvar, drop=FALSE]
    }
    predict(object = model, newdata = newdata, type = 'response')
  })
}
attr(lnr_rf, 'sl_lnr_name') <- 'rf'
attr(lnr_rf, 'sl_lnr_type') <- c('continuous', 'binary')


#' Linear Model Learner
#'
#' A wrapper for \code{lm()} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @param data A dataframe to train a learner / learners on.
#' @param formula A regression formula to use inside this learner.
#' @param weights Observation weights; see \code{?lm}
#' @param ... Any extra arguments that should be passed to the internal model
#'   for model fitting purposes.
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @export
#' @importFrom stats lm
lnr_lm <- function(data, formula, weights = NULL, ...) {
  model_args <- list(
    data = data,
    formula = formula)
  if (! is.null(weights)) {
    model_args$weights <- weights
  }
  model <- do.call(what = stats::lm, args = c(model_args, list(...)))

  predict_from_trained_lm <- function(newdata) {
    predict(model, newdata = newdata, type = 'response')
  }
  return(predict_from_trained_lm)
}
attr(lnr_lm, 'sl_lnr_name') <- 'lm'
attr(lnr_lm, 'sl_lnr_type') <- c('continuous', 'binary')


#' Earth Learner
#'
#' A wrapper for \code{earth::earth()} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @export
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @importFrom earth earth
lnr_earth <- function(data, formula,  weights = NULL, ...) {
  xdata <- model.frame(formula, data)
  y_variable <- as.character(formula)[[2]]
  if (y_variable %in% colnames(xdata)) {
  index_of_yvar_in_xdata <- which(colnames(xdata) == y_variable)
  xdata <- xdata[,-index_of_yvar_in_xdata,drop=FALSE]
  }
  index_of_yvar_in_data <- which(colnames(data) == y_variable)
  y <- data[[index_of_yvar_in_data]]
  fit_earth_model <- earth::earth(x = xdata, y = y, weights = weights, ...)

  predict_from_earth <- function(newdata) {
    if (y_variable %in% colnames(newdata)) {
      index_of_yvar_in_newdata <- which(colnames(newdata) == y_variable)
      newdata <- newdata[,-index_of_yvar_in_newdata,drop=FALSE]
    }
    as.vector(predict(fit_earth_model, newdata = newdata, type = 'response'))
  }
  return(predict_from_earth)
}
attr(lnr_earth, 'sl_lnr_name') <- 'earth'
attr(lnr_earth, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_earth, 'outcome_type_dependent_args') <- list(
  'binary' = list(glm = list(family = 'binomial')))


#' GLM Learner
#'
#' A wrapper for \code{stats::glm()} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @export
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @importFrom stats glm
lnr_glm <- function(data, formula, weights = NULL, ...) {
  model_args <- list(
    data = data,
    formula = formula)
  if (! is.null(weights) & is.numeric(weights) & length(weights) == nrow(data)) {
    model_args$weights <- weights
  }
  model <- do.call(what = stats::glm, args = c(model_args, list(...)))

  return(function(newdata) {
    predict(model, newdata = newdata, type = 'response')
  })
}
attr(lnr_glm, 'sl_lnr_name') <- 'glm'
attr(lnr_glm, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_glm, 'outcome_type_dependent_args') <- list(
  'binary' = list(family = binomial(link = 'logit')))

#' Generalized Additive Model Learner
#'
#' A wrapper for \code{mgcv::gam()} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @export
#' @importFrom mgcv gam
lnr_gam <- function(data, formula, weights = NULL, ...) {
  model_args <- list(
    data = data,
    formula = formula)
  if (! is.null(weights)) {
    model_args$weights <- weights
  }
  model <- do.call(what = mgcv::gam, args = c(model_args, list(...)))

  return(function(newdata) {
    as.vector(predict(model, newdata = newdata, type = 'response'))
  })
}
attr(lnr_gam, 'sl_lnr_name') <- 'gam'
attr(lnr_gam, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_gam, 'outcome_type_dependent_args') <- list(
  'binary' = list(family = binomial(link = 'logit')))

#' Random/Mixed-Effects (\code{lme4::lmer}) Learner
#'
#' A wrapper for \code{lme4::lmer} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @export
#' @importFrom lme4 lmer
lnr_lmer <- function(data, formula, weights = NULL, ...) {
  model <- lme4::lmer(formula = formula, data = data, weights = weights, ...)

  return(function(newdata) {
    predict(model, newdata = newdata, type = 'response', allow.new.levels = TRUE)
  })
}
attr(lnr_lmer, 'sl_lnr_name') <- 'lmer'
attr(lnr_lmer, 'sl_lnr_type') <- c('continuous', 'binary')


#' Generalized Linear Mixed-Effects (\code{lme4::glmer}) Learner
#'
#' A wrapper for \code{lme4::glmer()} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @export
#' @importFrom lme4 glmer
lnr_glmer <- function(data, formula, weights = NULL, ...) {
  model <- lme4::glmer(formula = formula, data = data, weights = weights, ...)

  return(function(newdata) {
    predict(model, newdata = newdata, type = 'response', allow.new.levels = TRUE)
  })
}
attr(lnr_glmer, 'sl_lnr_name') <- 'glmer'
attr(lnr_glmer, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_glmer, 'outcome_type_dependent_args') <- list(
  'binary' = list(family = binomial(link = 'logit')))


#' Highly Adaptive Lasso
#'
#' @seealso learners
#' @inheritParams lnr_glmnet
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @export
#' @importFrom hal9001 fit_hal
lnr_hal <- function(data, formula, weights = NULL, lambda = NULL, ...) {
  yvar <- as.character(formula[[2]])
  xdata <- model.matrix.lm(formula, data = data, na.action = 'na.pass')
  model <- hal9001::fit_hal(Y = data[[yvar]], X = xdata, lambda = lambda, weights = weights, ...)
  return(function(newdata) {
    # ensure the y-variable isn't required inside the model.matrix.default call
    if (length(formula) >= 3) {
      formula[2] <- NULL
    }
    xdata = model.matrix.lm(formula, data = newdata, na.action = 'na.pass')
    as.vector(predict(object = model, new_data = xdata, type = 'response'))
  })
}
attr(lnr_hal, 'sl_lnr_name') <- 'hal'
attr(lnr_hal, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_glmer, 'outcome_type_dependent_args') <- list(
  'binary' = list(family = 'binomial'))


#' XGBoost Learner
#'
#' A wrapper for \code{xgboost::xgboost()} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @param nrounds The max number of boosting iterations
#' @param verbose If verbose is \code{> 0} then \code{xgboost::xgboost()} will print out messages
#'   about its fitting process. See \code{?xgboost::xgboost}
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @export
#' @importFrom xgboost xgboost
lnr_xgboost <-
  function(data,
           formula,
           weights = NULL,
           nrounds = 1000,
           verbose = 0,
           ...) {

  xdata <- model.matrix.lm(formula, data, na.action = 'na.pass')
  yvar <- as.character(formula)[[2]]
  y <- data[[yvar]]

  model <-
    xgboost::xgboost(
      data = xdata,
      label = y,
      nrounds = nrounds,
      verbose = verbose,
      weight = weights,
      ...
    )

  return(function(newdata) {
    newdata_mat <- model.matrix.lm(formula, newdata, na.action = "na.pass")
    predict(model, newdata = newdata_mat)
  })
}
attr(lnr_xgboost, 'sl_lnr_name') <- 'xgboost'
attr(lnr_xgboost, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_xgboost, 'outcome_type_dependent_args') <- list(
  'binary' = list(objective = 'binary:logistic'))


#' Gradient Boosting Machines Learner
#'
#' A wrapper for \code{gbm::gbm()} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @param verbose (default: FALSE) if set to TRUE, information about the automatic
#'   outcome type inferred by \code{gbm} will be messaged to the console, as well as the number
#'   of trees used.
#' @param n.minobsinnode (default: 0) An integer specifying the minimum number of observations in the terminal nodes of the trees. See
#' the gbm documentation for more.  Set here to 0 to account for the potential of very small splits in cross-fitting.
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @export
#' @importFrom gbm gbm
#' @importFrom utils capture.output
lnr_gbm <-
  function(data,
           formula,
           verbose = FALSE,
           n.minobsinnode = 0,
           ...) {

    if (is.null(weights)) {
      weights <- rep(1, nrow(data))
    }

    capture.output({ # suppresses the "Distribution not specified, assuming ..."
      model <- gbm::gbm(
        formula = formula,
        data = data,
        verbose = verbose,
        n.minobsinnode = n.minobsinnode,
        ...
      )
    })

    return(function(newdata) {
      if (verbose) {
        predict(model, newdata = newdata, type = 'response')
      } else {
        suppressMessages({
          predict(model, newdata = newdata, type = 'response')
          })
      }
    })
  }
attr(lnr_gbm, 'sl_lnr_name') <- 'gbm'
attr(lnr_gbm, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_gbm, 'outcome_type_dependent_args') <- list(
  'continuous' = list(distribution = 'gaussian'),
  'binary' = list(distribution = 'bernoulli'))


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
#'  \item \code{lnr_hal}
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
#'  lnr_glm <- function(data, formula, weights = NULL, ...) {
#'   model <- stats::glm(formula = formula, data = data, weights = weights, ...)
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





