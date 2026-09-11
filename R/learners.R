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
#' @examples
#' lnr_mean(mtcars, mpg ~ hp)(mtcars)
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
#' @examples
#' lnr_ranger(mtcars, mpg ~ hp)(mtcars)
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
#' @examples
#' lnr_glmnet(mtcars, mpg ~ hp + disp + am + wt, lambda = .5)(mtcars)
lnr_glmnet <- function(data, formula, weights = NULL, lambda = .2, ...) {
  # glmnet takes Y and X separately, so we shall pull them out from the
  # data based on the formula
  yvar <- as.character(formula[[2]])
  formula_without_lhs <- formula
  formula_without_lhs[2] <- NULL
  # Preserve unused factor levels so their indicator columns are retained
  # (and contain only zeros when no observation has that level).
	factor_levels <- lapply(data, function(x) {
		if (is.factor(x)) levels(x) else NULL
	})
	factor_levels <- factor_levels[!vapply(factor_levels, is.null, logical(1))]
  xdata <- model.matrix.default(formula_without_lhs, data = data, xlev = factor_levels)
  if (yvar %in% colnames(xdata)) {
    yvar_idx <- which(colnames(xdata) == yvar)
    xdata <- xdata[,-yvar_idx]
  }

  # A conventional learner (which lnr_glmnet is taken to be) must return exactly
  # one prediction per row of newdata.
  #
  # glmnet with lambda = NULL or a vector of lambdas fits a whole automatically
  # selected path of lambdas and predict() returns an n x n_lambda matrix, which
  # breaks super_learner()'s bookkeeping. We consider this a mistake here and
  # error.
  if (is.null(lambda) || length(lambda) != 1 || !is.numeric(lambda)) {
    stop(
      "lnr_glmnet requires `lambda` to be a single numeric value so that ",
      "predictions are one-per-row of newdata rather than a matrix over a ",
      "lambda path.\n",
      "To fit over a grid of lambda values and predict at the cross-validation",
      " selected lambda, use `lnr_cvglmnet` or `lnr_glmnet_grid` instead."
    )
  }

  model <- glmnet::glmnet(y = data[[yvar]], x = xdata, lambda = lambda,
                          weights = weights, ...)
  return(function(newdata) {
    if (yvar %in% colnames(newdata)) {
      newdata[[yvar]] <- NULL
    }
    # if the y-variable (lhs) appears in the formula given to
    # model.matrix.default, then errors will be thrown if the same y-variable
    # doesn't appear in the newdata. hence we make a copy of the formula
    # without the lhs to use for constructing the model matrix for prediction.
    formula_without_lhs <- formula
    formula_without_lhs[2] <- NULL
    xdata <- model.matrix.default(formula_without_lhs, data = newdata, xlev = factor_levels)

    # use the s3 generic predict() here
    preds <- predict(model, newx = xdata, type = "response")

    # defensive: never silently flatten a multi-column prediction matrix
    if (ncol(preds) != 1) {
      stop(
        "lnr_glmnet produced ", ncol(preds), " columns of predictions ",
        "(one per lambda). Learners must return a single prediction per row; ",
        "use a single lambda, or use lnr_cvglmnet for lambda selection."
      )
    }
    as.vector(preds)
  })
}
attr(lnr_glmnet, "sl_lnr_name") <- "glmnet"
attr(lnr_glmnet, "sl_lnr_type") <- c("continuous", "binary")
attr(lnr_glmnet, "outcome_type_dependent_args") <- list(
  "binary" = list(family = binomial(link = "logit")))


#' cv.glmnet Learner
#'
#' A wrapper for \code{glmnet::cv.glmnet()} for use in \code{nadir::super_learner()}.
#'
#' The returning prediction function defaults to passing \code{s = "lambda.min"} to the \code{predict.cv.glmnet} method built into \code{glmnet} defaults to
#' predicting which says to use the minimum cross-validated loss lambda value from the CV grid
#' \code{cv.glmnet} sets up. The other option is to pass \code{s = "lambda.1se"} to the
#' returned prediction function which
#' returns the largest lambda estimated to be within one standard deviation of the
#' CV-optimal lambda according to the stored \code{cv.glmnet} object.
#'
#' @inheritParams lnr_lm
#' @param lambda The multiplier parameter grid for the penalty; see \code{?glmnet::cv.glmnet}
#' @seealso learners
#' @export
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @importFrom stats lm model.matrix
#' @importFrom glmnet cv.glmnet
#' @examples
#' lnr_cvglmnet(mtcars, mpg ~ hp + disp + am + wt)(mtcars)
lnr_cvglmnet <- function(data, formula, weights = NULL, lambda = NULL, ...) {
  # glmnet takes Y and X separately, so we shall pull them out from the
  # data based on the formula
  yvar <- as.character(formula[[2]])
  formula_without_lhs <- formula
  formula_without_lhs[2] <- NULL
  # Preserve unused factor levels so their indicator columns are retained
  # (and contain only zeros when no observation has that level).
	factor_levels <- lapply(data, function(x) {
		if (is.factor(x)) levels(x) else NULL
	})
	factor_levels <- factor_levels[!vapply(factor_levels, is.null, logical(1))]
  xdata <- model.matrix.default(formula_without_lhs, data = data, xlev = factor_levels)
  if (yvar %in% colnames(xdata)) {
    yvar_idx <- which(colnames(xdata) == yvar)
    xdata <- xdata[,-yvar_idx]
  }

  model <- glmnet::cv.glmnet(y = data[[yvar]], x = xdata, lambda = lambda, weights = weights, ...)
  return(function(newdata, s = 'lambda.min', ...) {
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
    xdata = model.matrix.default(formula_without_lhs, data = newdata, xlev = factor_levels)

    # construct the arguments for `predict.cv.glmnet`
    predict_args <- c(
      list(object = model, newx = xdata, s = s),
      list(...),
      list(type = 'response'))
    predict_args <- predict_args[!duplicated(names(predict_args))] # keeps 1st appearance

    # return the prediction results as a vector (they normally come out as a matrix,
    # which makes more sense with multiple values of lambda/s)
    as.vector(do.call(predict, predict_args))
  })
}
attr(lnr_cvglmnet, 'sl_lnr_name') <- 'glmnet'
attr(lnr_cvglmnet, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_cvglmnet, 'outcome_type_dependent_args') <- list(
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
#' @examples
#' lnr_rf(mtcars, mpg ~ hp + disp + am + wt, ntree = 20)(mtcars)
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
#' @examples
#' lnr_lm(mtcars, mpg ~ hp + disp + am + wt)(mtcars)
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
#' @examples
#' lnr_earth(mtcars, mpg ~ hp + disp + am + wt)(mtcars)
lnr_earth <- function(data, formula,  weights = NULL, ...) {
  x_formula <- formula
  x_formula[[2]] <- NULL                      # RHS-only: ~ x1 + x2 + ...
  xdata <- model.frame(x_formula, data)
  y <- data[[as.character(formula)[[2]]]]
  fit <- earth::earth(x = xdata, y = y, weights = weights, ...)
  function(newdata) {
    newx <- model.frame(x_formula, newdata, na.action = stats::na.pass)
    as.vector(predict(fit, newdata = newx, type = "response"))
  }
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
#' @examples
#' lnr_glm(mtcars, mpg ~ hp + disp + am + wt)(mtcars)
#' lnr_glm(mtcars, mpg ~ hp + disp + am + wt, family = Gamma)(mtcars)
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
#' @examples
#' lnr_gam(mtcars, mpg ~ s(hp) + disp + am + wt)(mtcars)
#' lnr_gam(mtcars, mpg ~ s(hp) + disp + am + wt, family = Gamma)(mtcars)
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
#' @examples
#' # random intercepts for each level of cyl column:
#' lnr_lmer(mtcars, mpg ~ (1|cyl) + disp + am + wt)(mtcars)
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
#' @examples
#' # random intercepts for each level of cyl column:
#' suppressMessages({
#' # singular fit, but that's ok if all you need is prediction:
#' lnr_glmer(mtcars, mpg ~ (1|cyl) + disp + wt, family = Gamma)(mtcars)
#' })
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
#' A wrapper for \code{hal9001::fit_hal()} for use in \code{nadir::super_learner()}.
#'
#' If a single \code{lambda} value is specified, \code{fit_control$cv_select}
#' is set to \code{FALSE}, since \code{fit_hal}'s default
#' (\code{cv_select = TRUE}) hands \code{lambda} to \code{cv.glmnet}, which
#' errors with "Need more than one value of lambda for cv.glmnet" — and inside
#' \code{super_learner()} that error would cause \code{lnr_hal} to be silently
#' dropped from the ensemble.
#'
#' If a vector of \code{lambda} values is specified (with the default
#' \code{fit_control}), the grid is treated as a search space:
#' \code{cv.glmnet} internally selects a single lambda, and only that model's
#' predictions are returned. To instead expose \emph{each} lambda value in a
#' grid to the meta-learning stage of \code{super_learner()} as its own
#' candidate learner, use \code{lnr_hal_grid}.
#'
#' @seealso learners lnr_hal_grid
#' @inheritParams lnr_glmnet
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @export
#' @importFrom hal9001 fit_hal
#' @examples
#' suppressWarnings({
#' # hal prints a lot of messages about some threads not reaching convergence
#' lnr_hal(mtcars, mpg ~ hp + am + cyl + disp)(mtcars)
#' })
lnr_hal <- function(data, formula, weights = NULL, lambda = NULL, ...) {
  yvar <- as.character(formula[[2]])

  # Preserve unused factor levels so their indicator columns are retained
  # (and contain only zeros when no observation has that level).
  factor_levels <- lapply(data, function(x) {
    if (is.factor(x)) levels(x) else NULL
  })
  factor_levels <- factor_levels[!vapply(factor_levels, is.null, logical(1))]

  xdata <- stats::model.matrix.default(formula,
                                       data = data,
                                       xlev = factor_levels,
                                       na.action = 'na.pass')

  # if the user specifies a single lambda value, cv_select needs to be FALSE:
  # fit_hal's default (cv_select = TRUE) gives lambda to cv.glmnet, which errors
  # as "Need more than one value of lambda for cv.glmnet" -- and inside
  # super_learner() this error would cause lnr_hal to silently be dropped from
  # the ensemble.
  dots <- list(...)
  if (! is.null(lambda) && length(lambda) == 1) {
    fit_control <- dots[['fit_control']]
    if (is.null(fit_control)) {
      fit_control <- list()
    }
    fit_control[['cv_select']] <- FALSE
    dots[['fit_control']] <- fit_control
  }

  model <- do.call(
    hal9001::fit_hal,
    c(list(Y = data[[yvar]], X = xdata, lambda = lambda, weights = weights),
      dots))
  return(function(newdata) {
    # ensure the y-variable isn't required inside the model.matrix.default call
    if (length(formula) >= 3) {
      formula[2] <- NULL
    }

    xdata <- stats::model.matrix.default(formula,
                                         data = newdata,
                                         na.action = 'na.pass',
                                         xlev = factor_levels)

    predictions <- predict(object = model, new_data = xdata, type = 'response')
    # if fit_hal retained fits at multiple lambda values (a grid of lambdas
    # plus user-supplied fit_control = list(cv_select = FALSE)), predictions
    # come back as an n-by-k matrix; as.vector() would silently flatten that
    # into a length n*k vector and corrupt the super_learner() meta-learning
    # stage, so we error informatively instead.
    if (is.matrix(predictions) && ncol(predictions) > 1) {
      stop("lnr_hal produced a matrix of predictions (one column per lambda),
likely because a grid of lambda values was passed alongside
fit_control = list(cv_select = FALSE). To expose each lambda value in a grid
to super_learner() as its own candidate learner, use lnr_hal_grid instead.")
    }
    as.vector(predictions)
  })
}
attr(lnr_hal, 'sl_lnr_name') <- 'hal'
attr(lnr_hal, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_hal, 'outcome_type_dependent_args') <- list(
  'binary' = list(family = 'binomial'))


#' XGBoost Learner
#'
#' A wrapper for \code{xgboost::xgb.train()} for use in
#' \code{nadir::super_learner()}.
#'
#' This version avoids the high-level \code{xgboost::xgboost()} wrapper and
#' instead uses the lower-level \code{xgb.DMatrix()} + \code{xgb.train()}
#' interface. This is more stable for programmatic Super Learner use because
#' \code{xgb.train()} supports the full range of XGBoost objectives.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @param weights Optional observation weights.
#' @param nrounds Number of boosting iterations.
#' @param xgb.params An \code{xgboost::xgb.params()} object, or a named list of
#'   parameters that can be passed to \code{xgboost::xgb.params()}.
#' @param objective Optional objective. If supplied, this is inserted into
#'   \code{xgb.params}. For binary Super Learner fits, \code{nadir} supplies
#'   \code{objective = "binary:logistic"} through
#'   \code{outcome_type_dependent_args}.
#' @param ... Additional arguments passed to \code{xgboost::xgb.params()}.
#'
#' @returns A prediction function that accepts \code{newdata} and returns a
#'   numeric vector of predictions.
#'
#' @export
#' @importFrom xgboost xgb.DMatrix xgb.params xgb.train
#'
#' @examples
#' lnr_xgboost(mtcars, mpg ~ hp, nrounds = 5)(mtcars)
#'
#' lnr_xgboost(
#'   mtcars,
#'   am ~ cyl + disp + hp,
#'   objective = "binary:logistic",
#'   nrounds = 5
#' )(mtcars)
lnr_xgboost <-
  function(data,
           formula,
           weights = NULL,
           nrounds = 1000,
           xgb.params = xgboost::xgb.params(),
           objective = NULL,
           ...) {

    yvar <- as.character(formula)[[2]]
    y <- data[[yvar]]

    # Use a right-hand-side-only formula for model.matrix().
    # This ensures that prediction does not require the outcome column to be
    # present in newdata.
    x_formula <- formula
    x_formula[[2]] <- NULL

    xdata <- stats::model.matrix.lm(
      object = x_formula,
      data = data,
      na.action = "na.pass"
    )

    # XGBoost does not need an intercept column for tree-based learners.
    if ("(Intercept)" %in% colnames(xdata)) {
      xdata <- xdata[, colnames(xdata) != "(Intercept)", drop = FALSE]
    }

    # xgb.DMatrix expects numeric labels.
    #
    # For binary outcomes, encode:
    #   FALSE / first factor level / 0 -> 0
    #   TRUE  / second factor level / 1 -> 1
    if (is.factor(y)) {
      if (nlevels(y) != 2L && identical(objective, "binary:logistic")) {
        stop(
          "For objective = 'binary:logistic', factor outcomes must have ",
          "exactly two levels."
        )
      }
      y <- as.integer(y) - 1L
    } else if (is.logical(y)) {
      y <- as.integer(y)
    } else {
      y <- as.numeric(y)
    }

    binary_objectives <- c(
      "binary:logistic",
      "binary:logitraw",
      "binary:hinge"
    )

    if (!is.null(objective) && objective %in% binary_objectives) {
      observed_y <- sort(unique(stats::na.omit(y)))

      if (!all(observed_y %in% c(0, 1))) {
        stop(
          "For binary XGBoost objectives, the outcome must be coded as ",
          "0/1, logical, or a two-level factor."
        )
      }
    }

    dtrain <- xgboost::xgb.DMatrix(
      data = xdata,
      label = y,
      weight = weights
    )

    # Merge parameters from three sources:
    #   1. xgb.params supplied by the user,
    #   2. objective supplied directly, often by outcome_type_dependent_args,
    #   3. additional named parameters in ...
    #
    # Later sources override earlier ones.
    xgb_params_list <- as.list(xgb.params)

    if (!is.null(objective)) {
      xgb_params_list$objective <- objective
    }

    dot_args <- list(...)
    if (length(dot_args) > 0L) {
      xgb_params_list <- utils::modifyList(xgb_params_list, dot_args)
    }

    xgb_params <- do.call(xgboost::xgb.params, xgb_params_list)

    model <- xgboost::xgb.train(
      params = xgb_params,
      data = dtrain,
      nrounds = nrounds
    )

    return(function(newdata) {
      newdata_mat <- stats::model.matrix.lm(
        object = x_formula,
        data = newdata,
        na.action = "na.pass"
      )

      if ("(Intercept)" %in% colnames(newdata_mat)) {
        newdata_mat <- newdata_mat[
          ,
          colnames(newdata_mat) != "(Intercept)",
          drop = FALSE
        ]
      }

      dnew <- xgboost::xgb.DMatrix(data = newdata_mat)

      as.numeric(predict(model, newdata = dnew))
    })
  }

attr(lnr_xgboost, "sl_lnr_name") <- "xgboost"
attr(lnr_xgboost, "sl_lnr_type") <- c("continuous", "binary")
attr(lnr_xgboost, "outcome_type_dependent_args") <- list(
  "binary" = list(objective = "binary:logistic")
)

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
#' @examples
#' lnr_gbm(mtcars, mpg ~ hp)(mtcars)
lnr_gbm <-
  function(data,
           formula,
           verbose = FALSE,
           n.minobsinnode = 0,
           ...) {
    if (is.null(weights)) {
      weights <- rep(1, nrow(data))
    }

    suppressMessages(capture.output({ # suppresses the "Distribution not specified, assuming ..."
      model <- gbm::gbm(
        formula = formula,
        data = data,
        verbose = verbose,
        n.minobsinnode = n.minobsinnode,
        ...
      )
    }))

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


#' LightGBM Learner
#'
#' A wrapper for \code{lightgbm::lgb.train()} for use in
#' \code{nadir::super_learner()}.
#'
#' \code{\{lightgbm\}} does not have a formula interface, so internally this
#' learner constructs a numeric design matrix via
#' \code{stats::model.matrix.lm()} (as in \code{lnr_xgboost}) and trains on
#' that.
#'
#' Note that \code{min_data_in_leaf} defaults to 1 here (rather than the
#' \code{\{lightgbm\}} default of 20) to account for the potential of very
#' small splits in cross-fitting; users can override this by passing
#' \code{min_data_in_leaf} explicitly.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @param weights Optional observation weights.
#' @param nrounds Number of boosting iterations.
#' @param objective Optional objective. For binary Super Learner fits,
#'   \code{nadir} supplies \code{objective = "binary"} through
#'   \code{outcome_type_dependent_args}. If unspecified, defaults to
#'   \code{"regression"}.
#' @param verbose (default: -1) Verbosity level passed to
#'   \code{lightgbm}; -1 suppresses lightgbm's messages, larger values
#'   (0, 1, 2) show progressively more output.
#' @param ... Additional named parameters passed into the \code{params} list
#'   of \code{lightgbm::lgb.train()}, e.g. \code{learning_rate},
#'   \code{num_leaves}, \code{min_data_in_leaf}.
#'
#' @returns A prediction function that accepts \code{newdata} and returns a
#'   numeric vector of predictions (probabilities of the outcome being 1
#'   when \code{objective = "binary"}), one for each row of \code{newdata}.
#'
#' @export
#' @importFrom lightgbm lgb.Dataset lgb.train
#'
#' @examples
#' lnr_lightgbm(mtcars, mpg ~ hp + wt, nrounds = 10)(mtcars)
#'
#' lnr_lightgbm(mtcars, am ~ cyl + disp + hp, objective = "binary",
#'   nrounds = 10)(mtcars)
lnr_lightgbm <-
  function(data,
           formula,
           weights = NULL,
           nrounds = 100,
           objective = NULL,
           verbose = -1,
           ...) {

    yvar <- as.character(formula)[[2]]
    y <- data[[yvar]]

    # Use a right-hand-side-only formula for model.matrix().
    # This ensures that prediction does not require the outcome column to be
    # present in newdata.
    x_formula <- formula
    x_formula[[2]] <- NULL

    xdata <- stats::model.matrix.lm(
      object = x_formula,
      data = data,
      na.action = "na.pass"
    )

    # LightGBM does not need an intercept column for tree-based learners.
    if ("(Intercept)" %in% colnames(xdata)) {
      xdata <- xdata[, colnames(xdata) != "(Intercept)", drop = FALSE]
    }

    # lgb.Dataset expects numeric labels.
    #
    # For binary outcomes, encode:
    #   FALSE / first factor level / 0 -> 0
    #   TRUE  / second factor level / 1 -> 1
    if (is.factor(y)) {
      if (nlevels(y) != 2L && identical(objective, "binary")) {
        stop(
          "For objective = 'binary', factor outcomes must have ",
          "exactly two levels."
        )
      }
      y <- as.integer(y) - 1L
    } else if (is.logical(y)) {
      y <- as.integer(y)
    } else {
      y <- as.numeric(y)
    }

    if (identical(objective, "binary")) {
      observed_y <- sort(unique(stats::na.omit(y)))
      if (!all(observed_y %in% c(0, 1))) {
        stop(
          "For objective = 'binary', the outcome must be coded as ",
          "0/1, logical, or a two-level factor."
        )
      }
    }

    # Merge parameters:
    #   1. defaults chosen for Super Learner use (small CV folds),
    #   2. objective supplied directly, often by outcome_type_dependent_args,
    #   3. additional named parameters in ...
    #
    # Later sources override earlier ones.
    params <- list(
      min_data_in_leaf = 1L,
      verbosity = verbose
    )

    params$objective <- if (is.null(objective)) "regression" else objective

    dot_args <- list(...)
    if (length(dot_args) > 0L) {
      params <- utils::modifyList(params, dot_args)
    }

    dtrain <- lightgbm::lgb.Dataset(
      data = xdata,
      label = y,
      weight = weights
    )

    model <- lightgbm::lgb.train(
      params = params,
      data = dtrain,
      nrounds = nrounds,
      verbose = verbose
    )

    return(function(newdata) {
      newdata_mat <- stats::model.matrix.lm(
        object = x_formula,
        data = newdata,
        na.action = "na.pass"
      )

      if ("(Intercept)" %in% colnames(newdata_mat)) {
        newdata_mat <- newdata_mat[
          ,
          colnames(newdata_mat) != "(Intercept)",
          drop = FALSE
        ]
      }

      # for objective = "binary", lightgbm returns probabilities by default,
      # so no type = 'response' analogue is needed.
      as.numeric(predict(model, newdata_mat))
    })
  }

attr(lnr_lightgbm, "sl_lnr_name") <- "lightgbm"
attr(lnr_lightgbm, "sl_lnr_type") <- c("continuous", "binary")
attr(lnr_lightgbm, "outcome_type_dependent_args") <- list(
  "continuous" = list(objective = "regression"),
  "binary" = list(objective = "binary")
)


#' k-Nearest Neighbors Learner
#'
#' A wrapper for \code{kknn::kknn()} for use in \code{nadir::super_learner()}.
#'
#' k-nearest neighbors is a "lazy" learner: no model is fit at training time.
#' Instead, the training data are stored and predictions for \code{newdata}
#' are formed as a (kernel-weighted) average of the outcomes of the \code{k}
#' nearest training observations in covariate space.  As a purely local,
#' instance-based method, kNN occupies a very different corner of the
#' bias-variance landscape than the global regression methods
#' (\code{lnr_lm}, \code{lnr_glmnet}, etc.) and the tree ensembles
#' (\code{lnr_rf}, \code{lnr_ranger}, \code{lnr_xgboost}), making it a useful
#' addition to a super learner library.
#'
#' Note that \code{kknn::kknn()} does not support observation weights, so no
#' \code{weights} argument is accepted here.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @param k The number of nearest neighbors to use; see \code{?kknn::kknn}.
#' @export
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @examples
#' lnr_knn(mtcars, mpg ~ hp + disp + wt)(mtcars)
#' lnr_knn(mtcars, mpg ~ hp + disp + wt, k = 5)(mtcars)
lnr_knn <- function(data, formula, k = 7, ...) {
  y_variable <- as.character(formula)[[2]]

  return(function(newdata) {
    # kknn constructs a model.frame on the test data, so the outcome column
    # must be present in newdata; its values are ignored in prediction.
    if (! y_variable %in% colnames(newdata)) {
      newdata[[y_variable]] <- data[[y_variable]][1]
    }
    fit <- kknn::kknn(
      formula = formula,
      train = data,
      test = newdata,
      k = k,
      ...)
    as.vector(fit$fitted.values)
  })
}
attr(lnr_knn, 'sl_lnr_name') <- 'knn'
attr(lnr_knn, 'sl_lnr_type') <- 'continuous'


#' Support Vector Machine Learner
#'
#' A wrapper for \code{e1071::svm()} for use in \code{nadir::super_learner()},
#' performing support vector (eps-)regression.
#'
#' Support vector regression fits a function in a kernel-induced feature space
#' (radial basis by default) that is at most \eqn{\varepsilon} away from the
#' observed outcomes wherever possible, yielding a flexible, margin-based
#' regression paradigm not otherwise represented in the built-in learners.
#' Kernel choice and hyperparameters (e.g., \code{kernel}, \code{cost},
#' \code{gamma}, \code{epsilon}) may be passed through \code{...}.
#'
#' Note that \code{e1071::svm()} does not support observation weights, so no
#' \code{weights} argument is accepted here.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @export
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @examples
#' lnr_svm(mtcars, mpg ~ hp + disp + wt)(mtcars)
#' lnr_svm(mtcars, mpg ~ ., kernel = 'polynomial', cost = 2)(mtcars)
lnr_svm <- function(data, formula, ...) {
  model <- e1071::svm(formula = formula, data = data, ...)
  y_variable <- as.character(formula)[[2]]

  return(function(newdata) {
    # predict.svm() applies na.omit to the full model frame including the
    # response, so NA outcome values in newdata would silently drop rows;
    # the response plays no role in prediction, so fill it with a dummy.
    if (y_variable %in% colnames(newdata) &&
        any(is.na(newdata[[y_variable]]))) {
      newdata[[y_variable]] <- data[[y_variable]][1]
    }
    as.vector(predict(model, newdata = newdata))
  })
}
attr(lnr_svm, 'sl_lnr_name') <- 'svm'
attr(lnr_svm, 'sl_lnr_type') <- 'continuous'


#' Recursive Partitioning (CART) Learner
#'
#' A wrapper for \code{rpart::rpart()} for use in \code{nadir::super_learner()}.
#'
#' A single regression tree is a classic member of super learner libraries:
#' it is fast, handles interactions and nonlinearities automatically, and its
#' predictions are piecewise-constant, complementing the smooth learners in
#' the library. Complexity may be controlled through \code{rpart.control}
#' arguments passed via \code{...} (e.g., \code{cp}, \code{minsplit},
#' \code{maxdepth}).
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @export
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @examples
#' lnr_rpart(mtcars, mpg ~ hp + disp + wt)(mtcars)
#' lnr_rpart(mtcars, mpg ~ ., cp = 0.05)(mtcars)
lnr_rpart <- function(data, formula, weights = NULL, ...) {
  model_args <- list(
    formula = formula,
    data = data,
    method = 'anova')
  if (! is.null(weights)) {
    model_args$weights <- weights
  }
  model <- do.call(rpart::rpart, args = c(model_args, list(...)))

  return(function(newdata) {
    as.vector(predict(model, newdata = newdata))
  })
}
attr(lnr_rpart, 'sl_lnr_name') <- 'rpart'
attr(lnr_rpart, 'sl_lnr_type') <- 'continuous'


#' Bayesian Additive Regression Trees (BART) Learner
#'
#' A wrapper for \code{dbarts::bart2()} for use in
#' \code{nadir::super_learner()}.
#'
#' BART is a Bayesian nonparametric sum-of-trees model with strong empirical
#' performance in the causal inference and prediction literature; predictions
#' returned are posterior mean predictions. Hyperparameters such as
#' \code{n.trees}, \code{n.samples}, and \code{n.burn} may be passed through
#' \code{...}.
#'
#' Categorical predictor levels are retained from the training data so that
#' prediction data are encoded using the same design matrix as the training
#' data, even when some factor levels are absent from \code{newdata}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @export
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#'
#' @examples
#' \donttest{
#' lnr_bart(mtcars, mpg ~ hp + disp + wt)(mtcars)
#' }
lnr_bart <- function(data, formula, weights = NULL, ...) {

  # 1. Fit BART model
  dots <- list(...)

  # Prediction from a fitted dbarts model requires the sampler/trees
  # to be retained.
  if ("keepTrees" %in% names(dots)) {
    if (!isTRUE(dots$keepTrees)) {
      warning(
        "lnr_bart requires keepTrees = TRUE for prediction; ",
        "the supplied keepTrees argument will be ignored."
      )
    }
    dots$keepTrees <- NULL
  }

  model_args <- list(
    formula = formula,
    data = data,
    keepTrees = TRUE,
    verbose = FALSE
  )

  if (!is.null(weights)) {
    model_args$weights <- weights
  }
  model <- do.call(dbarts::bart2, args = c(model_args, dots))

  # 2. Record categorical encoding from training data

  # dbarts stores the predictor names actually used by the fitted model here.
  predictor_names <- attr(model$fit$data@x, "term.labels")

  categorical_levels <- lapply(
    predictor_names,
    function(variable) {

      if (!variable %in% colnames(data)) {
        return(NULL)
      }

      x <- data[[variable]]

      if (is.factor(x)) {
        return(levels(x))
      }

      if (is.character(x)) {
        # This reproduces dbarts' conversion of character vectors to factors
        return(levels(factor(x)))
      }
      NULL
    }
  )

  names(categorical_levels) <- predictor_names

  categorical_levels <- categorical_levels[
    !vapply(categorical_levels, is.null, logical(1))
  ]


  # Return prediction function
  return(function(newdata) {
    newdata <- as.data.frame(newdata)

    # Restore the categorical structure used during training.
    for (variable in names(categorical_levels)) {

      if (!variable %in% colnames(newdata)) {
        stop(
          "Prediction data are missing predictor '",
          variable,
          "'.",
          call. = FALSE
        )
      }

      training_levels <- categorical_levels[[variable]]
      values <- as.character(newdata[[variable]])
      unseen_levels <- setdiff(
        unique(values[!is.na(values)]),
        training_levels
      )

      if (length(unseen_levels) > 0L) {
        stop(
          "Predictor '",
          variable,
          "' contains level(s) not present in the BART training data: ",
          paste(unseen_levels, collapse = ", "),
          ".",
          call. = FALSE
        )
      }
      newdata[[variable]] <- factor(values, levels = training_levels)
    }

    # Construct the test matrix using dbarts' stored training specification.
    #
    # This is preferable to model.matrix(), because dbarts has its own
    # factor/dummy-variable representation and stores the corresponding
    # column-dropping information in model$fit$data.
    newx <- withCallingHandlers(
      dbarts::makeTestModelMatrix(
        model$fit$data,
        newdata
      ),
      warning = function(w) {

        # dbarts normally falls back to matching columns by position if their
        # names differ. For our purposes in super learning, that is too dangerous.
        if (grepl(
          "column names of 'test' does not equal that of 'x'",
          conditionMessage(w),
          fixed = TRUE
        )) {
          stop(
            "Could not construct a BART prediction design matrix matching ",
            "the training design matrix.",
            call. = FALSE
          )
        }
      }
    )

    # Extra check: BART trees refer to predictors by column position,
    # so we require exact agreement with the training matrix.
    training_columns <- colnames(model$fit$data@x)

    if (!identical(colnames(newx), training_columns)) {
      stop(
        "The BART prediction design matrix does not match the ",
        "training design matrix.",
        call. = FALSE
      )
    }

    # predict.bart returns posterior samples as
    # posterior draws x observations when chains are combined.
    posterior_samples <- predict(
      model,
      newdata = newx,
      combineChains = TRUE
    )

    as.vector(colMeans(posterior_samples))
  })
}
attr(lnr_bart, 'sl_lnr_name') <- 'bart'
attr(lnr_bart, 'sl_lnr_type') <- 'continuous'


#' Projection Pursuit Regression Learner
#'
#' A wrapper for \code{stats::ppr()} for use in \code{nadir::super_learner()}.
#'
#' Projection pursuit regression models the outcome as a sum of smooth ridge
#' functions of linear combinations of the covariates,
#' \eqn{\hat{y} = \sum_m g_m(\alpha_m^\top x)}. It captures interactions and
#' nonlinearities along learned directions in covariate space, and being
#' part of base R's \code{stats} package, adds no new dependencies.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @param nterms Number of ridge terms to include in the final model;
#' see \code{?stats::ppr}.
#' @export
#' @importFrom stats ppr
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @examples
#' lnr_ppr(mtcars, mpg ~ hp + disp + wt)(mtcars)
#' lnr_ppr(mtcars, mpg ~ ., nterms = 4)(mtcars)
lnr_ppr <- function(data, formula, weights = NULL, nterms = 3, ...) {
  model_args <- list(
    formula = formula,
    data = data,
    nterms = nterms)
  if (! is.null(weights)) {
    model_args$weights <- weights
  }
  model <- do.call(stats::ppr, args = c(model_args, list(...)))

  return(function(newdata) {
    as.vector(predict(model, newdata = newdata))
  })
}
attr(lnr_ppr, 'sl_lnr_name') <- 'ppr'
attr(lnr_ppr, 'sl_lnr_type') <- 'continuous'


#' Gaussian Process Regression Learner
#'
#' A wrapper for \code{kernlab::gausspr()} for use in
#' \code{nadir::super_learner()}.
#'
#' Gaussian process regression is a Bayesian kernel method that places a
#' prior over functions and returns the posterior mean prediction. It is a
#' smooth, nonparametric paradigm distinct from both the tree ensembles and
#' the penalized regressions already in the library. The kernel and its
#' hyperparameters can be specified through \code{...} (e.g.,
#' \code{kernel = 'rbfdot'}, \code{kpar = list(sigma = 0.1)}).
#'
#' Note that \code{kernlab::gausspr()} does not support observation weights,
#' so no \code{weights} argument is accepted here.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @export
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @examples
#' lnr_gausspr(mtcars, mpg ~ hp + disp + wt)(mtcars)
lnr_gausspr <- function(data, formula, ...) {
  # kernlab::gausspr() cat()s a message about automatic sigma estimation on
  # every fit; capture it so cross-validation output stays clean.
  invisible(utils::capture.output(
    model <- kernlab::gausspr(x = formula, data = data, ...)
  ))

  return(function(newdata) {
    as.vector(kernlab::predict(model, newdata))
  })
}
attr(lnr_gausspr, 'sl_lnr_name') <- 'gausspr'
attr(lnr_gausspr, 'sl_lnr_type') <- 'continuous'


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
#'  lnr_glm <- function(data, formula, weights = NULL, ...) {
#'   model <- stats::glm(formula = formula, data = data, weights = weights, ...)
#'
#'   return(function(newdata) {
#'     predict(model, newdata = newdata, type = 'response')
#'   })
#'  }
#'
#' @rdname learners
#' @name learners
#' @keywords learners
NULL
