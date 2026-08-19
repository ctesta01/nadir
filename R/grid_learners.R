# Grid ("multi-predictor") learners --------------------------------------
#
# Some algorithms -- most notably glmnet and the highly adaptive lasso -- are
# dramatically cheaper to fit across an entire decreasing grid of lambda
# penalty values than they are to fit at each lambda value separately, because
# they exploit warm starts along the regularization path.
#
# To take advantage of this inside super_learner(), a learner may return a
# *multi-predictor*: a named list of prediction functions (one per lambda)
# instead of a single prediction function. super_learner() detects
# multi-predictors after the fitting stage and expands each one into distinct
# pseudo-learners named e.g. glmnet_grid_lambda_0.1, glmnet_grid_lambda_0.5, ...
#
# From the meta-learning stage onwards, these pseudo-learners are
# indistinguishable from ordinary learners: each contributes one column of
# held-out predictions, receives its own ensemble weight, shows up in
# compare_learners(), and so on. The only special-casing lives at the two
# points where models are fit (the CV stage and the final full-data fit).

#' Construct a Multi-Predictor
#'
#' A multi-predictor is a named list of prediction functions produced by a
#' single call to a learner, e.g. one prediction function per lambda value
#' along a jointly-fit regularization path. \code{super_learner()} expands
#' multi-predictors into distinct pseudo-learners (one per element) whose
#' names are formed as \code{paste(learner_name, names(predictors), sep = '_')}.
#'
#' Learner authors may wrap any algorithm that fits many models in one pass
#' (e.g., a lasso path, a boosting run saved at several iteration counts, etc.) with
#' this to expose each sub-model along the path to the meta-learning stage of
#' \code{super_learner()} at the cost of a single fit.
#'
#' The sub-model names must be identical across calls on different training
#' folds (i.e., deterministic given the learner arguments, not data-dependent),
#' since \code{super_learner()} aligns pseudo-learners across folds by name.
#'
#' @param predictors A uniquely-named list of functions, each of which accepts
#'   \code{newdata} and returns a numeric vector of predictions with one entry
#'   per row of \code{newdata}.
#' @returns The same list, classed as \code{nadir_multi_predictor}.
#' @seealso lnr_glmnet_grid lnr_hal_grid
#' @export
as_multi_predictor <- function(predictors) {
  if (! is.list(predictors) ||
      length(predictors) == 0 ||
      ! all(vapply(predictors, is.function, logical(1)))) {
    stop("as_multi_predictor() requires a nonempty list of functions.")
  }
  if (is.null(names(predictors)) ||
      any(names(predictors) == "") ||
      anyDuplicated(names(predictors)) > 0) {
    stop("as_multi_predictor() requires the list of prediction functions to
have unique, nonempty names, since sub-learners are aligned across
cross-validation folds by name.")
  }
  structure(predictors, class = 'nadir_multi_predictor')
}

#' Test for Multi-Predictors
#' @param x An object.
#' @returns Logical; whether \code{x} is a \code{nadir_multi_predictor}.
#' @keywords internal
is_multi_predictor <- function(x) {
  inherits(x, 'nadir_multi_predictor')
}

#' Format Lambda Values into Sub-Learner Names
#' @param lambda A numeric vector of penalty values.
#' @returns A character vector of unique labels like \code{lambda_0.1}.
#' @keywords internal
lambda_labels <- function(lambda) {
  labels <- paste0('lambda_', as.character(signif(lambda, digits = 6)))
  make.unique(labels, sep = '_')
}

#' Validate a User-Supplied Lambda Grid
#'
#' Grid learners require an explicit, fixed lambda grid: if lambda were left
#' for the underlying package to choose, each cross-validation fold would
#' generate a *different*, data-dependent lambda sequence, and the
#' "same-named" pseudo-learner would refer to different models in different
#' folds -- silently invalidating the cross-validated risk estimates used in
#' the meta-learning stage.
#'
#' @param lambda The lambda argument passed by the user.
#' @param learner_name Character name used in error messages.
#' @returns The lambda grid, deduplicated and sorted in decreasing order (the
#'   order in which pathwise coordinate descent proceeds).
#' @keywords internal
validate_lambda_grid <- function(lambda, learner_name) {
  if (missing(lambda) || is.null(lambda) || ! is.numeric(lambda) ||
      length(lambda) < 1 || any(is.na(lambda)) || any(lambda < 0)) {
    stop(paste0(learner_name, " requires an explicit numeric grid of lambda >= 0
values (e.g. lambda = exp(seq(log(1), log(.001), length.out = 50))).

An explicit grid is required because sub-learners are matched across
cross-validation folds by their lambda value; letting the underlying package
auto-generate a (data-dependent) lambda sequence would produce different
grids on different training folds."))
  }
  sort(unique(lambda), decreasing = TRUE)
}

#' glmnet Learner over a Grid of Lambda Values
#'
#' A wrapper for \code{glmnet::glmnet()} fit jointly over a decreasing grid of
#' lambda values, for use in \code{nadir::super_learner()}.
#'
#' \code{glmnet} exploits warm starts along the regularization path, so
#' fitting an entire grid of lambda values costs little more than fitting one
#' -- and is often *faster* than fitting a single small lambda from a cold
#' start. \code{lnr_glmnet_grid} fits the whole path once and returns a
#' multi-predictor: one prediction function per lambda value.
#' \code{super_learner()} expands these into distinct pseudo-learners (named
#' e.g. \code{glmnet_grid_lambda_0.1}) which each receive their own weight in
#' the meta-learning stage.
#'
#' Compare with \code{lnr_glmnet}, which fits a single lambda value, and
#' \code{lnr_cvglmnet}, which internally selects one lambda by
#' cross-validation and exposes only that model to the meta-learner.
#'
#' @inheritParams lnr_lm
#' @param lambda A numeric vector of penalty values (a grid). Must be
#'   explicitly specified; see \code{validate_lambda_grid} for why
#'   auto-generated grids are not supported. Values are deduplicated and
#'   sorted in decreasing order.
#' @seealso learners lnr_glmnet lnr_hal_grid as_multi_predictor
#' @export
#' @returns A \code{nadir_multi_predictor}: a named list of prediction
#'   functions, one per lambda value, each of which accepts \code{newdata} and
#'   returns a numeric vector of predictions.
#' @importFrom stats model.matrix
#' @importFrom glmnet glmnet predict.glmnet
#' @examples
#' multi_predictor <- lnr_glmnet_grid(
#'   mtcars, mpg ~ hp + disp + am + wt,
#'   lambda = c(0.01, 0.1, 0.5, 1))
#' names(multi_predictor)
#' multi_predictor[['lambda_0.5']](mtcars)
lnr_glmnet_grid <- function(data, formula, weights = NULL, lambda, ...) {
  lambda <- validate_lambda_grid(lambda, 'lnr_glmnet_grid')

  # glmnet takes Y and X separately, so we shall pull them out from the
  # data based on the formula
  yvar <- as.character(formula[[2]])
  formula_without_lhs <- formula
  formula_without_lhs[2] <- NULL
  xdata <- model.matrix.default(formula_without_lhs, data = data)
  if (yvar %in% colnames(xdata)) {
    yvar_idx <- which(colnames(xdata) == yvar)
    xdata <- xdata[,-yvar_idx]
  }

  # one fit for the entire path: this is where the warm-start savings happen
  model <- glmnet::glmnet(y = data[[yvar]], x = xdata, lambda = lambda,
                          weights = weights, ...)

  # a small memoization environment shared across the per-lambda prediction
  # closures: if the same newdata is passed consecutively (as happens when
  # super_learner() predicts each pseudo-learner on the same validation fold),
  # the model matrix and the full prediction matrix are computed only once.
  # this is purely an optimization; a cache miss is always safe.
  prediction_cache <- new.env(parent = emptyenv())
  prediction_cache$newdata <- NULL

  predict_matrix_for <- function(newdata) {
    if (yvar %in% colnames(newdata)) {
      newdata[[yvar]] <- NULL
    }
    if (! identical(prediction_cache$newdata, newdata)) {
      # the formula's lhs is removed so that newdata is not required to
      # contain the outcome variable at prediction time
      formula_without_lhs <- formula
      formula_without_lhs[2] <- NULL
      new_xdata <- model.matrix.default(formula_without_lhs, data = newdata)
      # note: the generic predict() is used (rather than
      # glmnet::predict.glmnet directly) so that S3 dispatch reaches
      # predict.lognet for binomial fits, where type = 'response' converts
      # from the link scale to probabilities
      predictions <- predict(model, newx = new_xdata, type = 'response')
      prediction_cache$newdata <- newdata
      prediction_cache$predictions <- predictions
    }
    prediction_cache$predictions
  }

  predictors <- lapply(seq_along(lambda), function(lambda_k) {
    force(lambda_k)
    function(newdata) {
      as.vector(predict_matrix_for(newdata)[, lambda_k])
    }
  })
  names(predictors) <- lambda_labels(lambda)
  as_multi_predictor(predictors)
}
attr(lnr_glmnet_grid, 'sl_lnr_name') <- 'glmnet_grid'
attr(lnr_glmnet_grid, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_glmnet_grid, 'outcome_type_dependent_args') <- list(
  'binary' = list(family = binomial(link = 'logit')))


#' Highly Adaptive Lasso over a Grid of Lambda Values
#'
#' A wrapper for \code{hal9001::fit_hal()} fit jointly over a decreasing grid
#' of lambda values, for use in \code{nadir::super_learner()}.
#'
#' Like \code{lnr_glmnet_grid}, this fits the entire lasso path (over the HAL
#' basis expansion) in a single call and returns a multi-predictor with one
#' prediction function per lambda value, which \code{super_learner()} expands
#' into distinct pseudo-learners. Internally \code{fit_hal} is called with
#' \code{fit_control = list(cv_select = FALSE)} so that the fits at every
#' lambda in the grid are retained rather than one being selected by
#' \code{cv.glmnet}.
#'
#' Since the HAL basis expansion (typically the expensive step at prediction
#' time) is shared across the whole grid, predictions for all lambda values on
#' the same \code{newdata} are computed once and cached.
#'
#' @inheritParams lnr_glmnet_grid
#' @seealso learners lnr_hal lnr_glmnet_grid as_multi_predictor
#' @export
#' @returns A \code{nadir_multi_predictor}: a named list of prediction
#'   functions, one per lambda value, each of which accepts \code{newdata} and
#'   returns a numeric vector of predictions.
#' @importFrom hal9001 fit_hal
#' @examples
#' \donttest{
#' suppressWarnings({
#' multi_predictor <- lnr_hal_grid(
#'   mtcars, mpg ~ hp + wt,
#'   lambda = c(0.01, 0.1, 1),
#'   max_degree = 1, num_knots = 3)
#' })
#' names(multi_predictor)
#' }
lnr_hal_grid <- function(data, formula, weights = NULL, lambda, ...) {
  lambda <- validate_lambda_grid(lambda, 'lnr_hal_grid')

  yvar <- as.character(formula[[2]])
  xdata <- stats::model.matrix.lm(formula, data = data, na.action = 'na.pass')

  # cv_select must be FALSE so that fit_hal retains the fit at every lambda in
  # the grid; merge it into any user-supplied fit_control rather than
  # clobbering the user's other fit_control options.
  dots <- list(...)
  fit_control <- dots[['fit_control']]
  if (is.null(fit_control)) {
    fit_control <- list()
  }
  if (isTRUE(fit_control[['cv_select']])) {
    warning("lnr_hal_grid sets fit_control$cv_select = FALSE (overriding the
user-specified value) since every lambda in the grid is exposed to the
meta-learning stage of super_learner() as its own pseudo-learner. To have
cv.glmnet select a single lambda internally, use lnr_hal instead.")
  }
  fit_control[['cv_select']] <- FALSE
  dots[['fit_control']] <- fit_control

  model <- do.call(
    hal9001::fit_hal,
    c(list(Y = data[[yvar]], X = xdata, lambda = lambda, weights = weights),
      dots))

  # shared memoization of the (expensive) HAL basis expansion + prediction
  # matrix across the per-lambda closures; see lnr_glmnet_grid for details.
  prediction_cache <- new.env(parent = emptyenv())
  prediction_cache$newdata <- NULL

  predict_matrix_for <- function(newdata) {
    if (! identical(prediction_cache$newdata, newdata)) {
      # ensure the y-variable isn't required inside the model matrix call
      prediction_formula <- formula
      if (length(prediction_formula) >= 3) {
        prediction_formula[2] <- NULL
      }
      new_xdata <- stats::model.matrix.lm(
        prediction_formula, data = newdata, na.action = 'na.pass')
      predictions <- predict(object = model, new_data = new_xdata,
                             type = 'response')
      # with a length-1 lambda grid, predict may return a vector; standardize
      predictions <- as.matrix(predictions)
      prediction_cache$newdata <- newdata
      prediction_cache$predictions <- predictions
    }
    prediction_cache$predictions
  }

  predictors <- lapply(seq_along(lambda), function(lambda_k) {
    force(lambda_k)
    function(newdata) {
      as.vector(predict_matrix_for(newdata)[, lambda_k])
    }
  })
  names(predictors) <- lambda_labels(lambda)
  as_multi_predictor(predictors)
}
attr(lnr_hal_grid, 'sl_lnr_name') <- 'hal_grid'
attr(lnr_hal_grid, 'sl_lnr_type') <- c('continuous', 'binary')
attr(lnr_hal_grid, 'outcome_type_dependent_args') <- list(
  'binary' = list(family = 'binomial'))


#' Expand Multi-Predictor Fits into Pseudo-Learner Rows
#'
#' Given the \code{trained_learners} tibble from \code{super_learner()} (one
#' row per learner-fold combination, with fitted predictors in the
#' \code{learned_predictor} list-column), expand every row whose fit is a
#' \code{nadir_multi_predictor} into one row per sub-model, named
#' \code{paste(learner_name, sub_model_name, sep = '_')}.
#'
#' If a multi-predictor learner failed to train on *any* fold (its fit is an
#' error object rather than a multi-predictor), none of its folds are
#' expanded: the learner is left as a single row per fold so that the existing
#' erring-learner machinery drops it wholesale, since its sub-learners could
#' not be aligned across all folds anyway.
#'
#' @param trained_learners A tibble with columns \code{.sl_fold},
#'   \code{learner_name}, and \code{learned_predictor}.
#' @returns A tibble of the same structure, with multi-predictor rows expanded.
#' @keywords internal
expand_multi_predictor_fits <- function(trained_learners) {
  any_multi <- any(vapply(trained_learners[['learned_predictor']],
                          is_multi_predictor, logical(1)))
  if (! any_multi) {
    return(trained_learners)
  }

  base_learner_names <- unique(trained_learners[['learner_name']])
  multi_learner_map <- list()

  expanded_blocks <- lapply(base_learner_names, function(base_name) {
    block_idx <- which(trained_learners[['learner_name']] == base_name)
    block <- trained_learners[block_idx, ]
    fits <- block[['learned_predictor']]
    fit_is_multi <- vapply(fits, is_multi_predictor, logical(1))

    # not a grid learner: pass through untouched
    if (! any(fit_is_multi)) {
      return(block)
    }

    # a grid learner that failed on >= 1 fold: leave unexpanded so the
    # erring-learner machinery drops it as a whole (see function docs)
    if (! all(fit_is_multi)) {
      return(block)
    }

    # sub-learners are aligned across folds by name, so the names must agree
    sub_names <- lapply(fits, names)
    if (length(unique(sub_names)) != 1) {
      stop(paste0(
        "The multi-predictor learner '", base_name, "' returned differently ",
        "named sub-models on different cross-validation folds. Sub-model ",
        "names must be deterministic given the learner arguments (e.g., an ",
        "explicit fixed lambda grid), not data-dependent."))
    }
    sub_names <- sub_names[[1]]
    multi_learner_map[[base_name]] <<- paste(base_name, sub_names, sep = '_')

    # one block of n_folds rows per sub-model, preserving fold order,
    # so that downstream pivoting sees them as ordinary learners
    do.call(rbind, lapply(sub_names, function(sub_name) {
      tibble::tibble(
        .sl_fold = block[['.sl_fold']],
        learner_name = paste(base_name, sub_name, sep = '_'),
        learned_predictor = lapply(fits, function(fit) fit[[sub_name]]))
    }))
  })

  expanded <- do.call(rbind, expanded_blocks)

  # every learner name (expanded or not) should appear exactly once per fold;
  # anything else indicates a name collision between an expanded sub-learner
  # and another learner
  per_fold_counts <- table(expanded[['learner_name']])
  n_folds <- length(unique(trained_learners[['.sl_fold']]))
  if (any(per_fold_counts != n_folds)) {
    stop(paste0(
      "After expanding multi-predictor learners, the following learner names ",
      "collide or are missing folds: ",
      paste(names(per_fold_counts)[per_fold_counts != n_folds], collapse = ', '),
      ". Rename your learners so that expanded sub-learner names are unique."))
  }

  attr(expanded, 'multi_learner_map') <- multi_learner_map
  expanded
}

#' Flatten Fitted Learners, Expanding Multi-Predictors
#'
#' Given the list of learners fit on the full dataset by
#' \code{super_learner()}, expand any \code{nadir_multi_predictor} entries
#' into their component prediction functions, named consistently with
#' \code{expand_multi_predictor_fits} (i.e.,
#' \code{paste(learner_name, sub_model_name, sep = '_')}). Non-function
#' entries (e.g. error objects captured during fitting) are passed through
#' under their original names.
#'
#' @param fit_learners A named list of fitted predictors (functions,
#'   multi-predictors, or error objects).
#' @returns A named flat list of predictors.
#' @keywords internal
flatten_fit_learners <- function(fit_learners) {
  flattened <- list()
  for (i in seq_along(fit_learners)) {
    fit <- fit_learners[[i]]
    learner_name <- names(fit_learners)[[i]]
    if (is_multi_predictor(fit)) {
      for (sub_name in names(fit)) {
        flattened[[paste(learner_name, sub_name, sep = '_')]] <- fit[[sub_name]]
      }
    } else {
      flattened[[learner_name]] <- fit
    }
  }
  flattened
}
