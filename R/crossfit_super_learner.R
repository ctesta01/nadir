#' Cross-Fit a Super Learner and Retain Fold-Specific Predictors
#'
#' Cross-fitting proceeds by:
#' \enumerate{
#'   \item splitting the data into \code{n_folds} outer folds,
#'   \item fitting a full \code{\link{super_learner}()} (with its own internal
#'     \code{inner_n_folds}-fold cross-validation to determine ensemble
#'     weights) on each outer training split,
#'   \item predicting on the corresponding outer held-out fold,
#'   \item retaining each fold-specific prediction function for later
#'     modified-data (e.g. intervention) prediction.
#' }
#'
#' Each observation's out-of-fold prediction therefore comes from an ensemble
#' whose base learners \emph{and} ensemble weights were estimated entirely
#' without that observation, which is the property required of cross-fitted
#' nuisance estimators in semiparametric workflows (AIPW, TMLE, DML). Typical
#' use is to obtain out-of-fold fits of an outcome model \eqn{m}, its
#' interventional counterparts \eqn{m_1, m_0} (via \code{$predict_modified()}),
#' and a propensity model \eqn{g}, each from one \code{crossfit_super_learner()}
#' call per nuisance.
#'
#' @section Parallelism:
#' Make sure to load the \code{future} package first using \code{library(future)}.
#'
#' Fitting is parallelized over outer folds through
#' \code{future.apply::future_lapply()}. With \code{plan(multisession)} only
#' the outer folds run in parallel and each inner \code{super_learner()} runs
#' sequentially inside its worker. To also
#' parallelize within each outer fold, declare a nested topology, e.g.
#' \code{plan(list(tweak(multisession, workers = 2),
#' tweak(multisession, workers = I(4))))}; the product across levels should
#' not exceed \code{future::availableCores()}.
#'
#' @srrstats {G2.0, G2.1} Lengths and types of n_folds, y_variable,
#'   cluster_ids, strata_ids, weights, learners are asserted with
#'   documented expectations.  [super_learner, cv_super_learner,
#'   crossfit_super_learner]
#' @srrstats {G2.3, G2.3a} Character option arguments are restricted via
#'   match.arg() (outcome_type, ensemble_or_discrete).
#' @srrstats {G2.13, G2.14, G2.14a, G2.14b} Missing data error by default
#'   with an informative message; use_complete_cases = TRUE opts into
#'   complete-case filtering with a message describing the filtering.
#'   [super_learner, crossfit_super_learner]
#' @srrstats {G2.15} Functions check for missingness rather than assuming
#'   non-missing inputs (complete.cases() guards; NA-weight checks).
#' @srrstats {RE4.0} The output of models fit with nadir are model classes: \code{nadir_sl_model},
#'   \code{nadir_crossfit_sl}, \code{nadir_cv_sl}, which themselves have supporting
#'   regression related S3 methods.
#'
#' @inheritParams super_learner
#' @param y_variable Optional character name of the outcome variable; inferred
#'   from \code{formulas} when omitted.
#' @param n_folds Number of \emph{outer} cross-fitting folds.
#' @param inner_n_folds Number of folds used by the inner
#'   \code{super_learner()} on each outer training split to estimate ensemble
#'   weights. Deliberately decoupled from \code{n_folds}.
#' @param cv_schema Function taking \code{(data, n_folds)} and returning a
#'   list with \code{training_data} and \code{validation_data}, each of length
#'   \code{n_folds}, used for the \emph{outer} split. If omitted and
#'   \code{cluster_ids} or \code{strata_ids} are supplied,
#'   \code{\link{cv_origami_schema}} is used so the outer split respects them;
#'   otherwise \code{\link{cv_random_schema}}.
#' @param inner_cv_schema Optional \code{cv_schema} for the inner
#'   \code{super_learner()} calls. If \code{NULL} (default),
#'   \code{super_learner()}'s own defaults apply (which respect
#'   \code{cluster_ids}/\code{strata_ids}, subset to each outer training fold).
#' @param determine_super_learner_weights Function used by the inner
#'   \code{super_learner()} calls to determine ensemble weights. If \code{NULL}
#'   (default), inferred from \code{outcome_type} via
#'   \code{default_determine_weights()}, matching \code{super_learner()}'s own
#'   inference.
#' @param loss_metric Optional loss function used only to report
#'   \code{cv_loss}, the empirical loss of the cross-fitted out-of-fold
#'   predictions. If \code{NULL} (default), inferred from \code{outcome_type}
#'   via \code{default_loss_metric()}.
#'
#' @returns An object of class \code{"nadir_crossfit_sl"}, a list with:
#' \describe{
#'   \item{\code{$oof_predictions()}}{Numeric vector of out-of-fold
#'     predictions in the original row order of \code{data} (after any
#'     complete-case filtering; see \code{$complete_rows}). Rows never held
#'     out by \code{cv_schema} are \code{NA}.}
#'   \item{\code{$predict_modified(modify)}}{Full-length out-of-fold
#'     predictions after applying \code{modify(newdata)} to each held-out
#'     fold, e.g. \code{modify = function(d) \{ d$treatment <- 1; d \}} to
#'     obtain cross-fitted \eqn{m_1}.}
#'   \item{\code{$predict_fold(newdata_list = NULL, modify = NULL)}}{List of
#'     per-fold prediction vectors; entry \code{i} is produced by the Super
#'     Learner trained without outer fold \code{i}.}
#'   \item{\code{$fold_assignments}}{Integer vector giving each row's outer
#'     validation fold (\code{NA} if never held out).}
#'   \item{\code{$fold_rows}}{List of row indices (into the analysis data)
#'     comprising each validation fold.}
#'   \item{\code{$sl_fits}}{List of the \code{n_folds} fitted
#'     \code{super_learner()} objects, e.g. for inspecting per-fold ensemble
#'     weights.}
#'   \item{\code{$cv_loss}}{Empirical loss of the cross-fitted predictions on
#'     held-out rows, or \code{NA} if it could not be computed.}
#'   \item{\code{$complete_rows}}{Indices into the originally supplied
#'     \code{data} retained after complete-case filtering (identity when no
#'     filtering occurred).}
#'   \item{\code{$warnings_from_fold_predictions}}{Present only if non-empty:
#'     warnings captured while predicting on the outer held-out folds.}
#'   \item{\code{$warnings_from_inner_super_learners}}{Present only if
#'     non-empty: a per-fold list of the warning fields captured by each
#'     inner \code{super_learner()} (see \code{?super_learner}).}
#'   \item{\code{$warning_learners}}{Present only if non-empty: names of
#'     learners that signaled warnings in any fold.}
#' }
#' plus \code{y_variable}, \code{outcome_type}, \code{n_folds},
#' \code{inner_n_folds}, \code{training_data}, and \code{validation_data}.
#'
#' @examples
#' \dontrun{
#' cf <- crossfit_super_learner(
#'   data = mtcars,
#'   formulas = mpg ~ disp + hp + am,
#'   learners = list(mean = lnr_mean, lm = lnr_lm, rf = lnr_rf),
#'   n_folds = 5
#' )
#'
#' cf$oof_predictions()          # cross-fitted \hat m(X_i)
#'
#' m1 <- cf$predict_modified(function(d) { d$am <- 1; d })  # \hat m(1, X_i)
#' m0 <- cf$predict_modified(function(d) { d$am <- 0; d })  # \hat m(0, X_i)
#'
#' cf$cv_loss                    # honest empirical loss
#' lapply(cf$sl_fits, `[[`, "learner_weights")  # per-fold ensemble weights
#' }
#'
#' @seealso super_learner cv_super_learner cv_origami_schema
#' @importFrom future.apply future_lapply
#' @importFrom stats complete.cases setNames
#' @export
crossfit_super_learner <- function(
    data,
    learners,
    formulas,
    y_variable = NULL,
    n_folds = 5,
    inner_n_folds = 5,
    determine_super_learner_weights = NULL,
    ensemble_or_discrete = c("ensemble", "discrete"),
    cv_schema = NULL,
    inner_cv_schema = NULL,
    outcome_type = c('continuous', 'binary', 'density', 'multiclass'),
    extra_learner_args = NULL,
    cluster_ids = NULL,
    strata_ids = NULL,
    weights = NULL,
    loss_metric = NULL,
    use_complete_cases = FALSE
) {

  ensemble_or_discrete <- match.arg(ensemble_or_discrete)
  outcome_type <- match.arg(outcome_type)

  # NOTE on NULL defaults: arguments referenced inside fit_one_fold() must
  # resolve to concrete values, because future.apply's static globals
  # inspection exports them to workers; a missing() promise there errors with
  # "argument ... is missing, with no default". So all optional arguments use
  # NULL defaults and are resolved to real values before the parallel region.

  # -------------------------------------------------------------------------
  # input validation
  # -------------------------------------------------------------------------
  if (length(n_folds) != 1L) stop("n_folds must be a length 1 numeric value.")
  n_folds <- as.integer(n_folds)
  if (is.na(n_folds) || n_folds < 2L) stop("n_folds must be an integer >= 2.")

  if (length(inner_n_folds) != 1L) stop("inner_n_folds must be a length 1 numeric value.")
  inner_n_folds <- as.integer(inner_n_folds)
  if (is.na(inner_n_folds) || inner_n_folds < 2L) stop("inner_n_folds must be an integer >= 2.")

  if (!is.list(learners)) stop("learners must be a list of learner functions. See ?learners.")
  if (!ensemble_or_discrete %in% c("ensemble", "discrete")) {
    stop("ensemble_or_discrete must be either 'ensemble' or 'discrete'.")
  }
  if (!outcome_type %in% c("continuous", "density", "binary", "multiclass")) {
    stop("outcome_type must be one of 'continuous', 'density', 'binary', 'multiclass'.")
  }
  if (!is.null(y_variable) && length(y_variable) != 1L) {
    stop("y_variable, if provided, must be a length 1 character string.")
  }
  if (!is.null(cluster_ids) && length(cluster_ids) != nrow(data)) {
    stop("cluster_ids must be equal in length to nrow(data).")
  }
  if (!is.null(strata_ids) && length(strata_ids) != nrow(data)) {
    stop("strata_ids must be equal in length to nrow(data).")
  }
  if (!is.null(weights) && (!is.numeric(weights) || length(weights) != nrow(data))) {
    stop("weights must be NULL or a numeric vector of length nrow(data).")
  }

  # G5.8a: zero-length and too-small data should error clearly, not fail
  # obscurely inside the CV fold construction.
  if (is.null(dim(data)) || nrow(data) == 0) {
    stop("data passed to nadir::super_learner() has zero rows.")
  }
  if (nrow(data) < n_folds) {
    stop("data passed to nadir::super_learner() has fewer rows (", nrow(data),
         ") than n_folds (", n_folds, "). ",
         "Reduce n_folds or provide more data.")
  }

  if (is.matrix(data)) data <- as.data.frame(data)

  # -------------------------------------------------------------------------
  # missing data handling (mirrors super_learner(), but done once, up front,
  # so cluster_ids / strata_ids / weights can be filtered consistently)
  # -------------------------------------------------------------------------
  complete_rows <- seq_len(nrow(data))
  if (!all(complete.cases(data))) {
    if (!use_complete_cases) {
      stop(
"nadir::crossfit_super_learner() does not have missing data imputation methods
built in. Pass use_complete_cases = TRUE to restrict to complete cases.")
    }
    message(
"Note that use_complete_cases = TRUE filters out rows with any missing data,
regardless of whether the missingness appears in a column referenced by the
formula(s) passed. Consider restricting data to the relevant columns first.")
    complete_rows <- which(complete.cases(data))
    data <- data[complete_rows, , drop = FALSE]
    if (!is.null(cluster_ids)) cluster_ids <- cluster_ids[complete_rows]
    if (!is.null(strata_ids))  strata_ids  <- strata_ids[complete_rows]
    if (!is.null(weights))     weights     <- weights[complete_rows]
  }
  n_obs <- nrow(data)

  y_variable <- extract_y_variable(
    formulas = formulas,
    data_colnames = colnames(data),
    learner_names = names(learners),
    y_variable = y_variable
  )

  # resolve outcome_type-dependent defaults eagerly (see NOTE above); these
  # helpers are the single source of truth shared with super_learner() et al.
  if (is.null(determine_super_learner_weights)) {
    determine_super_learner_weights <- default_determine_weights(outcome_type)
  }
  if (is.null(loss_metric)) {
    loss_metric <- default_loss_metric(outcome_type)
  }

  # -------------------------------------------------------------------------
  # outer cross-fitting split
  #
  # if the user did not supply a cv_schema but did supply cluster_ids or
  # strata_ids, the outer split must respect them -- route through
  # cv_origami_schema (this mirrors super_learner()'s own behavior).
  # -------------------------------------------------------------------------
  if (is.null(cv_schema)) {
    if (is.null(cluster_ids) && is.null(strata_ids)) {
      outer_schema <- cv_random_schema
    } else {
      outer_schema <- function(data, n_folds) {
        cv_origami_schema(
          data = data, n_folds = n_folds,
          fold_fun = origami::folds_vfold,
          cluster_ids = cluster_ids,
          strata_ids = strata_ids
        )
      }
    }
  } else {
    outer_schema <- cv_schema
    if (!is.null(cluster_ids) || !is.null(strata_ids)) {
      warning(
"A user-supplied cv_schema is being used for the outer cross-fitting split;
make sure it respects the cluster_ids/strata_ids you passed, since
crossfit_super_learner() cannot enforce that for arbitrary schemas.")
    }
  }

  if (".crossfit_rowid" %in% colnames(data)) {
    stop("data already has a .crossfit_rowid column; please rename it.")
  }
  data_cf <- data
  data_cf$.crossfit_rowid <- seq_len(n_obs)

  outer_splits <- outer_schema(data_cf, n_folds)
  training_data <- outer_splits$training_data
  validation_data <- outer_splits$validation_data

  if (length(training_data) != n_folds || length(validation_data) != n_folds) {
    stop("cv_schema(data, n_folds) must return training_data and validation_data lists of length n_folds.")
  }

  strip_rowid <- function(dat) { dat$.crossfit_rowid <- NULL; dat }

  training_rowids   <- lapply(training_data,   function(d) d[[".crossfit_rowid"]])
  validation_rowids <- lapply(validation_data, function(d) d[[".crossfit_rowid"]])

  # validation folds must be disjoint for out-of-fold predictions to be
  # well-defined; they need not cover every row (e.g. rolling-origin CV never
  # holds out the initial window), in which case uncovered rows are NA.
  all_val_rowids <- unlist(validation_rowids, use.names = FALSE)
  if (anyDuplicated(all_val_rowids) > 0L) {
    stop(
"The validation folds returned by cv_schema overlap; out-of-fold predictions
are ambiguous. Use a cv_schema whose validation sets are disjoint.")
  }
  covered <- sort(unique(all_val_rowids))
  if (length(covered) < n_obs) {
    warning(sprintf(
"%d row(s) never appear in a validation fold under this cv_schema; their
out-of-fold predictions will be NA.", n_obs - length(covered)))
  }

  # fitting copies with bookkeeping removed, so `y ~ .` never sees .crossfit_rowid
  training_data_clean   <- lapply(training_data,   strip_rowid)
  validation_data_clean <- lapply(validation_data, strip_rowid)

  # -------------------------------------------------------------------------
  # fit one full super_learner() per outer fold (parallel over outer folds)
  # -------------------------------------------------------------------------
  fit_one_fold <- function(i) {
    train_ids <- training_rowids[[i]]

    sl_args <- list(
      data = training_data_clean[[i]],
      learners = learners,
      formulas = formulas,
      y_variable = y_variable,
      n_folds = inner_n_folds,
      determine_super_learner_weights = determine_super_learner_weights,
      ensemble_or_discrete = ensemble_or_discrete,
      outcome_type = outcome_type,
      extra_learner_args = extra_learner_args,
      use_complete_cases = FALSE  # handled once, above
    )
    # cv_schema is only included when the user supplied one: super_learner()'s
    # cv_schema default is missing()-sensitive (it builds a cluster/strata-
    # aware origami schema only when cv_schema is absent), so we let its own
    # default logic run on the fold-subsetted ids below.
    if (!is.null(inner_cv_schema)) sl_args$cv_schema <- inner_cv_schema
    if (!is.null(cluster_ids)) sl_args$cluster_ids <- cluster_ids[train_ids]
    if (!is.null(strata_ids))  sl_args$strata_ids  <- strata_ids[train_ids]
    if (!is.null(weights))     sl_args$weights     <- weights[train_ids]

    sl_fit <- do.call(super_learner, sl_args)

    val_dat <- validation_data_clean[[i]]
    # capture any warnings signaled while predicting
    # on the held-out fold, rewriting their calls to be user-legible; errors
    # are deliberately not caught here. a fold whose predictions fail
    # outright should stop crossfitting loudly, as before
    captured_predictions <- capture_learner_conditions(
      as.numeric(sl_fit$predict(val_dat)),
      call. = substitute(
        sl_fits[[fold_i]]$predict(validation_data[[fold_i]]),
        list(fold_i = i)),
      catch_errors = FALSE)
    list(
      split = i,
      sl_fit = sl_fit,
      learned_predictor = sl_fit$predict,
      validation_rowid = validation_rowids[[i]],
      predictions = captured_predictions$value,
      prediction_warnings = captured_predictions$warnings
    )
  }

  fold_results <- future.apply::future_lapply(
    seq_len(n_folds),
    fit_one_fold,
    future.seed = TRUE
  )

  ###########################################################################
  # aggregate captured warnings: warnings signaled while predicting on the
  # outer held-out folds, plus the per-learner warnings each inner
  # super_learner() captured during its own training/prediction stages
  ###########################################################################
  warnings_from_fold_predictions <- unlist(
    lapply(fold_results, `[[`, "prediction_warnings"), recursive = FALSE)
  if (is.null(warnings_from_fold_predictions)) {
    warnings_from_fold_predictions <- list()
  }

  inner_sl_warning_fields <- c(
    "warnings_from_training_cv_stage1",
    "warnings_from_predicting_cv_stage2",
    "warnings_from_training_on_entire_data")
  warnings_from_inner_super_learners <- lapply(
    fold_results,
    function(fr) fr$sl_fit[intersect(inner_sl_warning_fields,
                                     names(fr$sl_fit))])
  names(warnings_from_inner_super_learners) <-
    paste0("fold_", seq_len(n_folds))
  any_inner_warnings <- any(vapply(
    warnings_from_inner_super_learners,
    function(x) length(x) > 0, logical(1)))

  warning_learners <- unique(unlist(
    lapply(fold_results, function(fr) fr$sl_fit$warning_learners)))


  ###########################################################################
  # prediction machinery
  ###########################################################################
  reconstruct_full_length <- function(predictions_by_fold) {
    out <- rep(NA_real_, n_obs)
    for (i in seq_len(n_folds)) {
      out[fold_results[[i]]$validation_rowid] <-
        as.numeric(predictions_by_fold[[i]])
    }
    out
  }

  # plain lapply: the per-fold work here is a handful of predict() calls, and
  # future_lapply() would serialize every fitted model out to workers.
  predict_fold <- function(newdata_list = NULL, modify = NULL) {
    if (is.null(newdata_list)) {
      newdata_list <- validation_data_clean
    }
    if (!is.list(newdata_list) || length(newdata_list) != n_folds) {
      stop("newdata_list must be a list of length n_folds.")
    }
    if (!is.null(modify) && !is.function(modify)) {
      stop("modify must be NULL or a function taking newdata and returning modified newdata.")
    }
    lapply(seq_len(n_folds), function(i) {
      nd <- newdata_list[[i]]
      if (!is.data.frame(nd)) stop(sprintf("newdata_list[[%d]] is not a data.frame.", i))
      if (!is.null(modify)) nd <- modify(nd)
      nd$.crossfit_rowid <- NULL
      as.numeric(fold_results[[i]]$learned_predictor(nd))
    })
  }

  oof_predictions <- function() {
    reconstruct_full_length(lapply(fold_results, `[[`, "predictions"))
  }

  predict_modified <- function(modify) {
    if (!is.function(modify)) {
      stop("modify must be a function taking newdata and returning modified newdata.")
    }
    reconstruct_full_length(predict_fold(modify = modify))
  }

  ###########################################################################
  # cross-fitted empirical loss on held-out rows
  ###########################################################################
  oof <- oof_predictions()
  cv_loss <- tryCatch({
    held_out <- !is.na(oof)
    if (outcome_type == "density") {
      loss_metric(oof[held_out])
    } else {
      loss_metric(oof[held_out], data[[y_variable]][held_out])
    }
  }, error = function(e) {
    warning("Could not compute cv_loss: ", conditionMessage(e))
    NA_real_
  })

  fold_assignments <- rep(NA_integer_, n_obs)
  for (i in seq_len(n_folds)) {
    fold_assignments[fold_results[[i]]$validation_rowid] <- i
  }

  output <- list(
    predict = function(newdata) {
      stop("A cross-fitted Super Learner has no single prediction function; use
$oof_predictions(), $predict_modified(modify), or $predict_fold(newdata_list).
If you want one predictor fit to all the data, use super_learner() instead.")
    },
    oof_predictions   = oof_predictions,
    predict_modified  = predict_modified,
    predict_fold      = predict_fold,
    sl_fits           = lapply(fold_results, `[[`, "sl_fit"),
    fold_assignments  = fold_assignments,
    fold_rows         = validation_rowids,
    cv_loss           = cv_loss,
    loss_metric       = loss_metric,
    y_variable        = y_variable,
    outcome_type      = outcome_type,
    n_folds           = n_folds,
    inner_n_folds     = inner_n_folds,
    training_data     = training_data_clean,
    validation_data   = validation_data_clean,
    complete_rows     = complete_rows
  )

  class(output) <- "nadir_crossfit_sl"

  # warning fields are only present when non-empty, mirroring how
  # super_learner() reports its error fields
  if (length(warnings_from_fold_predictions) > 0) {
    output$warnings_from_fold_predictions <- warnings_from_fold_predictions
  }
  if (any_inner_warnings) {
    output$warnings_from_inner_super_learners <-
      warnings_from_inner_super_learners
  }
  if (length(warning_learners) > 0) {
    output$warning_learners <- warning_learners
  }
  output
}


#' @export
print.nadir_crossfit_sl <- function(x, ...) {
  cat("Cross-fitted Super Learner (nadir_crossfit_sl)\n")
  cat("  outcome:      ", x$y_variable, " (", x$outcome_type, ")\n", sep = "")
  cat("  outer folds:  ", x$n_folds,
      "   inner CV folds: ", x$inner_n_folds, "\n", sep = "")
  n_na <- sum(is.na(x$fold_assignments))
  cat("  observations: ", length(x$fold_assignments),
      if (n_na > 0) sprintf(" (%d never held out; OOF = NA)", n_na), "\n", sep = "")
  if (!is.na(x$cv_loss)) {
    cat("  cross-fitted loss on held-out data: ",
        format(x$cv_loss, digits = 5), "\n", sep = "")
  }
  if (!is.null(x$warning_learners) && length(x$warning_learners) > 0) {
    cat("  note: warnings were captured during cross-fitting from: ",
        paste(x$warning_learners, collapse = ", "),
        "\n        see $warnings_from_inner_super_learners",
        if (!is.null(x$warnings_from_fold_predictions))
          " and $warnings_from_fold_predictions",
        "\n", sep = "")
  } else if (!is.null(x$warnings_from_fold_predictions)) {
    cat("  note: warnings were captured while predicting on held-out folds;",
        "\n        see $warnings_from_fold_predictions\n", sep = "")
  }
  cat("Methods: $oof_predictions(), $predict_modified(modify), ",
      "$predict_fold(newdata_list),\n        $sl_fits, $fold_assignments, $cv_loss\n",
      sep = "")
  invisible(x)
}


# nadir_crossfit_sl methods -------------------------------------------------
#
# S3 methods for the nadir_crossfit_sl class returned by
# crossfit_super_learner(). Standards: RE4.2 (coef), RE4.4 (formula),
# RE4.5 (nobs), RE4.9 (fitted), RE4.10 (residuals), RE4.18 (summary),
# RE6.0-RE6.2 (plot). print() (RE4.17) lives in crossfit_super_learner.R.

# internal helpers ----------------------------------------------------------

#' Observed outcomes aligned to the out-of-fold prediction vector
#'
#' Reconstructs the outcome column in the original row order of the
#' (complete-case-filtered) data from the per-fold validation sets. Rows
#' never held out by the cv_schema are NA, matching $oof_predictions().
#' @param x A \code{nadir_crossfit_sl}.
#' @returns A numeric vector of length \code{nobs(x)}.
#' @keywords internal
crossfit_observed_outcomes <- function(x) {
  y <- rep(NA_real_, length(x$fold_assignments))
  for (i in seq_len(x$n_folds)) {
    y[x$fold_rows[[i]]] <- as.numeric(x$validation_data[[i]][[x$y_variable]])
  }
  y
}

#' Per-fold ensemble weights aligned on the union of learner names
#'
#' @param x A \code{nadir_crossfit_sl}.
#' @returns A numeric matrix with one row per outer fold and one column per
#'   learner (union across folds); entries are NA where a learner was
#'   dropped (due to errors) in that fold.
#' @keywords internal
crossfit_weight_matrix <- function(x) {
  weight_list <- lapply(x$sl_fits, function(fit) fit$learner_weights)
  learner_names <- unique(unlist(lapply(weight_list, names)))
  out <- matrix(
    NA_real_, nrow = x$n_folds, ncol = length(learner_names),
    dimnames = list(paste0("fold_", seq_len(x$n_folds)), learner_names))
  for (i in seq_len(x$n_folds)) {
    out[i, names(weight_list[[i]])] <- weight_list[[i]]
  }
  out
}

#' Per-outer-fold held-out losses
#'
#' Applies \code{x$loss_metric} to each outer fold's out-of-fold
#' predictions, using the same density-vs-otherwise branching as the
#' overall \code{$cv_loss} computed by \code{crossfit_super_learner()}.
#' @param x A \code{nadir_crossfit_sl}.
#' @returns A data.frame with columns \code{fold}, \code{n_validation},
#'   \code{loss}.
#' @keywords internal
crossfit_fold_losses <- function(x) {
  oof <- x$oof_predictions()
  y <- crossfit_observed_outcomes(x)
  loss_for_fold <- function(i) {
    rows <- x$fold_rows[[i]]
    tryCatch({
      if (x$outcome_type == "density") {
        x$loss_metric(oof[rows])
      } else {
        x$loss_metric(oof[rows], y[rows])
      }
    }, error = function(e) NA_real_)
  }
  data.frame(
    fold = seq_len(x$n_folds),
    n_validation = vapply(x$fold_rows, length, integer(1)),
    loss = vapply(seq_len(x$n_folds), loss_for_fold, numeric(1)))
}

#############################################################################
# predict: fail with directions rather than falling through to
# predict.default's confusing error
#############################################################################

#' Predicting from a Cross-Fitted Super Learner
#'
#' A cross-fitted super learner deliberately has no single prediction
#' function: it is one fitted super learner \emph{per outer fold}, retained
#' so that each observation can be predicted by an ensemble trained without
#' it. This method therefore errors with directions to the fold-aware
#' interfaces: \code{$oof_predictions()} for out-of-fold predictions,
#' \code{$predict_modified(modify)} for interventional/modified-data
#' predictions, and \code{$predict_fold(newdata_list)} for arbitrary
#' per-fold newdata. To fit one predictor on all the data, use
#' \code{\link{super_learner}()}.
#'
#' @param object A \code{nadir_crossfit_sl}.
#' @param ... Ignored.
#' @returns Does not return; always signals an informative error.
#' @export
predict.nadir_crossfit_sl <- function(object, ...) {
  object$predict(NULL)
}

###########################################################
# methods: coef (RE4.2), fitted (RE4.9), residuals (RE4.10)
###########################################################

#' Per-Fold Ensemble Weights of a Cross-Fitted Super Learner
#'
#' Following the convention that a super learner's "coefficients" are its
#' meta-learned ensemble weights (see \code{\link{coef.nadir_sl_model}}),
#' the coefficients of a \emph{cross-fitted} super learner are the ensemble
#' weights of each outer fold's fit, returned as a folds-by-learners matrix.
#' Entries are \code{NA} for learners dropped (due to fitting errors) in a
#' given fold. Column-wise variability across rows is a useful diagnostic
#' of how stable the ensemble is across folds; see
#' \code{\link{summary.nadir_crossfit_sl}} and
#' \code{plot(x, type = "weights")}.
#'
#' @param object A \code{nadir_crossfit_sl}.
#' @param ... Ignored; included for compatibility with the generic.
#' @returns A numeric matrix (rows: outer folds; columns: learners); each
#'   row sums to 1 over its non-\code{NA} entries.
#' @importFrom stats coef
#' @export
coef.nadir_crossfit_sl <- function(object, ...) {
  crossfit_weight_matrix(object)
}

#' Out-of-Fold Fitted Values from a Cross-Fitted Super Learner
#'
#' Equivalent to \code{object$oof_predictions()}: each value is the
#' prediction for that observation from the outer fold whose training data
#' excluded it. Values are in the \emph{original row order} of the
#' (complete-case-filtered) data; \code{object$complete_rows} maps positions
#' back to the data as supplied. Rows never held out by the
#' \code{cv_schema} are \code{NA}.
#'
#' @param object A \code{nadir_crossfit_sl}.
#' @param ... Ignored; included for compatibility with the generic.
#' @returns A numeric vector of length \code{nobs(object)}.
#' @importFrom stats fitted
#' @export
fitted.nadir_crossfit_sl <- function(object, ...) {
  object$oof_predictions()
}

#' Out-of-Fold Residuals from a Cross-Fitted Super Learner
#'
#' Observed outcomes minus out-of-fold predictions, in the original row
#' order of the (complete-case-filtered) data. \code{NA} for rows never
#' held out by the \code{cv_schema}. Errors for density and multiclass
#' outcomes, where out-of-fold predictions are densities/probabilities of
#' the observed outcome rather than point predictions.
#'
#' @param object A \code{nadir_crossfit_sl}.
#' @param ... Ignored; included for compatibility with the generic.
#' @returns A numeric vector of length \code{nobs(object)}.
#' @importFrom stats residuals
#' @export
residuals.nadir_crossfit_sl <- function(object, ...) {
  if (object$outcome_type %in% c("density", "multiclass")) {
    stop("residuals() is not defined for outcome_type = '",
         object$outcome_type, "': out-of-fold predictions are ",
         "densities/probabilities of the observed outcome, not point ",
         "predictions.")
  }
  crossfit_observed_outcomes(object) - object$oof_predictions()
}

#########################################
# formula method (RE4.4) and nobs (RE4.5)
#########################################

#' Extract the Formula(s) from a Cross-Fitted Super Learner
#'
#' All outer folds share one specification by construction, so this
#' delegates to the first fold's fit: a single \code{formula} when all
#' learners share one, otherwise a named list of per-learner formulas.
#'
#' @param x A \code{nadir_crossfit_sl}.
#' @param ... Ignored; included for compatibility with the generic.
#' @returns A \code{formula} or a named list of formulas.
#' @importFrom stats formula
#' @export
formula.nadir_crossfit_sl <- function(x, ...) {
  formula(x$sl_fits[[1]])
}

#' Number of Observations Used in Cross-Fitting
#'
#' The number of rows of the (complete-case-filtered) data over which
#' cross-fitting was performed — i.e., the length of
#' \code{$oof_predictions()} and \code{$fold_assignments}.
#'
#' @param object A \code{nadir_crossfit_sl}.
#' @param ... Ignored; included for compatibility with the generic.
#' @returns An integer.
#' @importFrom stats nobs
#' @export
nobs.nadir_crossfit_sl <- function(object, ...) {
  length(object$fold_assignments)
}

###################################
# summary method (RE4.18) ---------
###################################

#' Summarise a Cross-Fitted Super Learner
#'
#' Reports (i) the held-out loss of each outer fold alongside the overall
#' cross-fitted loss, and (ii) a weight-stability table: the mean, standard
#' deviation, and range of each learner's ensemble weight across the outer
#' folds. Large across-fold variability in a learner's weight — or
#' \code{n_folds_present} below \code{n_folds}, indicating the learner
#' errored in some folds — is a signal that the ensemble is not stable
#' under resampling.
#'
#' @param object A \code{nadir_crossfit_sl}.
#' @param ... Ignored; included for compatibility with the generic.
#' @returns An object of class \code{summary.nadir_crossfit_sl}: a list with
#'   \code{$fold_losses} (data.frame: fold, n_validation, loss),
#'   \code{$weight_stability} (data.frame: learner, mean_weight, sd_weight,
#'   min_weight, max_weight, n_folds_present, ordered by mean weight),
#'   and the scalars \code{$cv_loss}, \code{$y_variable},
#'   \code{$outcome_type}, \code{$n_folds}, \code{$inner_n_folds},
#'   \code{$n_obs}, \code{$n_never_held_out}.
#' @export
summary.nadir_crossfit_sl <- function(object, ...) {
  w <- crossfit_weight_matrix(object)
  weight_stability <- data.frame(
    learner = colnames(w),
    mean_weight = apply(w, 2, mean, na.rm = TRUE),
    sd_weight = apply(w, 2, stats::sd, na.rm = TRUE),
    min_weight = apply(w, 2, min, na.rm = TRUE),
    max_weight = apply(w, 2, max, na.rm = TRUE),
    n_folds_present = apply(w, 2, function(col) sum(!is.na(col))))
  weight_stability <- weight_stability[
    order(weight_stability$mean_weight, decreasing = TRUE), ]
  rownames(weight_stability) <- NULL

  out <- list(
    fold_losses = crossfit_fold_losses(object),
    weight_stability = weight_stability,
    cv_loss = object$cv_loss,
    y_variable = object$y_variable,
    outcome_type = object$outcome_type,
    n_folds = object$n_folds,
    inner_n_folds = object$inner_n_folds,
    n_obs = length(object$fold_assignments),
    n_never_held_out = sum(is.na(object$fold_assignments)))
  class(out) <- "summary.nadir_crossfit_sl"
  out
}

#' @export
print.summary.nadir_crossfit_sl <- function(x, digits = 4, ...) {
  cat("Summary of Cross-fitted Super Learner\n")
  cat("  outcome: ", x$y_variable, " (", x$outcome_type, ")",
      ";  n = ", x$n_obs, sep = "")
  if (x$n_never_held_out > 0) {
    cat(" (", x$n_never_held_out, " never held out)", sep = "")
  }
  cat("\n  outer folds: ", x$n_folds,
      ";  inner CV folds: ", x$inner_n_folds, "\n\n", sep = "")

  cat("Held-out loss by outer fold:\n")
  fl <- x$fold_losses
  fl$loss <- signif(fl$loss, digits)
  print(fl, row.names = FALSE)
  if (!is.na(x$cv_loss)) {
    cat("Overall cross-fitted loss: ",
        format(x$cv_loss, digits = digits), "\n", sep = "")
  }

  cat("\nEnsemble weight stability across outer folds:\n")
  ws <- x$weight_stability
  num_cols <- c("mean_weight", "sd_weight", "min_weight", "max_weight")
  ws[num_cols] <- lapply(ws[num_cols], round, digits = digits)
  print(ws, row.names = FALSE)
  invisible(x)
}

#########################################
# plot method (RE6.0 - RE6.2) -----------
#########################################

#' Plot a Cross-Fitted Super Learner
#'
#' @description
#' Two plot types are provided:
#' \describe{
#'   \item{\code{type = "weights" (default)}}{Ensemble-weight stability: each
#'     learner's weight in each outer fold (open circles) with the
#'     across-fold mean (filled points). Tight clusters indicate an
#'     ensemble that is stable under resampling; missing circles indicate
#'     folds where a learner errored and was dropped.}
#'   \item{\code{type = "fitted"}}{Out-of-fold predictions against
#'     observed outcomes, colored by outer fold, with the identity line for
#'     reference. Every prediction comes from an ensemble trained entirely
#'     without that observation. Rows never held out by the
#'     \code{cv_schema} are omitted. Not defined for
#'     \code{outcome_type = "density"} or \code{"multiclass"}.}
#' }
#'
#' Requires the \pkg{ggplot2} package (listed in \code{Suggests}).
#'
#' @param x A \code{nadir_crossfit_sl} as returned by
#'   \code{\link{crossfit_super_learner}()}.
#' @param type One of \code{"fitted"} or \code{"weights"}.
#' @param ... Ignored; included for compatibility with the generic.
#' @returns A \code{ggplot} object.
#' @importFrom stats quantile
#' @examples
#' if (requireNamespace("ggplot2", quietly = TRUE)) {
#'   cf <- crossfit_super_learner(
#'     data = mtcars,
#'     formula = mpg ~ cyl + hp,
#'     n_folds = 2, inner_n_folds = 2,
#'     learners = list(mean = lnr_mean, lm = lnr_lm))
#'   plot(cf)                    # out-of-fold predictions vs. observed
#'   plot(cf, type = "weights")  # weight stability across folds
#' }
#' @export
plot.nadir_crossfit_sl <- function(x, type = c("weights", "fitted"), ...) {
  type <- match.arg(type)
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("plot.nadir_crossfit_sl() requires the {ggplot2} package. ",
         "Install it with install.packages('ggplot2').")
  }

  if (type == "fitted") {
    if (x$outcome_type %in% c("density", "multiclass")) {
      stop("type = 'fitted' is not defined for outcome_type = '",
           x$outcome_type, "': out-of-fold predictions are ",
           "densities/probabilities of the observed outcome, not point ",
           "predictions. Use type = 'weights' instead.")
    }
    df <- data.frame(
      observed = crossfit_observed_outcomes(x),
      predicted = x$oof_predictions(),
      fold = factor(x$fold_assignments))
    df <- df[stats::complete.cases(df), , drop = FALSE]
    return(
      ggplot2::ggplot(df,
                      ggplot2::aes(x = .data$observed, y = .data$predicted,
                                   color = .data$fold,
                                   fill = .data$fold)) +
        ggplot2::geom_point(alpha = 0.7) +
        ggplot2::geom_abline(slope = 1, intercept = 0,
                             linetype = "dashed", color = "grey40") +
        ggplot2::labs(
          x = paste0("Observed ", x$y_variable),
          y = "Out-of-fold prediction",
          color = "Outer fold",
          fill = "Outer fold",
          title = "Cross-fitted Super Learner: out-of-fold predictions vs. observed",
          caption = "Each prediction comes from an ensemble trained without that observation.") +
        ggplot2::theme_bw() +
        ggplot2::theme(plot.caption.position = "plot")
    )
  }

  # type == "weights"
  w <- crossfit_weight_matrix(x)
  long <- data.frame(
    fold = rep(rownames(w), times = ncol(w)),
    learner = rep(colnames(w), each = nrow(w)),
    weight = as.numeric(w))
  long <- long[!is.na(long$weight), , drop = FALSE]
  means <- tapply(long$weight, long$learner, mean)
  lowers <- tapply(long$weight, long$learner, quantile, 0.25)
  uppers <- tapply(long$weight, long$learner, quantile, 0.75)
  means_df <- data.frame(learner = names(means),
                         mean_weight = as.numeric(means),
                         upper_ci = uppers,
                         lower_ci = lowers)

  lvls <- means_df$learner[order(means_df$mean_weight)]
  long$learner <- factor(long$learner, levels = lvls)
  means_df$learner <- factor(means_df$learner, levels = lvls)

  ggplot2::ggplot(long,
                  ggplot2::aes(y = .data$learner, x = .data$weight,
                               fill = .data$learner)) +
    ggplot2::geom_col(data = means_df,
                      ggplot2::aes(x = .data$mean_weight, y = .data$learner,
                                   fill = .data$learner),
                      alpha = 0.5) +
    ggplot2::geom_jitter(height = 0.15, shape = "o") +
    ggplot2::geom_pointrange(
      data = means_df,
      mapping = ggplot2::aes(
        x = .data$mean_weight, y = .data$learner,
        xmax = .data$upper_ci, xmin = .data$lower_ci),
      alpha = .8) +
    ggplot2::labs(
      title = "Ensemble weight stability across outer folds",
      x = "Ensemble weight", y = NULL,
      caption = paste0(
        "Open circles: each outer fold's ensemble weight for the learner",
        "\nFilled points: across-fold mean. Intervals show 25th to 75th percentile.")) +
    ggplot2::theme_bw() +
    ggplot2::theme(plot.caption.position = "plot")
}
