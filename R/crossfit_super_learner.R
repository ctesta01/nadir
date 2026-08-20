# crossfit_super_learner.R --------------------------------------------------
# Replaces crossfit_super_learner{,2,3,4}. Semantics follow v4 (a full
# super_learner(), with its own internal CV, is fit on each outer training
# split), with the following fixes and changes relative to v4:
#
#   * .crossfit_rowid is stripped before the inner super_learner() is fit,
#     so `y ~ .` formulas no longer see the fold bookkeeping column.
#   * cluster_ids / strata_ids now (a) inform the *outer* cross-fitting split
#     (via cv_origami_schema when cv_schema is not user-supplied), and
#     (b) are subset per outer training fold before being handed to the inner
#     super_learner(), so inner CV also respects clustering/stratification.
#   * weights are likewise subset per outer training fold (previously they
#     were silently dropped by super_learner()'s length check).
#   * inner_n_folds / inner_cv_schema decouple the inner CV from the outer
#     cross-fitting folds.
#   * out-of-fold predictions are always reconstructed in original row order;
#     validation folds are checked to be disjoint (error) and covering
#     (warning + NA for never-validated rows, which permits e.g.
#     origami::folds_rolling_origin where the first window is never held out).
#   * the prediction-stage loops are plain lapply(): the work is trivial and
#     future_lapply() would serialize every fitted model to the workers.
#   * a cross-fitted empirical loss (cv_loss) is reported, with the loss
#     metric inferred from outcome_type when not supplied.

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
    list(
      split = i,
      sl_fit = sl_fit,
      learned_predictor = sl_fit$predict,
      validation_rowid = validation_rowids[[i]],
      predictions = as.numeric(sl_fit$predict(val_dat))
    )
  }

  fold_results <- future.apply::future_lapply(
    seq_len(n_folds),
    fit_one_fold,
    future.seed = TRUE
  )

  # -------------------------------------------------------------------------
  # prediction machinery
  # -------------------------------------------------------------------------
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

  # -------------------------------------------------------------------------
  # cross-fitted empirical loss on held-out rows
  # -------------------------------------------------------------------------
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
  cat("Access: $oof_predictions(), $predict_modified(modify), ",
      "$predict_fold(newdata_list),\n        $sl_fits, $fold_assignments, $cv_loss\n",
      sep = "")
  invisible(x)
}
