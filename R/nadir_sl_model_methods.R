# nadir_sl_model_methods.R ---------------------------------------------------
# S3 methods for the nadir_sl_model class returned by super_learner().
# Standards: RE4.2 (coef), RE4.4 (formula), RE4.5 (nobs), RE4.9 (fitted),
# RE4.10 (residuals), RE4.17 (print), RE4.18 (summary), RE6.0-RE6.2 (plot).

# .data is ggplot2's tidy-eval pronoun; declare it so R CMD check does not
# report "no visible binding for global variable '.data'".
utils::globalVariables(".data")

# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

#' Cross-validated ensemble predictions on the held-out folds
#'
#' The learner-weighted combination of each candidate learner's holdout
#' predictions; each value comes from learners trained without that
#' observation.
#' @param x A \code{nadir_sl_model}.
#' @returns A numeric vector, one entry per row of
#'   \code{x$holdout_predictions}.
#' @keywords internal
sl_holdout_ensemble_predictions <- function(x) {
  lw <- x$learner_weights
  pred_mat <- as.matrix(x$holdout_predictions[, names(lw), drop = FALSE])
  as.numeric(pred_mat %*% lw)
}

#' Per-fold held-out losses in long format
#'
#' One row per (learner, fold) pair — including a \code{super_learner} row
#' for the weighted ensemble — with the loss chosen by
#' \code{default_loss_metric(x$outcome_type)}.
#' @param x A \code{nadir_sl_model}.
#' @returns A data.frame with columns \code{learner}, \code{fold},
#'   \code{loss}.
#' @keywords internal
sl_per_fold_losses <- function(x) {
  hp <- x$holdout_predictions
  lw <- x$learner_weights
  loss_metric <- default_loss_metric(x$outcome_type)
  y <- hp[[x$y_variable]]
  folds <- hp[[".sl_fold"]]
  uf <- sort(unique(folds))

  apply_loss <- function(preds, yy) {
    # density/multiclass losses take predicted densities/probabilities of
    # the observed outcome only
    if (x$outcome_type %in% c("density", "multiclass")) {
      loss_metric(preds)
    } else {
      loss_metric(preds, yy)
    }
  }

  cols <- c(names(lw), "super_learner")
  ens <- sl_holdout_ensemble_predictions(x)
  out <- expand.grid(learner = cols, fold = uf,
                     KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  out$loss <- vapply(seq_len(nrow(out)), function(i) {
    idx <- folds == out$fold[i]
    preds <- if (out$learner[i] == "super_learner") ens[idx]
    else hp[[out$learner[i]]][idx]
    apply_loss(preds, y[idx])
  }, numeric(1))
  out
}

#' Map fold-ordered holdout values back to input-data row order
#' @keywords internal
sl_map_to_input_order <- function(x, values) {
  rid <- x$holdout_predictions[[".sl_rowid"]]
  if (is.null(rid) || all(is.na(rid))) {
    stop("The cv_schema used did not preserve the .sl_rowid bookkeeping ",
         "column, so held-out values cannot be mapped back to input rows. ",
         "Custom cv_schema functions should subset the data they are given ",
         "rather than rebuilding it.")
  }
  out <- rep(NA_real_, x$n_obs)
  if (anyDuplicated(stats::na.omit(rid)) > 0) {
    warning("Some rows were held out in multiple folds; returning the ",
            "per-row average across folds.")
    agg <- tapply(values, rid, mean)
    out[as.integer(names(agg))] <- as.numeric(agg)
  } else {
    ok <- !is.na(rid)
    out[rid[ok]] <- values[ok]
  }
  out
}

# ---------------------------------------------------------------------------
# print (RE4.17)
# ---------------------------------------------------------------------------

#' Print a \code{nadir_sl_model}
#'
#' Summarises the model inputs (outcome, outcome type, number of
#' observations and cross-validation folds) and outputs (ensemble weights),
#' plus any errors captured during training.
#'
#' @param x An object of class \code{nadir_sl_model} as returned by
#'   \code{\link{super_learner}()}.
#' @param digits Number of digits for the printed weights.
#' @param ... Ignored; included for compatibility with the generic.
#' @returns \code{x}, invisibly.
#' @export
print.nadir_sl_model <- function(x, digits = 3, ...) {
  cat("Super Learner (nadir_sl_model)\n")
  cat("  outcome:      ", x$y_variable, " (", x$outcome_type, ")\n", sep = "")
  if (!is.null(x$n_obs)) {
    cat("  observations: ", x$n_obs, sep = "")
    if (!is.null(x$n_folds)) cat("   CV folds: ", x$n_folds, sep = "")
    cat("\n")
  }
  cat("  ensemble weights:\n")
  w <- sort(x$learner_weights, decreasing = TRUE)
  for (nm in names(w)) {
    cat("    ", format(nm, width = max(nchar(names(w)))), "  ",
        format(round(w[[nm]], digits), nsmall = digits), "\n", sep = "")
  }
  # erroring learners:
  err_fields <- intersect(
    c("errors_from_training_cv_stage1", "errors_from_predicting_cv_stage2",
      "errors_from_training_on_entire_data"),
    names(x))
  if (length(err_fields) > 0) {
    cat("  note: errors were captured during training; see ",
        paste0("$", err_fields, collapse = ", "), "\n", sep = "")
  }
  if (!is.null(x$erring_learners) && length(x$erring_learners) > 0) {
    cat("  learners dropped due to errors: ",
        paste(x$erring_learners, collapse = ", "), "\n", sep = "")
  }
  # warning learners:
  warn_fields <- intersect(
    c("warnings_from_training_cv_stage1", "warnings_from_predicting_cv_stage2",
      "warnings_from_training_on_entire_data"),
    names(x))
  if (length(warn_fields) > 0) {
    cat("  note: warnings were captured during training; see ",
        paste0("$", warn_fields, collapse = ", "), "\n", sep = "")
  }
  if (!is.null(x$warning_learners) && length(x$warning_learners) > 0) {
    cat("  learners that signaled warnings: ",
        paste(x$warning_learners, collapse = ", "), "\n", sep = "")
  }
  cat("Methods: predict(x, newdata), plot(x), summary(x), coef(x), fitted(x)\n")
  invisible(x)
}

# ---------------------------------------------------------------------------
# summary (RE4.18)
# ---------------------------------------------------------------------------

#' Summarise a \code{nadir_sl_model}
#'
#' Combines the ensemble weights with each candidate learner's
#' cross-validated held-out loss, using the loss appropriate to the model's
#' \code{outcome_type} (mean squared error for continuous outcomes; negative
#' log loss for binary, multiclass, and density outcomes).
#'
#' @param object An object of class \code{nadir_sl_model}.
#' @param ... Ignored; included for compatibility with the generic.
#' @returns An object of class \code{summary.nadir_sl_model}: a list with a
#'   \code{$comparison} data.frame (one row per learner: weight and held-out
#'   loss, best first) plus \code{$y_variable}, \code{$outcome_type},
#'   \code{$n_obs}, and \code{$n_folds}.
#' @export
summary.nadir_sl_model <- function(object, ...) {
  loss_metric <- default_loss_metric(object$outcome_type)
  learner_losses <- compare_learners(object, loss_metric = loss_metric)
  comparison <- data.frame(
    learner = names(object$learner_weights),
    weight = as.numeric(object$learner_weights))
  comparison$cv_holdout_loss <-
    as.numeric(unlist(learner_losses[1, comparison$learner]))
  comparison <- comparison[order(comparison$cv_holdout_loss), ]
  rownames(comparison) <- NULL

  out <- list(
    comparison = comparison,
    y_variable = object$y_variable,
    outcome_type = object$outcome_type,
    n_obs = object$n_obs,
    n_folds = object$n_folds)
  class(out) <- "summary.nadir_sl_model"
  out
}

#' @export
print.summary.nadir_sl_model <- function(x, digits = 4, ...) {
  cat("Summary of Super Learner fit\n")
  cat("  outcome: ", x$y_variable, " (", x$outcome_type, ")", sep = "")
  if (!is.null(x$n_obs)) cat(";  n = ", x$n_obs, sep = "")
  if (!is.null(x$n_folds)) cat(";  folds = ", x$n_folds, sep = "")
  cat("\n\n")
  printable <- x$comparison
  printable$weight <- round(printable$weight, digits)
  printable$cv_holdout_loss <- signif(printable$cv_holdout_loss, digits)
  print(printable, row.names = FALSE)
  cat("\nLoss is evaluated on held-out (cross-validation) data; lower is better.\n")
  invisible(x)
}

# ---------------------------------------------------------------------------
# plot (RE6.0 - RE6.2)
# ---------------------------------------------------------------------------

#' Plot a \code{nadir_sl_model}
#'
#' @description
#' Two plot types are provided:
#' \describe{
#'   \item{\code{type = "comparison"} (default)}{For each candidate learner
#'     and for the weighted super learner ensemble: bars show the mean
#'     cross-validated held-out loss, open circles show each fold's held-out
#'     loss, and point-ranges show \eqn{\pm 1} standard deviation across
#'     folds. This automates the figure shown in the package README, with
#'     the ensemble added as its own row. Learners are ordered best (lowest
#'     loss) at top.}
#'   \item{\code{type = "fitted"}}{Cross-validated ensemble predictions on
#'     the held-out folds against the observed outcomes, with the identity
#'     line for reference. Each prediction comes from learners trained
#'     without that observation. Not defined for
#'     \code{outcome_type = "density"} or \code{"multiclass"}.}
#' }
#'
#' The loss is chosen by \code{outcome_type}: mean squared error for
#' continuous outcomes and negative log loss otherwise.
#'
#' Requires the \pkg{ggplot2} package (listed in \code{Suggests}).
#'
#' @param x An object of class \code{nadir_sl_model} as returned by
#'   \code{\link{super_learner}()}.
#' @param type One of \code{"comparison"} or \code{"fitted"}.
#' @param ... Ignored; included for compatibility with the generic.
#' @returns A \code{ggplot} object, which prints when returned to the
#'   console.
#' @examples
#' if (requireNamespace("ggplot2", quietly = TRUE)) {
#'   sl_model <- super_learner(
#'     data = mtcars,
#'     formula = mpg ~ cyl + hp,
#'     learners = list(mean = lnr_mean, lm = lnr_lm))
#'   plot(sl_model)                   # learner comparison
#'   plot(sl_model, type = "fitted")  # observed vs. CV ensemble predictions
#' }
#' @export
plot.nadir_sl_model <- function(x, type = c("comparison", "fitted"), ...) {
  type <- match.arg(type)
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("plot.nadir_sl_model() requires the {ggplot2} package. ",
         "Install it with install.packages('ggplot2').")
  }

  if (type == "fitted") {
    if (x$outcome_type %in% c("density", "multiclass")) {
      stop("type = 'fitted' is not defined for outcome_type = '",
           x$outcome_type, "': held-out predictions are ",
           "densities/probabilities of the observed outcome, not point ",
           "predictions. Use type = 'comparison' instead.")
    }
    df <- data.frame(
      observed = x$holdout_predictions[[x$y_variable]],
      predicted = sl_holdout_ensemble_predictions(x))
    return(
      ggplot2::ggplot(df,
                      ggplot2::aes(x = .data$observed, y = .data$predicted)) +
        ggplot2::geom_point(alpha = 0.7) +
        ggplot2::geom_abline(slope = 1, intercept = 0,
                             linetype = "dashed", color = "grey40") +
        ggplot2::labs(
          x = paste0("Observed ", x$y_variable),
          y = "Cross-validated ensemble prediction",
          title = "Super Learner: held-out predictions vs. observed") +
        ggplot2::theme_bw()
    )
  }

  # type == "comparison"
  fold_losses <- sl_per_fold_losses(x)
  means <- tapply(fold_losses$loss, fold_losses$learner, mean)
  sds <- tapply(fold_losses$loss, fold_losses$learner, stats::sd)
  summary_df <- data.frame(
    learner = names(means),
    mean_loss = as.numeric(means),
    sd_loss = as.numeric(sds))

  # order best (lowest mean loss) at the top of the y axis
  lvls <- summary_df$learner[order(summary_df$mean_loss, decreasing = TRUE)]
  summary_df$learner <- factor(summary_df$learner, levels = lvls)
  fold_losses$learner <- factor(fold_losses$learner, levels = lvls)

  loss_label <- switch(x$outcome_type,
                       continuous = "Cross-validated held-out MSE",
                       "Cross-validated held-out negative log loss")

  summary_df <- summary_df |> filter(learner != 'super_learner')
  fold_losses <- fold_losses |> filter(learner != 'super_learner')

  ggplot2::ggplot(summary_df,
                  ggplot2::aes(y = .data$learner, x = .data$mean_loss,
                               fill = .data$learner)) +
    ggplot2::geom_col(alpha = 0.5, show.legend = FALSE) +
    ggplot2::geom_jitter(
      data = fold_losses,
      mapping = ggplot2::aes(x = .data$loss, y = .data$learner),
      height = 0.15, shape = "o", inherit.aes = FALSE) +
    ggplot2::geom_pointrange(
      ggplot2::aes(xmin = .data$mean_loss - .data$sd_loss,
                   xmax = .data$mean_loss + .data$sd_loss),
      alpha = 0.5, show.legend = FALSE) +
    ggplot2::labs(
      title = "Comparison of Candidate Learners in the Super Learner Ensemble",
      x = loss_label, y = NULL,
      caption = paste0(
        "Bars and filled points show the mean held-out loss across CV folds;",
        "\nranges show +/-1 SD across folds; each open circle is one fold.")) +
    ggplot2::theme_bw() +
    ggplot2::theme(plot.caption.position = "plot")
}

# ---------------------------------------------------------------------------
# coef (RE4.2), fitted (RE4.9), residuals (RE4.10)
# ---------------------------------------------------------------------------

#' Ensemble Weights of a \code{nadir_sl_model}
#'
#' Following the convention of \code{coef.SuperLearner} in the
#' \pkg{SuperLearner} package, the "coefficients" of a super learner are the
#' meta-learned ensemble weights on the candidate learners.
#'
#' @param object An object of class \code{nadir_sl_model}.
#' @param ... Ignored; included for compatibility with the generic.
#' @returns A named numeric vector of ensemble weights summing to 1.
#' @importFrom stats coef
#' @export
coef.nadir_sl_model <- function(object, ...) {
  object$learner_weights
}

#' Cross-Validated Fitted Values from a \code{nadir_sl_model}
#'
#' Returns the ensemble's \emph{cross-validated} predictions on the held-out
#' folds: each value is the learner-weighted prediction for an observation,
#' produced by learners trained without that observation. These are honest
#' (out-of-fold) rather than in-sample fitted values.
#'
#' Values are returned in cross-validation fold order, matching the rows of
#' \code{$holdout_predictions} (which contains the corresponding observed
#' outcomes), not in the row order of the originally supplied data. For
#' out-of-fold predictions in original row order, use
#' \code{\link{crossfit_super_learner}()$oof_predictions()}.
#'
#' @srrstats {RE4.9} fitted() returns cross-validated modelled response
#'   values.
#' @param object An object of class \code{nadir_sl_model}.
#' @param ... Ignored; included for compatibility with the generic.
#' @returns A numeric vector of cross-validated ensemble predictions (or,
#'   for density/multiclass outcomes, ensemble mixture densities or
#'   probabilities of the observed outcomes).
#' @importFrom stats fitted
#' @export
fitted.nadir_sl_model <- function(object, ...) {
  sl_map_to_input_order(object, sl_holdout_ensemble_predictions(object))
}

#' Cross-Validated Residuals from a \code{nadir_sl_model}
#'
#' Observed outcomes minus the cross-validated ensemble predictions of
#' \code{\link{fitted.nadir_sl_model}}. I.e., honest out-of-fold residuals. See
#' that function's documentation for the row-ordering convention.
#'
#' @srrstats {RE4.10} residuals() returns out-of-fold residuals with
#'   documented definition.
#'
#' @param object An object of class \code{nadir_sl_model}.
#' @param ... Ignored; included for compatibility with the generic.
#' @returns A numeric vector of out-of-fold residuals. Errors for
#'   \code{outcome_type = 'density'} or \code{'multiclass'}, where held-out
#'   predictions are not point predictions.
#' @importFrom stats residuals
#' @export
residuals.nadir_sl_model <- function(object, ...) {
  if (object$outcome_type %in% c("density", "multiclass")) {
    stop("residuals() is not defined for outcome_type = '",
         object$outcome_type, "': held-out predictions are ",
         "densities/probabilities of the observed outcome, not point ",
         "predictions.")
  }
  y_fold_order <- object$holdout_predictions[[object$y_variable]]
  sl_map_to_input_order(
    object, y_fold_order - sl_holdout_ensemble_predictions(object))

}

# ---------------------------------------------------------------------------
# formula (RE4.4) and nobs (RE4.5)
# ---------------------------------------------------------------------------

#' Extract the Formula(s) from a \code{nadir_sl_model}
#'
#' Returns the regression formula shared by all learners when a single
#' formula was used, and otherwise the named list of per-learner formulas.
#'
#' @srrstats {RE4.4} formula() accessor.
#'
#' @param x An object of class \code{nadir_sl_model}.
#' @param ... Ignored; included for compatibility with the generic.
#' @returns A \code{formula} if all learners share one; otherwise a named
#'   list of formulas, one per learner.
#' @importFrom stats formula
#' @export
formula.nadir_sl_model <- function(x, ...) {
  if (is.null(x$formulas)) {
    stop("This nadir_sl_model was created by a version of super_learner() ",
         "that did not store formulas; re-fit to use formula().")
  }
  fs <- x$formulas
  deparsed <- vapply(fs, function(f) paste(deparse(f), collapse = " "),
                     character(1))
  if (length(unique(deparsed)) == 1L) {
    return(fs[[1]])
  }
  fs
}

#' Number of Observations Used to Fit a \code{nadir_sl_model}
#'
#' @srrstats {RE4.5} nobs() accessor.
#' @param object An object of class \code{nadir_sl_model}.
#' @param ... Ignored; included for compatibility with the generic.
#' @returns Integer number of rows of the (complete-case-filtered) training
#'   data.
#' @importFrom stats nobs
#' @export
nobs.nadir_sl_model <- function(object, ...) {
  if (is.null(object$n_obs)) {
    stop("This nadir_sl_model was created by a version of super_learner() ",
         "that did not store n_obs; re-fit to use nobs().")
  }
  as.integer(object$n_obs)
}
