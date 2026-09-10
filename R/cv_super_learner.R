#' Cross-Validating a `super_learner`
#'
#' Produce cv-rmse for a `super_learner` specified by a closure that
#' accepts data and returns a `super_learner` prediction function.
#'
#' The idea is that `cv_super_learner` splits the data into training/validation
#' splits, trains `super_learner` on each training split, and then
#' evaluates their predictions on the held-out validation data, calculating
#' a root-mean-squared-error on those held-out data.
#'
#' This function prints a message if the \code{loss_function} argument is
#' not set explicitly, letting the user know that the mean-squared-error will be
#' used by default. Pass in a loss function explicitly to
#' \code{super_learner()} if you'd like to suppress this message, or use a
#' similar approach for the appropriate loss function depending on context.
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
#' @param loss_metric A loss metric function, like the mean-squared-error or negative-log-loss to be
#'   used in evaluating the learners on held-out data and minimized through convex optimization.
#'   A loss metric should take two (vector) arguments:
#'   predictions, and true outcomes, and produce a single statistic summarizing the
#'   performance of each learner. Defaults to nadir's internal mean-squared-error function.
#' @param inner_n_folds Number of folds used by the inner
#'   \code{super_learner()} on each outer training split to estimate
#'   ensemble weights. Defaults to \code{n_folds}, matching the historical
#'   behavior in which one fold count governed both.
#' @param inner_cv_schema Optional \code{cv_schema} for the inner
#'   \code{super_learner()} calls; defaults to \code{cv_schema} when one is
#'   supplied, and otherwise to \code{super_learner()}'s own defaults.
#'
#' @returns A list containing \code{$trained_learners} and \code{$cv_loss} which
#'   respectively include 1) the trained super learner models on each fold of the data, their holdout predictions and,
#'   2) the cross-validated estimate of the risk (expected loss) on held-out data.
#' @examples
#'
#'   cv_super_learner(
#'     data = mtcars,
#'     formula = mpg ~ cyl + hp,
#'     learners = list(lnr_mean, lnr_lm))
#'
#' @export
cv_super_learner <- function(
    data,
    learners,
    formulas,
    y_variable = NULL,
    n_folds = 5,
    determine_super_learner_weights = NULL,
    ensemble_or_discrete = c('ensemble', 'discrete'),
    cv_schema = NULL,
    outcome_type = c('continuous', 'binary', 'density', 'multiclass'),
    extra_learner_args = NULL,
    cluster_ids = NULL,
    strata_ids = NULL,
    weights = NULL,
    rowids = NULL,
    loss_metric = NULL,
    use_complete_cases = FALSE,
    inner_n_folds = NULL,
    inner_cv_schema = NULL) {

  ensemble_or_discrete <- match.arg(ensemble_or_discrete)
  outcome_type <- match.arg(outcome_type)

  # legacy validations, retained verbatim: these exact messages are asserted
  # in tests/testthat/test-compare_and_cv_super_learner.R, and pre-validating
  # here errors earlier and more clearly than crossfit's equivalents.
  if (length(n_folds) > 1) {
    stop("n_folds must be a length 1 numeric value.")
  }
  if (! is.null(cluster_ids) & length(cluster_ids) != nrow(data)) {
    stop("the cluster_ids should be equal in length to nrow(data)")
  }
  if (! is.null(strata_ids) & length(strata_ids) != nrow(data)) {
    stop("the strata_ids should be equal in length to nrow(data)")
  }
  if (! is.null(y_variable) & length(y_variable) > 1) {
    stop("y_variable, if provided, must be a length 1 character string.")
  }

  # historical behavior: one fold count governed both the outer evaluation
  # split and the inner ensemble-weight estimation; keep that as the default
  # while letting users decouple them.
  if (is.null(inner_n_folds)) {
    inner_n_folds <- n_folds
  }
  # historical behavior: a user-supplied cv_schema applied to both the outer
  # split and the inner super_learner() calls.
  if (! is.null(cv_schema) && is.null(inner_cv_schema)) {
    inner_cv_schema <- cv_schema
  }

  if (is.null(loss_metric)) {
    message(
      paste0(
        "The loss_metric is being inferred based on the outcome_type=",
        outcome_type, " -> using ",
        switch(outcome_type,
               'continuous' = 'CV-MSE',
               'binary' = 'negative log likelihood loss',
               'density' = 'negative log density loss',
               'multiclass' = 'negative log likelihood loss')))
    loss_metric <- default_loss_metric(outcome_type)
  }

  # one engine: all fold construction, fitting, prediction, and loss
  # computation happens inside crossfit_super_learner(), so the two entry
  # points cannot drift apart.
  cf <- crossfit_super_learner(
    data = data,
    learners = learners,
    formulas = formulas,
    y_variable = y_variable,
    n_folds = n_folds,
    inner_n_folds = inner_n_folds,
    determine_super_learner_weights = determine_super_learner_weights,
    ensemble_or_discrete = ensemble_or_discrete,
    cv_schema = cv_schema,
    inner_cv_schema = inner_cv_schema,
    outcome_type = outcome_type,
    extra_learner_args = extra_learner_args,
    cluster_ids = cluster_ids,
    strata_ids = strata_ids,
    weights = weights,
    loss_metric = loss_metric,
    use_complete_cases = use_complete_cases)

  # reconstruct the historical cv_trained_learners tibble; per-fold held-out
  # predictions are recovered from the out-of-fold vector and the fold row
  # indices (no re-prediction needed).
  oof <- cf$oof_predictions
  per_fold_predictions <- lapply(cf$fold_rows, function(rows) oof[rows])

  cv_trained_learners <- tibble::tibble(
    split = seq_len(cf$n_folds),
    learned_predictor = lapply(cf$sl_fits, function(fit) fit$predict),
    predictions = per_fold_predictions)
  cv_trained_learners[[cf$y_variable]] <- lapply(
    cf$validation_data, function(d) d[[cf$y_variable]])

  output <- list(
    cv_trained_learners = cv_trained_learners,
    cv_loss = cf$cv_loss,
    crossfit = cf)
  # surface captured warnings at the top level (they are also reachable via
  # $crossfit), mirroring super_learner()'s only-present-when-non-empty fields
  for (warn_field in c("warnings_from_fold_predictions",
                       "warnings_from_inner_super_learners",
                       "warning_learners")) {
    if (!is.null(cf[[warn_field]])) {
      output[[warn_field]] <- cf[[warn_field]]
    }
  }
  class(output) <- "nadir_cv_sl"
  output
}

# internal helpers, for cv_super_learner or its methods -------------------------

#' Per-outer-fold held-out losses for candidates and the ensemble
#'
#' For each outer fold \code{i} of a \code{nadir_cv_sl}, the candidate
#' learners fit inside \code{$crossfit$sl_fits[[i]]} (each trained on outer
#' training fold \code{i}) are used to predict on
#' \code{$crossfit$validation_data[[i]]}, and the stored
#' \code{$crossfit$loss_metric} is applied. The ensemble's per-outer-fold
#' losses come from \code{crossfit_fold_losses()} and appear under
#' \code{learner = "super_learner"}. Every loss in the result is therefore
#' computed on the same held-out outer folds, which is what makes the
#' candidates-vs-ensemble comparison fair: within each outer fold, the
#' candidates \emph{and} the ensemble weights were estimated entirely
#' without the validation data they are scored on.
#'
#' Learners that errored when fitting on an outer training fold (and any
#' learner whose prediction or loss computation fails) contribute
#' \code{NA} losses for that fold rather than aborting the whole
#' computation, mirroring the defensive behavior of
#' \code{crossfit_fold_losses()}.
#'
#' @param x A \code{nadir_cv_sl} as returned by \code{cv_super_learner()}.
#' @returns A data.frame with columns \code{learner}, \code{fold},
#'   \code{loss}; one row per (learner, outer fold) pair, including
#'   \code{"super_learner"} rows for the ensemble.
#' @keywords internal
cv_sl_fold_losses <- function(x) {
  cf <- x$crossfit
  loss_metric <- cf$loss_metric
  if (is.null(loss_metric)) {
    loss_metric <- default_loss_metric(cf$outcome_type)
  }
  # density/multiclass losses take predicted densities/probabilities of the
  # observed outcome only (the observed outcome is a column of the newdata
  # handed to each predictor, so it is already accounted for in `preds`).
  density_like <- cf$outcome_type %in% c("density", "multiclass")

  candidate_rows <- lapply(seq_len(cf$n_folds), function(i) {
    fit <- cf$sl_fits[[i]]
    vd <- cf$validation_data[[i]]
    # as.numeric() to match crossfit_observed_outcomes(), so candidate and
    # ensemble losses are computed against identically-coerced outcomes
    y <- as.numeric(vd[[cf$y_variable]])
    # flatten so names line up 1-1 with names(fit$learner_weights), including
    # multi-predictor (e.g. grid learner) expansions
    predictors <- flatten_fit_learners(fit$fit_learners)
    learner_names <- names(fit$learner_weights)
    losses <- vapply(learner_names, function(nm) {
      predictor <- predictors[[nm]]
      # non-function entries are error objects captured during fitting
      if (!is.function(predictor)) {
        return(NA_real_)
      }
      tryCatch({
        preds <- as.numeric(predictor(vd))
        if (density_like) loss_metric(preds) else loss_metric(preds, y)
      }, error = function(e) NA_real_)
    }, numeric(1))
    data.frame(learner = learner_names, fold = i, loss = unname(losses),
               stringsAsFactors = FALSE)
  })

  ensemble <- crossfit_fold_losses(cf)
  ensemble_rows <- data.frame(
    learner = "super_learner",
    fold = ensemble$fold,
    loss = ensemble$loss,
    stringsAsFactors = FALSE)

  rbind(do.call(rbind, candidate_rows), ensemble_rows)
}


#' Build the bars + per-fold jitter + point-range loss-comparison plot
#'
#' Shared plot construction for \code{plot.nadir_sl_model(type =
#' "comparison")} and \code{plot.nadir_cv_sl(type = "comparison")}: for each
#' learner, a bar and filled point at the mean held-out loss across folds, an
#' open circle for each fold's held-out loss, and a range showing \eqn{\pm 1}
#' standard deviation of the fold losses. Learners are ordered best (lowest
#' mean loss) at the top of the y axis. Rows with \code{NA} loss (e.g.
#' learners that errored in a fold) are dropped, and any learner with no
#' computable losses at all is omitted.
#'
#' @param fold_losses A data.frame with columns \code{learner}, \code{fold},
#'   \code{loss}.
#' @param loss_label The x-axis label describing the loss.
#' @param title Plot title.
#' @param caption Plot caption.
#' @returns A \code{ggplot} object.
#' @keywords internal
sl_build_comparison_plot <- function(fold_losses, loss_label, title, caption) {
  fold_losses <- fold_losses[!is.na(fold_losses$loss), , drop = FALSE]
  if (nrow(fold_losses) == 0) {
    stop("No held-out losses could be computed, so there is nothing to ",
         "plot. Check for learner errors in the fitted object.")
  }

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
    ggplot2::labs(title = title, x = loss_label, y = NULL, caption = caption) +
    ggplot2::theme_bw() +
    ggplot2::theme(plot.caption.position = "plot")
}

#' Human-readable x-axis label for the stored loss metric
#' @param cf A \code{nadir_crossfit_sl}.
#' @returns A length-1 character string.
#' @keywords internal
cv_sl_loss_label <- function(cf) {
  if (is.null(cf$loss_metric) ||
      identical(cf$loss_metric, default_loss_metric(cf$outcome_type))) {
    switch(cf$outcome_type,
           continuous = "Cross-validated held-out MSE",
           "Cross-validated held-out negative log loss")
  } else {
    "Cross-validated held-out loss (user-supplied loss_metric)"
  }
}

# nadir_cv_sl methods ------------------------------------------------------------

#' Plot a Cross-Validated Super Learner
#'
#' @description
#' Three plot types are provided:
#' \describe{
#'   \item{\code{type = "comparison"} (default)}{A \emph{fair} comparison of
#'     the candidate learners against the super learner ensemble. For each
#'     learner — candidates and \code{super_learner} alike — bars and filled
#'     points show the mean held-out loss across the \emph{outer}
#'     cross-validation folds, open circles show each outer fold's held-out
#'     loss, and ranges show \eqn{\pm 1} standard deviation across folds.
#'     Learners are ordered best (lowest mean loss) at top. This automates
#'     the figure shown in the package README.}
#'   \item{\code{type = "weights"}}{Ensemble-weight stability across the
#'     outer folds; delegated to
#'     \code{\link{plot.nadir_crossfit_sl}(x$crossfit, type = "weights")}.}
#'   \item{\code{type = "fitted"}}{Out-of-fold ensemble predictions against
#'     observed outcomes; delegated to
#'     \code{\link{plot.nadir_crossfit_sl}(x$crossfit, type = "fitted")}.
#'     Not defined for \code{outcome_type = "density"} or
#'     \code{"multiclass"}.}
#' }
#'
#' @details
#' Why is this the fair comparison, rather than
#' \code{plot(super_learner(...))}? A \code{super_learner()} fit's ensemble
#' weights are chosen \emph{using} its candidates' holdout predictions, so
#' scoring the ensemble on those same holdouts is optimistically biased —
#' which is why \code{\link{plot.nadir_sl_model}} deliberately omits the
#' ensemble from its comparison. \code{cv_super_learner()} adds the outer
#' layer of cross-validation needed to score the ensemble honestly, and this
#' plot method scores every candidate on those \emph{same} outer validation
#' folds: within each outer fold, the candidate fits and the ensemble
#' weights were all estimated entirely without the validation data they are
#' evaluated on, and every learner is trained on the same outer training
#' data. All quantities are derived from the fitted object (the per-fold
#' \code{super_learner()} fits stored in \code{$crossfit}); nothing is
#' re-fit, though each candidate is re-\emph{predicted} once per outer fold.
#'
#' The loss is the \code{loss_metric} stored at fit time (by default, mean
#' squared error for continuous outcomes and negative log loss otherwise).
#' Learners that errored on an outer training fold contribute no loss for
#' that fold.
#'
#' Requires the \pkg{ggplot2} package (listed in \code{Suggests}).
#'
#' @srrstats {RE6.0, RE6.1} A default plot() generic method is provided for
#'   the nadir_cv_sl model class.
#' @srrstats {RE6.2} plot(x, type = "fitted") plots out-of-fold fitted
#'   values against observed responses.
#'
#' @param x An object of class \code{nadir_cv_sl} as returned by
#'   \code{\link{cv_super_learner}()}.
#' @param type One of \code{"comparison"}, \code{"weights"}, or
#'   \code{"fitted"}.
#' @param ... Ignored; included for compatibility with the generic.
#' @returns A \code{ggplot} object, which prints when returned to the
#'   console.
#' @examples
#' if (requireNamespace("ggplot2", quietly = TRUE)) {
#'   cv_sl <- cv_super_learner(
#'     data = mtcars,
#'     formula = mpg ~ cyl + hp,
#'     n_folds = 3, inner_n_folds = 2,
#'     learners = list(mean = lnr_mean, lm = lnr_lm))
#'   plot(cv_sl)                    # fair candidates-vs-ensemble comparison
#'   plot(cv_sl, type = "weights")  # weight stability across outer folds
#'   plot(cv_sl, type = "fitted")   # out-of-fold predictions vs. observed
#' }
#' @export
plot.nadir_cv_sl <- function(x, type = c("comparison", "weights", "fitted"),
                             ...) {
  type <- match.arg(type)
  if (is.null(x$crossfit)) {
    stop("This nadir_cv_sl was created by a version of cv_super_learner() ",
         "that did not store $crossfit; re-fit to use plot().")
  }
  if (type %in% c("weights", "fitted")) {
    # these are properties of the underlying cross-fit; delegate so the two
    # methods cannot drift apart
    return(plot(x$crossfit, type = type, ...))
  }

  # type == "comparison"
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("plot.nadir_cv_sl() requires the {ggplot2} package. ",
         "Install it with install.packages('ggplot2').")
  }

  fold_losses <- cv_sl_fold_losses(x)

  sl_build_comparison_plot(
    fold_losses = fold_losses,
    loss_label = cv_sl_loss_label(x$crossfit),
    title = "Comparison of Candidate Learners against the Super Learner",
    caption = paste0(
      "Bars and filled points show the mean held-out loss across the outer ",
      "CV folds;\nranges show +/-1 SD across folds; each open circle is one ",
      "outer fold.\nAll learners, including the ensemble (whose weights are ",
      "re-estimated within each\nouter training fold), are scored on the ",
      "same held-out outer validation folds."))
}

#' @export
print.nadir_cv_sl <- function(x, ...) {
  cf <- x$crossfit
  cat("Cross-validated Super Learner (nadir_cv_sl)\n")
  cat("  outcome:      ", cf$y_variable, " (", cf$outcome_type, ")\n", sep = "")
  cat("  outer folds:  ", cf$n_folds,
      "   inner CV folds: ", cf$inner_n_folds, "\n", sep = "")
  if (!is.na(x$cv_loss)) {
    cat("  cross-validated loss on held-out data: ",
        format(x$cv_loss, digits = 5), "\n", sep = "")
  }
  if (!is.null(x$warning_learners) && length(x$warning_learners) > 0) {
    cat("  note: warnings were captured during training from: ",
        paste(x$warning_learners, collapse = ", "),
        "\n        see $warnings_from_inner_super_learners\n", sep = "")
  }
  cat("Methods: $cv_trained_learners, $cv_loss, $crossfit\n")
  invisible(x)
}


#' Apply Cross-Validation to a Super Learner Closure
#'
#' Taking an \code{sl_closure}, a function that trains a super learner on one
#' argument \code{data} and produces a predictor function, \code{cv_super_learner_internal}
#' applies cross validation to this \code{sl_closure} with the data passed.
#'
#' @importFrom tidyr unnest
#' @importFrom methods is
#'
#' @inheritParams cv_super_learner
#' @param sl_closure A function that takes in data and produces a `super_learner` predictor.
#' @param y_variable The string name of the outcome column in `data`
#'
#' @keywords internal
#' @returns A list containing \code{$trained_learners} and \code{$cv_loss} which
#'   respectively include 1) the trained super learner models on each fold of the data, their holdout predictions and,
#'   2) the cross-validated estimate of the risk (expected loss) on held-out data.
#'
cv_super_learner_internal <- function(
    data,
    sl_closure,
    y_variable = NULL,
    n_folds = 5,
    cv_schema = cv_random_schema,
    loss_metric,
    outcome_type = 'continuous') {

  if (length(n_folds) > 1) {
    stop("n_folds must be a length 1 numeric value.")
  }

  if (! is.null(y_variable) & length(y_variable) > 1) {
    stop("y_variable, if provided, must be a length 1 character string.")
  }

  # set up training and validation data
  #
  # the training and validation data are lists of datasets,
  # where the training data are distinct (n-1)/n subsets of the data and the
  # validation data are the corresponding other 1/n of the data.
  training_and_validation_data <- cv_schema(data, n_folds)
  training_data <- training_and_validation_data$training_data
  validation_data <- training_and_validation_data$validation_data

  trained_learners <- tibble::tibble(split = 1:n_folds)

  # train each of the learners
  trained_learners$learned_predictor <- future_lapply(
    1:nrow(trained_learners), function(i) {
      sl_closure(training_data[[i]])$predict
    }, future.seed = TRUE)

  # produce predictions from each of the trained learners for the
  # validation data
  trained_learners$predictions <- future_lapply(
    1:nrow(trained_learners), function(i) {
      trained_learners$learned_predictor[[i]](
        validation_data[[i]]
      )
    }, future.seed = TRUE)

  # add in the corresponding validation data in a column with name given by yvar
  trained_learners[[y_variable]] <-
    future_lapply(1:nrow(trained_learners), function(i) {
      validation_data[[trained_learners$split[[i]]]][[y_variable]]
    }, future.seed = TRUE)

  # unnest only the predictions and validation/held-out data
  prediction_comparison_to_validation <- tidyr::unnest(trained_learners[,c('predictions', y_variable)], cols = c('predictions', !! y_variable))

  # calculate the cv-loss
  if (missing(loss_metric)) {
    # message("The default is to report CV-MSE if no other loss_metric is specified.")
    message(
      paste0(
        "The loss_metric is being inferred based on the outcome_type=",
        outcome_type,
        " -> ",
        "using ",
        switch(
          outcome_type,
          'continuous' = 'CV-MSE',
          'binary' = 'negative log likelihood loss',
          'density' = 'negative log density loss',
          'multiclass' = 'negative log likelihood loss'
        )
      )
    )
    loss_metric <- default_loss_metric(outcome_type)
  }
  cv_loss <- loss_metric(prediction_comparison_to_validation[['predictions']], prediction_comparison_to_validation[[y_variable]])

  return(list(
    cv_trained_learners = trained_learners,
    cv_loss = cv_loss))
}
