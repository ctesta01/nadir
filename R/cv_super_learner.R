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
  oof <- cf$oof_predictions()
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
  class(output) <- "nadir_cv_sl"
  output
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
  cat("Access: $cv_trained_learners, $cv_loss, $crossfit\n")
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
