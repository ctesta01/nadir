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
#' This function does print a message if the \code{loss_function} argument is
#' not set explicitly, letting the user know that the mean-squared-error will be
#' used by default. Pass in \code{loss_function = nadir:::mse} to
#' \code{super_learner()} if you'd like to suppress this message, or use a
#' similar approach for the appropriate loss function depending on context.
#'
#' @inheritParams super_learner
#' @param loss_metric A loss metric function, like the mean-squared-error or negative-log-loss to be
#'   used in evaluating the learners on held-out data and minimized through convex optimization.
#'   A loss metric should take two (vector) arguments:
#'   predictions, and true outcomes, and produce a single statistic summarizing the
#'   performance of each learner. Defaults to the mean-squared-error \code{nadir:::mse()}.
#'
#' @returns A list containing \code{$trained_learners} and \code{$cv_loss} which
#'   respectively include 1) the trained super learner models on each fold of the data, their holdout predictions and,
#'   2) the cross-validated estimate of the risk (expected loss) on held-out data.
#' @examples
#' \dontrun{
#'   cv_super_learner(
#'     data = mtcars,
#'     formula = mpg ~ cyl + hp,
#'     learners = list(lnr_mean, lnr_lm, lnr_rf))
#'
#'   cv_super_learner(
#'     data = mtcars,
#'     formula = am ~ cyl + hp,
#'     learners = list(lnr_mean, lnr_lm, lnr_logistic, lnr_rf_binary),
#'     outcome_type = 'binary')
#' }
#'
#' @export
cv_super_learner <- function(
    data,
    learners,
    formulas,
    y_variable = NULL,
    n_folds = 5,
    determine_super_learner_weights = determine_super_learner_weights_nnls,
    ensemble_or_discrete = 'ensemble',
    cv_schema = cv_random_schema,
    outcome_type = 'continuous',
    extra_learner_args = NULL,
    cluster_ids = NULL,
    strata_ids = NULL,
    weights = NULL,
    loss_metric,
    use_complete_cases = FALSE) {

  # extract the y-variable explicitly
  y_variable <- extract_y_variable(
    formulas = formulas,
    data_colnames = colnames(data),
    learner_names = names(learners),
    y_variable = y_variable
  )

  # build a closure version of the super learner specified
  sl_closure <- function(data) {
    super_learner(
      data = data,
      learners = learners,
      formulas = formulas,
      y_variable = y_variable,
      n_folds = n_folds,
      determine_super_learner_weights = determine_super_learner_weights,
      ensemble_or_discrete = ensemble_or_discrete,
      cv_schema = cv_schema,
      outcome_type = outcome_type,
      extra_learner_args = extra_learner_args,
      cluster_ids = cluster_ids,
      strata_ids = strata_ids,
      weights = weights,
      use_complete_cases = use_complete_cases)
  }

  # return the output of cv_super_learner_internal; i.e., run cross-validation
  # over sl_closure
  cv_super_learner_internal(
    data = data,
    sl_closure = sl_closure,
    y_variable = y_variable,
    cv_schema = cv_schema,
    loss_metric = loss_metric,
    outcome_type = outcome_type)
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
    y_variable,
    n_folds = 5,
    cv_schema = cv_random_schema,
    loss_metric,
    outcome_type = 'continuous') {

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
    switch(outcome_type,
           "continuous" = { loss_metric <- mse },
           "binary" = { loss_metric <- negative_log_loss_for_binary },
           "density" = { loss_metric <- negative_log_loss },
           "multiclass" = { loss_metric <- negative_log_loss }
    )
  }
  cv_loss <- loss_metric(prediction_comparison_to_validation[['predictions']], prediction_comparison_to_validation[[y_variable]])

  return(list(
    cv_trained_learners = trained_learners,
    cv_loss = cv_loss))
}
