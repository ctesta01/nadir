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
#' @inheritParams super_learner
#' @param sl_closure A function that takes in data and produces a `super_learner` predictor.
#' @param y_variable The string name of the outcome column in `data`
#' @param loss_metric A loss metric function, like the mean-squared-error or negative-log-loss to be
#'   used in evaluating the learners on held-out data and minimized through convex optimization.
#'   A loss metric should take two (vector) arguments:
#'   predictions, and true outcomes, and produce a single statistic summarizing the
#'   performance of each learner. Defaults to the mean-squared-error \code{nadir:::mse()}.
#'
#' @importFrom tidyr unnest
#' @importFrom methods is
#'
#' @export
cv_super_learner <- function(
    data,
    sl_closure,
    y_variable,
    n_folds = 5,
    cv_schema = cv_random_schema,
    loss_metric) {

  # set up training and validation data
  #
  # the training and validation data are lists of datasets,
  # where the training data are distinct (n-1)/n subsets of the data and the
  # validation data are the corresponding other 1/n of the data.
  training_and_validation_data <- cv_schema(data, n_folds)
  training_data <- training_and_validation_data$training_data
  validation_data <- training_and_validation_data$validation_data

  trained_learners <- tibble::tibble(split = 1:n_folds)

  # parallel_lapply basically just passes to future_lapply but with future.seed = TRUE enabled
  parallel_lapply <- if (methods::is(future::plan(), "sequential")) {
    function(X, FUN, ...) {
      lapply(X, FUN, ...)
    }
  } else {
    function(X, FUN, ...) {
      future.apply::future_lapply(X, FUN, ..., future.seed = TRUE)
    }
  }

  # train each of the learners
  trained_learners$learned_predictor <- parallel_lapply(
    1:nrow(trained_learners), function(i) {
      sl_closure(training_data[[i]])
    })

  # if the super learner was accidentally specified to be verbose,
  if ("nadir_sl_verbose_output" %in% class(trained_learners$learned_predictor[[1]])) {
    warning("Ideally, the sl_closure passed to cv_super_learner should not use the verbose = TRUE argument
            inside the sl_closure.")
    trained_learners$learned_predictor <- parallel_lapply(
      1:nrow(trained_learners), function(i) {
        sl_closure(training_data[[i]])$sl_predictor
      })

  }

  # produce predictions from each of the trained learners for the
  # validation data
  trained_learners$predictions <- parallel_lapply(
    1:nrow(trained_learners), function(i) {
      trained_learners$learned_predictor[[i]](
        validation_data[[i]]
      )
    }
  )

  # add in the corresponding validation data in a column with name given by yvar
  trained_learners[[y_variable]] <-
    parallel_lapply(1:nrow(trained_learners), function(i) {
      validation_data[[trained_learners$split[[i]]]][[y_variable]]
    })

  # unnest only the predictions and validation/held-out data
  prediction_comparison_to_validation <- tidyr::unnest(trained_learners[,c('predictions', y_variable)], cols = c('predictions', !! y_variable))

  # calculate the cv-loss
  if (missing(loss_metric)) {
    message("The default is to report CV-MSE if no other loss_metric is specified.")
    loss_metric <- mse
  }
  cv_loss <- loss_metric(prediction_comparison_to_validation[['predictions']], prediction_comparison_to_validation[[y_variable]])

  return(list(
    cv_trained_learners = trained_learners,
    cv_loss = cv_loss))
}
