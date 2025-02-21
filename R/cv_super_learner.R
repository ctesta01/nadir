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
#' @param yvar The string name of the outcome column in `data`
#'
#' @export
cv_super_learner <- function(
    data,
    sl_closure,
    yvar,
    n_folds = 5,
    cv_schema = cv_random_schema) {

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
  trained_learners$learned_predictor <- lapply(
    1:nrow(trained_learners), function(i) {
      sl_closure(training_data[[i]])
    })

  # produce predictions from each of the trained learners for the
  # validation data
  trained_learners$predictions <- lapply(
    1:nrow(trained_learners), function(i) {
      trained_learners$learned_predictor[[i]](
        validation_data[[i]]
      )
    }
  )

  # add in the corresponding validation data in a column with name given by yvar
  trained_learners[[yvar]] <-
    lapply(1:nrow(trained_learners), function(i) {
      validation_data[[trained_learners$split[[i]]]][[yvar]]
    })

  # unnest only the predictions and validation/held-out data
  prediction_comparison_to_validation <- tidyr::unnest(trained_learners[,c('predictions', yvar)], cols = c('predictions', !! yvar))

  # calculate the cv-rmse
  cv_mse <- (prediction_comparison_to_validation[['predictions']] - prediction_comparison_to_validation[[yvar]]) |>
    as.vector() |>
    mse()

  return(list(
    cv_trained_learners = trained_learners,
    cv_mse = cv_mse))
}
