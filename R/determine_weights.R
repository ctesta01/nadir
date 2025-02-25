#' Determine SuperLearner Weights with Nonnegative Least Squares
#'
#' This function accepts a dataframe that is structured to have
#' one column `Y` and other columns with unique names corresponding to
#' different model predictions for `Y`, and it will use nonnegative
#' least squares to determine the weights to use for a SuperLearner.
#'
#' @param data A data frame consisting of an outcome (y_variable) and
#' other columns corresponding to predictions from candidate learners.
#' @param yvar The string name of the outcome column in `data`.
#' @return A vector of weights to be used for each of the learners.
#'
#' @export
determine_super_learner_weights_nnls <- function(data, yvar) {
  # use nonlinear least squares to produce a weighting scheme
  index_of_yvar <- which(colnames(data) == yvar)[[1]]
  nnls_output <- nnls::nnls(
    A = as.matrix(data[,-index_of_yvar]),
    b = data[[yvar]])

  weights <- nnls_output$x
  weights <- weights / sum(weights)
  return(weights)
}



#' Determine Weights for Density Estimators for SuperLearner
#'
#' @param data A data.frame with columns corresponding to predicted densities from each learner and the true y_variable from held-out data
#' @param y_variable A character indicating the outcome variable in the data.frame.
#'
determine_weights_using_neg_log_lik <- function(data, y_variable) {
  # in density estimation, the estimates have already "looked at" the
  # y-variable by the time they've predicted a density estimate.
  if (y_variable %in% colnames(data)) {
    data[[y_variable]] <- NULL
  }

  weights_after_softmax <- rep(1/ncol(data), ncol(data))
  weights_before_softmax <- log(weights_after_softmax)

  data <- as.matrix(data)

  loss_fn <- function(presoftmax_weights) {
    weights <- softmax(presoftmax_weights)

    # apply the weights to each column
    weights_applied <- sapply(1:ncol(data), function(j) {
      weights[j] * data[,j]
    })
    # sum up each row of predicted densities across learners
    # this is now like a weighted average, and crucially the weights sum to 1
    # so it's still a conditional density.
    predicted_densities <- rowSums(weights_applied)

    # now take our loss function and return it, to optimize against it
    negative_log_lik_loss(predicted_densities)
  }

  weights_optim <- stats::optim(
    par = weights_before_softmax,
    fn = loss_fn,
    method = 'Nelder-Mead')

  weights <- softmax(weights_optim$par)

  return(weights)
}
