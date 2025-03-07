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
#' @importFrom nnls nnls
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
#' @export
#'
determine_weights_using_neg_log_loss <- function(data, y_variable) {
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
    negative_log_loss(predicted_densities)
  }

  weights_optim <- stats::optim(
    par = weights_before_softmax,
    fn = loss_fn,
    method = 'Nelder-Mead')

  weights <- softmax(weights_optim$par)

  return(weights)
}


#' Determine Weights Appropriately for Super Learner given Binary Outcomes
#'
#'
#' @export
#' @param data A data.frame with columns corresponding to predicted
#'   probabilities of 1 from each learner and the true y_variable from held-out
#'   data
#' @param y_variable A character indicating the outcome variable in the data.frame.
determine_weights_for_binary_outcomes <- function(data, y_variable) {

  # for binary outcomes, predictions on the response scale are the
  # probability of the outcome being = 1.
  #
  # therefore, to get the density of the observed outcome, we need to
  # replace the data in all but the y_variable column with
  # y*data + (1-y)*(1-data)
  y <- data[[y_variable]]
  y_index <- which(colnames(data) == y_variable)[[1]]

  for (i in 1:ncol(data)) {
    if (i == y_index) {
      # do nothing
    } else {
      data[[i]] <- max(min(1, data[[i]]), 0) # bound probabilities from 0 to 1
      data[[i]] <- data[[i]] * y + (1-data[[i]]) * (1 - y)
    }
  }

  determine_weights_using_neg_log_loss(data, y_variable)
}

