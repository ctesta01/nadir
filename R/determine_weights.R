#' Determine SuperLearner Weights with Nonnegative Least Squares
#'
#' This function accepts a dataframe that is structured to have
#' one column `Y` and other columns with unique names corresponding to
#' different model predictions for `Y`, and it will use nonnegative
#' least squares to determine the weights to use for a SuperLearner.
#'
#' @param data A data frame consisting of an outcome (y_variable) and
#' other columns corresponding to predictions from candidate learners.
#' @param y_variable The string name of the outcome column in `data`.
#' @param obs_weights A vector of weights for each observation that dictate
#'   how prediction should be more targeted to higher weighted observations.
#' @returns A vector of weights to be used for each of the learners.
#'
#' @importFrom nnls nnls
#'
#' @examples
#' # suppose that we have a data.frame of predictions from different candidate
#' # learners:
#' prediction_data <- data.frame(
#'   lm = lnr_lm(mtcars, mpg ~ hp)(mtcars),
#'   rf = lnr_rf(mtcars, mpg ~ hp)(mtcars),
#'   rf2 = lnr_rf(mtcars, mpg ~ hp, ntree = 20)(mtcars),
#'   earth = lnr_earth(mtcars, mpg ~ hp)(mtcars))
#' # make sure it includes the outcome y_variable
#' prediction_data$mpg <- mtcars$mpg
#'
#' # we can use determine_super_learner_weights() fn to apply the non-negative least
#' # squares algorithm to produce weights for averaging the learners
#' determine_super_learner_weights_nnls(
#'   data = prediction_data,
#'   y_variable = 'mpg')
#'
#' @export
determine_super_learner_weights_nnls <- function(data, y_variable, obs_weights = NULL) {
  # use nonlinear least squares to produce a weighting scheme
  index_of_y_variable <- which(colnames(data) == y_variable)[[1]]
  A <- as.matrix(data[,-index_of_y_variable])
  b <- data[[y_variable]]

  if (! is.null(obs_weights) & length(obs_weights) != nrow(data)) {
    stop("The vector of observation weights must be equal in length to the data being passed to nadir::super_learner().")
  }

  # if there are weights to use, we use the weights by multiplying A and b by
  # the square root of the weight vector
  if (! missing(obs_weights) & ! is.null(obs_weights) & is.numeric(obs_weights) & length(obs_weights) == nrow(A)) {
    A <- A * sqrt(obs_weights)
    b <- b * sqrt(obs_weights)
  }

  nnls_output <- nnls::nnls(
    A = A,
    b = b)

  model_weights <- nnls_output$x
  model_weights <- model_weights / sum(model_weights)
  return(model_weights)
}



#' Determine Weights for Density Estimators for SuperLearner
#'
#' @param data A data.frame with columns corresponding to predicted densities from each learner and the true y_variable from held-out data
#' @param y_variable A character indicating the outcome variable in the data.frame.
#' @param bound_eps A numeric value used for truncating probability (densities) away from 0.
#' @inheritParams determine_super_learner_weights_nnls
#' @returns A vector of weights to be used for each of the learners.
#'
#' @export
#'
#' @examples
#' predicted_densities <- data.frame(
#'   lm = lnr_lm_density(mtcars, mpg ~ hp)(mtcars),
#'   earth = lnr_homoskedastic_density(mtcars, mpg ~ hp, mean_lnr = lnr_earth)(mtcars),
#'   rf = lnr_homoskedastic_density(mtcars, mpg ~ hp, mean_lnr = lnr_rf)(mtcars),
#'   rf2 = lnr_homoskedastic_density(mtcars, mpg ~ hp, mean_lnr = lnr_rf,
#'     mean_lnr_args = list(ntree = 20))(mtcars),
#'   mpg = mtcars$mpg)
#' determine_weights_using_neg_log_loss(predicted_densities, y_variable = 'mpg')
determine_weights_using_neg_log_loss <- function(data, y_variable, obs_weights = NULL,
                                                 bound_eps = 1e-3) {
  # in density estimation, the estimates have already "looked at" the
  # y-variable by the time they've predicted a density estimate.
  if (y_variable %in% colnames(data)) {
    data[[y_variable]] <- NULL
  }

  # With a single learner, the ensemble weight is trivially 1
  # Running optim() here is (a) a constant objective, since softmax of a length-1
  # parameter is always 1, and (b) triggers Nelder-Mead's 1-D unreliability
  # warning.
  if (ncol(data) == 1) {
    return(1)
  }

  weights_after_softmax <- rep(1/ncol(data), ncol(data))
  weights_before_softmax <- log(weights_after_softmax)

  data <- as.matrix(data)

  if (! is.null(obs_weights) & length(obs_weights) != nrow(data)) {
    stop("The vector of observation weights must be equal in length to the data being passed to nadir::super_learner().")
  }

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

    # bound densities away from 0 before taking logs so a single
    # zero-density prediction cannot make the loss Inf/NaN and abort optim()
    # with "function cannot be evaluated at initial parameters".
    predicted_densities <- pmax(predicted_densities, bound_eps)

    # now take our loss function and return it, to optimize against it
    negative_log_predicted_densities <- -log(predicted_densities) # negative_log_loss(predicted_densities)

    if (! is.null(obs_weights) & length(weights) == nrow(data)) {
      negative_log_predicted_densities <- negative_log_predicted_densities * obs_weights
    }
    return(sum(negative_log_predicted_densities))
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
#' @param bound_eps A numeric value used to truncate probabilities away from 0 and 1.
#' @inheritParams determine_super_learner_weights_nnls
#' @returns A vector of weights to be used for each of the learners.
#' @examples
#' predicted_probabilities <- data.frame(
#'   logistic = lnr_logistic(mtcars, am ~ hp)(mtcars),
#'   nnet = lnr_nnet(mtcars, am ~ hp)(mtcars),
#'   am = mtcars$am)
#' determine_weights_for_binary_outcomes(predicted_probabilities, y_variable = 'am')
determine_weights_for_binary_outcomes <- function(data,
                                                  y_variable,
                                                  obs_weights = NULL,
                                                  bound_eps = 1e-3) {


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
      # Diagnostic: catch learners that are not returning probabilities
      # (e.g. link-scale predictions) instead of silently clamping them.
      if (any(data[[i]] < -1e-8 | data[[i]] > 1 + 1e-8, na.rm = TRUE)) {
        warning(
          "Column '", colnames(data)[i], "' contains values outside [0, 1]; ",
          "predictions for binary outcomes are expected to be probabilities. ",
          "Values will be truncated, but check that the learner predicts on ",
          "the response (probability) scale."
        )
      }
      # FIX: clamp to [eps, 1 - eps] rather than [0, 1], so that
      # -log(density) stays finite inside the optimizer.
      eps <- bound_eps

      data[[i]] <- pmax(pmin(1 - eps, data[[i]]), eps) # bound probabilities from 0 to 1
      data[[i]] <- data[[i]] * y + (1 - data[[i]]) * (1 - y)

    }
  }

  determine_weights_using_neg_log_loss(data, y_variable, obs_weights = obs_weights, bound_eps = bound_eps)
}

