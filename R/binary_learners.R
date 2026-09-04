#' Binary Learners in \code{\{nadir\}}
#'
#' \itemize{
#'  \item \code{lnr_nnet}
#'  \item \code{lnr_rf_binary}
#'  \item \code{lnr_logistic}
#' }
#'
#' The important thing to know about binary learners is that they
#' need to produce predictions that the outcome is \code{ == 1} or \code{TRUE}.
#'
#' Also, for binary outcomes, we should make sure to use the
#' \code{determine_weights_for_binary_outcomes} in our calls to
#' \code{super_learner()} which calculates the estimated probability of the observed
#' outcome (either 0 or 1) and then applies the negative log loss function
#' afterwards. This can be done automatically by declaring \code{outcome_type = 'binary'}
#' in calling \code{super_learner()}
#'
#' @examples
#' super_learner(
#'   data = mtcars,
#'   learners = list(logistic1 = lnr_logistic, logistic2 = lnr_logistic, lnr_rf_binary),
#'   formulas = list(
#'   .default = am ~ .,
#'   logistic2 = am ~ mpg * hp + .),
#'   outcome_type = 'binary'
#'   )
#'
#' @seealso density_learners learners
#'
#' @rdname binary_learners
#' @name binary_learners
#' @keywords binary_learners
NULL


#' Use nnet for Binary Classification
#'
#' @export
#' @inheritParams lnr_lm
#' @importFrom nnet nnet
#' @param size Size for neural network hidden layer
#' @param trace Whether nnet should print out its optimization success
#' @return A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @examples
#'
#' lnr_nnet(mtcars, am ~ ., size = 50)(mtcars)
#' lnr_nnet(iris, I(Species=='setosa') ~ ., size = 50)(iris)
#'
lnr_nnet <- function(data, formula, trace = FALSE, size, ...) {
  fit_nnet <- nnet::nnet.formula(
    formula = formula,
    data = data,
    size = if (! missing(size)) size else round(sqrt(nrow(data))),
    trace = trace,
    ...)

  return(function(newdata) {
    predictions <- predict(fit_nnet, newdata = newdata, type = 'raw')
    if (ncol(predictions) > 1) {
      warning("lnr_nnet is supposed to be used for binary outcomes.")
    }
    return(predictions)
  })
}
attr(lnr_nnet, 'sl_lnr_name') <- 'nnet'
attr(lnr_nnet, 'sl_lnr_type') <- 'binary'


#' ranger Learner for Binary Outcomes
#'
#' A wrapper for \code{ranger::ranger()} for use in \code{nadir::super_learner()}.
#'
#' @seealso learners
#' @inheritParams lnr_lm
#' @returns A prediction function that accepts \code{newdata},
#' which returns predictions (a numeric vector of values, one for each row
#' of \code{newdata}).
#' @export
#' @importFrom ranger ranger
#'
#' @examples
#' lnr_ranger_binary(mtcars, am ~ hp)(mtcars)
lnr_ranger_binary <- function(data, formula, weights = NULL, ...) {
  y_variable <- as.character(formula)[[2]]
  if (! is.factor(data[[y_variable]])) {
    data[[y_variable]] <- factor(data[[y_variable]])   # sorted levels: "0" < "1"
  }
  positive_class <- as.character(levels(data[[y_variable]])[[2]])
  model <- ranger::ranger(data = data, case.weights = weights,
                          formula = formula, probability = TRUE, ...)
  function(newdata) {
    # ranger's probability columns follow class-encounter order, so never
    # index positionally; the colnames carry the class labels
    predict(model, data = newdata)$predictions[, positive_class]
  }
}
attr(lnr_ranger_binary, 'sl_lnr_name') <- 'ranger'
attr(lnr_ranger_binary, 'sl_lnr_type') <- 'binary'


#' Use Random Forest for Binary Classification
#'
#' @inheritParams lnr_lm
#' @examples
#' lnr_rf_binary(data = mtcars, am ~ mpg)(mtcars)
#' @returns A prediction function that accepts \code{newdata}, which returns
#'   predictions for the probability of the outcome being 1/TRUE (a numeric
#'   vector of values, one for each row of \code{newdata}).
#' @export
#'
#' @examples
#' lnr_rf_binary(mtcars, am ~ hp)(mtcars)
lnr_rf_binary <- function(data, formula, weights = NULL, ...) {
  y_variable <- as.character(formula)[2]
  if (! is.factor(data[[y_variable]])) {
    data[[y_variable]] <- factor(data[[y_variable]])
  }
  positive_class <- as.character(levels(data[[y_variable]])[[2]])
  model <- randomForest::randomForest(formula = formula, data = data, weights = weights,
                                      type = 'classification', ...)
  return(function(newdata) {
    predict(model, newdata = newdata, type = 'prob')[,positive_class]
  })
}
attr(lnr_rf_binary, 'sl_lnr_name') <- 'rf_binary'
attr(lnr_rf_binary, 'sl_lnr_type') <- 'binary'


#' Standard Logistic Regression for Binary Classification
#'
#' A wrapper provided for convenience around \code{lnr_glm} that sets
#' \code{family = binomial(link = 'logit')}.
#'
#' @inheritParams lnr_lm
#' @importFrom stats binomial
#' @returns A prediction function that accepts \code{newdata}, which returns
#'   predictions for the probability of the outcome being 1/TRUE (a numeric
#'   vector of values, one for each row of \code{newdata}).
#' @export
#'
#' @examples
#' lnr_logistic(mtcars, am ~ hp)(mtcars)
lnr_logistic <- function(data, formula, weights = NULL, ...) {
  learned_predictor <- lnr_glm(
    data = data,
    formula = formula,
    weights = weights,
    family = binomial(link = 'logit'),
    ...
  )

  return(function(newdata) { learned_predictor(newdata) })
}
attr(lnr_logistic, 'sl_lnr_name') <- 'logistic'
attr(lnr_logistic, 'sl_lnr_type') <- 'binary'



#' Support Vector Machine Learner for Binary Classification
#'
#' A wrapper for \code{e1071::svm()} with \code{probability = TRUE} for use in
#' \code{nadir::super_learner()} with binary outcomes.
#'
#' Predicted probabilities are obtained via Platt scaling
#' (see \code{?e1071::svm}) and returned for the outcome being 1/TRUE.
#'
#' Note that \code{e1071::svm()} does not support observation weights, so no
#' \code{weights} argument is accepted here.
#'
#' @seealso binary_learners
#' @inheritParams lnr_lm
#' @export
#' @returns A prediction function that accepts \code{newdata}, which returns
#'   predictions for the probability of the outcome being 1/TRUE (a numeric
#'   vector of values, one for each row of \code{newdata}).
#' @examples
#' lnr_svm_binary(mtcars, am ~ hp + mpg)(mtcars)
lnr_svm_binary <- function(data, formula, ...) {
  y_variable <- as.character(formula)[[2]]
  if (! is.factor(data[[y_variable]])) {
    data[[y_variable]] <- as.factor(data[[y_variable]])
  }
  # the "positive" (1/TRUE) class is the highest sorted factor level
  positive_level <- levels(data[[y_variable]])[
    length(levels(data[[y_variable]]))]

  model <- e1071::svm(
    formula = formula,
    data = data,
    probability = TRUE,
    ...)

  return(function(newdata) {
    # see lnr_svm: predict.svm() na.omits rows with NA in the response column
    if (y_variable %in% colnames(newdata) &&
        any(is.na(newdata[[y_variable]]))) {
      newdata[[y_variable]] <- data[[y_variable]][1]
    }
    predictions <- predict(model, newdata = newdata, probability = TRUE)
    as.vector(attr(predictions, 'probabilities')[, positive_level])
  })
}
attr(lnr_svm_binary, 'sl_lnr_name') <- 'svm_binary'
attr(lnr_svm_binary, 'sl_lnr_type') <- 'binary'


#' k-Nearest Neighbors Learner for Binary Classification
#'
#' A wrapper for \code{kknn::kknn()} for use in \code{nadir::super_learner()}
#' with binary outcomes. Predicted probabilities for the outcome being 1/TRUE
#' are the (kernel-weighted) proportion of the \code{k} nearest neighbors
#' with outcome 1/TRUE.
#'
#' Note that \code{kknn::kknn()} does not support observation weights, so no
#' \code{weights} argument is accepted here.
#'
#' @seealso binary_learners
#' @inheritParams lnr_lm
#' @param k The number of nearest neighbors to use; see \code{?kknn::kknn}.
#' @export
#' @returns A prediction function that accepts \code{newdata}, which returns
#'   predictions for the probability of the outcome being 1/TRUE (a numeric
#'   vector of values, one for each row of \code{newdata}).
#' @examples
#' lnr_knn_binary(mtcars, am ~ hp + mpg)(mtcars)
lnr_knn_binary <- function(data, formula, k = 7, ...) {
  y_variable <- as.character(formula)[[2]]
  if (! is.factor(data[[y_variable]])) {
    data[[y_variable]] <- as.factor(data[[y_variable]])
  }
  positive_level <- levels(data[[y_variable]])[
    length(levels(data[[y_variable]]))]

  return(function(newdata) {
    # kknn constructs a model.frame on the test data, so the outcome column
    # must be present in newdata; its values are ignored in prediction.
    if (! y_variable %in% colnames(newdata)) {
      newdata[[y_variable]] <- data[[y_variable]][1]
    }
    fit <- kknn::kknn(
      formula = formula,
      train = data,
      test = newdata,
      k = k,
      ...)
    as.vector(fit$prob[, positive_level])
  })
}
attr(lnr_knn_binary, 'sl_lnr_name') <- 'knn_binary'
attr(lnr_knn_binary, 'sl_lnr_type') <- 'binary'


#' Recursive Partitioning (CART) Learner for Binary Classification
#'
#' A wrapper for \code{rpart::rpart()} with \code{method = 'class'} for use
#' in \code{nadir::super_learner()} with binary outcomes.
#'
#' Because classification trees can produce pure terminal nodes, raw predicted
#' probabilities of exactly 0 or 1 are possible, which yield infinite negative
#' log loss on held-out data where such predictions are wrong. To keep
#' \code{lnr_rpart_binary} compatible with
#' \code{determine_weights_for_binary_outcomes}, predicted probabilities are
#' bounded into \code{[bound, 1 - bound]}. Set \code{bound = 0} to disable
#' this behavior.
#'
#' @seealso binary_learners
#' @inheritParams lnr_lm
#' @param bound Predicted probabilities are truncated into
#' \code{[bound, 1 - bound]} to avoid infinite negative log loss from pure
#' terminal nodes.
#' @export
#' @returns A prediction function that accepts \code{newdata}, which returns
#'   predictions for the probability of the outcome being 1/TRUE (a numeric
#'   vector of values, one for each row of \code{newdata}).
#' @examples
#' lnr_rpart_binary(mtcars, am ~ hp + mpg)(mtcars)
lnr_rpart_binary <- function(data, formula, weights = NULL, bound = 0.0025, ...) {
  y_variable <- as.character(formula)[[2]]
  if (! is.factor(data[[y_variable]])) {
    data[[y_variable]] <- as.factor(data[[y_variable]])
  }
  positive_level <- levels(data[[y_variable]])[
    length(levels(data[[y_variable]]))]

  model_args <- list(
    formula = formula,
    data = data,
    method = 'class')
  if (! is.null(weights)) {
    model_args$weights <- weights
  }
  model <- do.call(rpart::rpart, args = c(model_args, list(...)))

  return(function(newdata) {
    predictions <- as.vector(
      predict(model, newdata = newdata, type = 'prob')[, positive_level])
    pmin(pmax(predictions, bound), 1 - bound)
  })
}
attr(lnr_rpart_binary, 'sl_lnr_name') <- 'rpart_binary'
attr(lnr_rpart_binary, 'sl_lnr_type') <- 'binary'
