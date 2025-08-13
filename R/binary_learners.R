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
#' \dontrun{
#'   super_learner(
#'     data = mtcars,
#'     learners = list(logistic1 = lnr_logistic, logistic2 = lnr_logistic, lnr_rf_binary),
#'     formulas = list(
#'     .default = am ~ .,
#'     logistic2 = am ~ mpg * hp + .),
#'     outcome_type = 'binary',
#'     verbose = TRUE
#'     )
#' }
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


#' Use Random Forest for Binary Classification
#'
#' @inheritParams lnr_lm
#' @examples
#' lnr_rf_binary(data = mtcars, am ~ mpg)(mtcars)
#' @returns A prediction function that accepts \code{newdata}, which returns
#'   predictions for the probability of the outcome being 1/TRUE (a numeric
#'   vector of values, one for each row of \code{newdata}).
#' @export
lnr_rf_binary <- function(data, formula, weights = NULL, ...) {
  y_variable <- as.character(formula)[2]
  if (! is.factor(data[[y_variable]])) {
    data[[y_variable]] <- as.factor(data[[y_variable]])
  }
  model <- randomForest::randomForest(formula = formula, data = data, weights = weights,
                                      type = 'classification', ...)
  return(function(newdata) {
    predict(model, newdata = newdata, type = 'prob')[,2]
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
