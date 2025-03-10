
#' Use Random Forst for Binary Classification
#'
#' @inheritParams lnr_lm
#' @examples
#' lnr_rf_binary(data = mtcars, am ~ mpg)(mtcars)
#' @export
lnr_rf_binary <- function(data, formula, ...) {
  y_variable <- as.character(formula)[2]
  if (! is.factor(data[[y_variable]])) {
    data[[y_variable]] <- as.factor(data[,y_variable])
  }
  model <- randomForest::randomForest(formula = formula, data = data, ...)
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
#' @export
lnr_logistic <- function(data, formula, ...) {
  learned_predictor <- lnr_glm(
    data = data,
    formula = formula,
    family = binomial(link = 'logit'),
    ...
  )

  return(function(newdata) { learned_predictor(newdata) })
}
attr(lnr_logistic, 'sl_lnr_name') <- 'logistic'
attr(lnr_logistic, 'sl_lnr_type') <- 'binary'
