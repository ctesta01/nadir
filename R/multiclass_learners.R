
#' Multiclass Learners in \code{\{nadir\}}
#'
#' \itemize{
#'  \item \code{lnr_multinomial_nnet}
#'  \item \code{lnr_multinomial_vglm}
#' }
#'
#' Suppose one of these is trained on some data and the fit learner is stored.
#' Suppose we are going to call it on \code{newdata} and \code{newdata$class} is
#' the outcome variable being predicting.
#'
#' The important thing to know about multiclass learners is that they
#' produce predictions that the outcome class is equal to
#' \code{newdata$class} given the covariates specified in
#' \code{newdata}.
#'
#' Similar to density estimation, we want to use
#' \code{determine_weights_using_neg_log_loss} in our calls to
#' \code{super_learner()}. This can be done automatically by declaring \code{outcome_type = 'multiclass'}
#' in calling \code{super_learner()}
#'
#' @examples
#' \dontrun{
#'   super_learner(
#'     data = iris,
#'     learners = list(lnr_multinomial_vglm, lnr_multinomial_vglm, lnr_multinomial_nnet),
#'     formulas = list(
#'     .default = Species ~ .,
#'     multinomial_vglm2 = Species ~ Petal.Length*Petal.Width + .),
#'     outcome_type = 'multiclass',
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


#' \code{VGAM::vglm} Multinomial Learner
#'
#' @inheritParams lnr_lm
#' @export
#' @examples
#' df <- mtcars
#' df$cyl <- as.factor(df$cyl)
#' lnr_multinomial_vglm(df, cyl ~ hp + mpg)(df)
#' lnr_multinomial_vglm(iris, Species ~ .)(iris)
lnr_multinomial_vglm <- function(data, formula, ...) {
  fit <- VGAM::vglm(
    formula = formula,
    data = data,
    family = VGAM::multinomial,
    ...)

  y_variable <- as.character(formula)[[2]]

  return(function(newdata) {
    # returns the density at the observed outcome in the newdata
    predicted_densities <- VGAM::predict(fit, newdata = newdata, type = 'response')
    predicted_densities <- sapply(1:nrow(newdata), function(i) {
      predicted_densities[i, newdata[[y_variable]][i]]
    })
    return(predicted_densities)
  })
}
attr(lnr_multinomial_vglm, "sl_lnr_name") <- "multinomial_vglm"
attr(lnr_multinomial_vglm, "sl_lnr_type") <- "multiclass"



#' \code{nnet::multinom} Multinomial Learner
#'
#' @inheritParams lnr_lm
#' @importFrom nnet multinom
#' @export
#' @examples
#' df <- mtcars
#' df$cyl <- as.factor(df$cyl)
#' lnr_multinomial_nnet(df, cyl ~ hp + mpg)(df)
#' lnr_multinomial_nnet(iris, Species ~ .)(iris)
lnr_multinomial_nnet <- function(data, formula, ...) {
  # trace in multinom is used to suppress messages
  fit <- nnet::multinom(formula = formula, data = data, trace = FALSE, ...)
  y_variable <- as.character(formula)[2]

  return(function(newdata) {
    predicted_densities <- predict(fit, newdata = newdata, type = 'probs')
    sapply(1:nrow(newdata), function(i) {
      predicted_densities[i, newdata[[y_variable]][i]]
    })
  })
}
attr(lnr_multinomial_nnet, "sl_lnr_name") <- "multinomial_nnet"
attr(lnr_multinomial_nnet, "sl_lnr_type") <- "multiclass"
