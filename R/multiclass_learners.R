
#' \code{VGAM::vglm} Multinomial Learner
#'
#' @inheritParams lnr_lm
#' @export
#' @examples
#' df <- mtcars
#' df$cyl <- as.factor(df$cyl)
#' lnr_multinomial_vglm(df, cyl ~ hp + mpg)(df)
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
