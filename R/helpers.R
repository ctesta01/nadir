
#' Mean Squared Error
#'
#' @keywords internal
mse <- function(x, y) {
  if (! is.numeric(x) || ! is.vector(x)) {
    stop("Argument x to mse is not a numeric vector.")
  }
  if (! is.numeric(y) || ! is.vector(y)) {
    stop("Argument y to mse is not a numeric vector.")
  }
  return(mean((x-y)^2))
}

#' Mean Squared
#'
#' @keywords internal
mean_squared <- function(x) {
  if (! is.numeric(x) || ! is.vector(x)) {
    stop("Argument x to mean_squared is not a numeric vector.")
  }
  return(mean(x^2))
}
