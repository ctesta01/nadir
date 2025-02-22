#' Root-Mean Squared Error
#'
#' @param x A numeric vector to take the square, then mean, and then
#' square-root of.
#'
#' @export
rmse <- function(x) {
  if (! is.numeric(x) || ! is.vector(x)) {
    stop("Argument x to rmse is not a numeric vector.")
  }
  return(sqrt(mean(x^2)))
}

#' @export
mse <- function(x) {
  if (! is.numeric(x) || ! is.vector(x)) {
    stop("Argument x to rmse is not a numeric vector.")
  }
  return(mean(x^2))
}

