
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


#' Round up or down randomly with probability equal to the decimal part of x
#'
#' @examples
#' for (i in 1:3) {
#'   print(nadir:::stochastic_round(c(1.01, 1.99, 1.5, 0.5, 1.6)))
#' }
#' #> [1] 1 2 2 0 2
#' #> [1] 1 2 1 1 2
#' #> [1] 1 2 1 0 1
#'
#' nadir:::stochastic_round(c(-1.01, 2.99, -5.5, 15.5, 51.6))
#' #> [1] -1  3 -5 15 51
#' @keywords internal
#' @param x A numeric vector
#' @importFrom stats rbinom
#' @returns A vector of integer values
stochastic_round <- function(x) {
  floor(x) + rbinom(n = length(x), prob = x %% 1, size = 1)
}


#' List Known Learners
#'
#' @param type One of 'any' or a supported outcome type in nadir including
#' at least 'continuous', 'binary', 'multiclass', 'density'. See \code{?super_learner()}.
#' @returns A character vector of functions that were automatically recognized as
#' nadir learners with the prediction/outcome type given.
#' @export
#' @examples
#' list_known_learners()
#' list_known_learners('continuous')
#' list_known_learners('binary')
#' list_known_learners('density')
#' list_known_learners('multiclass')
list_known_learners <- function(type = 'any') {
  ls_output <- c(ls(envir = .GlobalEnv),
                 ls(envir = environment(nadir::super_learner)))

  if (type == 'any') {
    return(ls_output[sapply(ls_output, \(x) ! is.null(attr(get(x), 'sl_lnr_type')))])
  } else if (type %in% nadir_supported_types) {
    return(ls_output[sapply(ls_output, \(x) type %in% attr(get(x), 'sl_lnr_type'))])
  }
}
