
#' Mean Squared Error
#'
#' @keywords internal
#' @returns A numeric value of the mean squared difference between x and y
mse <- function(x, y) {
  if (! is.numeric(x) || ! is.vector(x)) {
    stop("Argument x to mse is not a numeric vector.")
  }
  if (! is.numeric(y) || ! is.vector(y)) {
    stop("Argument y to mse is not a numeric vector.")
  }
  return(mean((x-y)^2))
}


#' Round up or down randomly with probability equal to the decimal part of x
#'
#' @keywords internal
#' @param x A numeric vector
#' @importFrom stats rbinom
#' @returns A vector of integer values
stochastic_round <- function(x) {
  # examples:
  # for (i in 1:3) {
  #   print(stochastic_round(c(1.01, 1.99, 1.5, 0.5, 1.6)))
  # }
  # #> [1] 1 2 2 0 2
  # #> [1] 1 2 1 1 2
  # #> [1] 1 2 1 0 1
  #
  # stochastic_round(c(-1.01, 2.99, -5.5, 15.5, 51.6))
  # #> [1] -1  3 -5 15 51
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


#' Validate that a formula has a simple left-hand side
#'
#' For example, a complex left-hand-side would be one that includes a transformation
#' like \code{log(y) ~ x1 + x2} or as is commonly done in survival modeling, a
#' survival outcome as in \code{Surv(time, death) ~ x1 + x2}.  Both of these
#' examples are considered "complex" left-hand-sides by \code{nadir} and are not
#' currently supported.  This function simply checks that the left-hand-side is
#' simple (as in, not complex), and returns `TRUE` in that case. An error is thrown
#' if the left-hand-side is complex. is not the case.
#'
#' @param formula A formula to be checked to ensure its left-hand-side (dependent/outcome) variable
#'   is not complex.
#' @returns Invisibly TRUE if okay; otherwise errors.
#' @keywords internal
check_simple_lhs <- function(formula) {
  # examples
  # check_simple_lhs(y ~ x)        # OK
  # testthat::expect_error(check_simple_lhs(log(y) ~ x))   # errors
  # testthat::expect_error(check_simple_lhs(cbind(y1,y2) ~ x))  # errors
  # testthat::expect_error(check_simple_lhs( ~ x1 + x2))   # errors because no lhs

  if (!inherits(formula, "formula")) {
    stop("`formula` must be a formula.", call. = FALSE)
  }
  ## only two-sided formulas have a true LHS
  if (length(formula) < 3) {
    stop(
      "The {nadir} package requires that the left-hand-sides of formulas be a column name from the data and not empty.",
      call. = FALSE)
  }
  if (length(formula) == 3) {
    lhs <- formula[[2]]
    ## we only allow a bare symbol:
    if (!is.name(lhs)) {
      stop(
        paste0("The {nadir} package does not support complex left-hand-sides of formulas.
",
               "For reference, the formula ", paste0(formula, collapse=' '), " was passed to {nadir}."),
        call. = FALSE
      )
    }
  }
  invisible(TRUE)
}
