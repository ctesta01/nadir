#' @keywords internal
"_PACKAGE"

# .sl_fold is a column name used internally and with some dplyr / tidy-eval style
# syntax in some of the code inside {nadir}.  Declaring it as a global variable
# within this package suppresses the message that .sl_fold is an undefined variable
# in the R CMD check or devtools::check() process.  Similarly for .data being used
# with the magrittr / dplyr toolkit.
utils::globalVariables(c(".sl_fold", ".data"))

## usethis namespace: start
## usethis namespace: end
NULL

#' Outcome types supported by {nadir}
#'
#' The following outcome types are supported in the {nadir}
#' package:
#'
#' \itemize{
#'  \item continuous
#'  \item binary
#'  \item multiclass
#'  \item density
#' }
#'
#' @seealso super_learner
#'
#' @export
nadir_supported_types <- c('continuous', 'binary', 'multiclass', 'density')
