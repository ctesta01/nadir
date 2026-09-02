#' The nadir R package
#'
#' @srrstats {G1.0} Primary references (van der Laan, Polley & Hubbard 2007)
#'   are cited in ?super_learner and the README.  [nadir-package.R]
#' @srrstats {G1.1} README's "Why reimplement super learner again?" states
#'   how this differs from {SuperLearner}, {sl3}, {mlr3superlearner}.
#' @srrstats {G1.2} Life Cycle Statement in CONTRIBUTING.md.  [see 6c]
#' @srrstats {G1.3} All terminology (ensemble vs. discrete super learning, cross-fitting,
#'   meta-learning, screeners) are defined in our roxygen2 documentation and vignettes.
#' @srrstats {G1.4, G1.4a} All functions documented with roxygen2; internal
#'   functions use @keywords internal.
#' @srrstats {G1.5, G1.6} Performance and comparison claims are reproducible
#'   via the Benchmarking and comparison_to_SuperLearner vignettes.
#' @keywords internal
#' @srrstats {G5.0, G5.1} Tests use standard datasets (mtcars, iris,
#'   Boston) and inline seeded simulations.
#' @srrstats {G5.2, G5.2a, G5.2b} Error/warning conditions carry unique
#'   messages asserted in the test suite.
#' @srrstats {G5.4, G5.4b} Correctness checked against {SuperLearner} and
#'   {sl3} in the comparison vignette.
#' @srrstats {G5.5} Correctness tests run under fixed seeds.
#' @srrstats {G5.6, G5.6a} Parameter recovery: obs_weights recovery tests
#'   in test-determine_weights.R; exact-relationship recovery in
#'   test-srr-correctness.R (see 6b).
#' @srrstats {RE4.0} The output of models fit with nadir are model classes: \code{nadir_sl_model},
#'   \code{nadir_crossfit_sl}, \code{nadir_cv_sl}, which themselves have supporting
#'   regression related S3 methods.

"_PACKAGE"

# .sl_fold is a column name used internally and with some dplyr / tidy-eval style
# syntax in some of the code inside {nadir}.  Declaring it as a global variable
# within this package suppresses the message that .sl_fold is an undefined variable
# in the R CMD check or devtools::check() process.  Similarly for .data being used
# with the magrittr / dplyr toolkit.
utils::globalVariables(c(".sl_fold", ".data"))

## usethis namespace: start
#' @importFrom lifecycle deprecated
## usethis namespace: end
NULL

#' Outcome types supported by \code{{nadir}}
#'
#' The following outcome types are supported in the \code{{nadir}}
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
