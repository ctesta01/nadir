# This file addresses some srr standards that do not apply to the {nadir}
# package and provides explanation

#' NA_standards
#'
#' Non-applicable standards are listed below with their tags set to
#' srrstatsNA along with explanations for why each of these standards have
#' been deemed not applicable.
#'
#' @srrstatsNA {G2.3b} we use match.arg throughout instead of using tolower() to
#'   ensure character input types match the code and authors expectations.
#' @srrstatsNA {G2.14c} Imputation of missing data is deliberately out of scope.
#'   Users must handle missingness explicitly upstream, or explicitly opt into
#'   complete-case filtering. We provide errors if users pass incomplete data
#'   without explicitly passing `complete_cases_only = TRUE` to functions like
#'   \code{super_learner()}, \code{cv_super_learner()},
#'   \code{crossfit_super_learner()}.
#' @srrstatsNA {G4.0} The package writes no outputs to local files.
#' @srrstatsNA {G5.4a} Not a new statistical method: the super learner
#'   algorithm has been previously established. Correctness is demonstrated
#'   against existing implementations (see also G5.4b).
#' @srrstatsNA {G5.4c} No stored/migrated published-paper outputs;
#'   comparison is against maintained implementations directly and done in
#'   a vignettes/articles/comparison_to_SuperLearner.Rmd
#' @srrstatsNA {G5.10, G5.11, G5.11a, G5.12} There are no extended (lengthy)
#'   tests included in the testthat/ suite: the full
#'   suite runs quickly with no large downloads, so no extended-test
#'   environment variable is needed.
#' @srrstatsNA {RE2.2} Separate missing-value handling for predictors vs.
#'   response is not provided: different learners may reference different
#'   columns via different formulas, so row-completeness is defined on the
#'   modeling frame as a whole. Prediction on new data does not require the
#'   response, so missing responses in newdata are permitted and supported.
#' @srrstatsNA {RE2.3} Centering/offsetting is not performed by nadir itself:
#'   formulas pass unmodified to learners, so users can center via formula
#'   transformations or upstream preprocessing, and learner-native offset()
#'   terms pass through to the underlying models.
#' @srrstatsNA {RE3.0, RE3.1, RE3.2, RE3.3} The ensemble-level algorithms
#'   are not iterative-convergence methods: NNLS (Lawson-Hanson) terminates
#'   exactly, and the simplex search uses stats::optim defaults documented
#'   upstream. Convergence warnings from wrapped learners (lme4, nnet, ...)
#'   propagate unmodified to the user through nadirs learner warning/error
#'   capturing system -- where the warnings/errors from learners are stored
#'   inside the returned nadir_sl_model object.
#' @srrstatsNA {RE4.1} We do not provide the means to realize an "unfitted"
#' super learned model object. Our view is that a not-yet-fit super learned model
#' is the model specification, i.e., the arguments to \code{super_learner()}.
#' At its simplest, this would be a list like so:
#' \code{
#' list(
#'    data = data,
#'    learners = list(lnr_lm, lnr_mean, lnr_rf),
#'    formulas = list(
#'      .default = y_col ~ xvar1 + xvar2
#'    )
#' )
#' }
#' This is an example of what we would call the specification for a
#' \code{nadir_sl_model} object that does not contain fitted values.
#'
#' @srrstatsNA {RE4.3, RE4.6, RE4.7} The Super Learner algorithm does not provide
#'   confidence intervals for the learner weights assigned to each candidate learner
#'   in the learner ensembling stage. We take the stance that calling
#'   \code{coef()} on a \code{nadir_sl_model} and similar super learned models
#'   should return the ensemble weights learned. Therefore, we would say a weighted ensemble of
#'   learners has no coefficient confidence intervals, parameter
#'   variance-covariance matrix, or convergence statistics.
#' @srrstatsNA {RE4.12} No transformation functions are applied to input
#'   data, so none are returned.
#' @srrstatsNA {RE4.13} Predictor data are intentionally not copied into
#'   the returned object to keep model objects lightweight; predictors are
#'   identified by formula() and remain in the user's data.
#' @srrstatsNA {RE4.14, RE4.15, RE6.3, RE7.4} nadir is not forecasting
#'   software: no forecast horizon exists, so forecast-value standards,
#'   forecast plots, and forecast-error tests do not apply.
#' @noRd
NULL
