#' Compare Learners
#'
#' Compare learners using the specified \code{loss_metric}
#'
#' @param sl_output Output from \code{nadir::super_learner()} with \code{verbose = TRUE}
#'
#' @export
#'
#' @examples
#' \dontrun{
#' sl_model <- super_learner(
#'   data = mtcars,
#'   learners = list(lm = lnr_lm, rf = lnr_randomForest, mean = lnr_mean),
#'   regression_formula = mpg ~ .,
#'   verbose = TRUE)
#'
#' compare_learners(sl_model)
#' }
compare_learners <- function(
    sl_output,
    y_variable,
    loss_metric) {

  if (! 'holdout_predictions' %in% names(sl_output)) {
    stop("compare_learners can only be run on output from nadir::super_learner
run with the verbose = TRUE option enabled.")
  }

  if (missing(y_variable)) {
    y_variable <- sl_output[['y_variable']]
  }

  if (missing(loss_metric)) {
    message("The default in nadir::compare_learners is to use CV-MSE for comparing learners.")
    message("Other metrics can be set using the loss_metric argument to compare_learners.")
    loss_metric <- mse
  }

  true_outcome <- sl_output$holdout_predictions[[y_variable]]

  sl_output$holdout_predictions |>
    dplyr::select(-{{ y_variable }}) |>
    dplyr::summarize(across(everything(), ~ loss_metric(., true_outcome)))
}
