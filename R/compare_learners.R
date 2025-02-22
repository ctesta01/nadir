#' Compare Learners
#'
#' @param sl_output Output from `nadir::super_learner()` with `verbose = TRUE`
#'
#' @export
compare_learners <- function(
    sl_output,
    y_variable,
    metric) {

  if (! 'holdout_predictions' %in% names(sl_output)) {
    stop("compare_learners can only be run on output from nadir::super_learner
run with the verbose = TRUE option enabled.")
  }

  if (missing(y_variable)) {
    y_variable <- sl_output[['y_variable']]
  }

  if (missing(metric)) {
    message("The default in nadir::compare_learners is to use CV-MSE for comparing learners.")
    message("Other metrics can be set using the metric argument to compare_learners.")
    metric <- mse
  }

  true_outcome <- sl_output$holdout_predictions[[y_variable]]

  sl_output$holdout_predictions |>
    dplyr::select(-{{ y_variable }}) |>
    dplyr::mutate(across(everything(), ~ as.vector(. - true_outcome))) |>
    dplyr::summarize(across(everything(), metric))
}
