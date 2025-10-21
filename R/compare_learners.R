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
#'   learners = list(lm = lnr_lm, rf = lnr_rf, mean = lnr_mean),
#'   formula = mpg ~ .,
#'   verbose = TRUE)
#'
#' compare_learners(sl_model)
#'
#' sl_model <- super_learner(
#'   data = mtcars,
#'   learners = list(lnr_logistic, lnr_rf_binary, mean = lnr_mean),
#'   formula = am ~ mpg,
#'   outcome_type = 'binary',
#'   verbose = TRUE)
#' compare_learners(sl_model)
#' }
#' @importFrom dplyr select summarize across everything
#' @param sl_output Output from running \code{super_learner()} with \code{verbose_output = TRUE}.
#' @param y_variable A character vector indicating the outcome variable.
#'   \code{y_variable} will be automatically inferred if it is missing and can
#'   be inferred from the \code{sl_output}.
#' @param loss_metric A loss metric, like the mean-squared-error or negative-log-loss to be
#'   used in comparing the learners. A loss metric should take two (vector) arguments:
#'   predictions, and true outcomes, and produce a single statistic summarizing the
#'   performance of each learner.
#' @returns A data.frame with the loss-metric on the held-out data for each learner.
compare_learners <- function(
    sl_output,
    y_variable,
    loss_metric) {

  if (length(y_variable) > 1) {
    stop("y_variable must be a length 1 character string.")
  }
  if (missing(y_variable)) {
    y_variable <- sl_output[['y_variable']]
  }

  if (missing(loss_metric)) {
    message("Inferring the loss metric for learner comparison based on the outcome type: ")
    message(paste0("outcome_type=", sl_output$outcome_type, " -> using ",
                   switch(sl_output$outcome_type,
                          'continuous' = 'mean squared error',
                          'density' = 'negative log loss',
                          'multiclass' = 'negative log loss',
                          'binary' = 'negative log loss'
                   )))

    switch(sl_output$outcome_type,
           'continuous' = { loss_metric <- mse },
           'density' = { loss_metric <- negative_log_loss },
           'multiclass' = { loss_metric <- negative_log_loss },
           'binary' = { loss_metric <- negative_log_loss_for_binary }
    )
  }

  true_outcome <- sl_output$holdout_predictions[[y_variable]]

  sl_output$holdout_predictions |>
    dplyr::select(-{{ y_variable }}, -.sl_fold) |>
    dplyr::summarize(dplyr::across(dplyr::everything(), ~ loss_metric(., true_outcome)))
}
