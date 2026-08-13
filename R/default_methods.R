# default_methods.R ----------------------------------------------------------
# Single source of truth for outcome_type-dependent defaults.
#
# Motivation: the outcome_type -> weights-method and outcome_type -> loss
# switches are currently inlined in several places (super_learner(),
# cv_super_learner(), and the crossfit variants), and they have already
# drifted from one another: super_learner() maps density/multiclass to
# determine_weights_using_neg_log_loss, while crossfit_super_learner3() mapped
# them to determine_super_learner_weights_nnls. These helpers pin the mapping
# in one place; the inline switches elsewhere in the package should be
# replaced with calls to these.
#
# Both helpers return actual function objects (not names), so their results
# are safe to close over and ship to {future} workers.

#' Default Ensemble-Weight Method for an Outcome Type
#'
#' Returns the \code{determine_super_learner_weights} function that
#' \code{\link{super_learner}()} and \code{\link{crossfit_super_learner}()}
#' use when the user does not supply one.
#'
#' @param outcome_type One of \code{'continuous'}, \code{'binary'},
#'   \code{'density'}, or \code{'multiclass'}.
#' @returns A function suitable for the
#'   \code{determine_super_learner_weights} argument.
#' @keywords internal
#' @export
default_determine_weights <- function(outcome_type) {
  switch(
    outcome_type,
    continuous = determine_super_learner_weights_nnls,
    binary     = determine_weights_for_binary_outcomes,
    density    = determine_weights_using_neg_log_loss,
    multiclass = determine_weights_using_neg_log_loss,
    stop("Unsupported outcome_type: ", outcome_type,
         ". Must be one of 'continuous', 'binary', 'density', 'multiclass'.")
  )
}

#' Default Loss Metric for an Outcome Type
#'
#' Returns the loss metric used for reporting cross-validated /
#' cross-fitted empirical loss when the user does not supply one.
#'
#' @inheritParams default_determine_weights
#' @returns A loss function.
#' @keywords internal
#' @export
default_loss_metric <- function(outcome_type) {
  switch(
    outcome_type,
    continuous = mse,
    binary     = negative_log_loss_for_binary,
    density    = negative_log_loss,
    multiclass = negative_log_loss,
    stop("Unsupported outcome_type: ", outcome_type,
         ". Must be one of 'continuous', 'binary', 'density', 'multiclass'.")
  )
}
