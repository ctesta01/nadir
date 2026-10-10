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
#' @examples
#' # the weight-determination function super_learner() uses when
#' # determine_super_learner_weights is not supplied:
#' default_determine_weights("continuous")   # non-negative least squares
#' default_determine_weights("density")      # negative-log-loss simplex
#' @export
default_determine_weights <- function(outcome_type =
    c("continuous", "binary", "density", "multiclass")) {
  outcome_type <- match.arg(outcome_type)
  switch(outcome_type,
    continuous = determine_super_learner_weights_nnls,
    binary = determine_weights_for_binary_outcomes,
    density = determine_weights_using_neg_log_loss,
    multiclass = determine_weights_using_neg_log_loss
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
#' @examples
#' # the loss used for cross-validated reporting when none is supplied;
#' # for continuous outcomes this is mean squared error:
#' continuous_loss <- default_loss_metric("continuous")
#' continuous_loss(c(1.5, 2.0), c(1, 2))  # (predicted, observed)
#'
#' # for binary outcomes, negative log loss:
#' binary_loss <- default_loss_metric("binary")
#' binary_loss(c(0.9, 0.2, 0.8), c(1, 0, 1))
#' @export
default_loss_metric <- function(outcome_type = c("continuous", "binary", "density", "multiclass")) {
  outcome_type <- match.arg(outcome_type)
  switch(outcome_type,
    continuous = mse,
    binary = negative_log_loss_for_binary,
    density = negative_log_loss,
    multiclass = negative_log_loss
  )
}
