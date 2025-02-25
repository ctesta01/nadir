
#' Parse Formulas for Super Learner
#'
#' @param regression_formulas Formulas to be passed to each learner of a super learner
#' @param learner_names The names of each of the learners passed to a super learner
parse_formulas <- function(
    regression_formulas,
    learner_names) {

  if (inherits(regression_formulas, 'formula')) {
    regression_formulas <- rep(c(regression_formulas), length(learner_names)) # repeat the regression formula
    names(regression_formulas) <- learner_names
    return(regression_formulas)
  }

  if (! is.vector(regression_formulas) && all(sapply(regression_formulas, class) == 'formula')) {
    stop("The regression_formulas must be passed as a vector, either a list() or c() vector of formulas.")
  }

  # if the length of the regression formulas matches the number of learners, and
  # the user did not name the regression formulas, then implicitly the user
  # has chosen to pass the regression formulas according to index-based-ordering
  if (length(regression_formulas) == length(learner_names) &&
      is.null(names(regression_formulas))) {
    names(regression_formulas) <- learner_names
    return(regression_formulas)
  }

  if (! is.null(names(regression_formulas))) {
    # either we require that there be as many regression formulas as there are learners
    if (all(learner_names %in% names(regression_formulas))) {
      # order according to learner names in this case
      regression_formulas <- regression_formulas[[learner_names]]
      names(regression_formulas) <- learner_names
      return(regression_formulas)
    }

    # or we require that .default be one of the formulas
    if (".default" %in% names(regression_formulas)) {
      regression_formulas <- lapply(
        learner_names,
        function(learner_name) {
          if (learner_name %in% names(regression_formulas)) {
            return(regression_formulas[[learner_name]])
          } else {
            return(regression_formulas[['.default']])
          }
        })
      names(regression_formulas) <- learner_names
      return(regression_formulas)
    }

    # one edge-case we do support is if the user has specified a vector of formulas,
    # some named, some not-named, but the indexing of the named formulas exactly matches
    # the names of the learners — in that case, we assume they have meant to provide
    # everything in index-based-ordering
    if (length(regression_formulas) == length(learner_names) &&
        all(
          sapply(1:length(regression_formulas), function(i) {
            names(regression_formulas)[i] %in% c("", learner_names[i])
          }))) {
      names(regression_formulas) <- learner_names
      return(regression_formulas)
    }
  }

  # if we've gotten here, none of the above cases applied, and we have a problem.
  #
  stop("Cannot appropriately match the regression_formulas to the learners.
Try making sure the names of the regression_formulas and learners match.
The regression_formulas must one of:
  * a single formula
  * a vector of formulas of the same length as the number of learners specified (with no names).
  * or a named vector of formulas including a '.default' formula and other formulas for specific learners by name.")
}

#' Extract Y Variable from a list of Regression Formulas and Learners
#'
#' @param regression_formulas A vector of formulas used for super learning
#' @param learner_names A character vector of names for the learners
#' @param data_colnames The column names of the dataset for super learning
#' @param y_variable (Optional) the y_variable specified by the user
#'
extract_y_variable <- function(
    regression_formulas,
    learner_names,
    data_colnames,
    y_variable) {
  # get all the y-variables mentioned
  y_variables <- sapply(regression_formulas, function(f) as.character(f)[[2]])

  # if the y_variable is missing and there's a unique y_variable common to
  # all formulas, then we use that
  if (missing(y_variable) & length(unique(y_variables)) == 1) {
    y_variable <- unique(y_variables)
    # if the y_variable is not common to all formulas, we cannot automatically
    # infer which y_variable we should use.
  } else if (missing(y_variable) & length(unique(y_variables)) > 1) {
    stop("Cannot infer the y-variable from the formulas passed.
Please pass y_variable = ... to nadir::super_learner.")
  }

  if (! y_variable %in% data_colnames) {
    stop("The left-hand-side of the regression formula given must appear as a column in the data passed.")
  }

  # if the y_variable matches with any of the learners, we have problems —
  # the output second_stage_SL_dataset wouldn't be interpretable.
  if (y_variable %in% learner_names) {
    stop("The outcome and names of all of the learners must be distinct, because the output
from super_learner is a data.frame with columns including the outcome variable and each of
the learners.")
  }

  return(y_variable)
}

#' Parse Extra Arguments
#'
#' @param extra_learner_args
#' @param learner_names
parse_extra_learner_arguments <- function(extra_learner_args, learner_names) {

  if (is.null(extra_learner_args)) {
    return(extra_learner_args)
  }

  if (all(learner_names %in% names(extra_learner_args))) {
    extra_learner_args <- extra_learner_args[learner_names]
    return(extra_learner_args)
  }

  if (is.null(names(extra_learner_args)) &&
      length(extra_learner_args) == length(learner_names)) {
    names(extra_learner_args) <- learner_names
    return(extra_learner_args)
  }

  if (all(names(extra_learner_args) %in% c(".default", learner_names))) {
    extra_learner_args <- lapply(
      learner_names, function(learner_name) {
        if (learner_name %in% names(extra_learner_args)) {
          extra_learner_args[[learner_name]]
        } else if ('.default' %in% names(extra_learner_args)) {
          extra_learner_args[['.default']]
        } else {
          NULL
        }
      })
    return(extra_learner_args)
  }

  stop("extra_learner_args must either be passed as:
    * NULL (the default)
    * a list() of extra arguments, in order, 1 for each learner
    * a named list() of extra arguments, 1 with each learners name
    * a named list() of extra arguments, a .default option and 1 learner for each individually specified")
}


#' Negative Log Likelihood Loss
#'
#' @details
#' \code{negative_log_lik_loss} encodes the logic:
#' if \eqn{\hat p_n} is a good model of the conditional densities, then it should minimize:
#'
#'    \deqn{ -\sum(\log(\hat p_n(X_i)) }
#'
#' @param predicted_densities
#'
negative_log_lik_loss <- function(predicted_densities, ...) {
  negative_log_predicted_densities <- -log(predicted_densities)
  # if there are 0 densities predicted, we replace them with .Machine$double.eps
  negative_log_predicted_densities[! is.finite(negative_log_predicted_densities)] <- -log(.Machine$double.eps)
  return(sum(negative_log_predicted_densities))
}

#' Softmax
#'
#' A common transformation used to go from a collection of
#' numbers from R to numbers in [0,1] such that they sum to 1.
#'
#' @param beta A vector of numeric values to transform
#'
softmax <- function(beta) {
  exp(beta) / sum(exp(beta))
}


