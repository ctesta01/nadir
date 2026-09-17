
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
list_known_learners <- function(type = c('any', 'continuous', 'binary', 'density', 'multiclass')) {
  type <- match.arg(type)
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


#' Helper to Truncate a Learner's Predictions
#'
#' Take in a learner, and return a learner that produces predictor functions
#' which are truncated to the (min, max) region (boundary inclusive).
#'
#' @param lnr a nadir learner
#' @param min the numeric minimum scalar value defining the minimum that can be predicted; Could be -Inf
#' @param max the numeric maximum scalar value defining the maximum that can be predicted; Can be Inf
#' @return A learner that produces predictor functions which trim their own output before returning
#' @export
#' @examples
#' lnr_truncated <- truncate_lnr(lnr_glm, min = 20, max = 22)
#' head(lnr_glm(mtcars, mpg ~ cyl + am + hp)(mtcars))
#' head(lnr_truncated(mtcars, mpg ~ cyl + am + hp)(mtcars))
truncate_lnr <- function(lnr, min, max) {
  truncate <- function(x, min, max) {
    pmax(pmin(x, max), min)
  }

  # needs to return a learner, so it returns a function that
  # takes in its inputs and returns a prediction function
  return(
    function(...) {
      predictor_fn <- lnr(...)
      truncated_predictor_fn <- function(...) {
        truncate(predictor_fn(...), min, max)
      }
      return(truncated_predictor_fn)
    }
  )
}


#' Validate that the outcome column is compatible with the declared outcome_type
#'
#' @param data The modeling data.
#' @param y_variable The outcome column name.
#' @param outcome_type One of 'continuous', 'binary', 'density', 'multiclass'.
#' @returns TRUE invisibly, or stops with an informative error.
#' @keywords internal
validate_outcome_type_matches_y <- function(data, y_variable, outcome_type) {
  y <- data[[y_variable]]
  y_class <- paste(class(y), collapse = "/")
  ok <- switch(outcome_type,
               'continuous' = is.numeric(y),
               'binary'     = (is.numeric(y) || is.logical(y)) && all(y %in% c(0, 1)),
               'density'    = is.numeric(y),
               'multiclass' = is.numeric(y) || is.factor(y) || is.character(y))
  if (! ok) {
    expected <- switch(outcome_type,
                       'continuous' = "a numeric vector",
                       'binary'     = "a numeric or logical vector with values in {0, 1}",
                       'density'    = "a numeric vector",
                       'multiclass' = "a numeric, factor, or character vector")
    stop("outcome_type = '", outcome_type, "' was indicated, but data[['",
         y_variable, "']] is not ", expected, " (got: ", y_class,
         if (outcome_type == 'binary' && (is.numeric(y) || is.logical(y)))
           " with values outside {0, 1}" else "", ").", call. = FALSE)
  }
  invisible(TRUE)
}



#' Extract the Variables Mentioned by a Formula's Right-Hand Side
#'
#' Expands \code{.} against the data (so \code{y ~ .} and \code{y ~ . - x}
#' resolve to concrete columns, with \code{- x} removals respected) and
#' extracts every variable name appearing in the right-hand-side terms,
#' including variables hidden inside transformations (\code{poly(x, 2)},
#' \code{I(x^2)}), interactions (\code{a:x}), and lme4-style bars
#' (\code{(1 | x)}).
#'
#' Used by \code{add_stratification()} to check that the stratifying
#' variable(s) are not mentioned by the regression formula, since within a
#' stratum the stratifying variable is constant and many learners error or
#' behave unexpectedly when handed a constant predictor.
#'
#' @param formula A regression formula.
#' @param data The data the formula will be used with (required to expand
#'   \code{.}).
#' @returns A character vector of variable names appearing in the RHS terms.
#' @keywords internal
rhs_variables <- function(formula, data) {
  tt <- tryCatch(
    stats::terms(formula, data = data),
    error = function(e) NULL)

  if (is.null(tt)) {
    # terms() could not parse the formula (some highly nonstandard learner
    # syntax). fall back to all.vars(), unless `.` is present, in which case
    # we cannot know what the formula mentions and refuse to guess.
    formula_vars <- all.vars(formula)
    if ('.' %in% formula_vars) {
      stop(
        "nadir::add_stratification() could not expand the `.` in the formula ",
        "against the data in order to verify that the formula does not ",
        "mention the stratifying variable(s). Please enumerate the ",
        "predictors explicitly.")
    }
    # drop the response variable(s)
    return(setdiff(formula_vars, all.vars(formula[[2]])))
  }

  term_labels <- attr(tt, 'term.labels')
  unique(unlist(lapply(
    term_labels,
    function(lbl) all.vars(str2lang(lbl)))))
}


#' Stratify a Learner by One or More Categorical Variables
#'
#' \code{add_stratification()} takes a learner and returns a new learner
#' that, when trained, fits the original learner separately within each
#' stratum of \code{stratify_by} and uses the stratum-specific fits for
#' prediction. This is useful for enriching a \code{super_learner()} library
#' with learners that do not smooth over pre-specified subgroups, e.g. when
#' estimating nuisance regressions for subgroup-specific (heterogeneous
#' treatment effect) parameters.
#'
#' @details
#' \strong{Formulas must not mention the stratifying variable(s).} Within a
#' stratum, the stratifying variable is constant, so including it as a
#' predictor is at best degenerate and at worst an error for many learners.
#' Rather than silently rewriting the user's formula, the stratified learner
#' errors at training time if the formula mentions any of \code{stratify_by}
#' anywhere in its right-hand side (including inside transformations like
#' \code{poly()}, interactions, and \code{(1 | x)} terms). Note that
#' \code{y ~ .} therefore errors, because its expansion includes the
#' stratifying variable, which must be a column of the data; write
#' \code{y ~ . - stratifying_var} or enumerate the predictors instead.
#'
#' \strong{Pooled fallback.} By default (\code{pooled_fallback = TRUE}), the
#' original learner is also fit once on the full (unstratified) training
#' data. This pooled fit is used:
#' \itemize{
#'  \item at training time, for strata with fewer than
#'    \code{min_stratum_size} observations, and for strata in which the
#'    stratum-specific fit errors; and
#'  \item at prediction time, for rows of \code{newdata} whose stratum was
#'    not present in the training data (which can happen for rare strata
#'    under cross-validation).
#' }
#' Each fallback signals a warning naming the affected strata; inside
#' \code{super_learner()} these warnings are captured (not printed) and
#' reported in the \code{$warnings_from_*} fields of the output. With
#' \code{pooled_fallback = FALSE}, all of the above situations are errors
#' instead, and inside \code{super_learner()} an erring learner is dropped
#' from the ensemble by the usual machinery.
#'
#' \strong{Cross-validation.} When using stratified learners inside
#' \code{super_learner()}, we strongly recommend passing the stratifying
#' variable as \code{strata_ids} so that every cross-validation training
#' fold contains observations from every stratum; e.g.
#' \code{super_learner(..., strata_ids = data$categorical_var)}. Note that
#' \code{strata_ids} (which balances CV folds) and \code{stratify_by} (which
#' stratifies a learner's fits) are related but distinct.
#'
#' \strong{Supported outcome types.} Currently only \code{'continuous'} and
#' \code{'binary'} learners are supported (prediction functions returning one
#' numeric value per row of \code{newdata}). Density and multiclass learners
#' are not yet supported by \code{add_stratification()}.
#'
#' @param learner A learner (see \code{?learners}) to be fit separately
#'   within each stratum.
#' @param stratify_by A character vector of one or more column names to
#'   stratify by. With more than one name, strata are the observed
#'   combinations of the variables (their interaction). Columns must be
#'   factor, character, logical, or integer-valued numeric; non-integer
#'   numeric columns error with a suggestion to discretize first.
#' @param min_stratum_size Strata with fewer than this many observations in
#'   the training data are not given their own fit; they use the pooled
#'   fallback (or error if \code{pooled_fallback = FALSE}). Defaults to 10.
#' @param pooled_fallback If \code{TRUE} (default), additionally fit the
#'   learner on the full training data and use that pooled fit for
#'   too-small strata, strata whose fits error, and strata unseen at
#'   training time. If \code{FALSE}, each of those situations errors.
#' @returns A new learner: a function taking \code{(data, formula,
#'   weights = NULL, ...)} and returning a prediction function of
#'   \code{newdata}. Predictions are returned in the row order of
#'   \code{newdata}. The new learner carries \code{sl_lnr_name} (e.g.
#'   \code{"lm_stratified_by_cyl"}) and \code{sl_lnr_type} attributes, and
#'   inherits the wrapped learner's \code{outcome_type_dependent_args}.
#'
#' @seealso learners add_screener super_learner
#' @export
#'
#' @examples
#' # a glm fit separately within each stratum of cyl:
#' lnr_lm_by_cyl <- add_stratification(lnr_lm, stratify_by = 'cyl',
#'                                     min_stratum_size = 5)
#' trained <- lnr_lm_by_cyl(mtcars, mpg ~ hp + wt)
#' trained(mtcars)
#'
#' # note that formulas mentioning the stratifying variable error, including
#' # mpg ~ . (because its expansion includes cyl); instead write:
#' trained2 <- lnr_lm_by_cyl(mtcars, mpg ~ . - cyl)
#'
#' # inside super_learner(), pass the stratifying variable as strata_ids so
#' # every training fold contains every stratum:
#' \donttest{
#' sl <- super_learner(
#'   data = mtcars,
#'   formulas = c(.default = mpg ~ hp + wt, lm_stratified_by_cyl = mpg ~ hp + wt),
#'   learners = list(lnr_lm, lnr_lm_by_cyl),
#'   strata_ids = mtcars$cyl,
#'   n_folds = 3)
#' }
add_stratification <- function(
    learner,
    stratify_by,
    min_stratum_size = 10,
    pooled_fallback = TRUE) {

  # construction-time validation
  if (! is.function(learner)) {
    stop("nadir::add_stratification() expects `learner` to be a function. See ?learners.")
  }
  if (! is.character(stratify_by) || length(stratify_by) < 1 ||
      anyNA(stratify_by) || any(!nzchar(stratify_by))) {
    stop("`stratify_by` must be a character vector of one or more column names.")
  }
  if (! is.numeric(min_stratum_size) || length(min_stratum_size) != 1 ||
      is.na(min_stratum_size) || min_stratum_size < 1) {
    stop("`min_stratum_size` must be a single number >= 1.")
  }
  if (! is.logical(pooled_fallback) || length(pooled_fallback) != 1 ||
      is.na(pooled_fallback)) {
    stop("`pooled_fallback` must be TRUE or FALSE.")
  }

  # only continuous/binary learners are supported for now: the stratified
  # prediction function reassembles one numeric prediction per row, which is
  # not how density and multiclass predictors behave.
  base_lnr_type <- attr(learner, 'sl_lnr_type')
  if (! is.null(base_lnr_type)) {
    new_lnr_type <- intersect(base_lnr_type, c('continuous', 'binary'))
    if (length(new_lnr_type) == 0) {
      stop(
        "nadir::add_stratification() currently supports learners of type ",
        "'continuous' and/or 'binary' only, but this learner declares ",
        "sl_lnr_type: ", paste(base_lnr_type, collapse = ", "), ".")
    }
  } else {
    new_lnr_type <- NULL
  }

  # a stratum key shared between training and prediction: the values of the
  # stratify_by columns pasted together, so multi-variable stratification is
  # by the interaction of the variables.
  stratum_key <- function(d) {
    missing_columns <- setdiff(stratify_by, colnames(d))
    if (length(missing_columns) > 0) {
      stop(
        "The stratifying variable(s) ",
        paste0("'", missing_columns, "'", collapse = ", "),
        " must appear as column(s) in the data passed to a learner ",
        "constructed with nadir::add_stratification().")
    }
    key_columns <- lapply(stratify_by, function(v) as.character(d[[v]]))
    do.call(paste, c(key_columns, list(sep = ' & ')))
  }

  # the new (stratified) learner
  new_stratified_learner <- function(data, formula, weights = NULL, ...) {

    # fit-time validation
    # stratifying columns must exist, be non-missing, and be categorical-ish
    for (v in stratify_by) {
      if (! v %in% colnames(data)) {
        stop(
          "The stratifying variable '", v, "' must appear as a column in ",
          "the data passed to a learner constructed with ",
          "nadir::add_stratification().")
      }
      column <- data[[v]]
      if (anyNA(column)) {
        stop("The stratifying variable '", v, "' contains missing values.")
      }
      column_is_categorical <-
        is.factor(column) || is.character(column) || is.logical(column) ||
        (is.numeric(column) && all(column == round(column)))
      if (! column_is_categorical) {
        stop(
          "The stratifying variable '", v, "' appears to be continuous ",
          "(non-integer numeric). nadir::add_stratification() stratifies on ",
          "categorical variables; please discretize or convert '", v,
          "' to a factor first.")
      }
    }
    if (! is.null(weights) &&
        (! is.numeric(weights) || length(weights) != nrow(data))) {
      stop("`weights` must be NULL or a numeric vector of length nrow(data).")
    }

    # the formula must not mention the stratifying variable(s): within a
    # stratum they are constant, and rather than rewriting the user's formula
    # we error and ask for a formula that does not mention them. see ?add_stratification.
    mentioned <- intersect(stratify_by, rhs_variables(formula, data))
    if (length(mentioned) > 0) {
      uses_dot <- '.' %in% all.vars(formula)
      stop(
        "The formula passed to a learner stratified by ",
        paste0("'", stratify_by, "'", collapse = ", "),
        " mentions the stratifying variable(s) ",
        paste0("'", mentioned, "'", collapse = ", "),
        " in its right-hand side. Within each stratum these variables are ",
        "constant, so they must not appear as predictors. ",
        if (uses_dot) {
          paste0(
            "Since the formula uses `.`, which expands to include the ",
            "stratifying variable(s), write e.g. `",
            deparse(formula[[2]]), " ~ . - ",
            paste(mentioned, collapse = " - "), "` instead.")
        } else {
          "Please remove them from the formula."
        })
    }

    fit_learner_on <- function(rows_or_null) {
      learner_args <- list(data = data, formula = formula)
      if (! is.null(rows_or_null)) {
        learner_args$data <- data[rows_or_null, , drop = FALSE]
        if (! is.null(weights)) {
          learner_args$weights <- weights[rows_or_null]
        }
      } else if (! is.null(weights)) {
        learner_args$weights <- weights
      }
      do.call(what = learner, args = c(learner_args, list(...)))
    }

    # training
    keys <- stratum_key(data)
    stratum_sizes <- table(keys)
    strata_present <- names(stratum_sizes)

    too_small_strata <- strata_present[
      stratum_sizes[strata_present] < min_stratum_size]
    strata_to_fit <- setdiff(strata_present, too_small_strata)

    if (length(too_small_strata) > 0 && ! pooled_fallback) {
      stop(
        "The following strata have fewer than min_stratum_size = ",
        min_stratum_size, " observations and pooled_fallback = FALSE: ",
        paste0("'", too_small_strata, "'", collapse = ", "),
        ". Either lower min_stratum_size, set pooled_fallback = TRUE, or ",
        "coarsen the stratifying variable(s).")
    }

    # the pooled fit is insurance against too-small strata, erring stratum
    # fits, and strata unseen at training time (all of which can arise under
    # cross-validation); it is only fit when pooled_fallback is enabled.
    pooled_fit <- if (pooled_fallback) fit_learner_on(NULL) else NULL

    stratum_fits <- list()
    erred_strata <- character(0)
    erred_strata_messages <- character(0)
    for (stratum in strata_to_fit) {
      stratum_rows <- which(keys == stratum)
      fit_or_error <- tryCatch(
        fit_learner_on(stratum_rows),
        error = function(e) e)
      if (inherits(fit_or_error, 'error')) {
        if (! pooled_fallback) {
          stop(
            "The stratum-specific fit for stratum '", stratum, "' erred ",
            "and pooled_fallback = FALSE. Original error: ",
            conditionMessage(fit_or_error))
        }
        erred_strata <- c(erred_strata, stratum)
        erred_strata_messages <- c(
          erred_strata_messages,
          paste0("'", stratum, "': ", conditionMessage(fit_or_error)))
      } else {
        stratum_fits[[stratum]] <- fit_or_error
      }
    }

    # one collapsed warning covering everything that fell back at training
    # time; inside super_learner() this is captured into $warnings_from_*
    fell_back_strata <- c(too_small_strata, erred_strata)
    if (length(fell_back_strata) > 0) {
      warning(
        "Stratified learner fell back to the pooled fit for ",
        length(fell_back_strata), " of ", length(strata_present),
        " strata: ",
        if (length(too_small_strata) > 0) {
          paste0(
            paste0("'", too_small_strata, "'", collapse = ", "),
            " (fewer than min_stratum_size = ", min_stratum_size,
            " observations)")
        } else { "" },
        if (length(too_small_strata) > 0 && length(erred_strata) > 0) {
          "; "
        } else { "" },
        if (length(erred_strata) > 0) {
          paste0(
            "stratum fits erred for ",
            paste(erred_strata_messages, collapse = "; "))
        } else { "" },
        if (length(fell_back_strata) == length(strata_present)) {
          ". Every stratum fell back, so this stratified learner is equivalent to the unstratified learner."
        } else { "" })
    }

    # prediction
    # split newdata by stratum, predict per stratum, and reassemble in the
    # original row order of newdata.
    predict_from_stratified_fits <- function(newdata) {
      newdata_keys <- stratum_key(newdata)
      newdata_strata <- unique(newdata_keys)

      # genuinely-unseen strata (not merely ones that already fell back with
      # a warning at training time) warrant a fresh prediction-time warning
      unseen_strata <- setdiff(
        newdata_strata, c(names(stratum_fits), fell_back_strata))
      strata_without_a_fit <- setdiff(newdata_strata, names(stratum_fits))
      if (length(strata_without_a_fit) > 0 && ! pooled_fallback) {
        stop(
          "newdata contains strata without a stratum-specific fit (",
          paste0("'", strata_without_a_fit, "'", collapse = ", "),
          ") and pooled_fallback = FALSE.")
      }
      if (length(unseen_strata) > 0) {
        warning(
          "newdata contains strata unseen at training time; predictions ",
          "for these rows use the pooled (unstratified) fit: ",
          paste0("'", unseen_strata, "'", collapse = ", "))
      }

      predictions <- rep(NA_real_, nrow(newdata))
      for (stratum in newdata_strata) {
        stratum_rows <- which(newdata_keys == stratum)
        stratum_predictor <-
          if (stratum %in% names(stratum_fits)) {
            stratum_fits[[stratum]]
          } else {
            pooled_fit
          }
        predictions[stratum_rows] <- as.numeric(
          stratum_predictor(newdata[stratum_rows, , drop = FALSE]))
      }
      predictions
    }

    return(predict_from_stratified_fits)
  }

  # attributes: name, type, and outcome_type dependent args pass through so
  # that e.g. lnr_glm-based stratified learners still receive
  # family = 'binomial' automatically when outcome_type = 'binary'.
  learner_name <-
    if (! is.null(attr(learner, 'sl_lnr_name'))) {
      attr(learner, 'sl_lnr_name')
    } else {
      'unnamed_lnr'
    }
  attr(new_stratified_learner, 'sl_lnr_name') <-
    paste0(learner_name, '_stratified_by_', paste(stratify_by, collapse = '_'))
  attr(new_stratified_learner, 'sl_lnr_type') <- new_lnr_type
  attr(new_stratified_learner, 'outcome_type_dependent_args') <-
    attr(learner, 'outcome_type_dependent_args')

  return(new_stratified_learner)
}
