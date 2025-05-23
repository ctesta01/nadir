
#' Add a Screener to a Learner
#'
#' @returns A modified learner that when called on data and a formula
#' now runs a screening stage before fitting the learner and returning
#' a prediction function.
#' @examples
#' \dontrun{
#'
#' # construct a learner where variables with less than .6 correlation are screened out
#' lnr_glm_with_cor_60_thresholding <-
#'   add_screener(
#'     learner = lnr_glm,
#'     screener = screener_cor,
#'     screener_extra_args = list(threshold = .6)
#'   )
#'
#' # train that on the mtcars dataset — also checking that extra arguments are properly passed to glm
#' lnr_glm_with_cor_60_thresholding(mtcars, formula = mpg ~ ., family = "gaussian")(mtcars)
#'
#' # if we've screened out variables with low correlation to mpg, one such variable is qsec,
#' # so changing qsec shouldn't modify the predictions from our learned algorithm
#' mtcars_but_qsec_is_changed <- mtcars
#' mtcars_but_qsec_is_changed$qsec <- rnorm(n = nrow(mtcars))
#'
#' identical(
#'   lnr_glm_with_cor_60_thresholding(mtcars, formula = mpg ~ .)(mtcars),
#'   lnr_glm_with_cor_60_thresholding(mtcars, formula = mpg ~ .)(mtcars_but_qsec_is_changed)
#'  )
#'
#' # earth version
#' lnr_earth_with_cor_60_thresholding <-
#'   add_screener(
#'     learner = lnr_earth,
#'     screener = screener_cor,
#'     screener_extra_args = list(threshold = .6)
#'   )
#' lnr_earth_with_cor_60_thresholding(mtcars, formula = mpg ~ .)(mtcars)
#'
#' identical(
#'   lnr_earth_with_cor_60_thresholding(mtcars, formula = mpg ~ .)(mtcars),
#'   lnr_earth_with_cor_60_thresholding(mtcars, formula = mpg ~ .)(mtcars)
#'  )
#'
#' # note that this 'test' does not pass for a learner like randomForest that has
#' # some randomness in its predictions.
#'
#' }
#' @param learner A learner to be modified by wrapping a screening stage on top of it.
#' @param screener A screener to be added on top of the learner
#' @param screener_extra_args Extra arguments to be passed to the screener
add_screener <- function(learner, screener, screener_extra_args = NULL) {
  # return a function that runs the screener and the learner on the data + formula given
  #
  # this is basically our new learner
  new_learner_with_screener <- function(data, formula, ...) {
    screener_output <-
      do.call(
        what = screener,
        args = c(list(
          data = data,
          formula = formula),
          screener_extra_args
        )
      )
    screened_data <- screener_output$data # extract updated data + formula
    screened_formula <- screener_output$formula

    # what we return is the output of calling learner on the updated data and formula
    return(do.call(
      what = learner,
      args = c(list(
        data = screened_data,
        formula = screened_formula),
        ...
      )
    ))
  }
  screener_name <-
  if (! is.null(attr(screener, 'sl_screener_name'))) {
    attr(screener, 'sl_screener_name')
  } else {
    'screened'
  }
  learner_name <-
    if (! is.null(attr(learner, 'sl_lnr_name'))) {
      attr(learner, 'sl_lnr_name')
    } else {
      'unnamed_lnr'
    }
  attr(new_learner_with_screener, 'sl_lnr_type') <- attr(learner, 'sl_lnr_type')
  attr(new_learner_with_screener, 'sl_lnr_name') <- paste(screener_name, learner_name, sep = "_")

  return(new_learner_with_screener)
}


#' Correlation Threshold Based Screening
#'
#' @details
#' If a variable used has little correlation with the outcome being predicted,
#' we might want to screen that variable out from the predictors.
#'
#' In large datasets, this is quite important, as having a huge number of
#' columns could be computationally intractable or frustratingly time-consuming
#' to run \code{super_learner()} with.
#'
#' @param data A dataframe intended to be used with \code{super_learner()}
#' @param formula The formula specifying the regression to be done
#' @param threshold The correlation coefficient cutoff, below which variables
#'   are screened out from the dataset and regression formula.
#' @param cor... An optional list of extra arguments to pass to \code{cor}. Use
#'   \code{method = 'spearman'} for the Spearman rank based correlation
#'   coefficient.
#'
#' @returns A list of \code{$data} with columns screened out,
#' \code{$formula} with variables screened out, and \code{$failed_to_correlate_names}
#' the names of variables that failed to correlate with the outcome at least at the threshold
#' level.
#'
#' @examples
#' \dontrun{
#' screener_cor(
#'   data = mtcars,
#'   formula = mpg ~ .,
#'   threshold = .5)
#'
#' # We're also showing how to specify that you want the Spearman rank-based
#' # correlation coefficient, to get away from the assumption of linearity.
#'
#' screener_cor(
#'   data = mtcars,
#'   formula = mpg ~ .,
#'   threshold = .5,
#'   cor... = list(method = 'spearman')
#'   )
#' }
#' @importFrom stats cor as.formula
screener_cor <- function(data, formula, threshold = .2, cor... = NULL) {
  tryCatch({
    model_frame <- model.frame(formula = formula, data = data)
  }, error = function(e) {
    stop("nadir::screener_cor() expects that it can use model.frame() to parse the formula and data.
Meaning, the formula should be of the type that lm can support to use nadir::screener_cor().")
  })

  # main logic, assuming model.frame succeeded:
  y_variable <- as.character(formula[2])
  if (! y_variable %in% colnames(model_frame)) {
    stop("nadir::screener_cor() only supports simple right-hand-sides of formulas that already appear as column names in data.")
  }
  if (length(y_variable) != 1) {
    stop("nadir::screener_cor() only supports single-column right-hand-sides of formulas.")
  }

  y_var_index <- which(colnames(model_frame) == y_variable)[[1]]
  xdata <- model_frame[,-y_var_index]

  # construct a list of the arguments to pass to stats:cor
  cor_args <- list(
    x = xdata,
    y = model_frame[[y_variable]])
  if (! is.null(cor...)){ # append cor... if necessary
    cor_args <- c(cor_args, cor...)
  }

  cor_vec <- do.call( # call stats::cor
    what = stats::cor,
    args = cor_args)

  failed_to_correlate <- which(abs(cor_vec) < threshold)
  failed_to_correlate_names <- colnames(xdata)[failed_to_correlate]
  if (length(failed_to_correlate) > 0) {
    xdata <- xdata[,-failed_to_correlate]
  }
  if (length(colnames(xdata)) == 0) {
    warning("Correlation threshold based screening screened out all variables from the right-hand-side.")
  }
  screened_data <- cbind.data.frame(model_frame[[y_variable]], xdata)
  colnames(screened_data)[1] <- y_variable
  screened_formula <- as.formula(paste0(y_variable, " ~ ", paste0(colnames(xdata), collapse = " + ")))

  return_list <- list(
    data = screened_data,
    formula = screened_formula
  )
  if (length(failed_to_correlate) > 0) {
    return_list[['failed_to_correlate_names']] <- failed_to_correlate_names
  } else {
    return_list[['failed_to_correlate_names']] <- NULL
  }
  return(return_list)
}
attr(screener_cor, 'sl_screener_name') <- 'cor_threshold_screened'


#' Correlation Threshold Based Screening
#'
#' @details
#' If a variable used has little correlation with the outcome being predicted,
#' we might want to screen that variable out from the predictors.
#'
#' In large datasets, this is quite important, as having a huge number of
#' columns could be computationally intractable or frustratingly time-consuming
#' to run \code{super_learner()} with.
#'
#' @inheritParams screener_cor
#' @param keep_n_terms Set to an integer value >=1, this indicates that the top
#' n terms in the model frame with greatest absolute correlation with the outcome will be kept.
#'
#' @returns A list of \code{$data} with columns screened out,
#' \code{$formula} with variables screened out, and \code{$failed_to_correlate_names}
#' the names of variables that failed to correlate with the outcome at least at the threshold
#' level.
#'
#' @examples
#' \dontrun{
#' screener_cor_top_n(
#'   data = mtcars,
#'   formula = mpg ~ .,
#'   keep_n_terms = 5)
#'
#' # We're also showing how to specify that you want the Spearman rank-based
#' # correlation coefficient, to get away from the assumption of linearity.
#'
#' screener_cor_top_n(
#'   data = mtcars,
#'   formula = mpg ~ .,
#'   keep_n_terms = 5,
#'   cor... = list(method = 'spearman')
#'   )

#' }
screener_cor_top_n <- function(data, formula, keep_n_terms, cor... = NULL) {
  tryCatch({
    model_frame <- model.frame(formula = formula, data = data)
  }, error = function(e) {
    stop("nadir::screener_cor_top_n() expects that it can use model.frame() to parse the formula and data.
Meaning, the formula should be of the type that lm can support to use nadir::screener_cor_top_n().")
  })

  # main logic, assuming model.frame succeeded:
  y_variable <- as.character(formula[2])
  if (! y_variable %in% colnames(model_frame)) {
    stop("nadir::screener_cor() only supports simple right-hand-sides of formulas that already appear as column names in data.")
  }
  if (length(y_variable) != 1) {
    stop("nadir::screener_cor() only supports single-column right-hand-sides of formulas.")
  }

  y_var_index <- which(colnames(model_frame) == y_variable)[[1]]
  xdata <- model_frame[,-y_var_index]

  # construct a list of the arguments to pass to stats:cor
  cor_args <- list(
    x = xdata,
    y = model_frame[[y_variable]])
  if (! is.null(cor...)){ # append cor... if necessary
    cor_args <- c(cor_args, cor...)
  }

  # calculate correlation between each term and the outcome
  cor_vec <- do.call( # call stats::cor
    what = stats::cor,
    args = cor_args)

  # helper function to make sure we either get top_n_terms or all of the terms
  # if n is less than the number of terms considered
  top_n_values <- function(cor_vec, keep_n_terms) {
    if (length(cor_vec) <= 1) {
      stop("screener_cor_top_n calculated a correlation matrix with <=1 terms")
    }
    tail_indices <- pmax(length(cor_vec)-keep_n_terms+1, 1):length(cor_vec)
    return(sort(cor_vec)[tail_indices])
  }

  # determine which failed to meet the top n absolute correlation
  abs_cor_vec <- abs(cor_vec)
  failed_to_correlate <- which(abs_cor_vec %in% top_n_values(abs_cor_vec, keep_n_terms))
  failed_to_correlate_names <- colnames(xdata)[failed_to_correlate]
  if (length(failed_to_correlate) > 0) {
    xdata <- xdata[,-failed_to_correlate]
  }
  if (length(colnames(xdata)) == 0) {
    warning("Correlation threshold based screening screened out all variables from the right-hand-side.")
  }
  screened_data <- cbind.data.frame(model_frame[[y_variable]], xdata)
  colnames(screened_data)[1] <- y_variable
  screened_formula <- as.formula(paste0(y_variable, " ~ ", paste0(colnames(xdata), collapse = " + ")))

  return_list <- list(
    data = screened_data,
    formula = screened_formula
  )
  if (length(failed_to_correlate) > 0) {
    return_list[['failed_to_correlate_names']] <- failed_to_correlate_names
  } else {
    return_list[['failed_to_correlate_names']] <- NULL
  }
  return(return_list)

}
attr(screener_cor_top_n, 'sl_screener_name') <- 'cor_top_n_screened'
