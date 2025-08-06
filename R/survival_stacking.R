
#' Repeat Observations for Survival Stacking
#'
#' Per the approach in *A review of survival stacking: a method to cast survival regression analysis as a classification problem* <https://www.degruyterbrill.com/document/doi/10.1515/ijb-2022-0055/html>
#' <https://arxiv.org/abs/2107.13480>, we provide \code{df_to_survival_stacked} as
#' a helper function for converting traditional survival data (one observation = one row) into
#' the survival stacked data structure, a repeated observations data structure where
#' multiple rows exist for each individual for each timepoint at which they were still in the
#' risk set up to and including their event time.
#'
#' @param data A data frame with survival -type outcomes including an event indicator and a time-to-event-or-censoring column
#' @param id_col (string) name of the id column that is unique to each observation in \code{data}. If one is not
#' specified, one will be created (called \code{.id}) assuming that each row is a unique observation.
#' @param time_col (string) name of the time‐to‐event column
#' @param status_col (string) name of the 0/1 event indicator column
#' @param covariate_cols  (string vector) names of your predictors
#' @param period_duration (numeric) length of each time-period (e.g. 1)
#' @param custom_times (numeric vector) [optional] A vector of the time-period breakpoints. If events could have occurred at any time after zero, this should begin with 0.
#'
#' @importFrom dplyr mutate
#' @importFrom dplyr row_number
#' @importFrom dplyr bind_rows
#' @importFrom tibble tibble
#'
#' @export
#'
df_to_survival_stacked <- function(
    data,
    id_col = NULL,
    time_col,
    status_col,
    covariate_cols,
    period_duration = 1,
    custom_times = NULL
    ) {

  if (! is.null(custom_times) & ! missing(custom_times)) {
    if (custom_times[1] != 0) {
      warning("custom_times does not begin with 0. Are you sure you want the first time-period to begin after time 0?")
    }

    if (max(custom_times) < max(data[[time_col]])) {
      warning("The maximum time in custom_times is less than the maximum time in the data frame. Are you sure you want this?")
    }
  } else {

    # get the end of the max time-period
    maxtime <- max(data[[time_col]])
    maxtime_rounding_factor <- (maxtime %% period_duration)
    # this is rounding logic:
    # say we have times up through 15.26 and we're rounding to time-periods of .25.
    # then the last time-period considered should be between 15.25 to 15.5.
    # if the max time falls on an exact multiple of the period_duration
    max_time_cutoff <- maxtime -
      (maxtime %% period_duration) +
      if (maxtime_rounding_factor == 0) 0 else period_duration

    custom_times <- seq(0, max_time_cutoff, by = period_duration)
  }

  # if no id_col provided, make one
  if (is.null(id_col)) {
    data <- data |> mutate(.id = row_number())
    id_col <- ".id"
  }

  # helper function for repeating observations the appropriate number of times
  repeat_row <- function(row_i) {

    had_event <- data[[row_i, status_col]]
    time <- data[[row_i, time_col]]

    # get the number of period cutoffs that is greater than their censoring/event time
    #
    # these are the number of complete observation periods:
    n_obs_times <- sum(custom_times <= time)
    if (had_event) {
      # in the case that their event time falls on exactly one of the cutoffs, we've
      # already counted properly;  otherwise add one (+1) for the partial observation period
      # in which they had an event
      if (! any(custom_times == time)) {
        n_obs_times <- n_obs_times + 1
      }
    }

    # these are the times we'll check-in on the observation for --
    # have they had the event in the last timestep?
    t_vec <- custom_times[1:n_obs_times]

    tibble::tibble(
      select(data[row_i,], !!! id_col),
      t = t_vec,
      event = as.integer(had_event == 1L & seq_len(n_obs_times) == n_obs_times),
      select(data[row_i,], !!! covariate_cols)
    )
  }

  # for each observation, repeat it as necessary
  new_data <- lapply(1:nrow(data), \(i) { repeat_row(i) })

  # bind the produced data together
  new_data <- bind_rows(new_data)

  return(new_data)
}

