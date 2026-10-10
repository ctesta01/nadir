suppressMessages(library(testthat))
suppressWarnings(library(future))

test_that("data frames with complex columns don't intrinsically cause errors", {
  #' @srrstats {G2.11} the nadir software accepts non-traditional table elements like a
  #' list column full of data frames in a tibble without breaking.  Since this isn't
  #' the typical use case, we set up a custom learner that deals with this type of
  #' unconventional data structure explicitly as a test case.  Users will have to
  #' do similar to use list() columns of non-traditional values.
  df <- tibble::tibble(
    a = LETTERS[1:10],
    b = lapply(1:10, function(i) {
      data.frame(x = "a", y = as.integer(16 + i)) # create a list column full of data frames
    }),
    c = rnorm(n = 10, mean = 17:26)
  )

  lnr_custom <- function(data, formula) {
    # this is just for testing functionality, so we 'instrument' or
    # fake the process of fitting some model by doing nothing with
    # the formula here, but use a deterministic process
    #
    # our 'custom' learner here uses data as a dictionary to transform
    # LETTERS[1:10] into 17:26

    mapping <- function(x) {
      which_idx <- which(data[["a"]] == x)
      if (length(which_idx) == 1) {
        return(data[["b"]][[which_idx]]$y)
      } else {
        return(rnorm(n = 1, mean = 22))
      }
    }

    prediction_fn <- function(newdata) {
      sapply(seq_len(nrow(newdata)), \(x) {
        mapping(newdata$a[x])
      })
    }
  }
  attr(lnr_custom, "sl_lnr_type") <- "continuous"
  attr(lnr_custom, "sl_lnr_name") <- "custom"

  learned_predictor <- lnr_custom(df, formula = c ~ .)
  output <- learned_predictor(newdata = df)

  testthat::expect_identical(output, 17:26)

  suppressMessages( # messages expected due to list column types
    sl_fit <- nadir::super_learner(
      data = df,
      learners = list(lnr_custom, lnr_mean),
      formula = c ~ .,
      outcome_type = "continuous"
    )
  )

  # meaning neither lnr_mean nor lnr_custom failed
  expect_true(length(sl_fit$learner_weights) == 2)

  #' @srrstats {G2.12} list columns are detected and produce an informative
  #'   message which is detected here rather than producing unexplained errors.
  #'  the fit still succeeds with learners designed for such data.
  suppressMessages(
    expect_message(
      sl_fit <- nadir::super_learner(
        data = df,
        learners = list(lnr_custom, lnr_mean),
        formula = c ~ .,
        outcome_type = "continuous"
      ),
      "list columns"
    )
  )
})
