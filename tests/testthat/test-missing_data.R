suppressWarnings(library(future))

# tests designed around ensuring nadir's handling of missing data is
# correct.

#
# nadir essentially takes the perspective that it will one of three things:
#   1) work off of complete cases if the user passes `use_complete_cases = TRUE`
#   2) check if there is missing data, and if there is none, proceed
#   3) check if there is missing data, and if there is any, error and tell the
#      user to either pass complete data or pass `use_complete_cases = TRUE`.

testthat::test_that("nadir's missing data handling system works", {
  # some data with missing data -- the penguins dataset
  df_raw <- datasets::penguins
  df_complete <- df_raw[complete.cases(df_raw), ]

  # first, expect missing data to cause an error
  testthat::expect_error(
    sl_fit <- nadir::super_learner(
      data = df_raw,
      formula = flipper_len ~ bill_dep,
      learners = list(lnr_lm, lnr_mean)
    ),
    "pass use_complete_cases = TRUE" # partial match on error
  )

  # second, expect that if there is no missing data, the algorithm runs
  testthat::expect_success({
    sl_fit <- nadir::super_learner(
      data = df_complete,
      formula = flipper_len ~ bill_dep,
      learners = list(lnr_lm, lnr_mean)
    )

    if (length(sl_fit$learner_weights) == 2) {
      # success! because all 2 of the above learners worked on the complete data
      testthat::succeed()
    }
  })

  # finally (third), if there is missing data but use.complete.cases is
  # passed, a warning is thrown and the data used is the complete cases data
  expect_message(
    sl_fit_complete <- nadir::super_learner(
      data = df_raw, # has missing data
      formula = flipper_len ~ bill_dep,
      learners = list(lnr_lm, lnr_mean),
      use_complete_cases = TRUE
    ),
    "use_complete_cases = TRUE" # tell the user about how this argument works
  )

  expect_true(
    # what super_learner() is doing is just df <- df[complete.cases(df),]
    # so test this to make sure that's transparently the case and
    # a verified feature of nadir
    identical(sl_fit_complete$training_data, df_complete)
  )
})
