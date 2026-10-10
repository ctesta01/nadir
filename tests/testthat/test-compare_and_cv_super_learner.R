# Tests for compare_learners.R and cv_super_learner.R

make_fake_sl_output <- function(outcome_type) {
  # a minimal object shaped like verbose super_learner output
  set.seed(1)
  hp <- data.frame(
    lm = c(0.4, 0.6, 0.5, 0.7),
    mean = c(0.5, 0.5, 0.5, 0.5),
    y = c(0, 1, 1, 0),
    .sl_fold = c(1, 1, 2, 2)
  )
  list(
    y_variable = "y",
    outcome_type = outcome_type,
    holdout_predictions = hp
  )
}

test_that("compare_learners compares learners from a fitted super learner", {
  set.seed(1)
  sl <- super_learner(
    data = mtcars,
    learners = list(lm = lnr_lm, mean = lnr_mean),
    formulas = mpg ~ hp,
    n_folds = 2
  )
  comparison <- suppressMessages(compare_learners(sl))
  expect_s3_class(comparison, "data.frame")
  expect_true(all(c("lm", "mean") %in% colnames(comparison)))
  # lm should beat the mean-only learner on MSE
  expect_lt(comparison$lm, comparison$mean)

  # an explicit y_variable and loss_metric produce no inference message
  expect_no_message(
    compare_learners(sl, y_variable = "mpg", loss_metric = nadir:::mse)
  )
})

test_that("compare_learners validates y_variable", {
  sl <- make_fake_sl_output("continuous")
  expect_error(
    compare_learners(sl, y_variable = c("a", "b")),
    "length 1 character string"
  )
})

test_that("compare_learners infers the loss metric from each outcome type", {
  for (ot in c("continuous", "binary", "density", "multiclass")) {
    sl <- make_fake_sl_output(ot)
    suppressMessages(
      expect_message(
        out <- compare_learners(sl),
        "Inferring the loss metric"
      )
    )
    expect_s3_class(out, "data.frame")
  }
})

test_that("cv_super_learner cross-validates a super learner", {
  set.seed(1)
  out <- suppressMessages(cv_super_learner(
    data = mtcars,
    learners = list(lm = lnr_lm, mean = lnr_mean),
    formulas = mpg ~ hp,
    n_folds = 2
  ))
  expect_true(is.numeric(out$cv_loss))
  expect_s3_class(out$cv_trained_learners, "data.frame")
  expect_equal(nrow(out$cv_trained_learners), 2)
})

test_that("cv_super_learner messages about the inferred loss metric", {
  set.seed(1)
  expect_message(
    cv_super_learner(
      data = mtcars,
      learners = list(lm = lnr_lm, mean = lnr_mean),
      formulas = mpg ~ hp,
      n_folds = 2
    ),
    "loss_metric is being inferred"
  )

  # an explicit loss_metric suppresses the message
  set.seed(1)
  expect_no_message(
    cv_super_learner(
      data = mtcars,
      learners = list(lm = lnr_lm, mean = lnr_mean),
      formulas = mpg ~ hp,
      n_folds = 2,
      loss_metric = nadir:::mse
    )
  )
})

test_that("cv_super_learner validates its inputs", {
  expect_error(
    cv_super_learner(mtcars, list(lm = lnr_lm), mpg ~ hp, n_folds = c(2, 3)),
    "length 1 numeric"
  )
  expect_error(
    cv_super_learner(mtcars, list(lm = lnr_lm), mpg ~ hp, cluster_ids = c(1, 2)),
    "cluster_ids should be equal in length"
  )
  expect_error(
    cv_super_learner(mtcars, list(lm = lnr_lm), mpg ~ hp, strata_ids = c(1, 2)),
    "strata_ids should be equal in length"
  )
  expect_error(
    cv_super_learner(mtcars, list(lm = lnr_lm), mpg ~ hp,
      y_variable = c("a", "b")
    ),
    "length 1 character string"
  )
})
