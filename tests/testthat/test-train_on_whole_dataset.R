# tests for super_learner(train_on_whole_dataset = FALSE):
# skipping the whole-dataset learner fits should
#   1) genuinely skip them (each learner trained exactly n_folds times),
#   2) leave every out-of-fold interface and holdout-based S3 method intact,
#   3) produce numerically identical CV-stage results to a default fit under
#      the same RNG state, and
#   4) make $predict()/predict() error informatively.

test_that("train_on_whole_dataset = FALSE skips whole-dataset fits and preserves OOF interfaces", {

  n_folds <- 3

  # a learner that counts how many times it is trained.
  # NOTE: counting via a shared environment assumes the (default) sequential
  # {future} plan used under testthat; under parallel plans worker-side
  # assignments would not propagate back.
  fit_counter <- new.env()
  fit_counter$n <- 0L
  lnr_counting_lm <- function(data, formula, ...) {
    fit_counter$n <- fit_counter$n + 1L
    model <- stats::lm(formula, data)
    function(newdata) { stats::predict(model, newdata) }
  }
  attr(lnr_counting_lm, 'sl_lnr_type') <- 'continuous'

  learners <- list(counting_lm = lnr_counting_lm, mean = lnr_mean)

  sl_no_full <- super_learner(
    data = mtcars,
    formulas = mpg ~ hp + cyl,
    learners = learners,
    n_folds = n_folds,
    train_on_whole_dataset = FALSE)

  # 1) the counting learner was trained once per fold and never on the
  # whole dataset
  expect_identical(fit_counter$n, as.integer(n_folds))

  # by contrast, the default trains n_folds + 1 times
  fit_counter$n <- 0L
  sl_default <- super_learner(
    data = mtcars,
    formulas = mpg ~ hp + cyl,
    learners = learners,
    n_folds = n_folds)
  expect_identical(fit_counter$n, as.integer(n_folds + 1))

  # 2) structure of the returned object
  expect_s3_class(sl_no_full, "nadir_sl_model")
  expect_null(sl_no_full$fit_learners)
  expect_false(sl_no_full$train_on_whole_dataset)
  expect_true(sl_default$train_on_whole_dataset)
  expect_type(sl_default$fit_learners, "list")

  # OOF interfaces all remain functional
  expect_length(sl_no_full$oof_predictions, nrow(mtcars))
  expect_true(all(is.finite(sl_no_full$oof_predictions)))

  oof_modified <- sl_no_full$oof_predict_modified(function(d) { d$hp <- d$hp + 10; d })
  expect_length(oof_modified, nrow(mtcars))
  expect_true(all(is.finite(oof_modified)))

  # modify = NULL agrees with the stored OOF predictions
  expect_equal(
    sl_no_full$oof_predict_modified(NULL),
    sl_no_full$oof_predictions,
    tolerance = 1e-10)

  fold_preds <- sl_no_full$oof_predict_fold()
  expect_length(fold_preds, n_folds)
  expect_true(all(vapply(fold_preds, is.numeric, logical(1))))

  # holdout-based S3 methods remain functional
  expect_s3_class(summary(sl_no_full), "summary.nadir_sl_model")
  expect_length(fitted(sl_no_full), nrow(mtcars))
  expect_length(residuals(sl_no_full), nrow(mtcars))
  expect_named(coef(sl_no_full))
  expect_output(print(sl_no_full), "train_on_whole_dataset = FALSE")
  expect_s3_class(
    suppressMessages(compare_learners(sl_no_full)),
    "data.frame")

  # 4) $predict() and predict() error informatively
  expect_error(
    sl_no_full$predict(mtcars),
    regexp = "train_on_whole_dataset = FALSE")
  expect_error(
    predict(sl_no_full, newdata = mtcars),
    regexp = "oof_predict")
})

test_that("train_on_whole_dataset = FALSE reproduces the default fit's CV-stage results", {
  learners <- list(lm = lnr_lm, mean = lnr_mean)

  set.seed(1234)
  sl_default <- super_learner(
    data = mtcars,
    formulas = mpg ~ hp + cyl,
    learners = learners,
    n_folds = 3)

  set.seed(1234)
  sl_no_full <- super_learner(
    data = mtcars,
    formulas = mpg ~ hp + cyl,
    learners = learners,
    n_folds = 3,
    train_on_whole_dataset = FALSE)

  # identical RNG state => identical folds, per-fold fits, meta-learned
  # weights, and out-of-fold predictions: the CV stages are unaffected by
  # skipping the whole-dataset fits
  expect_equal(sl_no_full$learner_weights, sl_default$learner_weights,
               tolerance = 1e-12)
  expect_equal(sl_no_full$oof_predictions, sl_default$oof_predictions,
               tolerance = 1e-12)
  expect_equal(sl_no_full$fold_assignments, sl_default$fold_assignments)
})

test_that("train_on_whole_dataset input validation", {
  learners <- list(lm = lnr_lm, mean = lnr_mean)
  expect_error(
    super_learner(
      data = mtcars, formulas = mpg ~ hp, learners = learners,
      train_on_whole_dataset = NA),
    regexp = "single TRUE or FALSE")
  expect_error(
    super_learner(
      data = mtcars, formulas = mpg ~ hp, learners = learners,
      train_on_whole_dataset = c(TRUE, FALSE)),
    regexp = "single TRUE or FALSE")
  expect_error(
    super_learner(
      data = mtcars, formulas = mpg ~ hp, learners = learners,
      train_on_whole_dataset = "yes"),
    regexp = "single TRUE or FALSE")
})
