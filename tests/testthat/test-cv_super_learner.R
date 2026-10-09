testthat::test_that("cv_super_learner() uses the right number of folds.", {
  cv_output <- suppressMessages(cv_super_learner(
    data = iris[1:30, ],
    formula = Petal.Length ~ Sepal.Length + Sepal.Width,
    n_folds = 6,
    learners = list(lnr_mean, lnr_lm),
  ))

  testthat::expect_equal(
    nrow(cv_output$cv_trained_learners),
    6
  )
})


test_that("cv_super_learner outputs contain no missing or undefined values", {
  #' @srrstats {G5.3} nadir_cv_sl objects' numeric outputs (cv_loss, the
  #'   per-fold held-out predictions and observed outcomes stored in
  #'   cv_trained_learners, the embedded crossfit's oof_predictions, and
  #'   the per-(learner, fold) losses reported by summary() via
  #'   cv_sl_fold_losses()) are explicitly checked to contain no NA, NaN,
  #'   or Inf values when fit on complete data with no erring learners.
  #'   (Learners that error on a fold intentionally contribute NA losses
  #'   rather than aborting; that defensive behaviour is tested with the
  #'   error-capture tests.)
  set.seed(1)
  out <- suppressMessages(cv_super_learner(
    data = mtcars,
    learners = list(lm = lnr_lm, mean = lnr_mean),
    formulas = mpg ~ hp + wt,
    n_folds = 2
  ))

  no_bad <- function(x) expect_false(any(is.na(x) | is.nan(x) | is.infinite(x)))

  no_bad(out$cv_loss)
  # per-fold held-out ensemble predictions and observed outcomes live in
  # list-columns of the cv_trained_learners tibble
  no_bad(unlist(out$cv_trained_learners$predictions))
  no_bad(unlist(out$cv_trained_learners$mpg))
  # the embedded crossfit object's stored out-of-fold vector
  no_bad(out$crossfit$oof_predictions)
  # the per-(learner, fold) loss table behind summary()/plot(type = "comparison")
  fold_losses <- nadir:::cv_sl_fold_losses(out)
  no_bad(fold_losses$loss)
})
