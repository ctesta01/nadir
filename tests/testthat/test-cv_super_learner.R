
testthat::test_that("cv_super_learner() uses the right number of folds.", {

  cv_output <- suppressMessages(cv_super_learner(
    data = iris[1:30,],
    formula = Petal.Length ~ Sepal.Length + Sepal.Width,
    n_folds = 6,
    learners = list(lnr_mean, lnr_lm),
  ))

  testthat::expect_equal(
    nrow(cv_output$cv_trained_learners),
    6)
})
