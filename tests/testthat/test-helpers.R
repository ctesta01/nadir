# Tests for helpers.R and loss/utility functions in super_learner_helpers.R

test_that("mse computes mean squared error and validates inputs", {
  expect_equal(mse(c(1, 2, 3), c(1, 2, 3)), 0)
  expect_equal(mse(c(0, 0), c(1, 3)), 5)
  expect_error(mse("a", c(1, 2)), "Argument x to mse is not a numeric vector")
  expect_error(mse(matrix(1:4, 2), c(1, 2)), "Argument x to mse is not a numeric vector")
  expect_error(mse(c(1, 2), "b"), "Argument y to mse is not a numeric vector")
  expect_error(mse(c(1, 2), matrix(1:4, 2)), "Argument y to mse is not a numeric vector")
})

test_that("stochastic_round rounds to floor or ceiling", {
  set.seed(1)
  x <- c(1.01, 1.99, 1.5, 0.5, 1.6, -1.01, 2.99)
  out <- stochastic_round(x)
  expect_true(all(out == floor(x) | out == ceiling(x)))
  # whole numbers are unchanged
  expect_equal(stochastic_round(c(1, 2, 5)), c(1, 2, 5))
})

test_that("softmax maps reals to a simplex", {
  out <- softmax(c(0, 0))
  expect_equal(out, c(0.5, 0.5))
  out2 <- softmax(c(-5, 2, 3))
  expect_equal(sum(out2), 1)
  expect_true(all(out2 >= 0 & out2 <= 1))
})

test_that("check_simple_lhs accepts simple formulas and rejects complex ones", {
  expect_invisible(check_simple_lhs(y ~ x))
  expect_true(check_simple_lhs(y ~ x))
  expect_error(check_simple_lhs("y ~ x"), "must be a formula")
  expect_error(check_simple_lhs(~ x1 + x2), "not empty")
  expect_error(check_simple_lhs(log(y) ~ x), "complex left-hand-sides")
  expect_error(check_simple_lhs(cbind(y1, y2) ~ x), "complex left-hand-sides")
})

test_that("list_known_learners returns learners by type", {
  any_learners <- list_known_learners()
  expect_true("lnr_lm" %in% any_learners)
  expect_true("lnr_lm_density" %in% any_learners)

  cont <- list_known_learners("continuous")
  expect_true("lnr_lm" %in% cont)
  expect_false("lnr_lm_density" %in% cont)

  bin <- list_known_learners("binary")
  expect_true("lnr_logistic" %in% bin)

  dens <- list_known_learners("density")
  expect_true("lnr_lm_density" %in% dens)

  mc <- list_known_learners("multiclass")
  expect_true("lnr_multinomial_nnet" %in% mc)

  # an unsupported type falls throws an unknown input error
  expect_error(list_known_learners("not_a_type"))
})

test_that("nadir_supported_types contains the four supported outcome types", {
  expect_setequal(
    nadir_supported_types,
    c("continuous", "binary", "multiclass", "density")
  )
})

test_that("negative_log_loss sums -log densities, replacing non-finite values", {
  expect_equal(negative_log_loss(c(1, 1)), 0)
  expect_equal(negative_log_loss(c(exp(-1), exp(-2))), 3)
  # zero densities produce non-finite -log values which get replaced
  out <- negative_log_loss(c(0, 1))
  expect_true(is.finite(out))
  expect_equal(out, -log(.Machine$double.eps))
  # extra args are accepted and ignored
  expect_equal(negative_log_loss(c(1, 1), c(0, 1)), 0)
})

test_that("negative_log_loss_for_binary computes loss of observed outcomes", {
  # predicted P(y=1) = 0.8 with true outcomes 1 and 0
  out <- negative_log_loss_for_binary(c(0.8, 0.8), c(1, 0))
  expect_equal(out, -log(0.8) - log(0.2))
})


test_that("outcome_type is validated against the outcome column's type", {
  d_num <- data.frame(y = rnorm(5))
  expect_silent(validate_outcome_type_matches_y(d_num, "y", "continuous"))
  expect_silent(validate_outcome_type_matches_y(
    data.frame(y = rpois(5, 3)), "y", "continuous"))       # integer y is numeric
  expect_error(validate_outcome_type_matches_y(
    data.frame(y = factor(letters[1:5])), "y", "continuous"), "not a numeric")
  expect_silent(validate_outcome_type_matches_y(
    data.frame(y = c(TRUE, FALSE)), "y", "binary"))
  expect_error(validate_outcome_type_matches_y(
    data.frame(y = c(0, 1, 3.7)), "y", "binary"), "values outside")
  expect_silent(validate_outcome_type_matches_y(
    data.frame(y = factor(c("a", "b"), ordered = TRUE)), "y", "multiclass"))
  expect_error(validate_outcome_type_matches_y(
    data.frame(y = as.Date("2020-01-01") + 0:3), "y", "multiclass"))
})
