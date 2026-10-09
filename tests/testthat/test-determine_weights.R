# Tests for determine_weights.R

test_that("determine_super_learner_weights_nnls returns normalized weights", {
  prediction_data <- data.frame(
    lm = lnr_lm(mtcars, mpg ~ hp)(mtcars),
    mean = lnr_mean(mtcars, mpg ~ hp)(mtcars),
    mpg = mtcars$mpg
  )
  w <- determine_super_learner_weights_nnls(prediction_data, y_variable = "mpg")
  expect_length(w, 2)
  expect_equal(sum(w), 1)
  expect_true(all(w >= 0))
})

test_that("determine_super_learner_weights_nnls validates and uses obs_weights", {
  prediction_data <- data.frame(
    lm = lnr_lm(mtcars, mpg ~ hp)(mtcars),
    mean = lnr_mean(mtcars, mpg ~ hp)(mtcars),
    mpg = mtcars$mpg
  )
  expect_error(
    determine_super_learner_weights_nnls(prediction_data, "mpg", obs_weights = c(1, 2)),
    "must be equal in length"
  )

  set.seed(1)
  obs_w <- runif(nrow(mtcars))
  w <- determine_super_learner_weights_nnls(prediction_data, "mpg", obs_weights = obs_w)
  expect_equal(sum(w), 1)
})

test_that("determine_weights_using_neg_log_loss returns simplex weights", {
  set.seed(1)
  predicted_densities <- data.frame(
    lm = lnr_lm_density(mtcars, mpg ~ hp)(mtcars),
    hd = lnr_homoskedastic_density(mtcars, mpg ~ hp, mean_lnr = lnr_lm)(mtcars),
    mpg = mtcars$mpg
  )
  w <- determine_weights_using_neg_log_loss(predicted_densities, y_variable = "mpg")
  expect_length(w, 2)
  expect_equal(sum(w), 1, tolerance = 1e-6)
  expect_true(all(w >= 0 & w <= 1))
})

test_that("determine_weights_using_neg_log_loss validates obs_weights length", {
  predicted_densities <- data.frame(
    lm = lnr_lm_density(mtcars, mpg ~ hp)(mtcars),
    lm2 = lnr_lm_density(mtcars, mpg ~ hp + cyl)(mtcars),
    mpg = mtcars$mpg
  )
  expect_error(
    determine_weights_using_neg_log_loss(predicted_densities, "mpg", obs_weights = c(1, 2, 3)),
    "must be equal in length"
  )
})

test_that("neg-log-loss weights respond to obs_weights on normally-shaped data", {
  # regression test: obs_weights used to be applied only when
  # ncol(data) == nrow(data); here 2 learners, 100 rows.
  # column values are predicted densities of the observed outcome:
  # learner a assigns high density on the first half of rows, low on the
  # second half; learner b is the reverse.
  n <- 100
  d <- data.frame(
    a = c(rep(0.9, n / 2), rep(0.1, n / 2)),
    b = c(rep(0.1, n / 2), rep(0.9, n / 2)),
    y = rnorm(n) # values unused; densities are already of the observed y
  )
  w_first <- c(rep(5, n / 2), rep(0.2, n / 2))
  w_second <- rev(w_first)

  w1 <- determine_weights_using_neg_log_loss(d, "y", obs_weights = w_first)
  w2 <- determine_weights_using_neg_log_loss(d, "y", obs_weights = w_second)

  expect_equal(sum(w1), 1, tolerance = 1e-6)
  expect_equal(sum(w2), 1, tolerance = 1e-6)
  # under equal weighting the learners are symmetric; the observation
  # weights must break the tie decisively
  expect_gt(w1[1], w2[1])
  expect_gt(w1[1], 0.5)
  expect_lt(w2[1], 0.5)
})

test_that("binary-outcome weights respond to obs_weights (delegation regression test)", {
  n <- 100
  y <- rep(c(1, 0), each = n / 2)
  d <- data.frame(
    # learner a is confident-and-right on the first half (y = 1),
    # uninformative on the second; learner b is the reverse
    a = c(rep(0.95, n / 2), rep(0.5, n / 2)),
    b = c(rep(0.5, n / 2), rep(0.05, n / 2)),
    y = y
  )
  w_first <- c(rep(5, n / 2), rep(0.2, n / 2))
  w_second <- rev(w_first)

  w1 <- determine_weights_for_binary_outcomes(d, "y", obs_weights = w_first)
  w2 <- determine_weights_for_binary_outcomes(d, "y", obs_weights = w_second)

  expect_equal(sum(w1), 1, tolerance = 1e-6)
  expect_gt(w1[1], w2[1])
})


test_that("determine_weights_for_binary_outcomes transforms and weights probabilities", {
  predicted_probabilities <- data.frame(
    logistic = lnr_logistic(mtcars, am ~ hp)(mtcars),
    mean = lnr_mean(mtcars, am ~ hp)(mtcars),
    am = mtcars$am
  )
  w <- determine_weights_for_binary_outcomes(predicted_probabilities, y_variable = "am")
  expect_length(w, 2)
  expect_equal(sum(w), 1, tolerance = 1e-6)

  # out-of-bounds probabilities are clipped to [0, 1]
  oob <- data.frame(
    a = c(-0.2, 1.4, 0.5, 0.5),
    b = c(0.5, 0.5, 0.5, 0.5),
    y = c(0, 1, 1, 0)
  )
  expect_warning(
    {
      w2 <- determine_weights_for_binary_outcomes(oob, y_variable = "y")
    },
    regexp = "Column 'a' contains values outside"
  )

  expect_equal(sum(w2), 1, tolerance = 1e-6)
})

test_that("nnls falls back to equal weights when all learners get zero weight", {
  # a learner whose predictions are anti-correlated with y earns an NNLS
  # coefficient of exactly 0; normalization must not produce NaN (0/0)
  set.seed(33)
  y <- rnorm(50)
  d <- data.frame(a = -y + rnorm(50, sd = 0.1), y = y)
  expect_warning(
    w <- determine_super_learner_weights_nnls(d, "y"),
    "zero weight to every learner"
  )
  expect_equal(unname(w), 1)
  expect_false(anyNA(w))
})
