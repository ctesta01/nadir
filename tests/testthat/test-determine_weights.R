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
    mpg = mtcars$mpg
  )
  expect_error(
    determine_weights_using_neg_log_loss(predicted_densities, "mpg", obs_weights = c(1, 2)),
    "must be equal in length"
  )
})

test_that("determine_weights_using_neg_log_loss applies obs_weights in square case", {
  # obs_weights only multiply the loss when length(weights) == nrow(data),
  # which requires as many rows as learners; craft such a case
  set.seed(1)
  square_data <- data.frame(
    a = c(0.5, 0.6, 0.7),
    b = c(0.4, 0.5, 0.6),
    c = c(0.3, 0.4, 0.5),
    y = c(1, 0, 1)
  )
  w <- determine_weights_using_neg_log_loss(
    square_data, y_variable = "y", obs_weights = c(1, 2, 3)
  )
  expect_length(w, 3)
  expect_equal(sum(w), 1, tolerance = 1e-6)
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
  w2 <- determine_weights_for_binary_outcomes(oob, y_variable = "y")
  expect_equal(sum(w2), 1, tolerance = 1e-6)
})
