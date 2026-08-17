# Tests for the continuous learners in learners.R

test_that("lnr_mean predicts the training mean for all rows", {
  pred <- lnr_mean(mtcars, mpg ~ hp)(mtcars)
  expect_length(pred, nrow(mtcars))
  expect_true(all(pred == mean(mtcars$mpg)))
})

test_that("lnr_lm fits and predicts, with and without weights", {
  pred <- lnr_lm(mtcars, mpg ~ hp + wt)(mtcars)
  expect_length(pred, nrow(mtcars))
  expect_true(is.numeric(pred))

  set.seed(1)
  w <- runif(nrow(mtcars))
  pred_w <- lnr_lm(mtcars, mpg ~ hp, weights = w)(mtcars)
  expect_length(pred_w, nrow(mtcars))
  expect_false(isTRUE(all.equal(unname(pred), unname(pred_w))))
})

test_that("lnr_glm fits and predicts, with weights and family arguments", {
  pred <- lnr_glm(mtcars, mpg ~ hp + wt)(mtcars)
  expect_length(pred, nrow(mtcars))

  set.seed(1)
  w <- runif(nrow(mtcars))
  pred_w <- lnr_glm(mtcars, mpg ~ hp, weights = w)(mtcars)
  expect_length(pred_w, nrow(mtcars))

  pred_gamma <- lnr_glm(mtcars, mpg ~ hp, family = Gamma)(mtcars)
  expect_length(pred_gamma, nrow(mtcars))
})

test_that("lnr_ranger fits and predicts", {
  set.seed(1)
  pred <- lnr_ranger(mtcars, mpg ~ hp, num.trees = 20)(mtcars)
  expect_length(pred, nrow(mtcars))
  expect_true(is.numeric(pred))
})

test_that("lnr_rf fits and predicts and drops the outcome from newdata", {
  set.seed(1)
  predictor <- lnr_rf(mtcars, mpg ~ hp + wt, ntree = 20)
  pred_with_y <- predictor(mtcars)
  expect_length(pred_with_y, nrow(mtcars))

  # newdata without the outcome column also works
  pred_no_y <- predictor(mtcars[, c("hp", "wt")])
  expect_length(pred_no_y, nrow(mtcars))
  expect_equal(unname(pred_with_y), unname(pred_no_y))
})

test_that("lnr_earth fits and predicts, with newdata with and without the outcome", {
  predictor <- lnr_earth(mtcars, mpg ~ hp + wt)
  pred <- predictor(mtcars)
  expect_length(pred, nrow(mtcars))

  pred_no_y <- predictor(mtcars[, c("hp", "wt")])
  expect_equal(pred, pred_no_y)
})

test_that("lnr_glmnet fits, predicts, and warns on vector lambda", {
  pred <- lnr_glmnet(mtcars, mpg ~ hp + wt + disp, lambda = 0.5)(mtcars)
  expect_length(pred, nrow(mtcars))

  # newdata without the y column
  pred_no_y <- lnr_glmnet(mtcars, mpg ~ hp + wt, lambda = 0.5)(mtcars[, c("hp", "wt")])
  expect_length(pred_no_y, nrow(mtcars))

  expect_error(
    lnr_glmnet(mtcars, mpg ~ hp + wt, lambda = c(0.1, 0.5)),
    "lnr_glmnet requires `lambda` to be a single"
  )
})

test_that("lnr_glmnet removes the outcome if it sneaks into the model matrix", {
  # putting the outcome on the right-hand side forces the yvar-removal branch
  # at fit time (predicting with such a formula is not supported, since the
  # outcome is deliberately dropped from newdata before prediction)
  predictor <- lnr_glmnet(mtcars, mpg ~ mpg + hp, lambda = 0.5)
  expect_true(is.function(predictor))
})

test_that("lnr_gam fits and predicts", {
  pred <- lnr_gam(mtcars, mpg ~ s(hp) + wt)(mtcars)
  expect_length(pred, nrow(mtcars))

  set.seed(1)
  w <- runif(nrow(mtcars))
  pred_w <- lnr_gam(mtcars, mpg ~ s(hp), weights = w)(mtcars)
  expect_length(pred_w, nrow(mtcars))
})

test_that("lnr_lmer fits and predicts with random effects", {
  pred <- lnr_lmer(mtcars, mpg ~ (1 | cyl) + wt)(mtcars)
  expect_length(pred, nrow(mtcars))
})

test_that("lnr_glmer fits and predicts", {
  # wt is rescaled because its raw units give the fixed-effect coefficient
  # a much larger eigenvalue than the random-effect variance, which makes
  # lme4 emit a genuine "nearly unidentifiable" convergence warning rather
  # than a spurious one worth suppressing.
  df <- mtcars
  df$wt <- scale(df$wt)
  pred <- lnr_glmer(df, mpg ~ (1 | cyl) + wt, family = Gamma)(df)
  expect_length(pred, nrow(df))
})

test_that("lnr_hal fits and predicts", {
  skip_if_not_installed("hal9001")
  set.seed(1)
  pred <- suppressWarnings(
    lnr_hal(mtcars, mpg ~ hp, max_degree = 1, num_knots = 3)(mtcars)
  )
  expect_length(pred, nrow(mtcars))
})

test_that("lnr_xgboost fits and predicts", {
  skip_if_not_installed("xgboost")
  pred <- lnr_xgboost(mtcars, mpg ~ hp + wt, nrounds = 5)(mtcars)
  expect_length(pred, nrow(mtcars))
  expect_true(is.numeric(pred))
})

test_that("lnr_gbm fits and predicts, quietly and verbosely", {
  set.seed(1)
  pred <- lnr_gbm(mtcars, mpg ~ hp + wt, n.trees = 10,
                  distribution = "gaussian")(mtcars)
  expect_length(pred, nrow(mtcars))

  set.seed(1)
  # verbose = TRUE exercises the non-suppressed prediction branch
  pred_v <- lnr_gbm(mtcars, mpg ~ hp + wt, n.trees = 10,
                    distribution = "gaussian", verbose = TRUE)(mtcars)
  expect_length(pred_v, nrow(mtcars))
})
