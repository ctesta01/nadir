# Tests for binary_learners.R, density_learners.R, and multiclass_learners.R

# ---- binary learners -------------------------------------------------------

test_that("lnr_nnet fits binary outcomes with default and explicit size", {
  set.seed(1)
  # explicit size
  pred <- lnr_nnet(mtcars, am ~ hp, size = 2)(mtcars)
  expect_equal(nrow(pred), nrow(mtcars))

  # missing size falls back to round(sqrt(nrow(data)))
  set.seed(1)
  pred_default <- lnr_nnet(mtcars, am ~ hp)(mtcars)
  expect_equal(nrow(pred_default), nrow(mtcars))
})

test_that("lnr_nnet warns when used on multiclass outcomes", {
  set.seed(1)
  expect_warning(
    lnr_nnet(iris, Species ~ ., size = 2)(iris),
    "supposed to be used for binary outcomes"
  )
})

test_that("lnr_ranger_binary predicts probabilities of the outcome being 1", {
  set.seed(1)
  pred <- lnr_ranger_binary(mtcars, am ~ hp, num.trees = 20)(mtcars)
  expect_length(pred, nrow(mtcars))
  expect_true(all(pred >= 0 & pred <= 1))
})

test_that("lnr_rf_binary casts numeric outcomes to factor and predicts probabilities", {
  set.seed(1)
  # numeric 0/1 outcome exercises the as.factor branch
  pred <- lnr_rf_binary(mtcars, am ~ hp, ntree = 20)(mtcars)
  expect_length(pred, nrow(mtcars))
  expect_true(all(pred >= 0 & pred <= 1))

  # already-factor outcome skips the cast
  df <- mtcars
  df$am <- as.factor(df$am)
  set.seed(1)
  pred2 <- lnr_rf_binary(df, am ~ hp, ntree = 20)(df)
  expect_length(pred2, nrow(df))
})

test_that("lnr_logistic predicts probabilities via glm with logit link", {
  pred <- lnr_logistic(mtcars, am ~ hp)(mtcars)
  expect_length(pred, nrow(mtcars))
  expect_true(all(pred >= 0 & pred <= 1))
})

# ---- density learners ------------------------------------------------------

test_that("lnr_lm_density produces conditional normal densities", {
  pred <- lnr_lm_density(mtcars, mpg ~ hp)(mtcars)
  expect_length(pred, nrow(mtcars))
  expect_true(all(pred >= 0))

  # weights branch
  set.seed(1)
  w <- runif(nrow(mtcars))
  pred_w <- lnr_lm_density(mtcars, mpg ~ hp, weights = w)(mtcars)
  expect_length(pred_w, nrow(mtcars))
})

test_that("lnr_glm_density produces conditional normal densities", {
  pred <- lnr_glm_density(mtcars, hp ~ mpg,
                          family = poisson(link = "identity"))(mtcars)
  expect_length(pred, nrow(mtcars))
  expect_true(all(pred >= 0))

  set.seed(1)
  w <- runif(nrow(mtcars))
  pred_w <- lnr_glm_density(mtcars, mpg ~ hp, weights = w)(mtcars)
  expect_length(pred_w, nrow(mtcars))
})

test_that("lnr_homoskedastic_density works with a mean learner and extra args", {
  pred <- lnr_homoskedastic_density(mtcars, mpg ~ hp, mean_lnr = lnr_lm)(mtcars)
  expect_length(pred, nrow(mtcars))
  expect_true(all(pred >= 0))

  # with mean_lnr_args, density_args, and weights
  set.seed(1)
  w <- runif(nrow(mtcars))
  pred2 <- lnr_homoskedastic_density(
    mtcars, mpg ~ hp,
    mean_lnr = lnr_rf,
    mean_lnr_args = list(ntree = 20),
    density_args = list(bw = 1),
    weights = w
  )(mtcars)
  expect_length(pred2, nrow(mtcars))
})

test_that("lnr_heteroskedastic_density predicts densities with modeled variance", {
  set.seed(1)
  fit <- lnr_heteroskedastic_density(
    mtcars, mpg ~ hp,
    mean_lnr = lnr_rf,
    var_lnr = lnr_lm,
    mean_lnr_args = list(ntree = 20),
    density_args = list(bw = 1)
  )
  pred <- fit(mtcars)
  expect_length(pred, nrow(mtcars))
  expect_true(all(pred >= 0))

  # prediction grid outside the training range exercises the variance floors
  grid <- expand.grid(
    mpg = seq(min(mtcars$mpg), max(mtcars$mpg), length.out = 10),
    hp = seq(0, max(mtcars$hp) * 2, length.out = 10)
  )
  pred_grid <- fit(grid)
  expect_length(pred_grid, nrow(grid))
  expect_true(all(is.finite(pred_grid)))
})

# ---- multiclass learners ---------------------------------------------------

test_that("lnr_multinomial_vglm predicts density at the observed class", {
  df <- mtcars
  df$cyl <- as.factor(df$cyl)
  # cyl is quasi-separated by hp + mpg, so VGAM emits many numerical
  # 'fitted probabilities 0 or 1' warnings; they are irrelevant to what
  # this test asserts about nadir's wrapper, so suppress them.
  pred <- suppressWarnings(lnr_multinomial_vglm(df, cyl ~ hp + mpg)(df))
  expect_length(pred, nrow(df))
  expect_true(all(pred >= 0 & pred <= 1))
})

test_that("lnr_multinomial_nnet predicts density at the observed class", {
  df <- mtcars
  df$cyl <- as.factor(df$cyl)
  pred <- lnr_multinomial_nnet(df, cyl ~ hp + mpg)(df)
  expect_length(pred, nrow(df))
  expect_true(all(pred >= 0 & pred <= 1))
})
