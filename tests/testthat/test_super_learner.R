suppressWarnings(library(future))

# super_learner() prefers the correct lm model ------

testthat::test_that(desc = "super_learner() prefers the correct lm outcome model",
{
  # we want to test that super_learner() picks out the right model.

  # we generate some fake data
  set.seed(1234)
  sample_size <- 1000

  # here we generate data with a quadratic term and fit an
  # intercept only term (lnr_mean), a linear model, and a model
  # with the right quadratic term, and we expect
  # super_learner() to pick the right one to weight highly.

  fake_data <- data.frame(
    x1 = rnorm(n = 1000),
    x2 = rnorm(n = 1000))
  fake_data$y <- fake_data$x1 + fake_data$x2^2 + rnorm(n = 1000)

  # train super_learner() on the fake data
  learned_predictor <- super_learner(
    data = fake_data,
    formula = list(
      .default = y ~ x1 + x2,
      lm2 = y ~ x1 + poly(x2, 2)), # pass the quadratic term to lm2
    learners = list(
      mean = lnr_mean,
      lm1 = lnr_lm,
      lm2 = lnr_lm
    )
  )

  # expect the correctly specified model to get all the weight
  testthat::expect_gte(learned_predictor$learner_weights['lm2'], .9)

})


testthat::test_that(desc = "super_learner() prefers the correct binary outcome model",
{
  # we want to test that super_learner() picks out the right model.

  # we generate some fake data
  set.seed(1234)
  sample_size <- 1000

  # here we generate data with a quadratic term and fit an
  # intercept only term (lnr_mean), a linear model, and a model
  # with the right quadratic term, and we expect
  # super_learner() to pick the right one to weight highly.

  fake_data <- data.frame(
    x1 = rnorm(n = 1000),
    x2 = rnorm(n = 1000))
  fake_data$y <- rbinom(
    n = 1000,
    size = 1,
    prob = plogis(fake_data$x1 + fake_data$x2^2 + rnorm(n = 1000)))

  # train super_learner() on the fake data
  learned_predictor <- super_learner(
    data = fake_data,
    formula = list(
      .default = y ~ x1 + x2,
      logistic2 = y ~ x1 + poly(x2, 2), # pass the quadratic term to logistic2
      logistic3 = y ~ x1),
    learners = list(
      mean = lnr_mean,
      logistic1 = lnr_logistic,
      logistic2 = lnr_logistic,
      logistic3 = lnr_logistic,
      rf = lnr_rf_binary
    ),
    outcome_type = 'binary'
  )

  # expect the correctly specified model to get all the weight
  testthat::expect_gte(learned_predictor$learner_weights['logistic2'], .9)

})



testthat::test_that(desc = "super_learner() prefers the correct lm outcome model",
{
  # we want to test that super_learner() picks out the right model.

  # we generate some fake data
  set.seed(1234)
  sample_size <- 1000

  # here we generate data with a quadratic term and fit an
  # intercept only term (lnr_mean), a linear model, and a model
  # with the right quadratic term, and we expect
  # super_learner() to pick the right one to weight highly.

  fake_data <- data.frame(
    x1 = rnorm(n = 1000),
    x2 = rnorm(n = 1000))
  fake_data$y <- fake_data$x1 + fake_data$x2^2 + rnorm(n = 1000)

  # train super_learner() on the fake data
  learned_predictor <- super_learner(
    data = fake_data,
    formula = list(
      .default = y ~ x1 + x2,
      lm2 = y ~ x1 + poly(x2, 2)), # pass the quadratic term to lm2
    learners = list(
      mean = lnr_mean,
      lm1 = lnr_lm,
      lm2 = lnr_lm
    )
  )

  # expect the correctly specified model to get all the weight
  testthat::expect_gte(learned_predictor$learner_weights['lm2'], .9)

})



# super_learner() prefers the correct lm density model -----

testthat::test_that(desc = "super_learner() prefers the correct lm density model",
{
# we want to test that super_learner() picks out the right model.

# we generate some fake data
set.seed(1234)
sample_size <- 1000

# here we generate data with a quadratic term and fit an
# intercept only term (lnr_mean), a linear model, and a model
# with the right quadratic term, and we expect
# super_learner() to pick the right one to weight highly.

fake_data <- data.frame(
  x1 = rnorm(n = 1000),
  x2 = rnorm(n = 1000))
fake_data$y <- fake_data$x1 + fake_data$x2^2 + rnorm(n = 1000)

# train super_learner() on the fake data
learned_predictor <- super_learner(
  data = fake_data,
  formula = list(
    .default = y ~ x1 + x2,
    lm2 = y ~ x1 + poly(x2, 2)), # pass the quadratic term to lm2
  learners = list(
    lm = lnr_lm_density,
    lm2 = lnr_lm_density
  ),
  outcome_type = 'density'
)

# expect the correctly specified model to get all the weight
testthat::expect_gte(learned_predictor$learner_weights['lm2'], .9)
})

# super_learner() outperforms naive lm ----------

testthat::test_that(desc = "verify that super_learner() really does outperform a simple linear model most of the time",
{
  # suppose you don't trust that the cross-validation system is working at all in {nadir}

  # then you might say, let me really hold out some data and do the evaluation myself.

  # this test is in that spirit.
  # example dataset
  data("Boston", package = "MASS")
  df <- Boston

  n_repetitions <- 3L
  results <- numeric(length = n_repetitions)

  for (i in 1:n_repetitions) {
    holdout_ids <- sample.int(n = nrow(df), size = 25)
    holdouts <- df[holdout_ids,]
    training <- df[-holdout_ids,]

    learned_predictor <- super_learner(
      data = training,
      formula = list(
        .default = medv ~ .,
        gam = medv ~ s(ptratio) + crim + zn + indus + s(nox) + rm + age + dis,
        lm2 = medv ~ age:zn + poly(nox, 2) + .),
      learners = list(
        mean = lnr_mean,
        lm = lnr_lm,
        lm2 = lnr_lm,
        gam = lnr_gam,
        earth = lnr_earth,
        rf = lnr_rf,
        xgboost = lnr_xgboost,
        glmnet = lnr_glmnet)
      )

    # now i would be truly astonished if we could not beat a simple lm model...
    simple_lm_model <- lm(medv ~ ., data = training)

    simple_lm_model_predictions <- predict(simple_lm_model, holdouts)
    super_learner_model_predictions <- learned_predictor$predict(holdouts)

    lm_heldout_mse <- nadir:::mse(holdouts$medv, simple_lm_model_predictions)
    sl_heldout_mse <- nadir:::mse(holdouts$medv, super_learner_model_predictions)

    # subtract the loss (mse) from the loss (mse) of the linear model on the held out data
    results[i] <- lm_heldout_mse - sl_heldout_mse
  }

  # if super_learner() is working well, we should be able to easily beat a
  # simple linear model in prediction performance.
  #
  # we take "beating a simple linear model" to mean that the heldout mse from the
  # lm should be > the heldout mse from the super learner, so in our repeated experiment
  # with recorded, we expect that at least half the time super_learner() outperforms
  # the simple lm model.
  testthat::expect_gte(mean(results), 0)
  testthat::expect_gte(mean(sign(results)), 0)

})


test_that(desc = "super_learner() contains at least
          predict(), holdout_predictions, y_variable, outcome_type, and learner_weights", {

learners <- list(
   glm = lnr_glm,
   rf = lnr_rf,
   glmnet = lnr_glmnet,
   lmer = lnr_lmer
)

# mtcars example ---
formulas <- c(
.default = mpg ~ cyl + hp, # first three models use same formula
lmer = mpg ~ (1 | cyl) + hp # lme4 uses different language features
)

# fit a super_learner
sl_model <- super_learner(
data = mtcars,
formula = formulas,
learners = learners)

expect_true('predict' %in% names(sl_model))
expect_true(is.function(sl_model$predict))
expect_true('holdout_predictions' %in% names(sl_model))
expect_true(is.data.frame(sl_model$holdout_predictions))
expect_true(sl_model$outcome_type %in% nadir_supported_types)
expect_true('learner_weights' %in% names(sl_model))
expect_true(is.numeric(sl_model$learner_weights))
expect_true(sum(sl_model$learner_weights) == 1L)
expect_true('y_variable' %in% names(sl_model))
expect_true('outcome_type' %in% names(sl_model))
})

test_that(desc = "super_learner() can use a character formula like 'y ~ x'", {

  learners <- list(
    glm = lnr_glm,
    glmnet = lnr_glmnet)

  formula <- 'hp ~ mpg'

  testthat::expect_no_error(
    sl_fit <- nadir::super_learner(
      data = mtcars, formula = formula, learners = learners)
  )
})


test_that(desc = "super_learner() doesn't need y to appear in predict(newdata)", {

  sl_fit <- nadir::super_learner(
    data = mtcars,
    formula = hp ~ mpg,
    learners = list(
      lnr_earth, lnr_gam, lnr_gbm, lnr_glm, lnr_glmnet, lnr_lm, lnr_mean,
      lnr_ranger, lnr_rf, lnr_xgboost))

  newdata <- mtcars
  newdata$hp <- NULL

  expect_no_error(sl_fit$predict(newdata))
})




fast_learners <- list(lm = lnr_lm, mean = lnr_mean)

test_that("super_learner fits a continuous ensemble and predicts", {
  set.seed(1)
  sl <- super_learner(
    data = mtcars,
    learners = fast_learners,
    formulas = mpg ~ hp + wt,
    n_folds = 2
  )
  expect_s3_class(sl, "nadir_sl_model")
  expect_equal(sl$y_variable, "mpg")
  expect_equal(sum(sl$learner_weights), 1, tolerance = 1e-8)
  expect_length(sl$predict(mtcars), nrow(mtcars))

  # the predict S3 method dispatches
  expect_equal(predict(sl, newdata = mtcars), sl$predict(mtcars))

  # calling predict with no newdata falls back to the training data
  expect_length(sl$predict(), nrow(mtcars))

  # newdata missing the outcome column gets an NA column added
  expect_length(sl$predict(mtcars[, c("hp", "wt")]), nrow(mtcars))
})

test_that("super_learner validates its inputs", {
  df_na <- mtcars
  df_na$mpg[1] <- NA
  expect_error(
    super_learner(df_na, fast_learners, mpg ~ hp, n_folds = 2),
    "does not have any missing data imputation"
  )

  expect_error(
    super_learner(mtcars, learners = lnr_lm, formulas = mpg ~ hp),
    "must be a list of learner functions"
  )

  expect_error(
    super_learner(mtcars, fast_learners, mpg ~ hp, outcome_type = "zzz")
  )
})

test_that("super_learner can filter to complete cases with a message", {
  df_na <- mtcars
  df_na$mpg[1] <- NA
  set.seed(1)
  expect_message(
    sl <- super_learner(df_na, fast_learners, mpg ~ hp, n_folds = 2,
                        use_complete_cases = TRUE),
    "use_complete_cases = TRUE will filter"
  )
  expect_length(sl$predict(mtcars), nrow(mtcars))
})

test_that("discrete super_learner picks a single learner, warning on ties", {
  set.seed(1)
  # deterministic tie via a custom weight function
  expect_warning(
    sl <- super_learner(
      mtcars, fast_learners, mpg ~ hp, n_folds = 2,
      determine_super_learner_weights = function(data, y_variable, obs_weights = NULL) c(0.5, 0.5),
      ensemble_or_discrete = "discrete"
    ),
    "tied for the maximum weight"
  )
  # learner_weights retain their names in the discrete branch (as in the
  # ensemble branch), since prediction is keyed by learner name
  expect_equal(unname(sort(sl$learner_weights)), c(0, 1))
  expect_false(is.null(names(sl$learner_weights)))

  # no tie
  set.seed(1)
  sl2 <- super_learner(
    mtcars, fast_learners, mpg ~ hp, n_folds = 2,
    determine_super_learner_weights = function(data, y_variable, obs_weights = NULL) c(0.3, 0.7),
    ensemble_or_discrete = "discrete"
  )
  expect_equal(sort(unname(sl2$learner_weights)), c(0, 1))

  # invalid option errors
  set.seed(1)
  expect_error(
    super_learner(mtcars, fast_learners, mpg ~ hp, n_folds = 2,
                  ensemble_or_discrete = "zzz")
  )
})

test_that("super_learner warns on NA weights and uses valid weights", {
  set.seed(1)
  w_na <- c(NA, rep(1, nrow(mtcars) - 1))
  expect_warning(
    super_learner(mtcars, fast_learners, mpg ~ hp, n_folds = 2, weights = w_na),
    "cannot be any NA weights"
  )

  set.seed(1)
  w <- runif(nrow(mtcars))
  sl <- super_learner(mtcars, fast_learners, mpg ~ hp, n_folds = 2, weights = w)
  expect_s3_class(sl, "nadir_sl_model")
  expect_length(sl$predict(mtcars), nrow(mtcars))
})

test_that("super_learner supports binary outcomes with outcome-type-dependent args", {
  set.seed(1)
  sl <- super_learner(
    data = mtcars,
    learners = list(glm = lnr_glm, mean = lnr_mean),
    formulas = am ~ hp,
    n_folds = 2,
    outcome_type = "binary"
  )
  pred <- sl$predict(mtcars)
  expect_true(all(pred >= 0 & pred <= 1))

  # if the family arg is already given, the outcome-dependent arg is skipped
  set.seed(1)
  sl2 <- super_learner(
    data = mtcars,
    learners = list(glm = lnr_glm, mean = lnr_mean),
    formulas = am ~ hp,
    n_folds = 2,
    outcome_type = "binary",
    extra_learner_args = list(glm = list(family = binomial(link = "logit")))
  )
  expect_s3_class(sl2, "nadir_sl_model")
})

test_that("super_learner supports density outcomes", {
  set.seed(1)
  sl <- suppressWarnings(super_learner(
    data = mtcars,
    learners = list(lm_dens = lnr_lm_density, hd = lnr_homoskedastic_density),
    formulas = mpg ~ hp,
    n_folds = 2,
    outcome_type = "density",
    extra_learner_args = list(hd = list(mean_lnr = lnr_lm))
  ))
  expect_equal(sum(sl$learner_weights), 1, tolerance = 1e-6)
  expect_true(all(sl$predict(mtcars) >= 0))
})

test_that("super_learner supports multiclass outcomes", {
  set.seed(1)
  df <- iris
  sl <- super_learner(
    data = df,
    learners = list(m1 = lnr_multinomial_nnet, m2 = lnr_multinomial_nnet),
    formulas = list(m1 = Species ~ Petal.Length, m2 = Species ~ Sepal.Length),
    n_folds = 2,
    outcome_type = "multiclass"
  )
  expect_equal(sum(sl$learner_weights), 1, tolerance = 1e-6)
})

test_that("super_learner builds an origami cv_schema when cluster or strata ids are given", {
  set.seed(1)
  sl_cl <- super_learner(
    mtcars, fast_learners, mpg ~ hp, n_folds = 2,
    cluster_ids = rep(1:8, each = 4)
  )
  expect_s3_class(sl_cl, "nadir_sl_model")

  set.seed(1)
  sl_st <- super_learner(
    mtcars, fast_learners, mpg ~ hp, n_folds = 2,
    strata_ids = rep(c(1, 2), 16)
  )
  expect_s3_class(sl_st, "nadir_sl_model")
})

test_that("super_learner records learner training errors and drops erring learners", {
  lnr_always_fails <- function(data, formula, ...) {
    stop("this learner always fails")
  }
  attr(lnr_always_fails, "sl_lnr_type") <- "continuous"
  attr(lnr_always_fails, "sl_lnr_name") <- "always_fails"

  set.seed(1)
  sl <- super_learner(
    mtcars,
    learners = list(lm = lnr_lm, mean = lnr_mean, bad = lnr_always_fails),
    formulas = mpg ~ hp,
    n_folds = 2
  )
  expect_true("errors_from_training_cv_stage1" %in% names(sl))
  expect_true("erring_learners" %in% names(sl))
  expect_true("bad" %in% sl$erring_learners)
  # the erring learner is excluded from the weights
  expect_false("bad" %in% names(sl$learner_weights))
  # predictions still work from surviving learners
  expect_length(sl$predict(mtcars), nrow(mtcars))
})

test_that("super_learner records prediction-stage errors", {
  lnr_bad_predictor <- function(data, formula, ...) {
    function(newdata) stop("prediction fails")
  }
  attr(lnr_bad_predictor, "sl_lnr_type") <- "continuous"
  attr(lnr_bad_predictor, "sl_lnr_name") <- "bad_predictor"

  set.seed(1)
  sl <- super_learner(
    mtcars,
    learners = list(lm = lnr_lm, mean = lnr_mean, badpred = lnr_bad_predictor),
    formulas = mpg ~ hp,
    n_folds = 2
  )
  expect_true("errors_from_predicting_cv_stage2" %in% names(sl))
  expect_true("badpred" %in% sl$erring_learners)
})

test_that("super_learner records errors from the final full-data fit", {
  # a learner that succeeds on CV training folds but fails on the full data
  lnr_fails_on_full_data <- function(data, formula, ...) {
    if (nrow(data) == nrow(mtcars)) stop("fails on the full dataset")
    lnr_lm(data, formula)
  }
  attr(lnr_fails_on_full_data, "sl_lnr_type") <- "continuous"
  attr(lnr_fails_on_full_data, "sl_lnr_name") <- "fails_full"

  set.seed(1)
  sl <- super_learner(
    mtcars,
    learners = list(lm = lnr_lm, flaky = lnr_fails_on_full_data),
    formulas = mpg ~ hp,
    n_folds = 2
  )
  expect_true("errors_from_training_on_entire_data" %in% names(sl))
})
