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
    ),
    verbose = TRUE
  )

  # expect the correctly specified model to get all the weight
  testthat::expect_gte(learned_predictor$learner_weights['lm2'], .9)

})


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
  determine_super_learner_weights = determine_weights_using_neg_log_loss,
  verbose = TRUE
)

# expect the correctly specified model to get all the weight
testthat::expect_gte(learned_predictor$learner_weights['lm2'], .9)
})


testthat::test_that(desc = "verify that super_learner() really does outperform a simple linear model most of the time",
{
  # suppose you don't trust that the cross-validation system is working at all in {nadir}

  # then you might say, let me really hold out some data and do the evaluation myself.

  # this test is in that spirit.
  # example dataset
  data("Boston", package = "MASS")
  df <- Boston

  n_repetitions <- 5L
  results <- numeric(length = n_repetitions)

  for (i in 1:n_repetitions) {
    cat('.')
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
        glmnet = lnr_glmnet),
      verbose = TRUE
      )

    # now i would be truly astonished if we could not beat a simple lm model...
    simple_lm_model <- lm(medv ~ ., data = training)

    simple_lm_model_predictions <- predict(simple_lm_model, holdouts)
    super_learner_model_predictions <- learned_predictor$sl_predictor(holdouts)

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

