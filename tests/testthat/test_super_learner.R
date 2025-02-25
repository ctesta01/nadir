testthat::test_that(desc = "super_learner() prefers the correct model",
{
  # we want to test that super_learner() picks out the right model.

  # here we generate data with a quadratic term and fit an
  # intercept only term (lnr_mean), a linear model, and a model
  # with the right quadratic term, and we expect
  # super_learner() to pick the right one to weight highly.

  # we generate some fake data
  set.seed(1234)
  sample_size <- 1000

  fake_data <- data.frame(
    x1 = rnorm(n = 1000),
    x2 = rnorm(n = 1000))
  fake_data$y <- fake_data$x1 + fake_data$x2^2 + rnorm(n = 1000)

  # train super_learner() on the fake data
  learned_predictor <- super_learner(
    data = fake_data,
    regression_formula = list(
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


# testthat::test_that(desc = "verify that super_learner() really does outperform at least some models",
# {
#   # suppose you don't trust that the cross-validation system is working at all in {nadir}
#
#   # then you might say, let me really hold out some data and do the evaluation
#   # myself.
#
#   # this test is in that spirit.
#
#   # example dataset
#   df <- palmerpenguins::penguins
#   df <- df[complete.cases(df),]
#
#   n_repetitions <- 25L
#   results <- numeric(length = n_repetitions)
#
#   for (i in 1:n_repetitions) {
#     cat('.')
#     holdout_ids <- sample.int(n = nrow(df), size = 25)
#     holdouts <- df[holdout_ids,]
#     training <- df[-holdout_ids,]
#
#     learned_predictor <- super_learner(
#       data = training,
#       regression_formula = list(
#         .default = flipper_length_mm ~ .,
#         gam = flipper_length_mm ~ s(body_mass_g, by = sex) + s(body_mass_g, by = species) + bill_depth_mm,
#         lm2 = flipper_length_mm ~ body_mass_g:sex + poly(body_mass_g, 2) + .,
#         lm3 = flipper_length_mm ~ interaction(sex, species) + poly(bill_length_mm, 2) + poly(body_mass_g, 2) + .),
#       learners = list(
#         mean = lnr_mean,
#         lm = lnr_lm,
#         lm2 = lnr_lm,
#         lm3 = lnr_lm,
#         gam = lnr_gam,
#         rf = lnr_randomForest)
#       )
#
#     # now i would be truly astonished if we could not beat a simple lm model...
#     simple_lm_model <- lm(flipper_length_mm ~ ., data = training)
#
#     simple_lm_model_predictions <- predict(simple_lm_model, holdouts)
#     super_learner_model_predictions <- learned_predictor(holdouts)
#
#     lm_heldout_mse <- nadir:::mse(holdouts$flipper_length_mm - simple_lm_model_predictions)
#     sl_heldout_mse <- nadir:::mse(holdouts$flipper_length_mm - super_learner_model_predictions)
#
#     # results[i] <- sl_heldout_mse < lm_heldout_mse
#     results[i] <- lm_heldout_mse - sl_heldout_mse
#   }
#
#   testthat::expect_gte(mean(results), 0)
#   testthat::expect_gte(mean(sign(results)), 0)
# })
#
