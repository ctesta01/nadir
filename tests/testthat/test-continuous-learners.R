
# continuous learners on mtcars -------------------------------------------

test_that(desc = "all continuous learners can be trained and predict on mtcars",
{
  # get all the known continuous learners
  known_continuous_learners <- list_known_learners(type = 'continuous')
  # handle lme4 separately because it demands that we actually use random effects
  known_continuous_learners <- setdiff(known_continuous_learners, c("lnr_lmer", 'lnr_glmer'))

  # get the learner functions from their names (i.e., "lnr_glm" -> lnr_glm)
  known_continuous_learners <- lapply(known_continuous_learners,
                                      \(lnr_name) {
                                        get(lnr_name, envir = environment(nadir::super_learner))
                                      })

  # train each of the learners on mtcars, mpg ~ hp + cyl
  trained_learners <- lapply(
    known_continuous_learners,
    \(learner) { learner(data = mtcars, formula = mpg ~ hp + cyl) })

  # make predictions from the trained learners on the mtcars dataset
  learner_predictions <- lapply(
    trained_learners,
    \(trained_learner) { trained_learner(mtcars) })

  # the predictions should all be numeric
  expect_true(all(sapply(learner_predictions, is.numeric)))

  learned_lme4 <- lnr_lmer(
    data = mtcars, formula = mpg ~ (1 | cyl) + hp)

  lme4_predictions <- learned_lme4(mtcars)

  expect_true(is.numeric(lme4_predictions))

})

test_that(desc = "all binary learners can be trained and predict on mtcars",
{
  # get all the known continuous learners
  known_binary_learners <- list_known_learners(type = 'binary')
  # handle lme4 separately because it demands that we actually use random effects
  known_binary_learners <- setdiff(known_binary_learners, c("lnr_lmer", 'lnr_glmer'))

  # get the learner functions from their names (i.e., "lnr_glm" -> lnr_glm)
  known_binary_learners <- lapply(known_binary_learners,
                                      \(lnr_name) {
                                        get(lnr_name, envir = environment(nadir::super_learner))
                                      })

  # train each of the learners on mtcars, am ~ hp + cyl
  trained_learners <- lapply(
    known_binary_learners,
    \(learner) { learner(data = mtcars, formula = am ~ hp + cyl + mpg) })

  # make predictions from the trained learners on the mtcars dataset
  learner_predictions <- lapply(
    trained_learners,
    \(trained_learner) { trained_learner(mtcars) })

  # the predictions should all be numeric
  expect_true(all(sapply(learner_predictions, is.numeric)))

  # learn a glmer model separately
  learned_glmer <- lnr_glmer(
    data = mtcars, formula = am ~ (1 | cyl) + hp, family = binomial)

  # apply the same test; glmer should produce numeric predictions
  expect_true(is.numeric(learned_glmer(mtcars)))

})

