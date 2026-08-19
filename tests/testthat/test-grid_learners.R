test_that("lnr_glmnet_grid returns a named multi-predictor and requires an explicit grid", {
  multi_predictor <- lnr_glmnet_grid(
    mtcars, mpg ~ hp + disp + am + wt, lambda = c(0.01, 0.1, 0.5, 1))

  expect_s3_class(multi_predictor, 'nadir_multi_predictor')
  expect_length(multi_predictor, 4)
  expect_setequal(
    names(multi_predictor),
    c('lambda_1', 'lambda_0.5', 'lambda_0.1', 'lambda_0.01'))

  predictions <- multi_predictor[['lambda_0.5']](mtcars)
  expect_length(predictions, nrow(mtcars))
  expect_true(is.numeric(predictions))

  # newdata without the outcome column
  predictions_no_y <- multi_predictor[['lambda_0.5']](
    mtcars[, c('hp', 'disp', 'am', 'wt')])
  expect_equal(unname(predictions), unname(predictions_no_y))

  # different lambda values yield genuinely different sub-models
  expect_false(isTRUE(all.equal(
    multi_predictor[['lambda_0.01']](mtcars),
    multi_predictor[['lambda_1']](mtcars))))

  # an explicit lambda grid is required, since auto-generated grids would
  # differ across cross-validation folds
  expect_error(lnr_glmnet_grid(mtcars, mpg ~ hp), 'explicit numeric grid')
  expect_error(lnr_glmnet_grid(mtcars, mpg ~ hp, lambda = NULL),
               'explicit numeric grid')
})

test_that("super_learner expands grid learners into weighted pseudo-learners", {
  set.seed(1)
  lambda_grid <- exp(seq(log(2), log(0.001), length.out = 10))

  sl_model <- super_learner(
    data = mtcars,
    formulas = mpg ~ hp + disp + am + wt,
    learners = list(mean = lnr_mean, lm = lnr_lm,
                    glmnet_grid = lnr_glmnet_grid),
    extra_learner_args = list(NULL, NULL, list(lambda = lambda_grid)))

  # 2 ordinary learners + 10 lambda pseudo-learners
  expect_length(sl_model$learner_weights, 12)
  expect_equal(sum(sl_model$learner_weights), 1, tolerance = 1e-6)
  expect_equal(
    sum(grepl('^glmnet_grid_lambda_', names(sl_model$learner_weights))), 10)

  # each pseudo-learner contributes its own holdout prediction column
  expect_equal(
    sum(grepl('^glmnet_grid_lambda_', colnames(sl_model$holdout_predictions))),
    10)

  predictions <- sl_model$predict(mtcars)
  expect_length(predictions, nrow(mtcars))
  expect_true(is.numeric(predictions))

  # compare_learners sees each lambda as a distinct learner
  learner_comparison <- suppressMessages(compare_learners(sl_model))
  expect_equal(
    sum(grepl('^glmnet_grid_lambda_', colnames(learner_comparison))), 10)
})

test_that("an erring grid learner is dropped wholesale without harming others", {
  lnr_bad_grid <- function(data, formula, weights = NULL, ...) {
    stop('deliberate failure')
  }
  attr(lnr_bad_grid, 'sl_lnr_name') <- 'bad_grid'
  attr(lnr_bad_grid, 'sl_lnr_type') <- 'continuous'

  sl_model <- suppressWarnings(super_learner(
    data = mtcars,
    formulas = mpg ~ hp + wt,
    learners = list(mean = lnr_mean, bad = lnr_bad_grid,
                    glmnet_grid = lnr_glmnet_grid),
    extra_learner_args = list(NULL, NULL, list(lambda = c(0.1, 0.5)))))

  expect_false('bad' %in% names(sl_model$learner_weights))
  expect_true('bad' %in% sl_model$erring_learners)
  expect_equal(
    sum(grepl('^glmnet_grid_lambda_', names(sl_model$learner_weights))), 2)
  expect_length(sl_model$predict(mtcars), nrow(mtcars))
})

test_that("discrete super_learner can select a single lambda pseudo-learner", {
  set.seed(2)
  sl_model <- super_learner(
    data = mtcars,
    formulas = mpg ~ hp + disp + am + wt,
    learners = list(mean = lnr_mean, glmnet_grid = lnr_glmnet_grid),
    extra_learner_args = list(
      NULL, list(lambda = exp(seq(log(2), log(0.001), length.out = 10)))),
    ensemble_or_discrete = 'discrete')

  expect_equal(sum(sl_model$learner_weights == 1), 1)
  expect_equal(sum(sl_model$learner_weights), 1)
  # names must be preserved on the one-hot weights so prediction can key by name
  expect_false(is.null(names(sl_model$learner_weights)))
  expect_length(sl_model$predict(mtcars), nrow(mtcars))
})

test_that("grid learners work with binary outcomes", {
  set.seed(3)
  sl_model <- suppressWarnings(super_learner(
    data = mtcars,
    formulas = am ~ hp + wt,
    learners = list(logistic = lnr_logistic, glmnet_grid = lnr_glmnet_grid),
    extra_learner_args = list(NULL, list(lambda = c(0.01, 0.05, 0.1))),
    outcome_type = 'binary'))

  predictions <- sl_model$predict(mtcars)
  expect_true(all(predictions >= -1e-8 & predictions <= 1 + 1e-8))
})

test_that("lnr_hal_grid returns a multi-predictor usable in super_learner", {
  skip_if_not_installed('hal9001')
  set.seed(4)
  suppressWarnings({
    multi_predictor <- lnr_hal_grid(
      mtcars, mpg ~ hp + wt, lambda = c(0.01, 0.1, 1),
      max_degree = 1, num_knots = 3)
  })
  expect_s3_class(multi_predictor, 'nadir_multi_predictor')
  expect_length(multi_predictor, 3)
  predictions <- multi_predictor[['lambda_0.1']](mtcars)
  expect_length(predictions, nrow(mtcars))

  suppressWarnings({
    sl_model <- super_learner(
      data = mtcars,
      formulas = mpg ~ hp + wt,
      learners = list(mean = lnr_mean, hal_grid = lnr_hal_grid),
      extra_learner_args = list(
        NULL, list(lambda = c(0.01, 0.1, 1), max_degree = 1, num_knots = 3)))
  })
  expect_equal(
    sum(grepl('^hal_grid_lambda_', names(sl_model$learner_weights))), 3)
  expect_length(sl_model$predict(mtcars), nrow(mtcars))
})

test_that("expand_multi_predictor_fits detects misaligned sub-model names", {
  trained_learners <- tibble::tibble(
    .sl_fold = c(1, 2),
    learner_name = c('grid', 'grid'),
    learned_predictor = list(
      as_multi_predictor(list(a = function(newdata) 1)),
      as_multi_predictor(list(b = function(newdata) 1))))

  expect_error(
    expand_multi_predictor_fits(trained_learners),
    'differently named sub-models')
})

test_that("as_multi_predictor validates its input", {
  expect_error(as_multi_predictor(list()), 'nonempty list of functions')
  expect_error(as_multi_predictor(list(function(x) x)), 'unique, nonempty names')
  expect_error(
    as_multi_predictor(list(a = function(x) x, a = function(x) x)),
    'unique, nonempty names')
})
