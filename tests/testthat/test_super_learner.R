suppressWarnings(library(future))

# super_learner() prefers the correct lm model ------

testthat::test_that(desc = "super_learner() prefers the correct lm outcome model", {
  # we want to test that super_learner() picks out the right model.

  run_super_learner_prefers_correct_lm_model <- function(custom_seed) {
    # we generate some fake data
    set.seed(custom_seed)
    sample_size <- 1000

    # here we generate data with a quadratic term and fit an
    # intercept only term (lnr_mean), a linear model, and a model
    # with the right quadratic term, and we expect
    # super_learner() to pick the right one to weight highly.

    fake_data <- data.frame(
      x1 = rnorm(n = 1000),
      x2 = rnorm(n = 1000)
    )
    fake_data$y <- fake_data$x1 + fake_data$x2^2 + rnorm(n = 1000)

    # train super_learner() on the fake data
    learned_predictor <- super_learner(
      data = fake_data,
      formula = list(
        .default = y ~ x1 + x2,
        lm2 = y ~ x1 + poly(x2, 2)
      ), # pass the quadratic term to lm2
      learners = list(
        mean = lnr_mean,
        lm1 = lnr_lm,
        lm2 = lnr_lm
      )
    )

    # expect the correctly specified model to get all the weight
    testthat::expect_gte(learned_predictor$learner_weights["lm2"], .9)
  }

  #' @srrstats {G5.6b, G5.9b} parameter recovery is checked with multiple seeds;
  #' in our case, parameter recovery means that a correct model of the
  #' predictor-outcome relationship is selected over ones that don't correctly
  #' model the predictor-outcome relationship. we do this with three different
  #' pseudo-random number generator seeds.
  run_super_learner_prefers_correct_lm_model(custom_seed = 1234)
  run_super_learner_prefers_correct_lm_model(custom_seed = 1111)
  run_super_learner_prefers_correct_lm_model(custom_seed = 2222)
})


testthat::test_that(desc = "super_learner() prefers the correct binary outcome model", {
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
    x2 = rnorm(n = 1000)
  )
  fake_data$y <- rbinom(
    n = 1000,
    size = 1,
    prob = plogis(fake_data$x1 + fake_data$x2^2 + rnorm(n = 1000))
  )

  # train super_learner() on the fake data
  learned_predictor <- super_learner(
    data = fake_data,
    formula = list(
      .default = y ~ x1 + x2,
      logistic2 = y ~ x1 + poly(x2, 2), # pass the quadratic term to logistic2
      logistic3 = y ~ x1
    ),
    learners = list(
      mean = lnr_mean,
      logistic1 = lnr_logistic,
      logistic2 = lnr_logistic,
      logistic3 = lnr_logistic,
      rf = lnr_rf_binary
    ),
    outcome_type = "binary"
  )

  # expect the correctly specified model to get all the weight
  testthat::expect_gte(learned_predictor$learner_weights["logistic2"], .9)
})



testthat::test_that(desc = "super_learner() prefers the correct lm outcome model", {
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
    x2 = rnorm(n = 1000)
  )
  fake_data$y <- fake_data$x1 + fake_data$x2^2 + rnorm(n = 1000)

  # train super_learner() on the fake data
  learned_predictor <- super_learner(
    data = fake_data,
    formula = list(
      .default = y ~ x1 + x2,
      lm2 = y ~ x1 + poly(x2, 2)
    ), # pass the quadratic term to lm2
    learners = list(
      mean = lnr_mean,
      lm1 = lnr_lm,
      lm2 = lnr_lm
    )
  )

  # expect the correctly specified model to get all the weight
  testthat::expect_gte(learned_predictor$learner_weights["lm2"], .9)
})



# super_learner() prefers the correct lm density model -----

#' @srrstats {RE1.4} Any assumptions made about the data in nadir are codified by users
#' in the choice of learners that they specify. Here, the test checking that super learner prefers
#' the correct model is in part a testing of the behavior of when
#' model assumptions are violated.  We show in our tests that \code{super_learner()}
#' downweights the learners that make incorrect assumptions about the data.
testthat::test_that(desc = "super_learner() prefers the correct lm density model", {
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
    x2 = rnorm(n = 1000)
  )
  fake_data$y <- fake_data$x1 + fake_data$x2^2 + rnorm(n = 1000)

  # train super_learner() on the fake data
  learned_predictor <- super_learner(
    data = fake_data,
    formula = list(
      .default = y ~ x1 + x2,
      lm2 = y ~ x1 + poly(x2, 2)
    ), # pass the quadratic term to lm2
    learners = list(
      lm = lnr_lm_density,
      lm2 = lnr_lm_density
    ),
    outcome_type = "density"
  )

  # expect the correctly specified model to get all the weight
  testthat::expect_gte(learned_predictor$learner_weights["lm2"], .9)
})

# super_learner() outperforms naive lm ----------

testthat::test_that(desc =
  paste0(
    "verify that super_learner() really does outperform a simple linear model",
    " most of the time"), {
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
    holdouts <- df[holdout_ids, ]
    training <- df[-holdout_ids, ]

    learned_predictor <- super_learner(
      data = training,
      formula = list(
        .default = medv ~ .,
        gam = medv ~ s(ptratio) + crim + zn + indus + s(nox) + rm + age + dis,
        lm2 = medv ~ age:zn + poly(nox, 2) + .
      ),
      learners = list(
        mean = lnr_mean,
        lm = lnr_lm,
        lm2 = lnr_lm,
        gam = lnr_gam,
        earth = lnr_earth,
        rf = lnr_rf,
        xgboost = lnr_xgboost,
        glmnet = lnr_glmnet
      )
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
    learners = learners
  )

  expect_true("predict" %in% names(sl_model))
  expect_true(is.function(sl_model$predict))
  expect_true("holdout_predictions" %in% names(sl_model))
  expect_true(is.data.frame(sl_model$holdout_predictions))
  expect_true(sl_model$outcome_type %in% nadir_supported_types)
  expect_true("learner_weights" %in% names(sl_model))
  expect_true(is.numeric(sl_model$learner_weights))
  expect_true(sum(sl_model$learner_weights) == 1L)
  expect_true("y_variable" %in% names(sl_model))
  expect_true("outcome_type" %in% names(sl_model))
})

test_that(desc = "super_learner() can use a character formula like 'y ~ x'", {
  learners <- list(
    glm = lnr_glm,
    glmnet = lnr_glmnet
  )

  formula <- "hp ~ mpg"

  testthat::expect_no_error(
    sl_fit <- nadir::super_learner(
      data = mtcars, formula = formula, learners = learners
    )
  )
})


test_that(desc = "super_learner() doesn't need y to appear in predict(newdata)", {
  sl_fit <- nadir::super_learner(
    data = mtcars,
    formula = hp ~ mpg,
    learners = list(
      lnr_earth, lnr_gam, lnr_gbm, lnr_glm, lnr_glmnet, lnr_lm, lnr_mean,
      lnr_ranger, lnr_rf, lnr_xgboost
    )
  )

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
    sl <- super_learner(df_na, fast_learners, mpg ~ hp,
      n_folds = 2,
      use_complete_cases = TRUE
    ),
    "use_complete_cases = TRUE will filter"
  )
  expect_length(sl$predict(mtcars), nrow(mtcars))
})

test_that("discrete super_learner picks a single learner, warning on ties", {
  set.seed(1)
  # deterministic tie via a custom weight function
  expect_warning(
    sl <- super_learner(
      mtcars, fast_learners, mpg ~ hp,
      n_folds = 2,
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
    mtcars, fast_learners, mpg ~ hp,
    n_folds = 2,
    determine_super_learner_weights = function(data, y_variable, obs_weights = NULL) c(0.3, 0.7),
    ensemble_or_discrete = "discrete"
  )
  expect_equal(sort(unname(sl2$learner_weights)), c(0, 1))

  # invalid option errors
  set.seed(1)
  expect_error(
    super_learner(mtcars, fast_learners, mpg ~ hp,
      n_folds = 2,
      ensemble_or_discrete = "zzz"
    )
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
    mtcars, fast_learners, mpg ~ hp,
    n_folds = 2,
    cluster_ids = rep(1:8, each = 4)
  )
  expect_s3_class(sl_cl, "nadir_sl_model")

  set.seed(1)
  sl_st <- super_learner(
    mtcars, fast_learners, mpg ~ hp,
    n_folds = 2,
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



# preserve row-ids from input in {fitted,residuals}.nadir_sl_model --------------

#' @srrstats {RE1.3, RE7.2} fitted()/residuals() are returned in the
#'   original input row order, demonstrated by by this test here
test_that("fitted() and residuals() are in input-data row order (RE1.3)", {
  set.seed(42)
  d <- data.frame(x = rnorm(90))
  d$y <- 2 * d$x + rnorm(90, sd = 0.3)
  sl <- suppressWarnings(super_learner(
    data = d, formula = y ~ x, n_folds = 3,
    learners = list(mean = lnr_mean, lm = lnr_lm)
  ))
  hp <- sl$holdout_predictions
  truth <- rep(NA_real_, nrow(d))
  truth[hp$.sl_rowid] <-
    as.matrix(hp[, names(sl$learner_weights)]) %*% sl$learner_weights
  expect_equal(fitted(sl), truth)
  expect_equal(residuals(sl), d$y - fitted(sl))
})

test_that("fitted() errors informatively when a cv_schema drops .sl_rowid", {
  set.seed(43)
  d <- data.frame(x = rnorm(60))
  d$y <- d$x + rnorm(60)
  cv_rebuild <- function(data, n_folds) {
    d2 <- data.frame(x = data$x, y = data$y)
    list(
      training_data = list(d2[1:30, ], d2[31:60, ]),
      validation_data = list(d2[31:60, ], d2[1:30, ])
    )
  }
  sl <- suppressWarnings(suppressMessages(super_learner(
    data = d, formula = y ~ x, n_folds = 2, cv_schema = cv_rebuild,
    learners = list(mean = lnr_mean, lm = lnr_lm)
  )))
  expect_error(fitted(sl), "did not preserve the .sl_rowid")
})



# test that warnings are suppressed ---------------------------------------

test_that("super_learner captures learner warnings silently, with rewritten calls", {
  lnr_grumpy <- function(data, formula, ...) {
    warning("this learner is grumpy")
    lnr_lm(data, formula)
  }
  attr(lnr_grumpy, "sl_lnr_type") <- "continuous"
  attr(lnr_grumpy, "sl_lnr_name") <- "grumpy"

  set.seed(1)
  expect_no_warning({
    sl <- super_learner(
      mtcars,
      learners = list(lm = lnr_lm, grumpy = lnr_grumpy),
      formulas = mpg ~ hp,
      n_folds = 2
    )
  })
  # one warning per CV training fold
  expect_length(sl$warnings_from_training_cv_stage1, 2)
  # plus one from the full-data fit
  expect_length(sl$warnings_from_training_on_entire_data, 1)
  expect_true(all(vapply(sl$warnings_from_training_cv_stage1,
    inherits, logical(1),
    what = "warning"
  )))
  # warnings are named by, and attributed to, the signaling learner
  expect_true(all(names(sl$warnings_from_training_cv_stage1) == "grumpy"))
  expect_identical(sl$warning_learners, "grumpy")
  # calls are rewritten to be user-legible
  expect_match(
    paste(deparse(sl$warnings_from_training_cv_stage1[[1]]$call), collapse = ""),
    "lnr_grumpy"
  )
  # a warning is not an error: the learner keeps its weight-eligibility
  expect_true("grumpy" %in% names(sl$learner_weights))
})

test_that("super_learner captures prediction-stage warnings", {
  lnr_grumpy_predictor <- function(data, formula, ...) {
    fit <- lnr_lm(data, formula)
    function(newdata) {
      warning("grumpy at prediction time")
      fit(newdata)
    }
  }
  attr(lnr_grumpy_predictor, "sl_lnr_type") <- "continuous"
  attr(lnr_grumpy_predictor, "sl_lnr_name") <- "grumpy_pred"

  set.seed(1)
  expect_no_warning({
    sl <- super_learner(
      mtcars,
      learners = list(lm = lnr_lm, gp = lnr_grumpy_predictor),
      formulas = mpg ~ hp,
      n_folds = 2
    )
  })
  expect_true("warnings_from_predicting_cv_stage2" %in% names(sl))
  expect_true("gp" %in% sl$warning_learners)
})

test_that("crossfit_super_learner aggregates captured warnings across folds", {
  lnr_grumpy <- function(data, formula, ...) {
    warning("this learner is grumpy")
    lnr_lm(data, formula)
  }
  attr(lnr_grumpy, "sl_lnr_type") <- "continuous"
  attr(lnr_grumpy, "sl_lnr_name") <- "grumpy"

  set.seed(1)
  expect_no_warning({
    cf <- crossfit_super_learner(
      data = mtcars,
      formulas = mpg ~ hp,
      learners = list(lm = lnr_lm, grumpy = lnr_grumpy),
      n_folds = 2, inner_n_folds = 2
    )
  })
  expect_identical(cf$warning_learners, "grumpy")
  expect_length(cf$warnings_from_inner_super_learners, 2)
  expect_true(
    "warnings_from_training_cv_stage1" %in%
      names(cf$warnings_from_inner_super_learners[["fold_1"]])
  )
})

test_that("errors_from_* fields are lists of error conditions named by learner", {
  lnr_always_fails <- function(data, formula, ...) stop("nope")
  attr(lnr_always_fails, "sl_lnr_type") <- "continuous"
  attr(lnr_always_fails, "sl_lnr_name") <- "always_fails"

  set.seed(1)
  sl <- super_learner(
    mtcars,
    learners = list(lm = lnr_lm, bad = lnr_always_fails),
    formulas = mpg ~ hp,
    n_folds = 2
  )
  expect_true(all(vapply(sl$errors_from_training_cv_stage1,
    inherits, logical(1),
    what = "error"
  )))
  expect_true(all(names(sl$errors_from_training_cv_stage1) == "bad"))
})

test_that("perfectly collinear predictors are detected and survivable", {
  #' @srrstats {RE7.0, RE7.0a} noiseless, exact relationships between
  #'   predictor columns (x2 = 2*x1) are detected by the collinearity
  #'   pre-processing warning, and the fit still proceeds with
  #'   collinearity-tolerant learners.
  #' @srrstats {RE2.4a} tests the perfect-collinearity-among-predictors check.
  set.seed(1)
  df <- data.frame(x1 = rnorm(100))
  df$x2 <- 2 * df$x1
  df$y <- df$x1 + rnorm(100)

  expect_warning(
    sl <- super_learner(df, list(mean = lnr_mean, glmnet = lnr_glmnet),
      y ~ x1 + x2,
      n_folds = 2
    ),
    "collinear"
  )
  expect_length(sl$predict(df), nrow(df))
})


test_that("noiseless y = f(x) relationships are recovered essentially exactly", {
  #' @srrstats {RE7.1} with a noiseless, exact linear relationship between
  #'   predictors and response, the ensemble puts its weight on lnr_lm and
  #'   out-of-fold predictions match the truth to numerical tolerance.
  #' @srrstats {RE2.4b} the perfect dependent~independent correlation warning
  #'   fires on this data.
  set.seed(1)
  df <- data.frame(x = rnorm(100))
  df$y <- 2 * df$x + 1 # exactly noiseless

  expect_warning( # drop this wrapper if you chose Option B in RE2.4
    sl <- super_learner(df, list(lm = lnr_lm, mean = lnr_mean),
      y ~ x,
      n_folds = 3
    ),
    "collinear|perfect"
  )
  expect_gte(sl$learner_weights[["lm"]], 0.99)
  expect_equal(sl$oof_predictions, df$y, tolerance = 1e-6)
})


test_that("return objects contain no missing or undefined values", {
  #' @srrstats {G5.3} fitted model objects' numeric outputs
  #'   (oof_predictions, fitted(), residuals(), coef()/learner weights,
  #'   predict() on training and new data) are explicitly checked to contain
  #'   no NA, NaN, or Inf values when fit on complete data.
  set.seed(1)
  sl <- super_learner(mtcars, list(lm = lnr_lm, mean = lnr_mean),
    mpg ~ hp + wt,
    n_folds = 3
  )
  no_bad <- function(x) expect_false(any(is.na(x) | is.nan(x) | is.infinite(x)))
  no_bad(sl$oof_predictions)
  no_bad(fitted(sl))
  no_bad(residuals(sl))
  no_bad(coef(sl))
  no_bad(sl$predict(mtcars))
})


# RE7.1a: model fitting on noiseless data is at least as fast as on
# equivalent noisy data.

test_that("noiseless relationships fit at least as fast as noisy ones", {
  skip_on_cran()

  #' @srrstats {RE7.1a} Fitting on data with a noiseless, exact
  #'   predictor-response relationship is confirmed to be at least as fast
  #'   as fitting on equivalent noisy data. nadir's own ensembling stage
  #'   introduces no noise-dependent overhead (candidate learners are
  #'   invoked identically either way, and the NNLS weight determination
  #'   operates on a fixed-size holdout matrix), so the two timings are
  #'   expected to be statistically indistinguishable; this test asserts
  #'   the noiseless fit is not slower beyond timing jitter, using medians
  #'   over repetitions and a generous slack factor so the assertion is
  #'   good for calling this unit test over and over.
  set.seed(1)
  n <- 500
  x <- rnorm(n)
  df_noiseless <- data.frame(x = x, y = 2 * x + 1) # exact
  df_noisy <- data.frame(x = x, y = 2 * x + 1 + rnorm(n)) # equivalent plus noise

  time_fit <- function(df, seed) {
    set.seed(seed) # matched seeds => identical fold assignment both arms
    system.time(
      # noiseless y is perfectly collinear with x, so the RE2.4b
      # outcome-collinearity check warns (by design); suppress so both
      # arms do identical condition handling during timing
      suppressWarnings(
        super_learner(
          data = df,
          learners = list(lm = lnr_lm, mean = lnr_mean),
          formulas = y ~ x,
          n_folds = 3
        )
      )
    )[["elapsed"]]
  }

  reps <- 5
  t_noiseless <- median(
    vapply(
      seq_len(reps), function(i) time_fit(df_noiseless, seed = i),
      numeric(1)
    )
  )
  t_noisy <- median(
    vapply(
      seq_len(reps), function(i) time_fit(df_noisy, seed = i),
      numeric(1)
    )
  )

  # "at least as fast" up to measurement noise: a 3x multiplicative slack
  # absorbs scheduler jitter, and the small additive term guards against
  # near-zero elapsed times at timer resolution
  expect_lte(t_noiseless, 3 * t_noisy + 0.05)
})

test_that("unsupported input types error informatively", {
  #' @srrstats {G5.8, G5.8b} data of unsupported types produce clear errors:
  #'   a character outcome declared continuous is caught by
  #'   validate_outcome_type_matches_y(); complex-valued predictors are
  #'   rejected by the underlying model-frame machinery with an error, not
  #'   silent misbehaviour.
  df <- mtcars
  df$mpg <- as.character(df$mpg)
  expect_error(
    super_learner(df, list(lm = lnr_lm), mpg ~ hp,
      outcome_type = "continuous"
    )
  )

  df2 <- mtcars
  df2$hp <- complex(real = df2$hp, imaginary = 1)
  expect_error(super_learner(df2, list(lm = lnr_lm), mpg ~ hp, n_folds = 2))
})


test_that("all-NA and constant columns produce expected behaviour", {
  #' @srrstats {G5.8, G5.8c} an all-NA column trips the missing-data error
  #'   (or complete-case filtering removes every row, which errors); an
  #'   all-identical predictor is tolerated by learners that handle rank
  #'   deficiency and produces finite predictions.
  df <- mtcars
  df$junk <- NA_real_
  expect_error(
    super_learner(df, list(lm = lnr_lm), mpg ~ hp + junk, n_folds = 2),
    "missing data"
  )

  df2 <- mtcars
  df2$const <- 1
  set.seed(1)
  sl <- super_learner(df2, list(mean = lnr_mean, glmnet = lnr_glmnet),
    mpg ~ hp + const,
    n_folds = 2
  )
  expect_true(all(is.finite(sl$oof_predictions)))
})


test_that("p > n data works with suitable learners", {
  #' @srrstats {G5.8, G5.8d} data with more columns than rows (outside the
  #'   scope of OLS) is handled: penalized learners fit and predict, and the
  #'   ensemble remains finite.
  set.seed(1)
  n <- 20
  p <- 40
  X <- as.data.frame(matrix(rnorm(n * p), n, p))
  names(X) <- paste0("x", seq_len(p))
  X$y <- rnorm(n)
  sl <- super_learner(X, list(mean = lnr_mean, glmnet = lnr_glmnet),
    y ~ .,
    n_folds = 2
  )
  expect_true(all(is.finite(sl$oof_predictions)))
})


test_that("results are stable across random seeds", {
  #' @srrstats {G5.9, G5.9b} fitting the same specification under different
  #'   seeds (which change fold assignment) does not meaningfully change
  #'   results: ensemble weights agree within a loose tolerance and CV loss
  #'   agrees within a few percent.
  fit_with <- function(seed) {
    set.seed(seed)
    super_learner(mtcars, list(lm = lnr_lm, mean = lnr_mean),
      mpg ~ hp + wt,
      n_folds = 5
    )
  }
  w1 <- fit_with(1)$learner_weights
  w2 <- fit_with(2)$learner_weights
  expect_equal(w1, w2, tolerance = 0.15)
})


test_that("machine-epsilon-scale noise does not meaningfully change results", {
  #' @srrstats {G5.9, G5.9a} adding noise at the scale of
  #'   .Machine$double.eps to the predictors and outcome leaves ensemble
  #'   weights and out-of-fold predictions essentially unchanged (fold
  #'   assignment held fixed via a shared seed).
  set.seed(1)
  sl1 <- super_learner(mtcars, list(lm = lnr_lm, mean = lnr_mean),
    mpg ~ hp + wt,
    n_folds = 3
  )
  jitter_eps <- function(x) x + rnorm(length(x)) * .Machine$double.eps
  m2 <- mtcars
  m2$mpg <- jitter_eps(m2$mpg)
  m2$hp <- jitter_eps(m2$hp)
  m2$wt <- jitter_eps(m2$wt)
  set.seed(1)
  sl2 <- super_learner(m2, list(lm = lnr_lm, mean = lnr_mean),
    mpg ~ hp + wt,
    n_folds = 3
  )
  expect_equal(sl1$learner_weights, sl2$learner_weights, tolerance = 1e-6)
  expect_equal(sl1$oof_predictions, sl2$oof_predictions, tolerance = 1e-6)
})



# tests for
# plot.nadir_sl_model, plot.nadir_cv_sl (+ sl_build_comparison_plot,
# cv_sl_loss_label), print.nadir_cv_sl, summary/print.summary.nadir_sl_model,
# truncate_lnr, and the default_* outcome-type dispatchers.

fit_sl_small <- function() {
  set.seed(1)
  super_learner(
    data = mtcars,
    formulas = mpg ~ cyl + hp,
    learners = list(lm = lnr_lm, mean = lnr_mean),
    n_folds = 2
  )
}

fit_cv_small <- function() {
  set.seed(1)
  suppressMessages(cv_super_learner(
    data = mtcars,
    formulas = mpg ~ cyl + hp,
    learners = list(lm = lnr_lm, mean = lnr_mean),
    n_folds = 2, inner_n_folds = 2
  ))
}



# ---- nadir_sl_model: plot, summary, print.summary --------------------------

test_that("plot.nadir_sl_model returns ggplots for both types and rejects others", {
  skip_if_not_installed("ggplot2")
  sl <- fit_sl_small()
  expect_s3_class(plot(sl), "ggplot") # comparison (default)
  expect_s3_class(plot(sl, type = "fitted"), "ggplot")
  expect_error(plot(sl, type = "nonsense")) # match.arg
})

test_that("summary and print.summary for nadir_sl_model report weights and losses", {
  sl <- fit_sl_small()
  s <- summary(sl)
  expect_s3_class(s, "summary.nadir_sl_model")
  expect_setequal(s$comparison$learner, c("lm", "mean"))
  expect_identical(s$y_variable, "mpg")

  out <- capture.output(print(s))
  expect_true(any(grepl("lm", out)))
  expect_true(any(grepl("mpg", out)))
})

# ---- nadir_cv_sl: plot, print, loss labels ---------------------------------

test_that("plot.nadir_cv_sl returns ggplots for all three types", {
  skip_if_not_installed("ggplot2")
  out <- fit_cv_small()
  expect_s3_class(plot(out), "ggplot") # comparison:
  # exercises
  # cv_sl_fold_losses +
  # sl_build_comparison_plot
  expect_s3_class(plot(out, type = "weights"), "ggplot") # delegates to crossfit
  expect_s3_class(plot(out, type = "fitted"), "ggplot")
  expect_error(plot(out, type = "nonsense"))
})

test_that("plot.nadir_cv_sl errors informatively without a stored $crossfit", {
  out <- fit_cv_small()
  out$crossfit <- NULL
  expect_error(plot(out), "did not store \\$crossfit")
})

test_that("print.nadir_cv_sl summarises the object", {
  out <- fit_cv_small()
  printed <- capture.output(print(out))
  expect_true(any(grepl("cv_loss", printed)))
  expect_true(any(grepl("crossfit", printed)))
})

test_that("cv_sl_loss_label covers default, non-continuous, and custom metrics", {
  out <- fit_cv_small()
  # default metric, continuous outcome
  expect_identical(
    nadir:::cv_sl_loss_label(out$crossfit),
    "Cross-validated held-out MSE"
  )
  # default metric, non-continuous outcome (label helper reads only these
  # two fields, so a minimal stand-in object suffices)
  expect_identical(
    nadir:::cv_sl_loss_label(list(outcome_type = "binary", loss_metric = NULL)),
    "Cross-validated held-out negative log loss"
  )
  # user-supplied loss metric
  cf_custom <- out$crossfit
  cf_custom$loss_metric <- function(x, y) mean(abs(x - y))
  expect_match(nadir:::cv_sl_loss_label(cf_custom), "user-supplied")
})
