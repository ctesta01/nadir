# test-learner-outcome-leakage.R
#
# Tests to check for outcome leakage through symbolic formulas like `y ~ .`
#
# Some learners, like lnr_xgboost and lnr_lightgbm construct design matrices
# internally (rather than passing the formula to a package with a
# formula interface) and could accidentally include the outcome
# among the predictors when the formula RHS contains `.`.
# This happened in prior (wrong) implementations of lnr_xgboost, lnr_lightgbm
# via a one-sided-formula approach `x_formula <- formula`, then
# `x_formula[[2]] <- NULL`, which makes `.` expand to every column of the
# data, outcome included. The detectable symptom from this now-corrected bug
# (which we test for below) is cross-validation/heldout MSE below what's
# theoretically achievable, i.e., impossibly good performance.
#
# Two complementary criteria are tested, both of which every
# learner should pass:
#
#   (1) PREDICTION INVARIANCE: a fitted learner's predictions must not change
#       when the outcome column of `newdata` is altered. Any dependence on
#       newdata's outcome means the outcome entered the feature set.
#
#   (2) NO CLAIRVOYANCE: on data where Y is pure noise, independent of all
#       predictors, no learner can achieve out-of-sample MSE meaningfully
#       below var(Y). (A leaking learner achieves MSE near 0 in that situation,
#       so we use a threshold of 0.5 * var(Y) to cleanly separate
#       cheating learners. No cheating!  Swiper no swiping!
#

make_leakage_canary_data <- function(n, p = 4, seed = 1) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), nrow = n)
  colnames(X) <- paste0("x", seq_len(p))
  # y is independent of X so any out-of-sample predictive skill is flagged as
  # leakage/cheating.
  data.frame(y = rnorm(n), X)
}

# TODO:  EXPAND THESE TESTS TO EVERY LEARNER *OR*
#   ADD AN ADDITIONAL SET OF OPTIONAL TESTS THAT DEVELOPERS CAN RUN
#   BUT WHICH AREN'T NECESSARY TO RUN AS FREQUENTLY ---
#
# Learners subject to the leakage tests, with fast fitting arguments.
# Every learner accepting (data, formula, ...) should eventually appear here;
# at minimum, every learner that builds its own design matrix must.
leakage_test_learners <- function() {
  list(
    xgboost  = list(learner = lnr_xgboost,  args = list(nrounds = 20),
                    package = "xgboost"),
    lightgbm = list(learner = lnr_lightgbm, args = list(nrounds = 20),
                    package = "lightgbm"),
    bart     = list(learner = lnr_bart,     args = list(n.samples = 100,
                                                        n.burn = 50),
                    package = "dbarts"),
    rf       = list(learner = lnr_rf,       args = list(ntree = 50),
                    package = "randomForest"),
    ranger   = list(learner = lnr_ranger,   args = list(num.trees = 50),
                    package = "ranger"),
    glmnet   = list(learner = lnr_glmnet,   args = list(lambda = 0.1),
                    package = "glmnet"),
    earth    = list(learner = lnr_earth,    args = list(),
                    package = "earth"),
    lm       = list(learner = lnr_lm,       args = list(),
                    package = NULL)
  )
}

test_that("fitted learners' predictions are invariant to newdata's outcome column (y ~ .)", {
  df <- make_leakage_canary_data(n = 150)

  for (learner_name in names(leakage_test_learners())) {
    spec <- leakage_test_learners()[[learner_name]]
    if (!is.null(spec$package)) skip_if_not_installed(spec$package)

    set.seed(2)
    fit <- do.call(
      spec$learner,
      c(list(data = df, formula = y ~ .), spec$args)
    )

    # Same predictors, radically different outcome column: predictions from
    # an outcome-clean learner must be bitwise identical.
    newdata_original  <- df
    newdata_perturbed <- df
    newdata_perturbed$y <- df$y + 1e6

    predictions_original  <- fit(newdata_original)
    predictions_perturbed <- fit(newdata_perturbed)

    expect_equal(
      predictions_original,
      predictions_perturbed,
      info = paste0(
        "Learner '", learner_name, "' produced predictions that depend on ",
        "the outcome column in newdata: the outcome is leaking into its ",
        "feature set."
      )
    )
  }
})

test_that("[no cheating/leakage:] learners predict noise out-of-sample poorly with (y ~ .)", {
  df_train <- make_leakage_canary_data(n = 300, seed = 10)
  df_test  <- make_leakage_canary_data(n = 300, seed = 11)

  noise_floor <- stats::var(df_test$y)

  for (learner_name in names(leakage_test_learners())) {
    spec <- leakage_test_learners()[[learner_name]]
    if (!is.null(spec$package)) skip_if_not_installed(spec$package)

    set.seed(3)
    fit <- do.call(
      spec$learner,
      c(list(data = df_train, formula = y ~ .), spec$args)
    )

    test_mse <- mean((fit(df_test) - df_test$y)^2)

    # A legitimate non-cheating learner's out-of-sample MSE on pure noise
    # w.p. one asymptotically concentrates around var(y) (or above,
    # if it overfit the training noise). MSE far below var(y) is
    # impossible without access to the heldout/test outcomes.
    expect_gt(
      test_mse,
      0.5 * noise_floor,
      label = paste0(
        "Learner '", learner_name, "' achieved out-of-sample MSE (",
        signif(test_mse, 3), ") far below the random noise floor (",
        signif(noise_floor, 3), "): the outcome is leaking into its ",
        "predictor feature set."
      )
    )
  }
})

# TODO:  EXPAND NOT JUST THIS TEST BUT ALSO THE ERROR FLAGGING FOR WHEN
# Y APPEARS IN PLACES IT SHOULDN'T -- SHOULDN'T BE HARD TO ADD TO A PLACE
# LIKE WHERE check_formulas_for_id_vars() IS CALLED IN super_learner()

test_that("design matrix constructing learners error when outcome in formula RHS", {
  # y ~ y + x should always be an error in a super learner; the
  # design matrix helper here can refuse it rather than silently leak.
  df <- make_leakage_canary_data(n = 50)

  skip_if_not_installed("xgboost")
  expect_error(
    lnr_xgboost(df, y ~ y + x1, nrounds = 5),
    "leak"
  )
})
