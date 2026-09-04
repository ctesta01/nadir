# Learner train/predict symmetry tests
#
# These tests exist because of a bug in lnr_earth: training passed covariates
# through model.frame(formula, data) (formula-ordered, formula-subset) while
# prediction passed raw newdata minus the outcome column (data-ordered, all
# columns). earth's x/y interface consumed the columns positionally, silently
# applying hinge functions to the wrong variables and producing predictions
# off by orders of magnitude.
#
# The invariants below hold for ANY correctly-wrapped learner, deterministic
# or stochastic, because they compare predictions from ONE fitted model under
# different presentations of the same newdata:
#
#   1. column-shuffle invariance:  pred(newdata) == pred(newdata[, shuffled])
#   2. extra-column invariance:    pred(newdata) == pred(cbind(newdata, junk))
#   3. formula-order sanity:       with a formula whose term order differs
#                                  from the data's column order, in-sample
#                                  predictions on an easy DGP must actually
#                                  fit (catches scale explosions without
#                                  demanding exact equality from stochastic
#                                  learners)
suppressWarnings(library(future))

# --- easy DGPs where every learner should do well -----------------------------
# NOTE: column order is deliberately DIFFERENT from the formula term order
# below (y first, then x3, x1, grp, x2) -- this ordering mismatch is exactly
# what detonated lnr_earth.
make_continuous_dgp <- function(n = 300, seed = 1) {
  set.seed(seed)
  x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n)
  grp <- factor(sample(c("a", "b", "c"), n, replace = TRUE))
  y <- 2 * x1 - 1.5 * x2 + 0.5 * x3 + (grp == "b") - (grp == "c") + rnorm(n, sd = .5)
  data.frame(y = y, x3 = x3, x1 = x1, grp = grp, x2 = x2)
}

make_binary_dgp <- function(n = 300, seed = 2) {
  set.seed(seed)
  x1 <- rnorm(n); x2 <- rnorm(n)
  grp <- factor(sample(c("a", "b"), n, replace = TRUE))
  y <- rbinom(n, 1, plogis(1.5 * x1 - x2 + (grp == "b")))
  data.frame(y = y, x1 = x1, grp = grp, x2 = x2)
}

# formulas whose term order differs from data column order
continuous_formula <- y ~ x1 + x2 + grp + x3
binary_formula     <- y ~ x1 + grp + x2

# --- learner registry ----------------------------------------------------------
# name -> list(lnr, type, pkg (for skip_if_not_installed), extra args)
# Add new learners here as they are added to the package; the tests below
# iterate over the whole registry.
learner_registry <- list(
  lm            = list(lnr = lnr_lm,            type = "continuous", pkg = NULL),
  mean          = list(lnr = lnr_mean,          type = "continuous", pkg = NULL),
  glm           = list(lnr = lnr_glm,           type = "continuous", pkg = NULL),
  earth         = list(lnr = lnr_earth,         type = "continuous", pkg = "earth"),
  rf            = list(lnr = lnr_rf,            type = "continuous", pkg = "randomForest",
                       args = list(ntree = 50)),
  ranger        = list(lnr = lnr_ranger,        type = "continuous", pkg = "ranger",
                       args = list(num.trees = 50)),
  gam           = list(lnr = lnr_gam,           type = "continuous", pkg = "mgcv"),
  gbm           = list(lnr = lnr_gbm,           type = "continuous", pkg = "gbm",
                       args = list(n.trees = 50, distribution = "gaussian")),
  glmnet        = list(lnr = lnr_glmnet,        type = "continuous", pkg = "glmnet",
                       args = list(lambda = .1)),
  cvglmnet      = list(lnr = lnr_cvglmnet,      type = "continuous", pkg = "glmnet"),
  hal           = list(lnr = lnr_hal,           type = "continuous", pkg = "hal9001",
                       args = list(lambda = .01)),
  xgboost       = list(lnr = lnr_xgboost,       type = "continuous", pkg = "xgboost",
                       args = list(nrounds = 20)),
  lightgbm      = list(lnr = lnr_lightgbm,      type = "continuous", pkg = "lightgbm",
                       args = list(nrounds = 20)),
  logistic      = list(lnr = lnr_logistic,      type = "binary", pkg = NULL),
  rf_binary     = list(lnr = lnr_rf_binary,     type = "binary", pkg = "randomForest",
                       args = list(ntree = 50)),
  ranger_binary = list(lnr = lnr_ranger_binary, type = "binary", pkg = "ranger",
                       args = list(num.trees = 50)),
  nnet          = list(lnr = lnr_nnet,          type = "binary", pkg = "nnet",
                       args = list(size = 3))
)

fit_registered_learner <- function(entry) {
  if (!is.null(entry$pkg)) testthat::skip_if_not_installed(entry$pkg)
  if (identical(entry$type, "binary")) {
    dat <- make_binary_dgp(); f <- binary_formula
  } else {
    dat <- make_continuous_dgp(); f <- continuous_formula
  }
  set.seed(99) # stochastic learners: fit once, reuse predictor across checks
  predictor <- do.call(entry$lnr, c(list(data = dat, formula = f), entry$args))
  list(dat = dat, predictor = predictor)
}

# =============================================================================
# 1 & 2. prediction invariance to newdata column order and irrelevant columns
# =============================================================================
testthat::test_that("learner predictions are invariant to newdata column order and extra columns", {
  for (nm in names(learner_registry)) {
    entry <- learner_registry[[nm]]
    fitted <- tryCatch(fit_registered_learner(entry), condition = function(c) c)
    if (inherits(fitted, "skip")) next # skip_if_not_installed signaled
    if (inherits(fitted, "error")) {
      testthat::fail(sprintf("learner '%s' errored during fitting: %s",
                             nm, conditionMessage(fitted)))
      next
    }
    dat <- fitted$dat; p <- fitted$predictor
    baseline <- as.numeric(p(dat))

    # (1) shuffle the newdata columns
    set.seed(7)
    shuffled <- dat[, sample(ncol(dat)), drop = FALSE]
    testthat::expect_equal(
      as.numeric(p(shuffled)), baseline, tolerance = 1e-10,
      label = sprintf("'%s' predictions under column-shuffled newdata", nm),
      expected.label = sprintf("'%s' baseline predictions", nm))

    # (2) append an irrelevant column
    with_junk <- dat
    with_junk$zzz_irrelevant <- rnorm(nrow(dat))
    testthat::expect_equal(
      as.numeric(p(with_junk)), baseline, tolerance = 1e-10,
      label = sprintf("'%s' predictions with an extra irrelevant column", nm),
      expected.label = sprintf("'%s' baseline predictions", nm))
  }
})

# =============================================================================
# 3. formula-order sanity: predictions must actually FIT the training data
#    when formula term order differs from data column order
#    (this is the check that catches the lnr_earth failure mode: it does not
#    require exact equality, so it is robust for stochastic learners, but a
#    positional column mix-up produces R^2 << 0 and fails loudly)
# =============================================================================
testthat::test_that("in-sample predictions fit the training data when formula order != data column order", {
  r2 <- function(p, y) 1 - sum((y - p)^2) / sum((y - mean(y))^2)
  for (nm in setdiff(names(learner_registry), "mean")) { # lnr_mean's R^2 is 0 by design
    entry <- learner_registry[[nm]]
    fitted <- tryCatch(fit_registered_learner(entry), condition = function(c) c)
    if (inherits(fitted, c("skip", "error"))) next # fitting failures caught above

    dat <- fitted$dat
    preds <- as.numeric(fitted$predictor(dat))
    y <- dat$y

    in_sample_r2 <- if (identical(entry$type, "binary")) {
      # for probability learners score against the 0/1 outcome; any correct
      # learner clears 0 easily on this DGP, a column mix-up will not
      r2(preds, y)
    } else {
      r2(preds, y)
    }

    testthat::expect_gt(
      in_sample_r2, 0.2,
      label = sprintf("'%s' in-sample R^2 (= %.3g)", nm, in_sample_r2))

    # scale sanity: a positional mix-up produces predictions wildly off the
    # outcome's scale; no legitimate learner does on this DGP
    testthat::expect_lt(
      max(abs(preds - mean(y))), 20 * sd(y),
      label = sprintf("'%s' max deviation of predictions from mean(y)", nm))
  }
})

# =============================================================================
# regression test pinned to the exact lnr_earth failure: penguins-like layout
# where the formula puts a trailing data column first
# =============================================================================
testthat::test_that("lnr_earth handles formula order != data order with a factor covariate (regression: penguins bug)", {
  testthat::skip_if_not_installed("earth")
  dat <- make_continuous_dgp()
  # trailing column x2 first in the formula, factor in the middle
  p <- lnr_earth(dat, y ~ x2 + grp + x1 + x3)
  preds <- as.numeric(p(dat))
  r2 <- 1 - sum((dat$y - preds)^2) / sum((dat$y - mean(dat$y))^2)
  testthat::expect_gt(r2, 0.5)
  # and invariance under shuffling, specifically for earth
  testthat::expect_equal(as.numeric(p(dat[, rev(seq_len(ncol(dat)))])), preds,
                         tolerance = 1e-10)
})

