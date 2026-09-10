suppressWarnings(library(future))


# tests for $oof_predictions and $oof_predict_modified() ------------------



make_sim_data <- function(n = 500, seed = 20260904) {
  set.seed(seed)
  W1 <- rnorm(n); W2 <- rnorm(n)
  A  <- rbinom(n, 1, plogis(0.4 * W1 - 0.3 * W2))
  Y  <- rbinom(n, 1, plogis(-0.5 + A + 0.8 * W1 - 0.5 * W2))
  data.frame(W1 = W1, W2 = W2, A = A, Y = Y)
}

testthat::test_that("oof_predictions equals weights applied to the stored holdout matrix", {
  d <- make_sim_data()
  sl <- super_learner(
    data = d, formulas = Y ~ A + W1 + W2,
    learners = list(mean = lnr_mean, glm = lnr_glm, rf = lnr_rf),
    outcome_type = "binary", n_folds = 5)

  hp <- sl$holdout_predictions
  learner_cols <- setdiff(colnames(hp), c(".sl_fold", ".sl_rowid", "Y"))
  manual <- rep(NA_real_, nrow(d))
  manual[hp$.sl_rowid] <-
    as.numeric(as.matrix(hp[, learner_cols]) %*% sl$learner_weights[learner_cols])

  testthat::expect_equal(sl$oof_predictions, manual)
  # every row held out under default CV
  testthat::expect_false(anyNA(sl$oof_predictions))
  # fold_assignments cover 1..n_folds
  testthat::expect_setequal(unique(sl$fold_assignments), 1:5)
})

testthat::test_that("oof_predict_modified(NULL) re-predicts and agrees with oof_predictions", {
  # strong internal-consistency check: repredicting the validation folds with
  # the retained fold fits must reproduce the stage-2 holdout predictions
  d <- make_sim_data()
  sl <- super_learner(
    data = d, formulas = Y ~ A + W1 + W2,
    learners = list(mean = lnr_mean, glm = lnr_glm),
    outcome_type = "binary", n_folds = 5)

  testthat::expect_equal(
    sl$oof_predict_modified(NULL), sl$oof_predictions, tolerance = 1e-10)
})

testthat::test_that("oof_predict_modified() respects the modification (out-of-fold Q(1,W) vs Q(0,W))", {
  d <- make_sim_data()
  sl <- super_learner(
    data = d, formulas = Y ~ A + W1 + W2,
    learners = list(glm = lnr_glm),
    outcome_type = "binary", n_folds = 5)

  Q1 <- sl$oof_predict_modified(function(dd) { dd$A <- 1; dd })
  Q0 <- sl$oof_predict_modified(function(dd) { dd$A <- 0; dd })
  # A has a strongly positive coefficient in the DGP
  testthat::expect_true(mean(Q1 - Q0) > 0)
  # per-observation predictions from a single glm must differ between arms
  testthat::expect_true(all(Q1 != Q0))
})

testthat::test_that("discrete super learner's oof_predictions equals the top learner's OOF column", {
  d <- make_sim_data()
  sl <- super_learner(
    data = d, formulas = Y ~ A + W1 + W2,
    learners = list(mean = lnr_mean, glm = lnr_glm),
    outcome_type = "binary", n_folds = 5,
    ensemble_or_discrete = "discrete")

  top <- names(sl$learner_weights)[sl$learner_weights == 1]
  hp <- sl$holdout_predictions
  manual <- rep(NA_real_, nrow(d))
  manual[hp$.sl_rowid] <- hp[[top]]
  testthat::expect_equal(sl$oof_predictions, manual)
})

testthat::test_that("shared-fold CV-TMLE recipe runs and epsilon solves the EIC equation", {
  # end-to-end check of the recipe in the patch notes: pooled fluctuation on
  # OOF values; after the update the empirical mean of the EIC is ~ 0
  d <- make_sim_data(n = 800)
  V <- 5
  fold_id <- sample(rep(seq_len(V), length.out = nrow(d)))
  schema <- function(data, n_folds) {
    list(
      training_data   = lapply(seq_len(n_folds), function(v) data[fold_id != v, , drop = FALSE]),
      validation_data = lapply(seq_len(n_folds), function(v) data[fold_id == v, , drop = FALSE]))
  }
  bound <- function(x, l = 0.005) pmin(pmax(x, l), 1 - l)

  Q_fit <- super_learner(d, formulas = Y ~ A + W1 + W2,
                         learners = list(glm = lnr_glm, mean = lnr_mean),
                         outcome_type = "binary", n_folds = V, cv_schema = schema)
  g_fit <- super_learner(d, formulas = A ~ W1 + W2,
                         learners = list(glm = lnr_glm, mean = lnr_mean),
                         outcome_type = "binary", n_folds = V, cv_schema = schema)

  A <- d$A; Y <- d$Y
  QAW <- bound(Q_fit$oof_predictions)
  Q1W <- bound(Q_fit$oof_predict_modified(function(dd) { dd$A <- 1; dd }))
  Q0W <- bound(Q_fit$oof_predict_modified(function(dd) { dd$A <- 0; dd }))
  gW  <- bound(g_fit$oof_predictions)

  H   <- A / gW - (1 - A) / (1 - gW)
  eps <- coef(glm(Y ~ -1 + H, offset = qlogis(QAW), family = binomial()))
  Q1s <- plogis(qlogis(Q1W) + eps / gW)
  Q0s <- plogis(qlogis(Q0W) - eps / (1 - gW))
  QAs <- plogis(qlogis(QAW) + eps * H)

  psi <- mean(Q1s - Q0s)
  IC  <- H * (Y - QAs) + (Q1s - Q0s) - psi

  # the logistic fluctuation's score equation implies mean(H * (Y - QAs)) ~ 0,
  # hence mean(IC) ~ 0
  testthat::expect_lt(abs(mean(IC)), 1e-8)
  # sanity: estimate is in a plausible range for this DGP (true ATE ~ 0.21)
  testthat::expect_gt(psi, 0.05)
  testthat::expect_lt(psi, 0.4)
})



# tests for $oof_predict --------------------------------------------------

# tests for $oof_predict(), rowids, and the oof_predictions vector conversion

fit_sl <- function(rowids = NULL) {
  super_learner(
    data = mtcars,
    formulas = mpg ~ cyl + hp,
    learners = list(lm = lnr_lm, glm = lnr_glm),
    rowids = rowids)
}

test_that("oof_predictions is a stored numeric vector in input-row order", {
  set.seed(1)
  sl <- fit_sl()
  expect_true(is.numeric(sl$oof_predictions))
  expect_false(is.function(sl$oof_predictions))
  expect_length(sl$oof_predictions, nrow(mtcars))
  expect_false(anyNA(sl$oof_predictions))
  # agrees with the holdout matrix mapped back manually (positional .sl_rowid)
  hp <- sl$holdout_predictions
  manual <- rep(NA_real_, nrow(mtcars))
  manual[hp$.sl_rowid] <- as.numeric(
    as.matrix(hp[, names(sl$learner_weights)]) %*% sl$learner_weights)
  expect_equal(sl$oof_predictions, manual, tolerance = 1e-10)
})

test_that(".sl_rowid bookkeeping stays positional even with user rowids", {
  set.seed(1)
  sl <- fit_sl(rowids = rownames(mtcars))
  expect_true(is.numeric(sl$holdout_predictions$.sl_rowid))
  expect_setequal(sl$holdout_predictions$.sl_rowid, seq_len(nrow(mtcars)))
  expect_identical(sl$rowids, rownames(mtcars))
})

test_that("oof_predict() with no arguments returns the stored vector", {
  set.seed(1)
  sl <- fit_sl()
  expect_identical(sl$oof_predict(), sl$oof_predictions)
})

test_that("positional matching warns once (classed) and agrees with stored OOF", {
  set.seed(1)
  sl <- fit_sl()
  expect_warning(
    p1 <- sl$oof_predict(mtcars))
  # second call on the same object: no further warning
  expect_no_warning(p2 <- sl$oof_predict(mtcars))
  expect_equal(p1, sl$oof_predictions, tolerance = 1e-8)
  expect_equal(p1, p2, tolerance = 1e-12)
})

test_that("positional path errors on wrong nrow", {
  set.seed(1)
  sl <- fit_sl()
  expect_error(sl$oof_predict(head(mtcars, 5)), "rowids")
})

test_that("predict-time rowids allow shuffled subsets without warnings", {
  set.seed(1)
  sl <- fit_sl()
  idx <- sample(nrow(mtcars), 10)
  expect_no_warning(p <- sl$oof_predict(mtcars[idx, ], rowids = idx))
  expect_equal(p, sl$oof_predictions[idx], tolerance = 1e-8)
})

test_that("fit-time rowids: required at predict time, matched by id", {
  set.seed(1)
  ids <- rownames(mtcars)  # character ids
  sl <- fit_sl(rowids = ids)
  # rowids required
  expect_error(sl$oof_predict(mtcars), "explicit rowids")
  # shuffled ids come back aligned
  idx <- sample(nrow(mtcars))
  p <- sl$oof_predict(mtcars[idx, ], rowids = ids[idx])
  expect_equal(p, sl$oof_predictions[idx], tolerance = 1e-8)
  # unknown id errors and is named in the message
  bad <- mtcars[1:2, ]
  expect_error(sl$oof_predict(bad, rowids = c(ids[1], "not_a_car")),
               "not_a_car")
})

test_that("fitted() is unaffected by user rowids (positional internals)", {
  set.seed(1)
  sl <- fit_sl(rowids = rownames(mtcars))
  expect_equal(fitted(sl), sl$oof_predictions, tolerance = 1e-12)
})

test_that("validate_rowids rejects bad inputs; accepts and coerces factors", {
  n <- nrow(mtcars)
  expect_error(fit_sl(rowids = as.list(1:n)), "list")
  expect_error(fit_sl(rowids = 1:5), "length")
  expect_error(fit_sl(rowids = rep(1, n)), "duplicates")
  expect_error(fit_sl(rowids = c(NA, 2:n)), "NA")
  sl <- fit_sl(rowids = factor(rownames(mtcars)))
  expect_type(sl$rowids, "character")
})

test_that("oof_predict_modified equals oof_predict on modified data", {
  set.seed(1)
  sl <- fit_sl()
  m <- function(d) { d$hp <- d$hp + 10; d }
  a <- sl$oof_predict_modified(m)
  b <- suppressWarnings(sl$oof_predict(m(mtcars)))
  expect_equal(a, b, tolerance = 1e-8)
  # row-dropping modify errors
  expect_error(sl$oof_predict_modified(function(d) d[-1, ]),
               "number of rows")
})

test_that("formula scanning: bookkeeping names error; id-like columns warn", {
  df <- mtcars
  df$my_id <- seq_len(nrow(df))
  expect_error(
    super_learner(df, formulas = mpg ~ .sl_rowid + hp,
                  learners = list(lm = lnr_lm, glm = lnr_glm)),
    "bookkeeping")
  expect_warning({
    super_learner(df, formulas = mpg ~ .,
                  learners = list(lm = lnr_lm, glm = lnr_glm),
                  rowids = df$my_id)})
  expect_no_warning(
    super_learner(df, formulas = mpg ~ cyl + hp,
                  learners = list(lm = lnr_lm, glm = lnr_glm),
                  rowids = df$my_id))
})

test_that("crossfit: vector oof_predictions, oof_predict, aliases", {
  set.seed(2)
  cf <- crossfit_super_learner(
    data = mtcars,
    formulas = mpg ~ cyl + hp,
    learners = list(lm = lnr_lm, glm = lnr_glm),
    n_folds = 3, inner_n_folds = 3,
    rowids = rownames(mtcars))
  expect_false(is.function(cf$oof_predictions))
  expect_length(cf$oof_predictions, nrow(mtcars))
  # rowids required at predict time
  expect_error(cf$oof_predict(mtcars), "explicit rowids")
  idx <- sample(nrow(mtcars), 8)
  p <- cf$oof_predict(mtcars[idx, ], rowids = rownames(mtcars)[idx])
  expect_equal(p, cf$oof_predictions[idx], tolerance = 1e-8)
})

test_that("cv_super_learner accepts and threads rowids", {
  set.seed(3)
  cvr <- suppressMessages(cv_super_learner(
    data = mtcars,
    formulas = mpg ~ cyl + hp,
    learners = list(lm = lnr_lm, glm = lnr_glm),
    n_folds = 3,
    rowids = rownames(mtcars)))
  expect_true(!is.null(cvr$cv_loss))
})
