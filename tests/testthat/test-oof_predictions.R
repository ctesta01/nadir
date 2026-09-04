suppressWarnings(library(future))

make_sim_data <- function(n = 500, seed = 20260904) {
  set.seed(seed)
  W1 <- rnorm(n); W2 <- rnorm(n)
  A  <- rbinom(n, 1, plogis(0.4 * W1 - 0.3 * W2))
  Y  <- rbinom(n, 1, plogis(-0.5 + A + 0.8 * W1 - 0.5 * W2))
  data.frame(W1 = W1, W2 = W2, A = A, Y = Y)
}

testthat::test_that("oof_predictions() equals weights applied to the stored holdout matrix", {
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

  testthat::expect_equal(sl$oof_predictions(), manual)
  # every row held out under default CV
  testthat::expect_false(anyNA(sl$oof_predictions()))
  # fold_assignments cover 1..n_folds
  testthat::expect_setequal(unique(sl$fold_assignments), 1:5)
})

testthat::test_that("oof_predict_modified(NULL) re-predicts and agrees with oof_predictions()", {
  # strong internal-consistency check: repredicting the validation folds with
  # the retained fold fits must reproduce the stage-2 holdout predictions
  d <- make_sim_data()
  sl <- super_learner(
    data = d, formulas = Y ~ A + W1 + W2,
    learners = list(mean = lnr_mean, glm = lnr_glm),
    outcome_type = "binary", n_folds = 5)

  testthat::expect_equal(
    sl$oof_predict_modified(NULL), sl$oof_predictions(), tolerance = 1e-10)
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

testthat::test_that("discrete super learner's oof_predictions() equals the top learner's OOF column", {
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
  testthat::expect_equal(sl$oof_predictions(), manual)
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
  QAW <- bound(Q_fit$oof_predictions())
  Q1W <- bound(Q_fit$oof_predict_modified(function(dd) { dd$A <- 1; dd }))
  Q0W <- bound(Q_fit$oof_predict_modified(function(dd) { dd$A <- 0; dd }))
  gW  <- bound(g_fit$oof_predictions())

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
