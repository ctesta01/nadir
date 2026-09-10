# test-crossfit_super_learner.R ---------------------------------------------
# Tests that pin down the properties modulatR depends on: row-order alignment,
# honesty of out-of-fold predictions, no bookkeeping leakage, and id handling.

library(testthat)

# Load {future.apply} up front: it otherwise loads lazily inside the first
# future_lapply() call, and any load-time warnings (e.g. "package 'future'
# was built under R version x.y.z") get relayed by future's condition
# machinery into whichever test happens to run first.
suppressWarnings(library(future.apply))

# deterministic outer split: fold of row r is ((r - 1) %% n_folds) + 1
cv_deterministic_schema <- function(data, n_folds) {
  fold <- ((seq_len(nrow(data)) - 1L) %% n_folds) + 1L
  list(
    training_data   = lapply(seq_len(n_folds), function(i) data[fold != i, , drop = FALSE]),
    validation_data = lapply(seq_len(n_folds), function(i) data[fold == i, , drop = FALSE])
  )
}

boston <- MASS::Boston[1:120, c("medv", "crim", "rm", "age")]

test_that("oof_predictions are full-length, in original row order", {
  set.seed(1)
  cf <- crossfit_super_learner(
    data = boston, formulas = medv ~ crim + rm + age,
    learners = list(lm = lnr_lm),
    n_folds = 4, cv_schema = cv_deterministic_schema
  )
  p <- cf$oof_predictions
  expect_length(p, nrow(boston))
  expect_false(anyNA(p))
  expect_setequal(cf$fold_assignments, 1:4)

  # shifting the outcome by a constant shifts every lm OOF prediction by it,
  # and must do so row-by-row -- this fails if reconstruction is
  # fold-concatenated rather than row-indexed.
  b2 <- boston; b2$medv <- b2$medv + 100
  set.seed(1)
  cf2 <- crossfit_super_learner(
    data = b2, formulas = medv ~ crim + rm + age,
    learners = list(lm = lnr_lm),
    n_folds = 4, cv_schema = cv_deterministic_schema
  )
  expect_equal(cf2$oof_predictions, p + 100, tolerance = 1e-8)
})

test_that("out-of-fold predictions are honest: perturbing fold i's outcomes
           does not move fold i's OOF predictions", {
  n_folds <- 4
  fold <- ((seq_len(nrow(boston)) - 1L) %% n_folds) + 1L

  set.seed(1)
  cf <- crossfit_super_learner(
    data = boston, formulas = medv ~ crim + rm + age,
    learners = list(lm = lnr_lm),   # single learner => weights are trivially 1
    n_folds = n_folds, cv_schema = cv_deterministic_schema
  )
  p <- cf$oof_predictions

  b_perturbed <- boston
  b_perturbed$medv[fold == 2] <- b_perturbed$medv[fold == 2] + 1000

  set.seed(1)
  cf_perturbed <- crossfit_super_learner(
    data = b_perturbed, formulas = medv ~ crim + rm + age,
    learners = list(lm = lnr_lm),
    n_folds = n_folds, cv_schema = cv_deterministic_schema
  )
  p2 <- cf_perturbed$oof_predictions

  # fold 2's predictor never saw fold 2: its predictions must be identical
  expect_equal(p2[fold == 2], p[fold == 2], tolerance = 1e-8)
  # every other fold trained on fold 2, so their predictions must move
  expect_gt(max(abs(p2[fold != 2] - p[fold != 2])), 1)
})

test_that("predict_modified(identity) reproduces oof_predictions", {
  set.seed(1)
  cf <- crossfit_super_learner(
    data = boston, formulas = medv ~ crim + rm + age,
    learners = list(lm = lnr_lm, mean = lnr_mean),
    n_folds = 4, cv_schema = cv_deterministic_schema
  )
  expect_equal(cf$oof_predict_modified(identity), cf$oof_predictions,
               tolerance = 1e-10)
})

test_that(".crossfit_rowid never reaches the learners", {
  lnr_paranoid <- function(data, formula, ...) {
    stopifnot(!".crossfit_rowid" %in% colnames(data))
    m <- stats::lm(formula, data)
    function(newdata) {
      stopifnot(!".crossfit_rowid" %in% colnames(newdata))
      stats::predict(m, newdata)
    }
  }
  attr(lnr_paranoid, "sl_lnr_name") <- "paranoid"
  attr(lnr_paranoid, "sl_lnr_type") <- "continuous"

  set.seed(1)
  expect_no_error({
    cf <- crossfit_super_learner(
      data = boston, formulas = medv ~ .,   # `.` would pick up a leaked rowid
      learners = list(paranoid = lnr_paranoid),
      n_folds = 3, cv_schema = cv_deterministic_schema
    )
    cf$oof_predictions
    cf$oof_predict_modified(function(d) { d$rm <- d$rm + 1; d })
  })
})

test_that("cluster_ids keep clusters intact across outer folds", {
  skip_if_not_installed("origami")
  cl <- rep(seq_len(30), each = 4)   # 30 clusters of 4 rows
  d <- boston[seq_along(cl), ]

  set.seed(1)
  cf <- crossfit_super_learner(
    data = d, formulas = medv ~ crim + rm,
    learners = list(lm = lnr_lm),
    n_folds = 3, cluster_ids = cl
  )
  # each cluster's rows must land in exactly one validation fold
  folds_per_cluster <- tapply(cf$fold_assignments, cl,
                              function(f) length(unique(f)))
  expect_true(all(folds_per_cluster == 1))
})

test_that("overlapping validation folds error; non-covering folds warn + NA", {
  cv_overlapping <- function(data, n_folds) {
    list(
      training_data   = replicate(n_folds, data[1:60, ], simplify = FALSE),
      validation_data = replicate(n_folds, data[61:120, ], simplify = FALSE)
    )
  }
  expect_error(
    crossfit_super_learner(
      data = boston, formulas = medv ~ crim,
      learners = list(lm = lnr_lm),
      n_folds = 2, cv_schema = cv_overlapping),
    "overlap"
  )

  cv_noncovering <- function(data, n_folds) {  # rolling-origin-like: row 1..20 never held out
    list(
      training_data   = list(data[1:60, ],   data[1:90, ]),
      validation_data = list(data[61:90, ],  data[91:120, ])
    )
  }
  expect_warning(
    cf <- crossfit_super_learner(
      data = boston, formulas = medv ~ crim,
      learners = list(lm = lnr_lm),
      n_folds = 2, cv_schema = cv_noncovering),
    "never appear"
  )
  p <- cf$oof_predictions
  expect_true(all(is.na(p[1:60])))
  expect_false(anyNA(p[61:120]))
})



# crossfit super learner helpers ------------------------------------------

fit_small_crossfit <- function() {
  set.seed(30)
  suppressWarnings(crossfit_super_learner(
    data = mtcars,
    formula = mpg ~ cyl + hp,
    n_folds = 3, inner_n_folds = 2,
    learners = list(mean = lnr_mean, lm = lnr_lm)))
}

test_that("predict() on a crossfit object errors with directions", {
  cf <- fit_small_crossfit()
  expect_error(predict(cf), "no single prediction function")
})

test_that("coef returns a folds-by-learners weight matrix (RE4.2)", {
  cf <- fit_small_crossfit()
  w <- coef(cf)
  expect_true(is.matrix(w))
  expect_equal(dim(w), c(3, 2))
  expect_setequal(colnames(w), c("mean", "lm"))
  expect_equal(unname(rowSums(w, na.rm = TRUE)), rep(1, 3),
               tolerance = 1e-6)
})

test_that("fitted and residuals align in original row order (RE4.9, RE4.10)", {
  cf <- fit_small_crossfit()
  f <- fitted(cf)
  r <- residuals(cf)
  expect_length(f, nrow(mtcars))
  expect_length(r, nrow(mtcars))
  expect_identical(f, cf$oof_predictions)
  # residuals = observed - oof, computed independently here from
  # fold_rows/validation_data
  y <- rep(NA_real_, nrow(mtcars))
  for (i in seq_len(cf$n_folds)) {
    y[cf$fold_rows[[i]]] <- cf$validation_data[[i]][[cf$y_variable]]
  }
  expect_equal(r, y - f)
})

test_that("formula and nobs accessors work (RE4.4, RE4.5)", {
  cf <- fit_small_crossfit()
  expect_equal(deparse(formula(cf)), deparse(mpg ~ cyl + hp))
  expect_identical(nobs(cf), nrow(mtcars))
})

test_that("summary reports fold losses and weight stability (RE4.18)", {
  cf <- fit_small_crossfit()
  s <- summary(cf)
  expect_s3_class(s, "summary.nadir_crossfit_sl")
  expect_equal(nrow(s$fold_losses), 3)
  expect_true(all(is.finite(s$fold_losses$loss)))
  expect_setequal(s$weight_stability$learner, c("mean", "lm"))
  expect_true(all(s$weight_stability$n_folds_present == 3))
  out <- capture.output(print(s))
  expect_true(any(grepl("weight stability", out)))
})

test_that("plot produces ggplot objects for both types (RE6.0-RE6.2)", {
  skip_if_not_installed("ggplot2")
  cf <- fit_small_crossfit()
  expect_s3_class(plot(cf), "ggplot")
  expect_s3_class(plot(cf, type = "weights"), "ggplot")
  expect_error(plot(cf, type = "nonsense"))
})

test_that("fitted-type plot and residuals refuse density outcomes clearly", {
  skip_if_not_installed("ggplot2")
  set.seed(31)
  cf_d <- suppressWarnings(suppressMessages(crossfit_super_learner(
    data = mtcars, formula = mpg ~ hp,
    n_folds = 2, inner_n_folds = 2,
    outcome_type = "density",
    learners = list(lmd = lnr_lm_density, glmd = lnr_glm_density))))
  expect_error(plot(cf_d, type = "fitted"), "not defined for outcome_type")
  expect_error(residuals(cf_d), "not defined for outcome_type")
  expect_s3_class(plot(cf_d, type = "weights"), "ggplot")
})

test_that("methods handle never-held-out rows (non-covering schemas)", {
  set.seed(32)
  boston_like <- data.frame(x = rnorm(120), y = rnorm(120))
  cv_noncovering <- function(data, n_folds) {
    list(training_data   = list(data[1:60, ],  data[1:90, ]),
         validation_data = list(data[61:90, ], data[91:120, ]))
  }
  cf <- suppressWarnings(crossfit_super_learner(
    data = boston_like, formulas = y ~ x,
    learners = list(lm = lnr_lm),
    n_folds = 2, inner_n_folds = 2, cv_schema = cv_noncovering))
  expect_true(all(is.na(fitted(cf)[1:60])))
  expect_true(all(is.na(residuals(cf)[1:60])))
  expect_false(anyNA(residuals(cf)[61:120]))
  s <- summary(cf)
  expect_equal(s$n_never_held_out, 60)
})
