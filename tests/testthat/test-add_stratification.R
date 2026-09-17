# tests for add_stratification() and the condition-synopsis print method.
#
# trained objects are reused across assertions within a block, and each block
# covers one region of the code (validation, training/prediction correctness,
# formula checking, fallback machinery, column validation, super_learner()
# integration, printing).

# ---- construction-time validation and attributes (no model fits) ------------

test_that("add_stratification validates its arguments and sets attributes", {
  expect_error(add_stratification("not a function", "cyl"),
               "expects `learner` to be a function")
  expect_error(add_stratification(lnr_lm, stratify_by = 1),
               "must be a character vector")
  expect_error(add_stratification(lnr_lm, "cyl", min_stratum_size = 0),
               "min_stratum_size")
  expect_error(add_stratification(lnr_lm, "cyl", pooled_fallback = NA),
               "pooled_fallback")

  # density/multiclass-only learners are unsupported (their predictors do
  # not return one numeric value per row)
  lnr_density_only <- function(data, formula, ...) lnr_lm(data, formula)
  attr(lnr_density_only, "sl_lnr_type") <- "density"
  expect_error(add_stratification(lnr_density_only, "cyl"),
               "'continuous' and/or 'binary'")

  # attributes: name, type, multi-variable names, unnamed-learner fallback,
  # and outcome_type_dependent_args passthrough (so e.g. a stratified lnr_glm
  # still receives family = 'binomial' under outcome_type = 'binary')
  lnr <- add_stratification(lnr_lm, "cyl")
  expect_equal(attr(lnr, "sl_lnr_name"), "lm_stratified_by_cyl")
  expect_equal(attr(lnr, "sl_lnr_type"), c("continuous", "binary"))
  expect_equal(
    attr(add_stratification(lnr_lm, c("cyl", "am")), "sl_lnr_name"),
    "lm_stratified_by_cyl_am")

  lnr_anon <- function(data, formula, weights = NULL, ...) lnr_lm(data, formula)
  attr(lnr_anon, "sl_lnr_type") <- c("continuous", "binary")
  attr(lnr_anon, "outcome_type_dependent_args") <-
    list(binary = list(family = "binomial"))
  lnr_anon_stratified <- add_stratification(lnr_anon, "cyl")
  expect_equal(attr(lnr_anon_stratified, "sl_lnr_name"),
               "unnamed_lnr_stratified_by_cyl")
  expect_equal(attr(lnr_anon_stratified, "outcome_type_dependent_args"),
               list(binary = list(family = "binomial")))
})

# ---- training/prediction correctness -----------------------------------------

test_that("stratified fits equal manual per-stratum fits, subset weights, and preserve row order", {
  set.seed(2)
  obs_weights <- runif(nrow(mtcars))
  lnr <- add_stratification(lnr_lm, "cyl", min_stratum_size = 2)

  # one weighted fit covers per-stratum exactness AND per-stratum weights
  # subsetting at once
  expect_no_warning(
    trained <- lnr(mtcars, mpg ~ hp + wt, weights = obs_weights))
  preds <- trained(mtcars)
  expect_length(preds, nrow(mtcars))
  for (s in unique(mtcars$cyl)) {
    rows <- mtcars$cyl == s
    stratum_weights <- obs_weights[rows]
    manual_fit <- lm(mpg ~ hp + wt, data = mtcars[rows, , drop = FALSE],
                     weights = stratum_weights)
    expect_equal(
      unname(preds[rows]),
      unname(predict(manual_fit, newdata = mtcars[rows, , drop = FALSE])))
  }
  expect_error(lnr(mtcars, mpg ~ hp + wt, weights = obs_weights[1:3]),
               "length nrow")

  # row order: reuse the trained predictor, no new fits needed
  shuffled_rows <- sample(nrow(mtcars))
  expect_equal(trained(mtcars[shuffled_rows, , drop = FALSE]),
               preds[shuffled_rows])

  # multi-variable stratification stratifies by the interaction; the
  # per-stratum machinery is proven above, so verifying one cell (the
  # smallest, cyl == 8 & am == 1, n = 2) suffices to pin the key logic
  lnr_two <- add_stratification(lnr_lm, c("cyl", "am"), min_stratum_size = 2)
  expect_no_warning(trained_two <- lnr_two(mtcars, mpg ~ hp))
  preds_two <- trained_two(mtcars)
  cell <- mtcars$cyl == 8 & mtcars$am == 1
  cell_fit <- lm(mpg ~ hp, data = mtcars[cell, , drop = FALSE])
  expect_equal(
    unname(preds_two[cell]),
    unname(predict(cell_fit, newdata = mtcars[cell, , drop = FALSE])))
})

# ---- aggressive formula checking ---------------------------------------------

test_that("formulas mentioning the stratifying variable error aggressively", {
  lnr <- add_stratification(lnr_lm, "cyl", min_stratum_size = 2)
  # plain mention, and a mention hidden inside a transformation (poly()
  # exercises the same term-parsing path as I(), interactions, and (1 | x))
  expect_error(lnr(mtcars, mpg ~ hp + cyl), "mentions the stratifying variable")
  expect_error(lnr(mtcars, mpg ~ poly(cyl, 2) + hp),
               "mentions the stratifying variable")
  # y ~ . expands to include the stratifying variable and the error suggests
  # the `y ~ . - var` fix verbatim ...
  expect_error(lnr(mtcars, mpg ~ .), "mpg ~ . - cyl", fixed = TRUE)
  # ... which works:
  expect_length(lnr(mtcars, mpg ~ . - cyl)(mtcars), nrow(mtcars))
})

# ---- pooled fallback machinery -----------------------------------------------

test_that("small, erring, and unseen strata fall back to the pooled fit (or error)", {
  # (a) too-small stratum: cyl == 6 has 7 rows, below min_stratum_size = 10
  lnr <- add_stratification(lnr_lm, "cyl", min_stratum_size = 10)
  expect_warning(trained <- lnr(mtcars, mpg ~ hp + wt),
                 "fewer than min_stratum_size")
  # strata that fell back at training time do not re-warn at prediction time
  expect_no_warning(preds <- trained(mtcars))
  pooled <- lnr_lm(mtcars, mpg ~ hp + wt)
  rows_small <- mtcars$cyl == 6
  expect_equal(unname(preds[rows_small]),
               unname(pooled(mtcars[rows_small, , drop = FALSE])))
  # adequately-sized strata still use their stratum-specific fits
  rows_big <- mtcars$cyl == 8
  big_fit <- lm(mpg ~ hp + wt, data = mtcars[rows_big, , drop = FALSE])
  expect_equal(
    unname(preds[rows_big]),
    unname(predict(big_fit, newdata = mtcars[rows_big, , drop = FALSE])))

  # (b) erring stratum fit: falls back with a warning naming the error
  lnr_fails_on_4cyl <- function(data, formula, weights = NULL, ...) {
    if (all(data$cyl == 4)) stop("cannot fit this stratum")
    lnr_lm(data, formula, weights = weights, ...)
  }
  attr(lnr_fails_on_4cyl, "sl_lnr_type") <- "continuous"
  lnr_err <- add_stratification(lnr_fails_on_4cyl, "cyl", min_stratum_size = 2)
  expect_warning(trained_err <- lnr_err(mtcars, mpg ~ hp + wt),
                 "stratum fits erred")
  rows4 <- mtcars$cyl == 4
  expect_equal(unname(trained_err(mtcars)[rows4]),
               unname(pooled(mtcars[rows4, , drop = FALSE])))

  # (c) stratum unseen at training time: pooled prediction with a warning
  training_data <- mtcars[mtcars$cyl != 8, , drop = FALSE]
  lnr2 <- add_stratification(lnr_lm, "cyl", min_stratum_size = 2)
  trained_sub <- lnr2(training_data, mpg ~ hp + wt)
  expect_warning(preds_sub <- trained_sub(mtcars), "unseen at training")
  pooled_sub <- lnr_lm(training_data, mpg ~ hp + wt)
  expect_equal(unname(preds_sub[rows_big]),
               unname(pooled_sub(mtcars[rows_big, , drop = FALSE])))

  # (d) with pooled_fallback = FALSE, each of (a)-(c) is an error instead
  expect_error(
    add_stratification(lnr_lm, "cyl", min_stratum_size = 10,
                       pooled_fallback = FALSE)(mtcars, mpg ~ hp + wt),
    "pooled_fallback = FALSE")
  expect_error(
    add_stratification(lnr_fails_on_4cyl, "cyl", min_stratum_size = 2,
                       pooled_fallback = FALSE)(mtcars, mpg ~ hp + wt),
    "cannot fit this stratum")
  trained_strict <- add_stratification(
    lnr_lm, "cyl", min_stratum_size = 2,
    pooled_fallback = FALSE)(training_data, mpg ~ hp + wt)
  expect_error(trained_strict(mtcars), "pooled_fallback = FALSE")
})

# ---- stratifying column validation -------------------------------------------

test_that("stratifying columns must exist, be non-missing, and be categorical-ish", {
  # continuous (non-integer numeric) columns are rejected; factor, character,
  # logical, and integer-valued numeric are accepted. one fit covers logical
  # and character via multi-variable stratification (factor and integer
  # numeric are already exercised throughout via cyl / cyl-derived keys)
  expect_error(add_stratification(lnr_lm, "wt")(mtcars, mpg ~ hp),
               "appears to be continuous")
  augmented <- mtcars
  augmented$am_logical <- as.logical(augmented$am)
  augmented$gear_chr <- as.character(augmented$gear)
  augmented$cyl_fct <- factor(augmented$cyl)
  lnr_types <- add_stratification(
    lnr_lm, c("am_logical", "gear_chr", "cyl_fct"), min_stratum_size = 1)
  expect_length(lnr_types(augmented, mpg ~ hp)(augmented), nrow(augmented))

  # missing columns and NA stratum values error, at training and prediction
  expect_error(add_stratification(lnr_lm, "not_a_column")(mtcars, mpg ~ hp),
               "must appear as")
  with_na <- mtcars
  with_na$cyl[1] <- NA
  expect_error(add_stratification(lnr_lm, "cyl")(with_na, mpg ~ hp),
               "contains missing values")
  trained <- add_stratification(lnr_lm, "cyl", min_stratum_size = 2)(
    mtcars, mpg ~ hp + wt)
  expect_error(trained(mtcars[, c("mpg", "hp", "wt"), drop = FALSE]),
               "must appear as")
})

# ---- inside super_learner() ---------------------------------------------------

test_that("stratified learners work inside super_learner()", {
  set.seed(3)
  lnr_lm_by_cyl <- add_stratification(lnr_lm, "cyl", min_stratum_size = 2)
  # passing the stratifying variable as strata_ids (the recommended usage)
  # keeps every stratum represented in every training fold
  expect_no_warning({
    sl <- super_learner(
      data = mtcars,
      formulas = mpg ~ hp + wt,
      learners = list(mean = lnr_mean, lm_by_cyl = lnr_lm_by_cyl),
      strata_ids = mtcars$cyl,
      n_folds = 2)
  })
  expect_setequal(names(sl$learner_weights), c("mean", "lm_by_cyl"))
  preds <- sl$predict(mtcars)
  expect_length(preds, nrow(mtcars))
  expect_true(all(is.finite(preds)))
})

# ---- condition_synopsis and the print method ----------------------------------

test_that("print.nadir_sl_model shows a deduplicated, elided synopsis of captured conditions", {
  x <- structure(list(
    y_variable = "mpg", outcome_type = "continuous", n_obs = 32, n_folds = 5,
    learner_weights = c(lm = 0.7, mean = 0.3),
    errors_from_training_cv_stage1 = list(bad = simpleError("nope, cannot fit")),
    erring_learners = "bad",
    warnings_from_training_cv_stage1 = stats::setNames(
      rep(list(simpleWarning("this learner is grumpy")), 5), rep("grumpy", 5)),
    warning_learners = "grumpy"
  ), class = "nadir_sl_model")

  printed <- paste(capture.output(print(x)), collapse = "\n")
  expect_match(printed, "captured conditions")
  expect_match(printed, "\\[error\\] bad @ cv-training: nope, cannot fit")
  # the same warning signaled once per fold is deduplicated with a count
  expect_match(printed,
               "\\[warning\\] grumpy @ cv-training: this learner is grumpy \\(x5\\)")
  expect_match(printed, "learners dropped due to errors: bad")

  # synopsis internals: errors sort first, long messages truncate, overflow
  # past max_lines is elided with a count, and no conditions -> no lines
  y <- list(
    warnings_from_training_cv_stage1 = stats::setNames(
      lapply(1:6, function(i) simpleWarning(paste("warning number", i))),
      rep("w", 6)),
    errors_from_training_on_entire_data = list(
      bad = simpleError(paste(rep("verylongword", 20), collapse = " "))))
  synopsis_lines <- condition_synopsis(y, max_lines = 4)
  expect_length(synopsis_lines, 5)                    # 4 shown + 1 elision line
  expect_match(synopsis_lines[1], "^    \\[error\\]") # errors before warnings
  expect_match(synopsis_lines[1], "\\.\\.\\.")        # truncation
  expect_match(synopsis_lines[5], "and 3 more unique conditions")
  expect_length(condition_synopsis(list()), 0)

  # print output is unchanged when nothing was captured
  clean <- structure(list(
    y_variable = "mpg", outcome_type = "continuous", n_obs = 32, n_folds = 5,
    learner_weights = c(lm = 1)), class = "nadir_sl_model")
  expect_false(any(grepl("captured conditions", capture.output(print(clean)))))
})
