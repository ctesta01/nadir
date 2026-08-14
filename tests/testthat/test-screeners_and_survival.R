# Tests for screeners.R and survival_stacking.R

# ---- screener_cor ----------------------------------------------------------

test_that("screener_cor screens out weakly correlated variables", {
  out <- screener_cor(mtcars, mpg ~ ., threshold = 0.5)
  expect_named(out, c("data", "formula", "failed_to_correlate_names"),
               ignore.order = TRUE)
  expect_true("qsec" %in% out$failed_to_correlate_names)
  expect_false("qsec" %in% colnames(out$data))
  expect_true(inherits(out$formula, "formula"))

  # extra correlation arguments (spearman) are passed through
  out_sp <- screener_cor(mtcars, mpg ~ ., threshold = 0.5,
                         cor... = list(method = "spearman"))
  expect_true(inherits(out_sp$formula, "formula"))
})

test_that("screener_cor keeps everything at threshold 0", {
  out <- screener_cor(mtcars, mpg ~ ., threshold = 0)
  expect_null(out$failed_to_correlate_names)
  expect_equal(ncol(out$data), ncol(mtcars))
})

test_that("screener_cor warns when everything is screened out", {
  # a threshold above 1 screens out every variable; the resulting empty
  # right-hand-side is an error, but the warning fires first
  expect_error(
    expect_warning(
      screener_cor(mtcars, mpg ~ hp + wt, threshold = 1.1),
      "screened out all variables"
    )
  )
})

test_that("screener_cor errors when model.frame cannot parse the formula", {
  expect_error(
    screener_cor(mtcars, mpg ~ not_a_column, threshold = 0.5),
    "expects that it can use model.frame"
  )
})

# ---- screener_cor_top_n ----------------------------------------------------

test_that("screener_cor_top_n keeps the top n correlated variables", {
  out <- screener_cor_top_n(mtcars, mpg ~ ., keep_n_terms = 5)
  # 10 predictors, keep 5 -> 5 screened out
  expect_length(out$failed_to_correlate_names, 5)
  expect_equal(ncol(out$data), 6)  # y + 5 predictors

  out_sp <- screener_cor_top_n(mtcars, mpg ~ ., keep_n_terms = 5,
                               cor... = list(method = "spearman"))
  expect_length(out_sp$failed_to_correlate_names, 5)
})

test_that("screener_cor_top_n keeps everything when n exceeds the predictors", {
  out <- screener_cor_top_n(mtcars, mpg ~ hp + wt, keep_n_terms = 10)
  expect_null(out$failed_to_correlate_names)
  expect_equal(ncol(out$data), 3)
})

test_that("screener_cor_top_n errors with a single predictor or unparseable formula", {
  expect_error(
    screener_cor_top_n(mtcars, mpg ~ hp, keep_n_terms = 1),
    "<=1 terms"
  )
  expect_error(
    screener_cor_top_n(mtcars, mpg ~ not_a_column, keep_n_terms = 2),
    "expects that it can use model.frame"
  )
})

# ---- screener_t_test -------------------------------------------------------

test_that("screener_t_test screens on p-values and t statistics", {
  # p-value threshold only
  out_p <- screener_t_test(mtcars, mpg ~ ., p_value_threshold = 0.0001)
  expect_true(length(out_p$failed_to_pass_threshold) >= 1)

  # t statistic threshold only
  out_t <- screener_t_test(mtcars, mpg ~ ., t_statistic_threshold = 8)
  expect_true(length(out_t$failed_to_pass_threshold) >= 1)

  # both thresholds at once
  out_both <- screener_t_test(mtcars, mpg ~ .,
                              p_value_threshold = 0.05,
                              t_statistic_threshold = 2)
  expect_s3_class(out_both$data, "data.frame")

  # lenient thresholds screen out nothing
  out_none <- screener_t_test(mtcars, mpg ~ hp + wt, p_value_threshold = 1)
  expect_false("failed_to_pass_threshold" %in% names(out_none))
})

test_that("screener_t_test validates its inputs", {
  expect_error(
    screener_t_test(mtcars, mpg ~ .),
    "At least one of the p_value_threshold or t_statistic_threshold"
  )
  expect_error(
    screener_t_test(mtcars, mpg ~ not_a_column, p_value_threshold = 0.05),
    "expects that it can use model.frame"
  )
})

# ---- add_screener ----------------------------------------------------------

test_that("add_screener wraps a learner with a screening stage", {
  lnr_screened <- add_screener(
    learner = lnr_lm,
    screener = screener_cor,
    screener_extra_args = list(threshold = 0.6)
  )
  expect_equal(attr(lnr_screened, "sl_lnr_name"), "cor_threshold_screened_lm")
  expect_equal(attr(lnr_screened, "sl_lnr_type"), attr(lnr_lm, "sl_lnr_type"))

  predictor <- lnr_screened(mtcars, formula = mpg ~ .)
  expect_length(predictor(mtcars), nrow(mtcars))

  # screened-out variables no longer affect predictions
  mtcars_qsec_changed <- mtcars
  mtcars_qsec_changed$qsec <- 0
  expect_equal(predictor(mtcars), predictor(mtcars_qsec_changed))

  # extra learner arguments pass through the ... path; lnr_glm (unlike
  # lnr_lm) actually accepts `family`, so this exercises the passthrough
  # without lm.fit's "extra argument disregarded" warning
  lnr_screened_glm <- add_screener(
    learner = lnr_glm,
    screener = screener_cor,
    screener_extra_args = list(threshold = 0.6)
  )
  predictor2 <- lnr_screened_glm(mtcars, formula = mpg ~ ., family = "gaussian")
  expect_length(predictor2(mtcars), nrow(mtcars))
})

test_that("add_screener falls back to default names when attributes are missing", {
  anonymous_screener <- function(data, formula, ...) {
    screener_cor(data, formula, threshold = 0.6)
  }
  anonymous_learner <- function(data, formula, ...) lnr_lm(data, formula)

  lnr_screened <- add_screener(anonymous_learner, anonymous_screener)
  expect_equal(attr(lnr_screened, "sl_lnr_name"), "screened_unnamed_lnr")
  predictor <- lnr_screened(mtcars, formula = mpg ~ .)
  expect_length(predictor(mtcars), nrow(mtcars))
})

# ---- df_to_survival_stacked ------------------------------------------------

surv_df <- data.frame(
  id = 1:6,
  time = c(1, 2.5, 3, 4, 2, 5),
  status = c(1, 1, 0, 1, 0, 1),
  age = c(50, 60, 55, 45, 65, 70),
  sex = c(0, 1, 0, 1, 0, 1)
)

test_that("df_to_survival_stacked repeats observations per risk period", {
  out <- df_to_survival_stacked(
    data = surv_df,
    id_col = "id",
    time_col = "time",
    status_col = "status",
    covariate_cols = c("age", "sex"),
    period_duration = 1
  )
  expect_s3_class(out, "data.frame")
  expect_true(all(c("id", "t", "event", "age", "sex") %in% colnames(out)))
  # each subject with an event has exactly one event row
  expect_equal(sum(out$event[out$id == 1]), 1)
  # censored subjects have no event rows
  expect_equal(sum(out$event[out$id == 3]), 0)
  # subject 1 (event at t = 1, on a cutoff) is observed at t = 0 and t = 1
  expect_equal(out$t[out$id == 1], c(0, 1))
  # subject 2 (event at t = 2.5, between cutoffs) gets a partial period row
  # in addition to the three complete periods starting at 0, 1, and 2
  expect_equal(out$t[out$id == 2], c(0, 1, 2, 3))
})

test_that("df_to_survival_stacked creates an id column when none is given", {
  out <- df_to_survival_stacked(
    data = surv_df,
    time_col = "time",
    status_col = "status",
    covariate_cols = c("age")
  )
  expect_true(".id" %in% colnames(out))
})

test_that("df_to_survival_stacked handles non-integer max times", {
  # max time 5 with period 0.75: max time is not a multiple of the period,
  # exercising the rounding-up branch
  out <- df_to_survival_stacked(
    data = surv_df,
    id_col = "id",
    time_col = "time",
    status_col = "status",
    covariate_cols = c("age"),
    period_duration = 0.75
  )
  expect_s3_class(out, "data.frame")
})

test_that("df_to_survival_stacked warns about questionable custom_times", {
  expect_warning(
    df_to_survival_stacked(
      data = surv_df,
      id_col = "id",
      time_col = "time",
      status_col = "status",
      covariate_cols = c("age"),
      custom_times = c(1, 2, 3, 4, 5)
    ),
    "does not begin with 0"
  )
  expect_warning(
    df_to_survival_stacked(
      data = surv_df,
      id_col = "id",
      time_col = "time",
      status_col = "status",
      covariate_cols = c("age"),
      custom_times = c(0, 1, 2, 3)
    ),
    "less than the maximum time"
  )
})
