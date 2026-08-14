# Tests for cv_schemas.R

test_that("cv_random_schema partitions data into folds", {
  set.seed(1)
  out <- cv_random_schema(mtcars, n_folds = 3)
  expect_named(out, c("training_data", "validation_data"))
  expect_length(out$training_data, 3)
  expect_length(out$validation_data, 3)
  expect_equal(sum(sapply(out$validation_data, nrow)), nrow(mtcars))
  # training + validation fold sizes each sum to nrow
  expect_true(all(
    sapply(1:3, function(i) {
      nrow(out$training_data[[i]]) + nrow(out$validation_data[[i]]) == nrow(mtcars)
    })
  ))
})

test_that("cv_random_schema accepts matrices and validates inputs", {
  set.seed(1)
  m <- as.matrix(mtcars)
  out <- cv_random_schema(m, n_folds = 2)
  expect_true(is.data.frame(out$training_data[[1]]))

  expect_error(cv_random_schema(mtcars, n_folds = c(2, 3)), "length 1 numeric")

  bad <- mtcars
  bad$.sl_fold <- 1
  expect_error(cv_random_schema(bad, n_folds = 2), "already has a .sl_fold column")
})

test_that("cv_character_and_factors_schema validates its inputs", {
  expect_error(
    cv_character_and_factors_schema(mtcars, n_folds = c(2, 3)),
    "length 1 numeric"
  )

  # no character/factor columns
  expect_error(
    cv_character_and_factors_schema(mtcars, n_folds = 2),
    "must be character/factor column types"
  )

  # constant factor column
  d_const <- data.frame(y = rnorm(10), g = rep("a", 10))
  expect_error(
    cv_character_and_factors_schema(d_const, n_folds = 2),
    "constant"
  )

  # a level that appears exactly once
  d_once <- data.frame(y = rnorm(10), g = c(rep("a", 9), "b"))
  expect_error(
    cv_character_and_factors_schema(d_once, n_folds = 2),
    "only appear once"
  )

  # levels appearing exactly twice with check_validation_datasets_too = TRUE
  d_two <- data.frame(y = rnorm(8), g = rep(c("a", "b"), each = 4))
  d_two$g[1:2] <- "c"  # c appears twice
  expect_error(
    cv_character_and_factors_schema(d_two, n_folds = 2,
                                    check_validation_datasets_too = TRUE),
    "2 or fewer times"
  )

  # levels appearing exactly three times with cv_sl_mode and validation checks
  d_three <- data.frame(y = rnorm(9), g = rep(c("a", "b", "c"), each = 3))
  expect_error(
    cv_character_and_factors_schema(d_three, n_folds = 2, cv_sl_mode = TRUE,
                                    check_validation_datasets_too = TRUE),
    "3 or fewer times"
  )
})

test_that("cv_character_and_factors_schema produces valid splits", {
  set.seed(2)
  d <- data.frame(y = rnorm(60), g = rep(c("a", "b", "c"), each = 20))
  out <- cv_character_and_factors_schema(d, n_folds = 3)
  expect_length(out$training_data, 3)
  # every level appears in every training split
  expect_true(all(sapply(out$training_data, function(df) {
    all(c("a", "b", "c") %in% df$g)
  })))
  # cv_sl_mode = FALSE path with no validation checking
  set.seed(2)
  out2 <- cv_character_and_factors_schema(
    d, n_folds = 3, cv_sl_mode = FALSE, check_validation_datasets_too = FALSE
  )
  expect_length(out2$validation_data, 3)
})

test_that("cv_character_and_factors_schema messages after repeated resampling", {
  # a configuration where valid splits are rare, forcing 5+ resamples:
  # 3 levels appearing 5 times each, 5 folds, and validation checks on
  d <- data.frame(y = seq_len(15), g = rep(c("a", "b", "c"), each = 5))
  set.seed(1)
  msgs <- character(0)
  out <- withCallingHandlers(
    cv_character_and_factors_schema(
      d, n_folds = 5, cv_sl_mode = FALSE, check_validation_datasets_too = TRUE
    ),
    message = function(m) {
      msgs <<- c(msgs, conditionMessage(m))
      invokeRestart("muffleMessage")
    }
  )
  expect_true(any(grepl("5\\+ cross-validation splits", msgs)))
  expect_true(any(grepl("Successfully generated splits", msgs)))
  expect_length(out$training_data, 5)
})

test_that("cv_origami_schema validates inputs and splits data", {
  expect_error(cv_origami_schema(mtcars, n_folds = c(2, 3)), "length 1 numeric")
  expect_error(
    cv_origami_schema(mtcars, n_folds = 2, cluster_ids = c(1, 2)),
    "cluster_ids should be equal in length"
  )
  expect_error(
    cv_origami_schema(mtcars, n_folds = 2, strata_ids = c(1, 2)),
    "strata_ids should be equal in length"
  )

  set.seed(1)
  out <- cv_origami_schema(mtcars, n_folds = 3)
  expect_length(out$training_data, 3)
  expect_equal(sum(sapply(out$validation_data, nrow)), nrow(mtcars))

  # cluster ids keep clusters intact across splits
  set.seed(1)
  clusters <- rep(1:8, each = 4)
  out_cl <- cv_origami_schema(mtcars, n_folds = 2, cluster_ids = clusters)
  expect_length(out_cl$validation_data, 2)

  # strata ids balance strata
  set.seed(1)
  strata <- rep(c(1, 2), 16)
  out_st <- cv_origami_schema(mtcars, n_folds = 2, strata_ids = strata)
  expect_length(out_st$validation_data, 2)
})

test_that("cv_origami_schema supports fold functions without a V argument", {
  # folds_loo takes only n, exercising the non-V branch
  out <- cv_origami_schema(mtcars[1:5, ], n_folds = 5,
                           fold_fun = origami::folds_loo)
  expect_length(out$validation_data, 5)
  expect_true(all(sapply(out$validation_data, nrow) == 1))
})
