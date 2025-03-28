test_that("cv_origami_schema respects clusters", {

  # generate synthetic clustered data
  n_clusters <- 25
  n_participants <- 100
  cluster_ids <- sample.int(25, 100, replace = TRUE)
  age <- sample.int(100, n_participants, replace = TRUE)
  drug <- sample(x = c(0, 1), n_participants, replace = TRUE)

  cluster_mean_outcomes <- rnorm(mean = 25, sd = 5, n = n_clusters)

  participant_outcomes <-
    cluster_mean_outcomes[cluster_ids] +
    drug * 15 +
    age

  df <- data.frame(
    cluster_id = cluster_ids,
    age = age,
    drug = drug,
    outcome = participant_outcomes)

  # call origami schema with cluster_ids
  df_splits <- cv_origami_schema(
    data = df,n_folds = 5, fold_fun = origami::folds_vfold, cluster_ids = df$cluster_id)

  unisort <- \(x) sort(unique(x))

  train1_ids <- unisort(df_splits$training_data[[1]]$cluster_id)
  validate1_ids <- unisort(df_splits$validation_data[[1]]$cluster_id)
  expect_true(! any(validate1_ids %in% train1_ids))

  train2_ids <- unisort(df_splits$training_data[[2]]$cluster_id)
  validate2_ids <- unisort(df_splits$validation_data[[2]]$cluster_id)
  expect_true(! any(validate2_ids %in% train2_ids))
})

test_that("cv_origami_schema respects strata", {

  # generate synthetic clustered data
  n_clusters <- 25
  n_participants <- 100
  strata_ids <- rep(1:20, each = 5)
  age <- sample.int(100, n_participants, replace = TRUE)
  drug <- sample(x = c(0, 1), n_participants, replace = TRUE)

  participant_outcomes <-
    drug * 15 +
    age

  df <- data.frame(
    strata_id = strata_ids,
    age = age,
    drug = drug,
    outcome = participant_outcomes)

  # call origami schema with strata_ids
  df_splits <- cv_origami_schema(
    data = df,n_folds = 5, fold_fun = origami::folds_vfold, strata_ids = df$strata_id)

  unisort <- \(x) sort(unique(x))

  train1_ids <- unisort(df_splits$training_data[[1]]$cluster_id)
  validate1_ids <- unisort(df_splits$validation_data[[1]]$cluster_id)
  expect_true(all(validate1_ids %in% train1_ids))

  train2_ids <- unisort(df_splits$training_data[[2]]$cluster_id)
  validate2_ids <- unisort(df_splits$validation_data[[2]]$cluster_id)
  expect_true(all(validate2_ids %in% train2_ids))
})
