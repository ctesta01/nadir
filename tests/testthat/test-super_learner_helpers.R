test_that("super_learner_helpers work as intended", {

  # section on validate_learner_types :
  #
  # we should get warnings if we use the wrong learner types and no warnings if
  # we use the right learner types
  expect_warning({
    validate_learner_types(
      list(mean = lnr_mean, lm = lnr_lm), 'density')
  })
  expect_no_warning({
    validate_learner_types(
      list(lnr_mean, lnr_lm), 'continuous')
  })
  expect_no_warning({
    validate_learner_types(
      list(lnr_lm_density, lnr_homoskedastic_density), 'density')
  })
  expect_no_warning({
    validate_learner_types(
      list(lnr_glm, lnr_mean), 'binary')
  })
})

test_that("cv_random_schema produces good splits", {
  withr::local_seed(20260105)
  # produce synthetic data
  df <- data.frame(id = 1:100,
                   x = sample.int(n = 100, size = 100, replace = FALSE))

  # using a weird number of folds just to make sure everything works even
  # when n_folds isn't one of the common choices like 5 or 10.
  n_folds <- 12
  cv_splits <- cv_random_schema(df, n_folds = n_folds)

  # check that there is no "leakage" across training/test splits
  validation_data_appears_in_training_data <-
    sapply(1:length(cv_splits$training_data), function(i) {
      any(
        cv_splits$validation_data[[i]][['id']] %in%
        cv_splits$training_data[[i]][['id']])
    })
  training_data_appears_in_validation_data <-
    sapply(1:length(cv_splits$training_data), function(i) {
      any(
        cv_splits$training_data[[i]][['id']] %in%
        cv_splits$validation_data[[i]][['id']])
    })
  expect_false(any(validation_data_appears_in_training_data))
  expect_false(any(training_data_appears_in_validation_data))


  # check the sizes of the splits
  validation_data_sizes <- sapply(
    1:length(cv_splits$validation_data),
    function(i) {
      nrow(cv_splits$validation_data[[i]])
    })
  training_data_sizes <- sapply(
    1:length(cv_splits$training_data),
    function(i) {
      nrow(cv_splits$training_data[[i]])
    })

  # the validation data splits should not be far from nrow(df) / n_folds in size
  expect_true(
    all(validation_data_sizes >= nrow(df) / n_folds - 3),
    info = paste("validation sizes:", paste(validation_data_sizes, collapse = ", ")))
  expect_true(all(validation_data_sizes <= nrow(df) / n_folds + 3),
              info = paste("validation sizes:", paste(validation_data_sizes, collapse = ", ")))
  # the training data splits should not be far from nrow(df) * (n_folds - 1) / n_folds in size
  expect_true(
    all(training_data_sizes >= nrow(df) * (n_folds - 1)/ n_folds - 3),
    info = paste("training sizes:", paste(training_data_sizes, collapse = ", ")))
  expect_true(all(training_data_sizes <= nrow(df) * (n_folds - 1)/ n_folds + 3),
    info = paste("training sizes:", paste(training_data_sizes, collapse = ", "))
  )
})
