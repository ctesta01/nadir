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

  # extract_y_variable had a bug where it couldn't handle formulas
  # being input as characters
  expect_no_error(
    extract_y_variable(
      formulas = 'y ~ x',
      learner_names = c('a', 'b'),
      data_colnames = c('y', 'x'),
      y_variable = NULL)
    )


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


# Tests for make_learner_names_unique, validate_learner_types, parse_formulas,
# extract_y_variable, parse_extra_learner_arguments

test_that("make_learner_names_unique handles named, unnamed, and duplicate learners", {
  # fully unnamed list where learners carry sl_lnr_name attributes
  learners <- list(lnr_mean, lnr_rf, lnr_rf, lnr_glm)
  out <- make_learner_names_unique(learners)
  expect_equal(names(out), c("mean", "rf_1", "rf_2", "glm"))

  # mixture of named, attribute-named, and anonymous learners
  learners <- list(
    mean = lnr_mean,
    rf = lnr_rf,
    rf = lnr_rf,
    lnr_glm,
    function(data, formula) {},
    function(data, formula) {}
  )
  out <- make_learner_names_unique(learners)
  expect_equal(
    names(out),
    c("mean", "rf_1", "rf_2", "glm", "unnamed_lnr_1", "unnamed_lnr_2")
  )

  # fully unnamed anonymous learners (no names, no sl_lnr_name attributes)
  learners <- list(function(data, formula) {}, function(data, formula) {})
  out <- make_learner_names_unique(learners)
  expect_equal(names(out), c("unnamed_lnr_1", "unnamed_lnr_2"))

  # already-unique names pass through untouched
  learners <- list(a = lnr_mean, b = lnr_lm)
  expect_equal(names(make_learner_names_unique(learners)), c("a", "b"))
})

test_that("validate_learner_types is silent on matches and warns on mismatches", {
  expect_invisible(
    validate_learner_types(list(lm = lnr_lm, mean = lnr_mean), "continuous")
  )
  expect_null(
    suppressWarnings(validate_learner_types(list(lm = lnr_lm), "continuous"))
  )

  # a density learner does not match a continuous outcome (named case)
  expect_warning(
    validate_learner_types(list(dens = lnr_lm_density), "continuous"),
    "do not have attr"
  )

  # unnamed learners also warn
  expect_warning(
    validate_learner_types(list(lnr_lm_density), "continuous"),
    "do not have attr"
  )
})

test_that("parse_formulas handles a single formula", {
  out <- parse_formulas(mpg ~ hp, learner_names = c("a", "b"))
  expect_length(out, 2)
  expect_equal(names(out), c("a", "b"))
  expect_true(all(sapply(out, inherits, "formula")))
})

test_that("parse_formulas handles character formulas", {
  out <- parse_formulas("mpg ~ hp", learner_names = c("a", "b"))
  expect_length(out, 2)
  expect_equal(names(out), c("a", "b"))
  expect_true(all(sapply(out, inherits, "formula")))

  # multiple character formulas convert and match by position
  out2 <- parse_formulas(c("mpg ~ hp", "mpg ~ wt"), learner_names = c("a", "b"))
  expect_length(out2, 2)
  expect_equal(names(out2), c("a", "b"))
})

test_that("parse_formulas handles unnamed equal-length lists positionally", {
  out <- parse_formulas(list(mpg ~ hp, mpg ~ wt), learner_names = c("a", "b"))
  expect_equal(names(out), c("a", "b"))
  expect_equal(out[["b"]], mpg ~ wt)
})

test_that("parse_formulas handles fully-named formula lists", {
  out <- parse_formulas(
    list(b = mpg ~ wt, a = mpg ~ hp),
    learner_names = c("a", "b")
  )
  # reordered to match the learner names
  expect_equal(names(out), c("a", "b"))
  expect_equal(out[["a"]], mpg ~ hp)
})

test_that("parse_formulas supports .default with learner-specific overrides", {
  out <- parse_formulas(
    list(.default = mpg ~ hp, b = mpg ~ wt),
    learner_names = c("a", "b")
  )
  expect_equal(out[["a"]], mpg ~ hp)
  expect_equal(out[["b"]], mpg ~ wt)
})

test_that("parse_formulas supports partially-named index-matched formulas", {
  out <- parse_formulas(
    list(a = mpg ~ hp, mpg ~ wt),
    learner_names = c("a", "b")
  )
  expect_equal(names(out), c("a", "b"))
  expect_equal(out[["b"]], mpg ~ wt)
})

test_that("parse_formulas errors on unmatched formulas", {
  expect_error(
    parse_formulas(
      list(c = mpg ~ hp, d = mpg ~ wt),
      learner_names = c("a", "b")
    ),
    "Cannot appropriately match the formulas to the learners"
  )

  # a non-vector collection of formulas errors
  weird <- structure(list(mpg ~ hp, mpg ~ wt), class = "not_a_vector")
  expect_error(
    parse_formulas(weird, learner_names = c("a", "b")),
    "must be passed as a vector"
  )
})

test_that("extract_y_variable infers the outcome from formulas", {
  expect_equal(
    extract_y_variable(mpg ~ hp, learner_names = "a", data_colnames = colnames(mtcars)),
    "mpg"
  )
  expect_equal(
    extract_y_variable("mpg ~ hp", learner_names = "a", data_colnames = colnames(mtcars)),
    "mpg"
  )
  expect_equal(
    extract_y_variable(
      list(a = mpg ~ hp, b = mpg ~ wt),
      learner_names = c("a", "b"),
      data_colnames = colnames(mtcars)
    ),
    "mpg"
  )
  # explicit y_variable is honored
  expect_equal(
    extract_y_variable(mpg ~ hp, learner_names = "a",
                       data_colnames = colnames(mtcars), y_variable = "mpg"),
    "mpg"
  )
})

test_that("extract_y_variable uses .default when outcomes differ", {
  expect_equal(
    extract_y_variable(
      list(.default = mpg ~ hp, b = wt ~ hp),
      learner_names = c("a", "b"),
      data_colnames = colnames(mtcars)
    ),
    "mpg"
  )
})

test_that("extract_y_variable errors informatively", {
  # differing outcomes without .default
  expect_error(
    extract_y_variable(
      list(a = mpg ~ hp, b = wt ~ hp),
      learner_names = c("a", "b"),
      data_colnames = colnames(mtcars)
    ),
    "Cannot infer the y-variable"
  )

  # outcome must appear in the data
  expect_error(
    extract_y_variable(zzz ~ hp, learner_names = "a", data_colnames = colnames(mtcars)),
    "must appear as a column in the data"
  )

  # outcome must not collide with a learner name
  expect_error(
    extract_y_variable(mpg ~ hp, learner_names = c("mpg"), data_colnames = colnames(mtcars)),
    "must be distinct"
  )
})

test_that("parse_extra_learner_arguments handles NULL, named, positional, and .default", {
  # NULL -> list of NULLs
  out <- parse_extra_learner_arguments(NULL, c("a", "b"))
  expect_length(out, 2)
  expect_true(all(sapply(out, is.null)))

  # all learner names present (in scrambled order)
  out <- parse_extra_learner_arguments(
    list(b = list(x = 2), a = list(x = 1)),
    c("a", "b")
  )
  expect_equal(out[["a"]], list(x = 1))
  expect_equal(out[["b"]], list(x = 2))

  # unnamed positional
  out <- parse_extra_learner_arguments(list(list(x = 1), list(x = 2)), c("a", "b"))
  expect_equal(names(out), c("a", "b"))
  expect_equal(out[["b"]], list(x = 2))

  # .default with an override; unmatched learner gets the default
  out <- parse_extra_learner_arguments(
    list(.default = list(x = 0), b = list(x = 2)),
    c("a", "b", "c")
  )
  expect_equal(out[[1]], list(x = 0))
  expect_equal(out[[2]], list(x = 2))
  expect_equal(out[[3]], list(x = 0))

  # names that cannot be matched error
  expect_error(
    parse_extra_learner_arguments(list(zzz = list(x = 1)), c("a", "b")),
    "extra_learner_args must either be passed as"
  )
})

test_that("parse_extra_learner_arguments .default branch returns NULL without .default", {
  # names subset of learner names, no .default: unmatched learners get NULL
  out <- parse_extra_learner_arguments(
    list(b = list(x = 2)),
    c("a", "b")
  )
  expect_null(out[[1]])
  expect_equal(out[[2]], list(x = 2))
})


