#' Assign Data to One of n_folds Randomly and Produce Training/Validation Data Lists
#'
#' Each row in the data are assigned to one of `1:n_folds` at random.
#' Then for each of `i` in `1:n_folds`, the `training_data[[i]]`
#' is comprised of the data with `sl_fold != i`, i.e., capturing
#' roughly `(n-folds-1)/n_folds` proportion of the data.  The validation data
#' is a list of dataframes, each comprising of roughly `1/n_folds` proportion of the
#' data.
#'
#' Since the assignment to folds is random, the proportions are not
#' exact or guaranteed and there is some variability in the size of
#' each `training_data` data frame, and likewise for the `validation_data`
#' data frames.
#'
#' @param data a data.frame (or similar) to split into training and validation datasets.
#' @param n_folds The number of `training_data` and `validation_data` data frames to make.
#' @return a named list of two lists, each being a list of `n_folds` data frames.
#' @examples
#' \dontrun{
#'   data(Boston, package = 'MASS')
#'   training_validation_data <- cv_random_schema(Boston, n_folds = 3)
#'   # take a look at what's in the output:
#'   str(training_validation_data, max.level = 2)
#' }
#' @importFrom dplyr filter select
#' @importFrom utils str
#' @export
cv_random_schema <- function(data, n_folds = 5) {
  # check if the data already has .sl_fold and error
  if ('.sl_fold' %in% colnames(data)) {
    stop("The data passed to make_folds already has a .sl_fold column")
  }

  if (is.matrix(data)) { # cast to data frame
    data <- as.data.frame(data)
  }
  data[['.sl_fold']] <- sample.int(n = n_folds, size = nrow(data), replace = TRUE)

  # sample.int — but do make sure every validation fold contains >0 observations
  resampling_counts <- 0
  while(any(table(data[['.sl_fold']]) < 1)) {
    data[['.sl_fold']] <- sample.int(n = n_folds, size = nrow(data), replace = TRUE)
    resampling_counts <- resampling_counts + 1
    if (resampling_counts >= 5) {
      warning("cv_random_schema has made 5+ attempts to make sure every fold of validation data is non-empty;
You may want to write your own cv_random_schema if constructing cv folds continues to take a while...")
    }
  }

  # split into an n_folds length list of training_data and validation_data:
  # training_data contains n_folds-1 folds of data, validation_data contains 1 fold
  training_data <- lapply(
    1:n_folds, function(i) {data |> dplyr::filter(.data$.sl_fold != i) |> dplyr::select(-".sl_fold")})
  validation_data <- lapply(
    1:n_folds, function(i) {data |> dplyr::filter(.data$.sl_fold == i) |> dplyr::select(-".sl_fold")})

  return(list(
    training_data = training_data,
    validation_data = validation_data
  ))
}


#' Cross Validation Training/Validation Splits with Characters/Factor Columns
#'
#' Designed to handle cross-validation on models like randomForest, ranger,
#' glmnet, etc., where the model matrix of newdata must match eactly the model
#' matrix of the training dataset, this function intends to answer the need "The
#' training datasets need to have every level of every discrete-type column that
#' appears in the data."
#'
#' The fundamental idea is to check if the unique levels of character and/or factor
#' columns are represented in every training dataset.
#'
#' Above and beyond this, this function is designed to support cv_super_learner,
#' which inherently involves two layers of cross-validation.  As a result, more stringent
#' conditions are specified when the `cv_sl_mode` is enabled.  For convenience this
#' mode is enabled by default
#'
#' @inheritParams super_learner
#' @param cv_sl_mode A binary (default: TRUE) indicator for if the output
#'   training/validation data lists will be used inside another `super_learner`
#'   call. If so, then the training data needs to have every level appear at
#'   least twice so that the data can be put into further training/validation
#'   splits.
#' @param check_validation_datasets_too Enforce that the validation datasets
#'   produced also have every level of every character / factor type column
#'   present. This is particularly useful for learners like `glmnet` which
#'   require that the `newx` have the exact same shape/structure as the training
#'   data, down to binary indicators for every level that appears.
#'
#' @export
#'
#' @examples
#'
#' \dontrun{
#' require(palmerpenguins)
#' training_validation_splits <- cv_character_and_factors_schema(
#'   palmerpenguins::penguins)
#'
#' # we can see the population breakdown across all the training
#' # splits:
#' sapply(training_validation_splits$training_data, function(df) {
#'   table(df$species)
#'   })
#' # notably, none of them are empty! this is crucial for certain
#' # types of learning algorithms that must see all levels appear in the
#' # training data, like random forests.
#'
#' # certain models like glmnet require that the prediction dataset
#' # newx have the _exact_ same shape as the training data, so it
#' # can be important that every level appears in the validation data
#' # as well.  check that by looking into these types of tables:
#' sapply(training_validation_splits$validation_data, function(df) {
#'   table(df$species)
#'   })
#'
#' # if you don't need this level of stringency, but you just want
#' # to make cv_splits where every level appears in the training_data,
#' # you can do so using the check_validation_datasets_too = FALSE
#' # argument.
#' penguins_small <- palmerpenguins::penguins[c(1:3, 154:156, 277:279), ]
#' penguins_small <- penguins_small[complete.cases(penguins_small),]
#'
#' training_validation_splits <- cv_character_and_factors_schema(
#'   penguins_small,
#'   cv_sl_mode = FALSE,
#'   n_folds = 5,
#'   check_validation_datasets_too = FALSE)
#'
#' sapply(training_validation_splits$training_data, function(df) {
#'   table(df$species)
#'   })
#'
#' # now you can see plenty of non-appearing levels in the validation data:
#' sapply(training_validation_splits$validation_data, function(df) {
#'   table(df$species)
#'   })
#' }
cv_character_and_factors_schema <- function(
    data, n_folds = 5,
    cv_sl_mode = TRUE,
    check_validation_datasets_too = TRUE) {

  # check where the characters/factors are located
  chr_fct_col_indices <- which(sapply(data, class) %in% c("character", "factor"))

  if (is.null(chr_fct_col_indices) || length(chr_fct_col_indices) == 0) {
    stop("There must be character/factor column types to use with cv_character_and_factors_schema.")
  }

  # get the unique levels for each character/factor column
  unique_levels <- lapply(1:length(chr_fct_col_indices), function(i) {
    unique(data[[chr_fct_col_indices[i]]])
  })
  # error if any chr/fct columns only have one level
  if (any(sapply(unique_levels, length) <= 1)) {
    stop("There are character/factor levels in the data that are constant, and therefore cannot be included in every training/test split")
  }

  # levels that only appear once pose an issue
  level_frequencies <- lapply(1:length(chr_fct_col_indices), function(i) {
    table(data[[chr_fct_col_indices[i]]])
  })
  if (any(sapply(level_frequencies, min) == 1)) {
    stop("There are character/factor levels in the data that only appear once.")
  }

  # get a randomized training/test split
  cv_random_schema_output <- cv_random_schema(data, n_folds)

  # if there are no character/factor type columns, there's nothing
  # more to do -- so we just run the random splitting as default.
  if (length(chr_fct_col_indices) == 0) {
    return(cv_random_schema_output)
  }

  # determine if any of the chr/fct columns have levels that appear 2
  # or fewer times
  two_or_fewer_levels <- sapply(chr_fct_col_indices, function(i) {
    any(table(data[[i]]) <= 2)
  })

  # if we are going to check the validation datasets as well as the training data
  # for having every level present, then if there are any levels that appear two or
  # fewer times, we have a problem
  if (any(two_or_fewer_levels) & check_validation_datasets_too) {
    which_two_or_fewer <- which(two_or_fewer_levels)
    problematic_colnames <- colnames(data)[which_two_or_fewer]
    stop(paste0(
      "There are character/factor columns that have levels only appearing 2 or fewer times.
If check_validation_datasets_too = TRUE, then this is too few appearances of those levels for
it to be possible that they appear in every training and validation split. This poses
problems for prediction models like glmnet where it is required that the input newx matrix
has the same shape every time. The following columns had two or fewer levels: ",
problematic_colnames)
    )
  }

  # if we are going to use this function to produce splits for cv_super_learner
  # then we need that within the training data, each level appears 2+ times so that
  # a splits can be made when training super_learner on each training split.
  #
  # this means if cv_sl_mode and check_validation_datasets_too are both
  # engaged, then we need to see every level appear 2+ times in the training
  # split and at least once in the validation data.  And hence every level
  # needs to appear 3+ times.
  if (cv_sl_mode & check_validation_datasets_too) {
    three_or_fewer_levels <- any(sapply(chr_fct_col_indices, function(i) {
      any(table(data[[i]]) <= 3)
    }))

    if (any(three_or_fewer_levels)) {
      which_three_or_fewer <- which(two_or_fewer_levels)
      problematic_colnames <- colnames(data)[which_three_or_fewer]
      stop(paste0(
        "There are character/factor columns that have levels only appearing 3 or fewer times.
If check_validation_datasets_too = TRUE and cv_sl_mode = TRUE, then this is too few appearances of those levels for
it to be possible that they appear in every training dataset 2+ times and in every validation split.
When cv_sl_mode = TRUE, we require that there be 2+ appearances of every level
in each training_split so that a further layer of cv can be performed when super_learner is called
on each training_split. The following columns had three or fewer levels: ",
problematic_colnames)
      )
    }
  }

  # set a false success condition until we verify that all levels are represented
  # in each training/test split
  success_condition <- FALSE

  # check if every level appearing in the data appears in each of the
  # data list passed
  determine_success_condition <- function(training_or_validation_data_list) {
    sapply(1:length(chr_fct_col_indices), function(i) {
      sapply(1:length(training_or_validation_data_list), function(dataset_j) {
        all(unique_levels[[i]] %in% training_or_validation_data_list[[dataset_j]][[chr_fct_col_indices[i]]])
      })
    })
  }

  # check if every level appearing in the data appears in each of the
  # data list passed at least twice
  determine_2plus_entries_present_success_condition <- function(training_or_validation_data_list) {
    sapply(1:length(chr_fct_col_indices), function(i) {
      sapply(1:length(training_or_validation_data_list), function(dataset_j) {
        all(unique_levels[[i]] %in% training_or_validation_data_list[[dataset_j]][[chr_fct_col_indices[i]]]) &&
          all(table(training_or_validation_data_list[[dataset_j]][[chr_fct_col_indices[i]]]) >= 2)
      })
    })
  }

  # we're going to keep track of how many times we call cv_random_schema
  # so that we can write a message if it seems high.
  #
  # it may be the case that a more sophisticated cv_schema is appropriate,
  # in which case the user should be instructed to implement it in a message.
  cv_resampling_count <- 0

  # if the success condition is not met, re-run the random_schema to get another
  # training/split
  while (! all(success_condition)) {  # get a vector of if each chr or fct column has all of its levels represented
    cv_random_schema_output <- cv_random_schema(data, n_folds)

    # increment how many times cv_random_schema was called
    cv_resampling_count <- cv_resampling_count + 1
    if (cv_resampling_count == 5) {
      message("5+ cross-validation splits have been randomly generated and not been satisfactory for use.
You may want to consider writing your own cv_schema type of function to handle setting up training/validation splits
yourself instead. See ?cv_character_and_factors_schema and ?cv_random_schema.")
    }

    if (cv_sl_mode) {
      # for cv_sl_mode, make sure every level appears at least twice in the training data
      success_condition <- determine_2plus_entries_present_success_condition(cv_random_schema_output$training_data)
    } else {
      # get a vector of if each chr or fct column has all of its levels represented
      success_condition <- determine_success_condition(cv_random_schema_output$training_data)
    }

    # if we also require that every level appear in the validation datasets,
    # check those too
    if (check_validation_datasets_too) {
      success_condition <- c(
        success_condition,
        determine_success_condition(cv_random_schema_output$validation_data)
      )
    }
  }

  return(cv_random_schema_output)
}


#' Cross-Validation with Origami
#'
#' @examples
#' \dontrun{
#'
#' # to use origami::folds_vfold behind the scenes, just tell nadir::super_learner
#' # you want to use cv_origami_schema.
#'
#' sl_model <- super_learner(
#'   data = mtcars,
#'   formula = mpg ~ cyl + hp,
#'   learners = list(rf = lnr_rf, lm = lnr_lm, mean = lnr_mean),
#'   cv_schema = cv_origami_schema,
#'   verbose = TRUE
#'  )
#'
#' # if you want to use a different origami::folds_* function, pass it into cv_origami_schema
#' sl_model <- super_learner(
#'   data = mtcars,
#'   formula = mpg ~ cyl + hp,
#'   learners = list(rf = lnr_rf, lm = lnr_lm, mean = lnr_mean),
#'   cv_schema = \(data, n_folds) {
#'     cv_origami_schema(data, n_folds, fold_fun = origami::folds_loo)
#'     },
#'   verbose = TRUE
#'  )
#' }
#' @importFrom origami folds_vfold make_folds
#' @importFrom methods formalArgs
#' @inheritParams cv_random_schema
#' @param fold_fun An \code{origami::folds_*} function
#' @param cluster_ids A vector of cluster ids. Clusters are treated as a unit –
#'   that is, all observations within a cluster are placed in either the
#'   training or validation set. See \code{?origami::make_folds}.
#' @param strata_ids A vector of strata ids. Strata are balanced: insofar as
#'   possible the distribution in the sample should be the same as the
#'   distribution in the training and validation sets. See \code{?origami::make_folds}.
#' @param ... Extra arguments to be passed to \code{origami::make_folds()}
cv_origami_schema <- function(
    data = data,
    n_folds = 5,
    fold_fun = origami::folds_vfold,
    cluster_ids = NULL,
    strata_ids = NULL,
    ...
) {

  # use methods::formalArgs to determine if the fold function passed takes
  # V as an argument — if so, we want to make sure we pass n_folds as V.
  if ("V" %in% methods::formalArgs(fold_fun)) {
    folds <- origami::make_folds(n = nrow(data),
                                 fold_fun = fold_fun,
                                 cluster_ids = cluster_ids,
                                 strata_ids = strata_ids,
                                 V = n_folds,
                                 ...)
  } else {
    folds <- origami::make_folds(n = nrow(data),
                                 fold_fun = fold_fun,
                                 cluster_ids = cluster_ids,
                                 strata_ids = strata_ids,
                                 ...)
  }

  # split into an n_folds length list of training_data and validation_data:
  # training_data contains n_folds-1 folds of data, validation_data contains 1 fold
  training_data <- lapply(
    1:n_folds, function(i) {
      data[folds[[i]]$training_set,]
      }
    )
  validation_data <- lapply(
    1:n_folds, function(i) {
      data[folds[[i]]$validation_set,]
      })

  return(list(
    training_data = training_data,
    validation_data = validation_data
  ))
}

