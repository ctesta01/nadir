#' Super Learner: Cross-Validation Based Ensemble Learning
#'
#' Super learning with functional programming!
#'
#' The goal of any super learner is to use cross-validation and a
#' set of candidate learners to 1) evaluate how the learners perform
#' on held out data and 2) to use that evaluation to produce a weighted
#' average (for continuous super learner) or to pick a best learner (for
#' discrete super learner) of the specified candidate learners.
#'
#' Super learner and its statistically desirable properties have been written
#' about at length, including at least the following references:
#'
#'   * <https://biostats.bepress.com/ucbbiostat/paper222/>
#'   * <https://www.stat.berkeley.edu/users/laan/Class/Class_subpages/BASS_sec1_3.1.pdf>
#'
#' `nadir::super_learner` adopts several user-interface design-perspectives
#' that will be useful to know in understanding what it does and how it works:
#'
#'   * The specification of learners should be _very flexible_, really only
#'   constrained by the fact that candidate learners should be designed
#'   for the same prediction problem but their details can wildly vary
#'   from learner to learner.
#'   * It should be easy to specify a customized or new learner.
#'
#' `nadir::super_learner` at its core accepts `data`,
#' a `regression_formula` (a single one passed to `regression_formulas` is fine),
#' and a list of `learners`.
#'
#' `learners` are taken to be lists of functions of the following specification:
#'
#'   * a learner must accept a `data` and `regression_formula` argument,
#'   * a learner may accept more arguments, and
#'   * a learner must return a prediction function that accepts `newdata` and
#' produces a vector of prediction values given `newdata`.
#'
#' In essence, a learner is specified to be a function taking (`data`, `regression_formula`, ...)
#' and returning a _closure_ (see <http://adv-r.had.co.nz/Functional-programming.html#closures> for an introduction to closures)
#' which is a function accepting `newdata` returning predictions.
#'
#' Since many candidate learners will have hyperparameters that should be tuned,
#' like depth of trees in random forests, or the `lambda` parameter for `glmnet`,
#' extra arguments can be passed to each learner via the `extra_learner_args`
#' argument. `extra_learner_args` should be a list of lists, one list of
#' extra arguments for each learner. If no additional arguments are needed
#' for some learners, but some learners you're using do require additional
#' arguments, you can just put a `NULL` value into the `extra_learner_args`.
#' See the examples.
#'
#' In order to seamlessly support using features implemented by extensions
#' to the formula syntax (like random effects formatted like random intercepts or slopes that use the
#' `(age | strata)` syntax in
#' `lme4` or splines like `s(age | strata)` in `mgcv`), we allow for the
#' `regression_formulas` argument to either be one fixed formula that
#' `super_learner` will use for all the models, or a vector of formulas,
#' one for each learner specified.
#'
#' Note that in the examples a root-mean-squared-error (rmse) is calculated on
#' the same training/test set, and this is only useful as a crude diagnostic to
#' see that super_learner is working. A more rigorous performance metric to
#' evaluate `super_learner` on is the cv-rmse produced by cv_super_learner.
#'
#' @param data Data to use in training a `super_learner`.
#' @param learners A list of predictor/closure-returning-functions. See Details.
#' @param regression_formulas Either a single regression formula or a vector of regression formulas.
#' @param y_variable Typically `y_variable` can be inferred automatically from the `regression_formulas`, but if needed, the y_variable can be specified explicitly.
#' @param n_folds The number of cross-validation folds to use in constructing the `super_learner`.
#' @param determine_super_learner_weights A function/method to determine the weights for each of the candidate `learners`. The default is to use `determine_super_learner_weights_nnls`.
#' @param continuous_or_discrete Defaults to `'continuous'`, but can be set to `'discrete'`.
#' @param cv_schema A function that takes `data`, `n_folds` and returns a list containing `training_data` and `validation_data`, each of which are lists of `n_folds` data frames.
#' @param extra_learner_args A list of equal length to the `learners` with additional arguments to pass to each of the specified learners.
#' @param verbose_output If `verbose_output = TRUE` then return a list containing the fit learners with their predictions on held-out data as well as the
#' prediction function closure from the trained `super_learner`.
#'
#' @examples
#' \dontrun{
#'
#' learners <- list(
#'      glm = lnr_glm,
#'      rf = lnr_rf,
#'      glmnet = lnr_glmnet,
#'      lmer = lnr_lmer
#'   )
#'
#' # mtcars example ---
#' regression_formulas <- c(
#'   rep(c(mpg ~ cyl + hp), 3), # first three models use same formula
#'   mpg ~ (1 | cyl) + hp # lme4 uses different language features
#'   )
#'
#' # fit a super_learner
#' sl_model <- super_learner(
#'   data = mtcars,
#'   regression_formula = regression_formulas,
#'   learners = learners)
#'
#' # produce super_learner predictions
#' sl_model_predictions <- sl_model(mtcars)
#' # compare against the predictions from the individual learners
#' fit_individual_learners <- lapply(1:length(learners), function(i) { learners[[i]](data = mtcars, regression_formula = regression_formulas[[i]]) } )
#' individual_learners_rmse <- lapply(fit_individual_learners, function(fit_learner) { rmse(fit_learner(mtcars) - mtcars$mpg) })
#' names(individual_learners_rmse) <- names(learners)
#'
#' print(paste0("super-learner rmse: ", rmse(sl_model_predictions - mtcars$mpg)))
#' individual_learners_rmse
#'
#'
#' # iris example ---
#' sl_model <- super_learner(
#'   data = iris,
#'   regression_formula = Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width,
#'   learners = learners[1:3])
#'
#' # produce super_learner predictions and compare against the individual learners
#' sl_model_predictions <- sl_model(iris)
#' fit_individual_learners <- lapply(learners[1:3], function(learner) { learner(data = iris, regression_formula = Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width) } )
#' individual_learners_rmse <- lapply(fit_individual_learners, function(fit_learner) { rmse(fit_learner(iris) - iris$Sepal.Length) })
#'
#' print(paste0("super-learner rmse: ", rmse(sl_model_predictions - iris$Sepal.Length)))
#' individual_learners_rmse
#' }
#'
#' @seealso cv_super_learner
#'
#' @export
super_learner <- function(
  data,
  learners,
  regression_formulas,
  y_variable,
  n_folds = 5,
  determine_super_learner_weights = determine_super_learner_weights_nnls,
  continuous_or_discrete = 'continuous',
  cv_schema = cv_random_schema,
  extra_learner_args = NULL,
  verbose_output = FALSE) {

  # throw an error if the learners are not a named list
  if (! is.list(learners) | length(unique(names(learners))) != length(learners)) {
    stop("The learners passed to lmpti::super_learner must have (unique) names.")
  }

  # set up training and validation data
  #
  # the training and validation data are lists of datasets,
  # where the training data are distinct (n-1)/n subsets of the data and the
  # validation data are the corresponding other 1/n of the data.
  training_and_validation_data <- cv_schema(data, n_folds)
  training_data <- training_and_validation_data$training_data
  validation_data <- training_and_validation_data$validation_data

  # make a tibble/dataframe to hold the trained learners:
  # one for each combination of a specific fold and a specific model
  trained_learners <- tibble::tibble(
    split = rep(1:n_folds, length(learners)),
    learner_name = rep(names(learners), each = n_folds))

  # handle vectorized regression_formulas argument
  #
  # if the regression_formulas is just a single formula, then we repeat it
  # in a vector length(learners) times to make it simple to just pass the ith
  # learner regression_formula[[i]].
  #
  # TODO: Abstract this to a parse_formulas(regression_formulas, learners)
  # function call to handle index AND name-based syntax.
  regression_formulas <- parse_formulas(regression_formulas = regression_formulas,
                                        learner_names = names(learners))

  # for each i in 1:n_folds and each model, train the model
  trained_learners$learned_predictor <- lapply(
    1:nrow(trained_learners), function(i) {
      # calculate which learner has the name for this row and use
      # the appropriate regression formula as well as the right
      # extra_learner_args
      learner_index <- which(names(learners) == trained_learners$learner_name[[i]])[[1]]

      # train the learner — the returned output is the prediction function from
      # the trained learner
      do.call(
        what = learners[[trained_learners[[i,'learner_name']]]],
        args = c(list(
          data = training_data[[trained_learners[[i,'split']]]],
          regression_formula = regression_formulas[[
            learner_index
          ]]),
          extra_learner_args[[learner_index]]
        )
      )
    }
  )

  # predict from each fold+model combination on the held-out data
  trained_learners$predictions_for_testset <- lapply(
    1:nrow(trained_learners), function(i) {
      trained_learners[[i,'learned_predictor']][[1]](validation_data[[trained_learners[[i, 'split']]]])
    }
  )

  # from here forward, we just need to use the split + model name + predictions on the test-set
  # to regress against the held-out (validation) data to determine the ensemble weights
  second_stage_SL_dataset <- trained_learners[,c('split', 'learner_name', 'predictions_for_testset')]

  # pivot it into a wider format, with one column per model, with columnname model_name
  second_stage_SL_dataset <- tidyr::pivot_wider(
    second_stage_SL_dataset,
    names_from = 'learner_name',
    values_from = 'predictions_for_testset')

  # Extract the Y-variable (its character name)
  #
  # This only supports simple Y variables, nothing like a survival right-hand-side or
  # a transformed right-hand-side.
  #
  y_variable <- extract_y_variable(
    regression_formulas = regression_formulas,
    learner_names = names(learners),
    data_colnames = colnames(data),
    y_variable = y_variable
  )


  # insert the validation Y data in another column next to the predictions
  second_stage_SL_dataset[[y_variable]] <- lapply(1:nrow(second_stage_SL_dataset), function(i) {
    validation_data[[second_stage_SL_dataset[[i, 'split']]]][[y_variable]]
  })

  # unnest all of the data (each cell prior to this contained a vector of either
  # predictions or the validation data)
  second_stage_SL_dataset <- tidyr::unnest(second_stage_SL_dataset, cols = colnames(second_stage_SL_dataset))

  # drop the split column so we can simplify the following regression formula
  second_stage_SL_dataset$split <- NULL

  # regress the validation data on the predictions from every model with no intercept.
  # notice this is now for all of the folds
  #
  # TODO: Here we assume a continuous Y-variable and use a linear regression to
  # determine the SuperLearner weights;  we may want to support other types of
  # Y-variables like binary, count, and survival.  My theory on how to support
  # these most flexibly is to abstract the logic of the
  # model-weight-determination to a secondary function that eats
  # second_stage_SL_dataset and produces weights; that way the user can swap out
  # whatever they'd like instead, but several handy defaults are supported and
  # already coded up for users.
  #
  # TODO: An option for handling count outcomes / weighting the
  # outcomes/observations -- What may be a solution is multiplying the
  # rows by the square root of the desired weights...
  learner_weights <- determine_super_learner_weights(second_stage_SL_dataset, y_variable)

  # adjust weights according to if using continuous or discrete super-learner
  if (continuous_or_discrete == 'continuous') {
    # nothing needs to be done; leave the learner_weights as-is
  } else if (continuous_or_discrete == 'discrete') {
    max_learner_weight <- which(learner_weights == max(learner_weights))
    if (length(max_learner_weight) > 1) {
      warning("Multiple learners were tied for the maximum weight. Since discrete super-learner was specified, the first learner with the maximum weight will be used.")
      learner_weights <- rep(0, length(learner_weights))
      learner_weights[max_learner_weight[1]] <- 1
    }
  } else {
    stop("Argument continuous_or_discrete must be one of 'continuous' or 'discrete'")
  }

  # fit all of the learners on the entire dataset
  fit_learners <- lapply(
    1:length(learners), function(i) {
      do.call(
        what = learners[[i]],
        args = c(list(
          data = data, regression_formula = regression_formulas[[i]]
          ),
          extra_learner_args[[i]]
        )
      )
    })

  # construct a function that predicts using all of the learners combined using
  # SuperLearned weights
  #
  # this is a closure that will be returned from this function
  predict_from_super_learned_model <- function(newdata) {
    # for each model, predict on the newdata and apply the model weights
    lapply(1:length(fit_learners), function(i) {
      fit_learners[[i]](newdata) * learner_weights[[i]]
    }) |>
      Reduce(`+`, x = _) # aggregate across the weighted model predictions
  }

  if (verbose_output) {
    output <- list(
      sl_predictor = predict_from_super_learned_model,
      y_variable = y_variable,
      holdout_predictions = second_stage_SL_dataset
      )
    class(output) <- c(class(output), "nadir_sl_verbose_output")
    return(output)
  } else {
    class(predict_from_super_learned_model) <- c(class(predict_from_super_learned_model), "nadir_sl_predictor")
    return(predict_from_super_learned_model)
  }
}

#' Parse Formulas for Super Learner
#'
#'
parse_formulas <- function(
    regression_formulas,
    learner_names) {

  if (inherits(regression_formulas, 'formula')) {
    regression_formulas <- rep(c(regression_formulas), length(learner_names)) # repeat the regression formula
    names(regression_formulas) <- learner_names
    return(regression_formulas)
  }

  if (! is.vector(regression_formulas) && all(sapply(regression_formulas, class) == 'formula')) {
    stop("The regression_formulas must be passed as a vector, either a list() or c() vector of formulas.")
  }

  # if the length of the regression formulas matches the number of learners, and
  # the user did not name the regression formulas, then implicitly the user
  # has chosen to pass the regression formulas according to index-based-ordering
  if (length(regression_formulas) == length(learner_names) &&
      is.null(names(regression_formulas))) {
    names(regression_formulas) <- learner_names
    return(regression_formulas)
  }

  if (! is.null(names(regression_formulas))) {
    # either we require that there be as many regression formulas as there are learners
    if (all(learner_names %in% names(regression_formulas))) {
      # order according to learner names in this case
      regression_formulas <- regression_formulas[[learner_names]]
      names(regression_formulas) <- learner_names
      return(regression_formulas)
    }

    # or we require that .default be one of the formulas
    if (".default" %in% names(regression_formulas)) {
      regression_formulas <- lapply(
        learner_names,
        function(learner_name) {
          if (learner_name %in% names(regression_formulas)) {
            return(regression_formulas[[learner_name]])
          } else {
            return(regression_formulas[['.default']])
          }
        })
      names(regression_formulas) <- learner_names
      return(regression_formulas)
    }

    # one edge-case we do support is if the user has specified a vector of formulas,
    # some named, some not-named, but the indexing of the named formulas exactly matches
    # the names of the learners — in that case, we assume they have meant to provide
    # everything in index-based-ordering
    if (length(regression_formulas) == length(learner_names) &&
        all(
          sapply(1:length(regression_formulas), function(i) {
            names(regression_formulas)[i] %in% c("", learner_names[i])
          }))) {
      names(regression_formulas) <- learner_names
      return(regression_formulas)
    }
  }

  # if we've gotten here, none of the above cases applied, and we have a problem.
  #
  stop("Cannot appropriately match the regression_formulas to the learners.
Try making sure the names of the regression_formulas and learners match.
The regression_formulas must one of:
  * a single formula
  * a vector of formulas of the same length as the number of learners specified (with no names).
  * or a named vector of formulas including a '.default' formula and other formulas for specific learners by name.")
}

#' Extract Y Variable from a list of Regression Formulas and Learners
#'
#' @param regression_formulas A vector of formulas used for super learning
#' @param learner_names A character vector of names for the learners
#' @param data_colnames The column names of the dataset for super learning
#' @param y_variable (Optional) the y_variable specified by the user
#'
extract_y_variable <- function(
    regression_formulas,
    learner_names,
    data_colnames,
    y_variable) {
  # get all the y-variables mentioned
  y_variables <- sapply(regression_formulas, function(f) as.character(f)[[2]])

  # if the y_variable is missing and there's a unique y_variable common to
  # all formulas, then we use that
  if (missing(y_variable) & length(unique(y_variables)) == 1) {
    y_variable <- unique(y_variables)
    # if the y_variable is not common to all formulas, we cannot automatically
    # infer which y_variable we should use.
  } else if (missing(y_variable) & length(unique(y_variables)) > 1) {
    stop("Cannot infer the y-variable from the formulas passed.
Please pass y_variable = ... to nadir::super_learner.")
  }

  if (! y_variable %in% data_colnames) {
    stop("The left-hand-side of the regression formula given must appear as a column in the data passed.")
  }

  # if the y_variable matches with any of the learners, we have problems —
  # the output second_stage_SL_dataset wouldn't be interpretable.
  if (y_variable %in% learner_names) {
    stop("The outcome and names of all of the learners must be distinct, because the output
from super_learner is a data.frame with columns including the outcome variable and each of
the learners.")
  }

  return(y_variable)
}


#' Determine SuperLearner Weights with Nonnegative Least Squares
#'
#' This function accepts a dataframe that is structured to have
#' one column `Y` and other columns with unique names corresponding to
#' different model predictions for `Y`, and it will use nonnegative
#' least squares to determine the weights to use for a SuperLearner.
#'
#' @param data A data frame consisting of an outcome (y_variable) and
#' other columns corresponding to predictions from candidate learners.
#' @param yvar The string name of the outcome column in `data`.
#' @return A vector of weights to be used for each of the learners.
#'
#' @export
determine_super_learner_weights_nnls <- function(data, yvar) {
  # use nonlinear least squares to produce a weighting scheme
  index_of_yvar <- which(colnames(data) == yvar)[[1]]
  nnls_output <- nnls::nnls(
    A = as.matrix(data[,-index_of_yvar]),
    b = data[[yvar]])

  return(nnls_output$x)
}
