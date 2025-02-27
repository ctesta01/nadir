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
#' a `formula` (a single one passed to `formulas` is fine),
#' and a list of `learners`.
#'
#' `learners` are taken to be lists of functions of the following specification:
#'
#'   * a learner must accept a `data` and `formula` argument,
#'   * a learner may accept more arguments, and
#'   * a learner must return a prediction function that accepts `newdata` and
#' produces a vector of prediction values given `newdata`.
#'
#' In essence, a learner is specified to be a function taking (`data`, `formula`, ...)
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
#' `formulas` argument to either be one fixed formula that
#' `super_learner` will use for all the models, or a vector of formulas,
#' one for each learner specified.
#'
#' Note that in the examples a mean-squared-error (mse) is calculated on
#' the same training/test set, and this is only useful as a crude diagnostic to
#' see that super_learner is working. A more rigorous performance metric to
#' evaluate `super_learner` on is the cv-rmse produced by cv_super_learner.
#'
#' @param data Data to use in training a `super_learner`.
#' @param learners A list of predictor/closure-returning-functions. See Details.
#' @param formulas Either a single regression formula or a vector of regression formulas.
#' @param y_variable Typically `y_variable` can be inferred automatically from the `formulas`, but if needed, the y_variable can be specified explicitly.
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
#' formulas <- c(
#'   .default = mpg ~ cyl + hp, # first three models use same formula
#'   lmer = mpg ~ (1 | cyl) + hp # lme4 uses different language features
#'   )
#'
#' # fit a super_learner
#' sl_model <- super_learner(
#'   data = mtcars,
#'   formula = formulas,
#'   learners = learners,
#'   verbose = TRUE)
#'
#' # We recommend taking a look at this object, and comparing it to the sole function
#' # returned when verbose = FALSE.  tip: It's the $sl_predictor function in the
#' # verbose output.
#' sl_model
#'
#' compare_learners(sl_model)
#'
#' # iris example ---
#' sl_model <- super_learner(
#'   data = iris,
#'   formula = list(
#'     .default = Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width,
#'     lmer = Sepal.Length ~ (Sepal.Width | Species) + Petal.Length),
#'   learners = learners,
#'   verbose = TRUE)
#'
#' # produce super_learner predictions and compare against the individual learners
#' compare_learners(sl_model)
#'
#' @seealso cv_super_learner
#'
#' @export
super_learner <- function(
    data,
    learners,
    formulas,
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
    .sl_fold = rep(1:n_folds, length(learners)),
    learner_name = rep(names(learners), each = n_folds))

  # handle vectorized formulas argument
  #
  # if the formulas is just a single formula, then we repeat it
  # in a vector length(learners) times to make it simple to just pass the ith
  # learner formula[[i]].
  #
  # TODO: Abstract this to a parse_formulas(formulas, learners)
  # function call to handle index AND name-based syntax.
  formulas <- parse_formulas(formulas = formulas,
                                        learner_names = names(learners))

  # handle named extra arguments:
  #   * extra arguments can be passed with a .default option and otherwise named
  #      entries for each learner
  #   * they can be passed as a 1:length(learners) list of extra arguments in order
  #   * they can be passed as a 1:length(learners) list of extra arguments where the names
  #      match 1-1 with the names(learners).
  extra_learner_args <- parse_extra_learner_arguments(
    extra_learner_args = extra_learner_args,
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
          data = training_data[[trained_learners[[i,'.sl_fold']]]],
          formula = formulas[[
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
      trained_learners[[i,'learned_predictor']][[1]](validation_data[[trained_learners[[i, '.sl_fold']]]])
    }
  )

  # from here forward, we just need to use the split + model name + predictions on the test-set
  # to regress against the held-out (validation) data to determine the ensemble weights
  second_stage_SL_dataset <- trained_learners[,c('.sl_fold', 'learner_name', 'predictions_for_testset')]

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
    formulas = formulas,
    learner_names = names(learners),
    data_colnames = colnames(data),
    y_variable = y_variable
  )


  # insert the validation Y data in another column next to the predictions
  second_stage_SL_dataset[[y_variable]] <- lapply(1:nrow(second_stage_SL_dataset), function(i) {
    validation_data[[second_stage_SL_dataset[[i, '.sl_fold']]]][[y_variable]]
  })

  # unnest all of the data (each cell prior to this contained a vector of either
  # predictions or the validation data)
  second_stage_SL_dataset <- tidyr::unnest(second_stage_SL_dataset, cols = colnames(second_stage_SL_dataset))

  # drop the split column so we can simplify the following regression formula
  split_col_index <- which(colnames(second_stage_SL_dataset) == '.sl_fold')

  # regress the validation data on the predictions from every model with no intercept.
  # notice this is now for all of the folds
  #
  # TODO: Here we assume a continuous Y-variable and use non-negative least squares by default to
  # determine the SuperLearner weights;  we may want to think about other types of
  # Y-variables like binary, count, and survival.  My theory on how to support
  # these most flexibly is to abstract the logic of the
  # model-weight-determination to a secondary function that eats
  # second_stage_SL_dataset and produces weights; that way the user can swap out
  # whatever they'd like instead, but several handy defaults are supported and
  # already coded up for users.
  #
  # TODO: An option for handling binary/count outcomes / weighting the
  # outcomes/observations.  I'm not sure how weighting observations fits
  # into this either yet.
  learner_weights <- determine_super_learner_weights(second_stage_SL_dataset[,-split_col_index], y_variable)
  names(learner_weights) <- names(learners)

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
          data = data, formula = formulas[[i]]
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
      learner_weights = learner_weights,
      holdout_predictions = second_stage_SL_dataset
      )
    class(output) <- c(class(output), "nadir_sl_verbose_output")
    return(output)
  } else {
    class(predict_from_super_learned_model) <- c(class(predict_from_super_learned_model), "nadir_sl_predictor")
    return(predict_from_super_learned_model)
  }
}

