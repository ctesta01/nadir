# nadir 1.0.0

This release marks `{nadir}`'s readiness to submit to rOpenSci for peer review, 
and incorporates everything described under 0.0.2 below (which was a development
version and was never released to CRAN).

## Breaking changes

* Learner backend packages (e.g., `glmnet`, `ranger`, `xgboost`, `lightgbm`,
  `dbarts`, `e1071`, `earth`, `gbm`, `hal9001`, `kernlab`, `kknn`, `lme4`,
  `mgcv`, `nnet`, `randomForest`, `rpart`, `VGAM`, and others) have been moved
  from `Imports` to `Suggests`, in order to reduce `{nadir}`'s
  installation footprint. Users only need to install the backends for the
  learners they actually use. A new internal `require_backend()` check is
  called throughout, so attempting to use a learner whose backend is not
  installed produces an informative error naming the package to install.

## Documentation and vignettes

* New Glossary article defining the technical terms used throughout the
  package documentation (cross-fitting, out-of-fold predictions, metalearning,
  and so on).
* New Translation Guide article for users arriving from `{SuperLearner}` or
  `{sl3}`, which supersedes and replaces the previous comparison article.
* Expanded the Guidance for Developers article and FAQs.

## Internal changes

* Completed `pkgcheck::pkgcheck()` compatibility in preparation for rOpenSci
  submission.
* Tests for Suggests-backed learners are now skipped gracefully when the
  corresponding backend package is not installed.

# nadir 0.0.2

## New features

* `crossfit_super_learner()` is a newly available method for cross-fitting
  super learners; `cv_super_learner()` has been reworked to use
  `crossfit_super_learner()` internally.
* `super_learner()` output gains `$oof_predictions()` and
  `$oof_predict_modified()` for accessing and working with out-of-fold
  predictions, alongside `$oof_predict()` and `$oof_predict_fold()`.
* New `train_on_whole_dataset` argument to `super_learner()` (default `TRUE`)
  allows users to skip refitting learners on the full dataset after
  cross-validation.
* `add_stratification()` added for stratified learner construction.
* `truncate_lnr()` added for bounding a learner's predictions between
  user-specified minimum and maximum values.
* Learners that error or warn during fitting now have their messages captured
  and stored for user inspection rather than cluttering the console; a new
  vignette covers timing learners via warnings.
* Support for *multi-predictors*: learners that return lists of multiple
  predictors, via `as_multi_predictor()`.
* Formulas may now be passed as character strings (including a single string
  recycled across learners), which are handled via `as.formula()`.
* Fit learners are now returned as part of the super learner object
  (`$fit_learners`).
* `$predict()` on super learned models improved, including support for
  `newdata` that omits the outcome column.

## New learners

* `lnr_lightgbm()` (LightGBM)
* `lnr_bart()` (Bayesian Additive Regression Trees), with careful handling of
  prediction closures when categorical variables are present
* `lnr_knn()` and `lnr_knn_binary()` (k-nearest neighbors)
* `lnr_svm()` and `lnr_svm_binary()` (support vector machines)
* `lnr_cvglmnet()` (cross-validated glmnet)

## New S3 methods

* `nadir_sl_model` and `nadir_crossfit_sl` objects gain `print()`, `plot()`,
  `summary()`, `coef()`, `nobs()`, `formula()`, `fitted()`, and `residuals()`
  methods.
* `plot()` method added for `nadir_cv_sl` objects.
* `fitted()` and `residuals()` are returned in the row-order of the input data
  via an internal `.sl_rowid`, with tests ensuring correctness on
  out-of-order and shuffled inputs.

## Bug fixes

* Fixed predictions on data whose columns are out of order relative to the
  training data.
* Fixed an `obs_weights` bug in weight determination.
* `lnr_glmnet()` (and `lnr_glmnet_grid()`) now work even when some factor
  levels contain zero observations; rare-factor handling also added to
  `lnr_hal()` and `lnr_hal_grid()`.
* Fixed a bug in `lnr_gbm()` when a `weights = NULL` argument was passed.
* Fixed bugs in `lnr_xgboost()` and `lnr_lightgbm()`; updated `lnr_xgboost()`
  so binary learners work again.
* Fixed `n_folds` handling in `cv_super_learner()`.
* `cv_origami_schema()` now errors informatively on non-discrete
  `cluster_ids` / `strata_ids`.
* The negative log loss for binary outcomes now truncates predicted
  probabilities (following the same truncation scheme as `sl3`) to avoid
  `-log(0) = Inf`.
* Weight determination now checks edge cases: a single learner, and
  all-zero model weights (falling back to equal weights), with informative
  errors added throughout `determine_weights_*()`.
* Outcome variables are now validated against the specified outcome type.
* `match.arg()` is now used on all categorical arguments.
* Added a check for datasets that are too small to cross-validate (srr G5.8a).

## Documentation and vignettes

* New vignette on glmnet and HAL approaches to determining weights.
* Updated the basic examples vignette to be more illustrative, and clarified
  the multiclass prediction vignette.
* Removed `super_learner()` from the comparison plot shown by
  `plot.nadir_sl_model()` to avoid overstating performance.
* Many documentation improvements: all arguments documented, ORCID iD format
  updated, removal of links and `:::` usage flagged by CRAN, and shorter
  examples to respect CRAN's runtime limits.

## rOpenSci preparation and internal changes

* Added `srrstats` standards tags throughout the package in preparation for
  rOpenSci submission.
* Package passes `pkgcheck::pkgcheck()`; removed unused code and improved
  docs in response.
* Added `lintr` configuration and resolved lints (`&&`/`||` in `if`,
  `seq_len()`/`seq_along()`, removal of commented-out and unused code,
  line-width limits); `lintr` also caught a never-called validation function,
  which is now wired in.
* Ran `styler::style_pkg()` across the package.
* Expanded the testing suite substantially, including new tests for
  cross-fitting methods, learner leakage, character formulas, and input
  ordering.
* Continuous Integration: removed slow ubuntu-devel/oldrel workflows that spent 
  excessive time installing dependencies.

# nadir 0.0.1

* Initial CRAN submission.
