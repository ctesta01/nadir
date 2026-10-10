# helper files run before any test file: load parallelization namespaces up
# front so one-time load warnings (e.g. "built under R version X") can never
# leak into expect_no_warning() / expect_warning() assertions in tests.
suppressWarnings({
  library(future)
  requireNamespace("future.apply", quietly = TRUE)
})

# load the learner backends
learner_backends <- c(
  lnr_glmnet = "glmnet", lnr_cvglmnet = "glmnet",
  lnr_glmnet_grid = "glmnet", lnr_hal = "hal9001",
  lnr_hal_grid = "hal9001", lnr_xgboost = "xgboost",
  lnr_lightgbm = "lightgbm", lnr_gbm = "gbm", lnr_bart = "dbarts",
  lnr_earth = "earth", lnr_gam = "mgcv", lnr_lmer = "lme4",
  lnr_glmer = "lme4", lnr_gausspr = "kernlab", lnr_svm = "e1071",
  lnr_svm_binary = "e1071", lnr_knn = "kknn", lnr_knn_binary = "kknn",
  lnr_ranger = "ranger", lnr_ranger_binary = "ranger",
  lnr_multinomial_ranger = "ranger", lnr_rf = "randomForest",
  lnr_rf_binary = "randomForest", lnr_rpart = "rpart",
  lnr_rpart_binary = "rpart", lnr_nnet = "nnet",
  lnr_multinomial_nnet = "nnet", lnr_multinomial_vglm = "VGAM"
)

backend_available <- function(lnr_name) {
  pkg <- learner_backends[lnr_name]
  is.na(pkg) || requireNamespace(pkg, quietly = TRUE)
}
