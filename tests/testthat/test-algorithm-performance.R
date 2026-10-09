suppressWarnings(library(future))
library(testthat)

# our goal is to show the correct performance of the algorithm
#
# we run in a monte carlo fashion 10 replicated simulations in
# 3 scenarios where as scenarios progress 1->2->3 it is more and more
# evident which candidate learner should win all of the weight from
# \code{nadir::super_learner()}.
#

test_that("the nadir::super_learner() makes better decisions with more data.", {

  scenarios <- c('low info', 'med info', 'high info')

  dgp <- function(n, seed, scenario) {
    # check the scenario
    scenario <- match.arg(scenario, choices = scenarios)

    set.seed(seed)

    X1 <- rnorm(n = n)
    X2 <- rnorm(n = n)
    beta <- switch(scenario,
                   'low info' = 1,
                   'med info' = 2,
                   'high info' = 3)
    Y <- rnorm(n = n, mean = X1 + beta * X2^2)

    data.frame(X1, X2, Y)
  }

  fit_nadir_sl_model <- function(n, seed, scenario) {
    df <- dgp(n = n, seed = seed, scenario = scenario)

    sl_fit <- nadir::super_learner(
      data = df,
      learners = list(lnr_mean,
                      lm1 = lnr_lm,
                      lm2 = lnr_lm),
      formulas = list(
        .default = Y ~ X1 + X2,
        lm2 = Y ~ X1 + poly(X2, 2)),
      n_folds = 2,
      train_on_whole_dataset = FALSE)

    return(sl_fit)
  }

  check_lm2_weight <- function(sl_fit) {
    # lm2 with the quadratic term should get all the weight
    return(sl_fit$learner_weights[['lm2']])
  }

  repeat_sl_experiment <- function(ntimes, n, scenario) {
    sapply(1:ntimes, function(nth_time) {
      sl_fit <- fit_nadir_sl_model(n = n, seed = nth_time, scenario = scenario)
      check_lm2_weight(sl_fit)
    })
  }

  # run_experiment_over_sample_sizes
  sample_sizes <- c(10, 15, 20)
  experiment_results <- sapply(sample_sizes, \(sample_size) {
    repeat_sl_experiment(ntimes = 4, n = sample_size, scenario = 'low info')
  })

  # get the average weight by sample size
  experiment_avg_lm2_weight_by_sample_size <- experiment_results |> colMeans()

  #' @srrstats {G5.7} test that as SAMPLE SIZE increases the fit improves
  expect_true(
    experiment_avg_lm2_weight_by_sample_size[2] >= experiment_avg_lm2_weight_by_sample_size[1]
  )
  expect_true(
    experiment_avg_lm2_weight_by_sample_size[3] >= experiment_avg_lm2_weight_by_sample_size[2]
  )

  experiment2_results <- sapply(scenarios, \(scenario) {
    repeat_sl_experiment(ntimes = 4, n = 15, scenario = scenario)
  })

  # get the average lm2 weight as beta increases
  #' @srrstats {G5.7} we test that as the strength of beta increases on a
  #' quadratic term, the nadir::super_learner() does a better and better job
  #' of picking the model that includes a quadratic term.
  experiment2_avg_lm2_weight_by_scenario <- experiment2_results |> colMeans()

  expect_true(
    experiment2_avg_lm2_weight_by_scenario[2] >= experiment2_avg_lm2_weight_by_scenario[1]
  )
  expect_true(
    experiment2_avg_lm2_weight_by_scenario[3] >= experiment2_avg_lm2_weight_by_scenario[2]
  )

})
