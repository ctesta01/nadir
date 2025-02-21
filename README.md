`{nadir}`
================

`{nadir}` implements the Super Learner[^1][^2] algorithm. To quote [^3]:

> SuperLearner is an algorithm that uses cross-validation to estimate
> the performance of multiple machine learning models, or the same model
> with different settings. It then creates an optimal weighted average
> of those models, aka an “ensemble”, using the test data performance.
> This approach has been proven to be asymptotically as accurate as the
> best possible prediction algorithm that is tested.

## Why `{nadir}` and why reimplement Super Learner again?

In previous implementations
([`{SuperLearner}`](https://github.com/ecpolley/SuperLearner),
[`{sl3}`](https://github.com/tlverse/sl3/),
[`{mlr3superlearner}`](https://cran.r-project.org/web/packages/mlr3superlearner/mlr3superlearner.pdf)),
support for *flexible formula-based syntax* has been limited, instead
opting for specifying learners as models on an $X$ matrix and $Y$
outcome vector. Many popular R packages such as `lme4` and `mgcv` (for
random effects and generalized additive models) use formulas extensively
to specify models using syntax like `(age | strata)` to specify random
effects on age by strata, or `s(age, income)` to specify a smoothing
term on `age` and `income` simultaneously.

At present, it is difficult to use these kinds of features in
`{SuperLearner}`, `{sl3}` and `{ml3superlearner}`.

For example, it is easy to imagine the Super Learner algorithm being
appealing to modelers fond of random effects based models because they
may want to hedge on the exact nature of the random effects models, not
sure if random intercepts are enough or if random slopes should be
included, etc.it is easy to imagine the Super Learner algorithm being
appealing to epidemiologists because they may want to hedge on the exact
nature of the random effects models, not sure if random intercepts are
enough or if random slopes should be included, and similar other
modeling decisions in other languages.

Therefore, the `{nadir}` package takes as its charges to:

- Implement a syntax in which it is easy to specify *different formulas*
  for each of many candidate learners.
- To make it easy to pass new learners to the Super Learner algorithm.

# Demonstration

``` r
library(nadir)

learners <- list(
     glm = lnr_glm,
     rf = lnr_rf,
     glmnet = lnr_glmnet,
     lmer = lnr_lmer
  )

# mtcars example ---
regression_formulas <- c(
  rep(c(mpg ~ cyl + hp), 3), # first three models use same formula
  mpg ~ (1 | cyl) + hp # lme4 uses different language features
  )

# fit a super_learner
sl_model <- super_learner(
  data = mtcars,
  regression_formula = regression_formulas,
  learners = learners)

# produce super_learner predictions
sl_model_predictions <- sl_model(mtcars)
# compare against the predictions from the individual learners
fit_individual_learners <- lapply(1:length(learners), function(i) { learners[[i]](data = mtcars, regression_formula = regression_formulas[[i]]) } )
individual_learners_mse <- lapply(fit_individual_learners, function(fit_learner) { mse(fit_learner(mtcars) - mtcars$mpg) })
names(individual_learners_mse) <- names(learners)

print(paste0("super-learner mse: ", mse(sl_model_predictions - mtcars$mpg)))
```

    ## [1] "super-learner mse: 4.66650603458354"

``` r
individual_learners_mse
```

    ## $glm
    ## [1] 9.124205
    ## 
    ## $rf
    ## [1] 4.899838
    ## 
    ## $glmnet
    ## [1] 9.167678
    ## 
    ## $lmer
    ## [1] 8.744686

``` r
# iris example ---
sl_model <- super_learner(
  data = iris,
  regression_formula = Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width,
  learners = learners[1:3])

# produce super_learner predictions and compare against the individual learners
sl_model_predictions <- sl_model(iris)
fit_individual_learners <- lapply(learners[1:3], function(learner) { learner(data = iris, regression_formula = Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width) } )
individual_learners_mse <- lapply(fit_individual_learners, function(fit_learner) { mse(fit_learner(iris) - iris$Sepal.Length) })

print(paste0("super-learner mse: ", mse(sl_model_predictions - iris$Sepal.Length)))
```

    ## [1] "super-learner mse: 0.0795156530281833"

``` r
individual_learners_mse
```

    ## $glm
    ## [1] 0.0963027
    ## 
    ## $rf
    ## [1] 0.04204984
    ## 
    ## $glmnet
    ## [1] 0.2035002

[^1]: van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007). Super
    Learner. In Statistical Applications in Genetics and Molecular
    Biology (Vol. 6, Issue 1). Walter de Gruyter GmbH.
    <https://doi.org/10.2202/1544-6115.1309>
    <https://pubmed.ncbi.nlm.nih.gov/17910531/>

[^2]: Guide to `{SuperLearner}`:
    <https://cran.r-project.org/web/packages/SuperLearner/vignettes/Guide-to-SuperLearner.html>

[^3]: Guide to `{SuperLearner}`:
    <https://cran.r-project.org/web/packages/SuperLearner/vignettes/Guide-to-SuperLearner.html>
