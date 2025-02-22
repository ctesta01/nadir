`{nadir}`
<a href='https://github.com/ctesta01/nadir/'><img src='logo/nadir_logo.png' align='right' height='138' /></a>
================

*nadir* (noun): nā-dir

> the lowest point.

Fitting with the *minimum loss based estimation*[^1][^2] literature,
`{nadir}` is an implementation of the Super Learner algorithm with
improved support for flexible formula based syntax and that is fond of
functional programming solutions such as closures and currying.

------------------------------------------------------------------------

`{nadir}` implements the Super Learner[^3] algorithm. To quote *the
Guide to SuperLearner*[^4]:

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
included, etc., and similar other modeling decisions in other
frameworks.

Therefore, the `{nadir}` package takes as its charges to:

- Implement a syntax in which it is easy to specify *different formulas*
  for each of many candidate learners.
- To make it easy to pass new learners to the Super Learner algorithm.

# Installation Instructions

At present, `{nadir}` is only available on GitHub.

``` r
devtools::install_github("ctesta01/nadir")
```

<b><i>Warning: this package is currently under active development and
may be wrong!</i> Do not use this for serious applications until this
message has been removed, likely at the time of a future release.</b>

# Demonstration

First, let’s start with the simplest possible use case of
`nadir::super_learner()`, which is where the user would like to feed in
data, a specification for some regression formula(s), specify a library
of learners, and get back a prediction function that is suitable for
plugging into downstream analyses, like in Targeted Learning or for
pure-prediction applications.

Here is a demo of an extremely simple application of using
`nadir::super_learner`:

``` r
library(nadir)

# we'll use a few basic learners
learners <- list(
     glm = lnr_glm,
     rf = lnr_rf,
     glmnet = lnr_glmnet
  )
# more learners are available, see ?learners

sl_model <- super_learner(
  data = mtcars,
  regression_formula = mpg ~ cyl + hp,
  learners = learners)

# the output from super_learner is a prediction function:
# here we are producing predictions based on a weighted combination of the
# trained learners. 
sl_model(mtcars) |> head()
```

    ##         Mazda RX4     Mazda RX4 Wag        Datsun 710    Hornet 4 Drive 
    ##          20.54049          20.54049          25.05098          20.54049 
    ## Hornet Sportabout           Valiant 
    ##          16.54871          20.19491

### One Step Up: Fancy Formula Features

Continuing with our `mtcars` example, suppose the user would really like
to use random effects or similar types of fancy formula language
features. One easy way to do so with `nadir::super_learner` is using the
following syntax:

``` r
learners <- list(
     glm = lnr_glm,
     rf = lnr_rf,
     glmnet = lnr_glmnet,
     lmer = lnr_lmer,
     gam = lnr_gam
  )
  
regression_formulas <- c(
  .default = mpg ~ cyl + hp,   # our first three learners use same formula
  lmer = mpg ~ (1 | cyl) + hp, # both lme4::lmer and mgcv::gam have 
  gam = mpg ~ s(hp) + cyl      # specialized formula syntax
  )

# fit a super_learner
sl_model <- super_learner(
  data = mtcars,
  regression_formulas = regression_formulas,
  learners = learners)
  
sl_model(mtcars) |> head()
```

    ##         Mazda RX4     Mazda RX4 Wag        Datsun 710    Hornet 4 Drive 
    ##          20.33912          20.33912          24.98573          20.33912 
    ## Hornet Sportabout           Valiant 
    ##          16.72556          19.97616

### How should we assess performance of `nadir::super_learner()`?

To put the learners and the super learner algorithm on a level playing
field, it’s important that learners and super learner both be evaluated
on *held-out validation/test data* that the algorithms have not seen
before.

Using the `verbose = TRUE` output from `nadir::super_learner()`, we can
call `compare_learners()` to see the mean-squared-error (MSE) on the
held-out data, also called CV-MSE, for each of the candidate learners
specified.

``` r
# construct our super learner with verbose = TRUE
sl_model <- super_learner(
  data = mtcars,
  regression_formulas = regression_formulas,
  learners = learners,
  verbose = TRUE)
  
compare_learners(sl_model)
```

    ## The default in nadir::compare_learners is to use CV-MSE for comparing learners.

    ## Other metrics can be set using the metric argument to compare_learners.

    ## # A tibble: 1 × 5
    ##     glm    rf glmnet  lmer   gam
    ##   <dbl> <dbl>  <dbl> <dbl> <dbl>
    ## 1  10.6  9.18   10.7  10.9  11.6

Now how should we go about getting the CV-MSE from a super learned
model? We will have to [*curry*](https://en.wikipedia.org/wiki/Currying)
our super learner into a function that only takes in data (with all of
its additional specification built into it) and which returns a
prediction function (i.e., a closure).

Don’t let all this complicated language scare you; it’s fairly
straightforward. Essentially you just need to wrap your super learner
specification inside `sl_closure <- function(data) { ... }`, make sure
you specify `data = data` inside the inner `super_learner()` call, and
you’re done.

The return value from such a function is a closure is because what
`super_learner()` returns is already a closure that eats in `newdata`
and returns predictions.

``` r
sl_closure_mtcars <- function(data) {
  nadir::super_learner(
  data = data,
  regression_formulas = regression_formulas,
  learners = learners)
}

cv_super_learner(data = mtcars, sl_closure_mtcars, yvar = 'mpg')$cv_mse
```

    ## [1] 10.55626

``` r
# iris example ---
sl_model_iris <- super_learner(
  data = iris,
  regression_formula = Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width,
  learners = learners[1:3],
  verbose = TRUE)
  
compare_learners(sl_model_iris)
```

    ## The default in nadir::compare_learners is to use CV-MSE for comparing learners.

    ## Other metrics can be set using the metric argument to compare_learners.

    ## # A tibble: 1 × 3
    ##     glm    rf glmnet
    ##   <dbl> <dbl>  <dbl>
    ## 1 0.103 0.143  0.215

``` r
sl_closure_iris <- function(data) {
  nadir::super_learner(
  data = data,
  regression_formula = Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width,
  learners = learners[1:3])
}

cv_super_learner(data = iris, sl_closure_iris, yvar = 'Sepal.Length')$cv_mse
```

    ## [1] 0.1003356

### What about model hyperparameters or extra arguments?

Model hyperparameters are easy to handle in `{nadir}`. Two easy
solutions are available to users:

- `nadir::super_learner()` has an `extra_learner_args` parameter that
  can be passed a list of extra arguments for each learner.
- Users can always build new learners (which allows for building in the
  hyperparameter specification), and using the `...` syntax, it’s easy
  to build new learners from the learners already provided by `{nadir}`.

Here’s some examples showing each approach.

#### Using `extra_learner_args`:

``` r
# when using extra_learner_args, it's totally okay to use the 
# same learner multiple times as long as their hyperparameters differ.

sl_model <- nadir::super_learner(
  data = mtcars,
  regression_formula = mpg ~ .,
  learners = c(
    glmnet0 = lnr_glmnet,
    glmnet1 = lnr_glmnet,
    glmnet2 = lnr_glmnet,
    rf0 = lnr_rf,
    rf1 = lnr_rf,
    rf2 = lnr_rf
    ),
  extra_learner_args = list(
    glmnet0 = list(lambda = 0.01),
    glmnet1 = list(lambda = 0.1),
    glmnet2 = list(lambda = 1),
    rf0 = list(ntree = 30),
    rf1 = list(ntree = 30),
    rf2 = list(ntree = 30)
    ),
  verbose = TRUE
)

compare_learners(sl_model)
```

    ## The default in nadir::compare_learners is to use CV-MSE for comparing learners.

    ## Other metrics can be set using the metric argument to compare_learners.

    ## # A tibble: 1 × 6
    ##   glmnet0 glmnet1 glmnet2   rf0   rf1   rf2
    ##     <dbl>   <dbl>   <dbl> <dbl> <dbl> <dbl>
    ## 1    11.4    8.62    9.33  7.59  5.99  7.38

#### Building New Learners Programmatically

When does it make more sense to build new learners with the
hyperparameters built into them rather than using the
`extra_learner_args` parameter?

One instance when building new learners may make sense is when the user
would like to produce a large number of hyperparameterized learners
programmatically, for example over a grid of hyperparameter values.
Below we show such an example for a 1-d grid of hyperparameters with
`glmnet`.

``` r
# produce a "grid" of glmnet learners with lambda set to 
# exp(-1 to 1 in steps of .1)
hyperparameterized_learners <- lapply(
  exp(seq(-1, 1, by = .1)), 
  function(lambda) { 
    return(
      function(data, regression_formula, ...) {
        lnr_glmnet(data, regression_formula, lambda = lambda, ...)
        })
  })
  
# give them names because nadir::super_learner requires that the 
# learners argument be named.
names(hyperparameterized_learners) <- paste0('glmnet', 1:length(hyperparameterized_learners))

# fit the super_learner with 20 glmnets with different lambdas
sl_model_glmnet <- nadir::super_learner(
  data = mtcars,
  learners = hyperparameterized_learners,
  regression_formula = mpg ~ .,
  verbose = TRUE)

compare_learners(sl_model_glmnet)
```

    ## The default in nadir::compare_learners is to use CV-MSE for comparing learners.

    ## Other metrics can be set using the metric argument to compare_learners.

    ## # A tibble: 1 × 21
    ##   glmnet1 glmnet2 glmnet3 glmnet4 glmnet5 glmnet6 glmnet7 glmnet8 glmnet9
    ##     <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>
    ## 1    8.69    8.57    8.48    8.34    8.20    8.11    8.08    8.10    8.14
    ## # ℹ 12 more variables: glmnet10 <dbl>, glmnet11 <dbl>, glmnet12 <dbl>,
    ## #   glmnet13 <dbl>, glmnet14 <dbl>, glmnet15 <dbl>, glmnet16 <dbl>,
    ## #   glmnet17 <dbl>, glmnet18 <dbl>, glmnet19 <dbl>, glmnet20 <dbl>,
    ## #   glmnet21 <dbl>

## Coming Down the Pipe

- Automated tests that try to ensure validity/correctness of the
  implementation!
- Reworking some of the internals to use
  - `{future}` and `{future.apply}`
  - `{origami}`
- Adding support for named `extra_learner_args`. Currently support for
  named arguments is only built out for the `regression_formulas`, which
  makes it possible to have the following nice syntax, with options to
  specify options using either index-based or names-based parameters:

``` r
  regression_formulas = list(
    .default = Y ~ .,
    gam = Y ~ s(smoothing_term) + ...,
    lme4 = Y ~ (random|effect) + ...
    )
```

- So far, support for named-sub-arguments has only been implemented for
  the formulas but not for the `extra_learner_args`.
- Hopefully a `pkgdown` website and more vignettes soon.

[^1]: van der Laan, Mark J. and Dudoit, Sandrine, “Unified
    Cross-Validation Methodology For Selection Among Estimators and a
    General Cross-Validated Adaptive Epsilon-Net Estimator: Finite
    Sample Oracle Inequalities and Examples” (November 2003). U.C.
    Berkeley Division of Biostatistics Working Paper Series. Working
    Paper 130. <https://biostats.bepress.com/ucbbiostat/paper130>

[^2]: Zheng, W., & van der Laan, M. J. (2011). Cross-Validated Targeted
    Minimum-Loss-Based Estimation. In Springer Series in Statistics
    (pp. 459–474). Springer New York.
    <https://doi.org/10.1007/978-1-4419-9782-1_27>

[^3]: van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007). Super
    Learner. In Statistical Applications in Genetics and Molecular
    Biology (Vol. 6, Issue 1). Walter de Gruyter GmbH.
    <https://doi.org/10.2202/1544-6115.1309>
    <https://pubmed.ncbi.nlm.nih.gov/17910531/>

[^4]: Guide to `{SuperLearner}`:
    <https://cran.r-project.org/web/packages/SuperLearner/vignettes/Guide-to-SuperLearner.html>
