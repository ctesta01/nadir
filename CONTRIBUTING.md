# Contributing to `{nadir}`

#' @srrstats {G1.2} Life Cycle Statement 

# Lifecycle Statement

This package is in a stable state of development, with active subsequent
development planned primarily in response to user feedback and as 
envisioned by the primary authors.


# Developer Conventions

In order to facilitate consistency across the codebase, an effort is made to 
notate what conventions we intend to follow throughout. 

## On Function Arguments 

### Referring to Columns vs. Supplying Vectors

Two related conventions, and the distinction matters:

* Arguments that refer to a **column by name** (and hence take a
  string) end in `_col`, `_cols`, `_var`, or `_variable`. `_variable`
  is used for variables with contextual meaning, like `y_variable`,
  where $y$ has the implied meaning of being the outcome variable.
* Arguments that take a **vector of values** (one per observation) end
  in `_ids`: `rowids`, `cluster_ids`, `strata_ids`. These are the data
  themselves, not a column name.

Standardized names used throughout — non-standard variants of these are
discouraged:

* `data` — always the `data.frame`-like input
* `y_variable`, `id_col`, `covariate_cols`
* `rowids`, `cluster_ids`, `strata_ids`
* `outcome_type` — one of `nadir_supported_types`
* `n_folds` (and `inner_n_folds` for cross-fitting)
* `loss_metric`, `cv_schema`, `extra_learner_args`
### Optional Singleton or List Arguments

As of now, there is only **one** acceptable place where we readily and 
often use partial argument matching (<https://stackoverflow.com/a/14155259/3161979>),
and that for the `formulas` argument to `nadir::super_learner()`. 

The reason partial argument matching is used throughout much of the documentation
with the `formulas` argument to `super_learner()` is that in the case when 
only one formula will be used across all the learners, the user may (either
by preference or without thinking) only pass `formula = <...>` to `super_learner()`. 

We make a concerted effort to not use partial argument matching in 
examples or documentation except for with regard to the `formulas` argument
to `super_learner()`. In part, the usage of a partial argument match `formula`
in `super_learner()` is acceptable because there is control-flow/logic that
detects if a single formula was passed (rather than a named list of formulas). 

### Protected Arguments

Certain arguments should always follow a standard convention, and are privileged
above others. 

  * We prefer to use `data` in all of our function arguments to refer to `data.frame`s. 
  * We explicitly recognize `weights` as an argument in all learners where possible. 
  Moreover, if `weights` are passed, then explicit code to handle them appropriately given the underlying 
  model's syntax is included in each learner that comes with `nadir` so that 
  `nadir::super_learner()` can pass observation weights to all included candidate learners 
  and rely on them being handled properly. 

## Code Style 

* We used `styler::style_pkg()` to get the code in shape before enforcing linting
  and our `.lintr` configuration. 
* We often use an explicit `return()` at function exits for code clarity and 
  easier readability.

## Packaging and Sending to CRAN 

Useful guidance on releasing to CRAN is available here: 

  * https://r-pkgs.org/release.html 

On the NEWS.md structure:  

  * https://blog.r-hub.io/2020/05/08/pkg-news/
  
## Please make sure to also look at `vignettes/articles/Guidance-for-Developers.Rmd`!
