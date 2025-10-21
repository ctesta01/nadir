
# Developer Conventions

In order to facilitate consistency across the codebase, an effort is made to 
notate what conventions we intend to follow throughout. 

## On Function Arguments 

### Referring to Column Names

Where functions take an argument of a column name, we expect string/character
values to be passed. In order to make it clear to users, arguments that 
refer to a column name (and hence should be a string value) will end in `_col`,
`_cols`, `_var` or `_variable` depending on the context.  `_variable` 
is used to refer to variables with contextual meaning, like `y_variable` where 
$y$ has the implied meaning of being the outcome variable. 

There are a few standardized names that we will try to use throughout, so 
non-standard variants of the following are discouraged: 

  * `id_col` 
  * `y_variable`
  * `covariate_cols`

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

