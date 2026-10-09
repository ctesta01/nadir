# helper files run before any test file: load parallelization namespaces up
# front so one-time load warnings (e.g. "built under R version X") can never
# leak into expect_no_warning() / expect_warning() assertions in tests.
suppressWarnings({
  library(future)
  requireNamespace("future.apply", quietly = TRUE)
})
