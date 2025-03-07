test_that("super_learner_helpers work as intended", {

  # section on validate_learner_types :
  #
  # we should get warnings if we use the wrong learner types and no warnings if
  # we use the right learner types
  expect_warning({
    validate_learner_types(
      list(mean = lnr_mean, lm = lnr_lm), 'density')
  })
  expect_no_warning({
    validate_learner_types(
      list(lnr_mean, lnr_lm), 'continuous')
  })
  expect_no_warning({
    validate_learner_types(
      list(lnr_lm_density, lnr_homoskedastic_density), 'density')
  })
  expect_no_warning({
    validate_learner_types(
      list(lnr_glm, lnr_mean), 'binary')
  })
})
