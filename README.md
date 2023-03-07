# rextendr_value_iteration_gridworld

devtools::install_github("cobriant/rextendr_value_iteration_gridworld")

library(rust)

reward <- rep(0.0, 25)

reward[5] <- 1.0

rust::value_iteration(reward, 4L)
