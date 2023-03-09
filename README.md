# rextendr_value_iteration_gridworld

This does value iteration to find the value of being in each cell of a 5x5 gridworld given a reward function and an end state (cell in which movement on the grid terminates). Movement is windy, so you only get to the cell you want w.p. 0.7. The discount factor beta = 0.95.

Use:

devtools::install_github("cobriant/rextendr_value_iteration_gridworld")

library(rust)

reward <- rep(0.0, 25)

reward[5] <- 1.0

This finds the value of being in each cell on the grid where the upper right hand corner (cell 4 where the 0th cell is the upper left hand corner) is the end state and the only cell with nonzero reward:

rust::value_iteration(reward, obstacles = c(0L, 2L), end_cell = 4L)
