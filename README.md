# rextendr_value_iteration_gridworld

This does value iteration to find the value of being in each cell of a 5x5 gridworld given a reward function. Movement may be windy and time steps can be discounted by a discount factor beta. Movement can be deterred with a vector of obstacles. If you want no obstacles, set obstacles = -1L.

Use:

devtools::install_github("cobriant/rextendr_value_iteration_gridworld")

library(rust)

reward <- rep(0.0, 25)

reward[5] <- 1.0

This finds the value of being in each cell on the grid where the upper right hand corner (cell 4 where the 0th cell is the upper left hand corner) is the only cell with nonzero reward:

v <- value_iteration(reward, obstacles = -1L, wind = 0.7, beta = 0.95)

The value function:
v[1:25]

value_action:
matrix(v[26:125], nrow = 25, byrow = T)
