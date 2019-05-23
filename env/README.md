# Discard Simulation Environment 

Gym environment (following tutorial at https://github.com/apoddar573/Tic-Tac-Toe-Gym_Environment/blob/master/gym-tictac4/gym_tictac4/envs/tictac4_env.py). 

## State

### Neighborhood Observation

All observations are local (i.e. smooth space). First component of observation vector is M x N x 2 tensor containing food and flint in each coordinate of observable grid. 

Then there is a 2-element state vector with nourishment level and flint level.

### Rules

1. Nourishment level decreases every timestep
2. Reward is equal to nourishment level. The game ends if nourishment level reaches 0.
3. Flint and food appear near pre-determined hotspots (sites) on the grid. Flint appears very rarely, food appears regularly.
4. If flint level is 0, food cannot be consumed.
5. The agent can move. The higher the level of carried flint, the slower the movement. 
6. Consuming food reduces the carried flint level (at a slow rate).
