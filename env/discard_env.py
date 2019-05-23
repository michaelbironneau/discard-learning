import gym 
from gym import error, spaces, utils 
from gym.utils import seeding 
import numpy

class Discard(gym.Env):
    metadata = {'render.modes': ['human']}
    WIDTH = 100
    HEIGHT = 100
    SITES = [numpy.array([int(WIDTH*0.25), int(HEIGHT*0.25)]), numpy.array([int(WIDTH*0.75), int(HEIGHT*0.75)])]
    FOOD_PER_STEP = 5  # per site
    FLINT_PER_STEP = 1
    DECAY_PER_STEP = 10
    INITIAL_NOURISHMENT = 100
    MAX_NOURISHMENT_PER_STEP = 5  # The max number of food that can be consumed in one timestep
    INITIAL_FLINT = 10
    MAX_MOVEMENT_PER_STEP = numpy.linalg.norm(SITES[0]-SITES[1])*0.25  # Optimal movement can reach second site in 4 steps
    FLINT_WEIGHT_COEFF = 1  # The higher this is, the slower movement with flint


    def __init__(self):
        self.state = []

    def step(self, action):
        pass 

    def _random_point_in_grid(self, center, std):
        p = numpy.random.normal(center, std)
        p = numpy.array([numpy.clip(int(p[0]), [0, Discard.WIDTH-1]), numpy.clip(int(p[1]), [0, Discard.HEIGHT-1])])
        return p

    def _add_to_state(self, site=0, flint=False):
        assert site == 0 or site == 1
        p = self._random_point_in_grid(Discard.SITES[site], int(0.1*WIDTH))
        if flint:
            self.state[0][p[0], p[1], 1] += 1
        else:
            # food
            self.state[0][p[0], p[1], 0] += 1
        return 
        
    def _decay(self):
        self.state[1] = self.state[1] - Discard.DECAY_PER_STEP
        if self.state[1] <= 0:
            self.state[1] = 0

    def _move(self, action):
        # Action contains four directions which the agent can move in [x_1, x_2, x_3, x_4], with x_n \in [0,1]
        # The total movement is the sum of these four vectors weighted by the flint carried
        # Thus it is possible to stand still.
        weight = self.state[2]*Discard.FLINT_WEIGHT_COEFF
        if weight <= 0:
            weight = 1
        new_p = self.state[3] + (1.0/weight)*(action[0]*numpy.array((0,1)) + action[1]*numpy.array((1,0)) + action[2]*numpy.array((0,-1)) + action[3]*numpy.array((-1,0)))
        self.state[2] = new_p

    def reset(self):
        self.state = [numpy.zeros((Discard.WIDTH, Discard.HEIGHT, 2)), Discard.INITIAL_NOURISHMENT, Discard.INITIAL_FLINT, numpy.random.uniform([0,0],[10,10], 2)]

    def render(self, mode='human', close=False):
        pass

    