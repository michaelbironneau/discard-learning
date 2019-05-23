import gym 
from gym import error, spaces, utils 
from gym.utils import seeding 
import numpy

class Discard(gym.Env):
    metadata = {'render.modes': ['human']}
    WIDTH = 100
    HEIGHT = 100
    SITES = [numpy.array([int(WIDTH*0.25), int(HEIGHT*0.25)]), numpy.array([int(WIDTH*0.75), int(HEIGHT*0.75)])]
    FOOD_PER_STEP = 5  # per site, should be an integer
    FLINT_PER_STEP = 0.1  # per site, should be between 0 and 1
    DECAY_PER_STEP = 10
    INITIAL_NOURISHMENT = 100
    MAX_NOURISHMENT_PER_STEP = 5  # The max number of food that can be consumed in one timestep
    INITIAL_FLINT = 10
    MAX_MOVEMENT_PER_STEP = numpy.linalg.norm(SITES[0]-SITES[1])*0.25  # Optimal movement can reach second site in 4 steps
    FLINT_WEIGHT_COEFF = 1  # The higher this is, the slower movement with flint
    FLINT_PER_FOOD = 0.1  # Units of flint consumed per food


    def __init__(self):
        self.state = []
        self.reward = 0
        self.reset()

    def step(self, action):
        """Perform one step in simulation. The action is a 5-element array with each element between 0 and 1: 
            1-4: Movement vectors
            5: How much flint to discard (as proportion of total flint carried)
        """
        # 1. First, move. Don't add anything to the grid yet as it would not have been observable.
        self._move(action)

        #2. Eat and discard flint.
        food_before = self.state[1]
        self._eat_if_possible()
        reward = self.state[1] - food_before  # Alternative - clip to 0

        self._discard_flint_if_necessary(action)

        #3. Pick up flint if possible. This should be done after the discard, as the discard will be in relation to currently carried flint.
        self._pick_up_flint_if_possible()

        #4. Check if we're dead. If not, calculate reward and return. Otherwise, end simulation.
        dead = self._decay()

        if dead:
            return self.state, -100, 1, {}
        else:
            # Modify state with new food and flint
            for i in range(len(Discard.SITES)):
                for _ in range(Discard.FOOD_PER_STEP):
                    self._add_to_state(i, False)

            # Now, add flint
            for i in range(len(Discard.SITES)):
                r = numpy.random.uniform()
                if r < Discard.FLINT_PER_FOOD:
                    self._add_to_state(i, True)            

            return self.state, reward, 0, {}


    def _random_point_in_grid(self, center, std):
        p = numpy.random.normal(center, std)
        p = numpy.array([numpy.clip(int(p[0]), 0, Discard.WIDTH-1), numpy.clip(int(p[1]), 0, Discard.HEIGHT-1)])
        return p

    def _add_to_state(self, site=0, flint=False):
        """Every timestep some flint and food is added at random points concentrated around the sites"""
        assert site == 0 or site == 1
        p = self._random_point_in_grid(Discard.SITES[site], int(0.1*min(Discard.WIDTH, Discard.HEIGHT)))
        if flint:
            self.state[0][p[0], p[1], 1] += 1
        else:
            # food
            self.state[0][p[0], p[1], 0] += 1
        return 
        
    def _decay(self):
        """Every timestep the nourishment level decreases"""
        self.state[1] = self.state[1] - Discard.DECAY_PER_STEP
        if self.state[1] <= 0:
            self.state[1] = 0
            return True 
        return False

    def _eat_if_possible(self):
        """After movement is complete, the agent may eat up to MAX_NOURISHMENT_PER_STEP food provided it has flint"""
        food_here = self.state[0][self.state[3]][0]
        food_here = min(food_here, Discard.MAX_NOURISHMENT_PER_STEP)
        self.state[0][self.state[3]][0] -= food_here
        self.state[1] += food_here

    def _pick_up_flint_if_possible(self):
        """After movement is complete, the agent may pick up any flint in location"""
        flint_here = self.state[0][self.state[3]][1]
        self.state[0][self.state[3]][1] -= flint_here
        self.state[2] += flint_here     

    def _discard_flint_if_necessary(self, action):
        """The agent can discard flint in any location"""
        self.state[0][self.state[3]][1] += action[4]*self.state[2]

    def _move(self, action):
        """The agent can move in any direction it wants, or stay still"""
        # Action contains four directions which the agent can move in [x_1, x_2, x_3, x_4], with x_n \in [0,1]
        # The total movement is the sum of these four vectors weighted by the flint carried
        # Thus it is possible to stand still.
        weight = self.state[2]*Discard.FLINT_WEIGHT_COEFF
        if weight <= 0:
            weight = 1
        new_p = self.state[3] + (1.0/weight)*(action[0]*numpy.array((0,1)) + action[1]*numpy.array((1,0)) + action[2]*numpy.array((0,-1)) + action[3]*numpy.array((-1,0)))
        self.state[3] = new_p

    def reset(self):
        self.state = [numpy.zeros((Discard.WIDTH, Discard.HEIGHT, 2)), Discard.INITIAL_NOURISHMENT, Discard.INITIAL_FLINT, numpy.random.uniform([0,0],[10,10], 2)]
        self.reward = 0
        return self.state

    def render(self, mode='human', close=False):
        pass

    