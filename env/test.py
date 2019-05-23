import unittest 
from discard_env import Discard
import numpy

class TestDiscardEnv(unittest.TestCase):
    def test_initial(self):
        d = Discard()
        d.reset()
        self.assertEqual(numpy.sum(d.state[0][:,:,0]), Discard.FOOD_PER_STEP*2) # *2 because there are two sites
        self.assertEqual(d.state[1], Discard.INITIAL_NOURISHMENT)
        self.assertEqual(d.state[2], Discard.INITIAL_FLINT)
        self.assertTrue(d.state[3][0]>= 0 and d.state[3][0] <= Discard.WIDTH)
        self.assertTrue(d.state[3][1]>= 0 and d.state[3][0] <= Discard.HEIGHT)

    def test_movement(self):
        d = Discard() 
        d.reset() 
        d.state[2] = 0  # No flint

        d.state[3] = numpy.array((1,1))
        d._move([1,0,0,0,0])
        self.assertTrue(numpy.array_equal(d.state[3], numpy.array([1,2])) )

        d.state[3] = numpy.array((1,1))
        d._move([0,1,0,0,0])
        self.assertTrue(numpy.array_equal(d.state[3], numpy.array([2,1])) )

        d.state[3] = numpy.array((1,1))
        d._move([0,0,1,0,0])
        self.assertTrue(numpy.array_equal(d.state[3], numpy.array([1,0])) )

        d.state[3] = numpy.array((1,1))
        d._move([0,0,0,1,0])
        self.assertTrue(numpy.array_equal(d.state[3], numpy.array([0,1])) )

    def test_eating(self):
        d = Discard() 
        d.reset() 
        d.state[2] = 0  # No flint so can't eat
        d.state[0][0,0,0] = 1
        d.state[3] = numpy.array([0,0])
        d.state[1] = 0

        d._eat_if_possible()
        self.assertEqual(d.state[1], 0)  # Can't eat if we have no flint
        self.assertEqual(d.state[0][0,0,0], 1)  # The food should not have disappeared as we didn't eat it

        d.state[2] = 1
        d._eat_if_possible() 
        self.assertEqual(d.state[1], 1)  # We should be able to eat it all
        self.assertTrue(d.state[2] < 1)  # Flint should go down
        self.assertEqual(d.state[0][0,0,0], 0)  # Food should have disappeared as we ate it!

    def test_pickup_flint(self):
        d = Discard() 
        d.reset() 
        d.state[2] = 0  # No flint 
        d.state[0][0,0,1] = 1
        d.state[3] = numpy.array([0,0])
        d._pick_up_flint_if_possible()

        self.assertEqual(d.state[2], 1)
        self.assertEqual(d.state[0][0,0,1],0)    

    def test_discard(self):
        d = Discard() 
        d.reset() 
        d.state[2] = 4
        d.state[3] = numpy.array([0,0])
        d.state[0][0,0,1] = 0
        d._discard_flint_if_necessary([0,0,0,0,0.75])
        self.assertEqual(d.state[2], 1)
        self.assertEqual(d.state[0][0,0,1],3)


    def test_decay(self):
        d = Discard()
        d.reset() 
        d._decay()
        self.assertLess(d.state[1], Discard.INITIAL_NOURISHMENT)  # Nourishment should decay
        self.assertEqual(d.state[2], Discard.INITIAL_FLINT)  # Flint should not decay


if __name__ == '__main__':
    unittest.main()