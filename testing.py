import unittest
import random
import numpy.random as R
import numpy as N
import pylab as P
import utils
import model
import inference

TEST_FILE = "/home/sm/sync/mscthesis/data/thesis/testing/bar_hillel_demo_short.txt"
SEED_VAL = 25


class TestWalks(unittest.TestCase):
    
    def setUp(self):
        data_file = utils.load_file(TEST_FILE)
        self.data = data_file[:,2:].T
        self.data_time = data_file[:,1].T*1000
        self.params = model.DiagonalConjugate.HyperParams(
                a=4,
                b=1,
                mu0=0,
                n0=0.1,
                dims=2
                )
        
        self.inference_params = inference.InferenceParams(
                rho=0.985,
                alpha=0.001,
                p_uniform_deletion=0.99999,
                r_abs = 2
                )
        
        self.m = model.DiagonalConjugate(self.params)

    def testCompareStorage(self):
        R.seed(SEED_VAL)
        pf1 = inference.ParticleFilter(
                self.m,
                self.data,
                self.data_time,
                self.inference_params,
                10,
                utils.FixedSizeStoreRing
                )
        pf1.run()
        labeling1 = pf1.get_labeling()
        
        R.seed(SEED_VAL)
        pf2 = inference.ParticleFilter(
                self.m,
                self.data,
                self.data_time,
                self.inference_params,
                10,
                utils.FixedSizeStore
                )
        pf2.run()
        labeling2 = pf2.get_labeling()
        self.assert_(N.all(labeling1 == labeling2))

        R.seed(SEED_VAL)
        pf3 = inference.ParticleFilter(
                self.m,
                self.data,
                self.data_time,
                self.inference_params,
                10,
                utils.ArrayOfLists
                )
        pf3.run()
        labeling3 = pf3.get_labeling()
        self.assert_(N.all(labeling3 == labeling1))
        self.assert_(N.all(labeling3 == labeling2))

if __name__ == '__main__':
    unittest.main()


