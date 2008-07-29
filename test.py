#!/usr/bin/python
import numpy.random as R
import numpy as N
import pylab as P
from matplotlib.colors import no_norm
import matplotlib.cm
import logging
from optparse import OptionParser
import cProfile

from utils import *
import model
import inference
import preprocessing
from plotting import *

N.set_printoptions(edgeitems=30)

# seed the RNG (this 
R.seed(24)

def test():
    a = N.arange(-5,5,0.1)
    P.plot(a,pstudent(a,array(0),array(0.1/1.1*4,dtype=N.float64),array(8,dtype=N.float64)))
    b = N.zeros_like(a)
    c = N.zeros_like(a)
    m.set_data(array(1.))
    print m.mun
    for i in range(len(a)):
        b[i] = m.p_predictive(a[i])
        c[i] = m.p_prior(a[i])
    P.plot(a,b)
    P.plot(a,c)
    P.grid()
    P.show()

def get_options():
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename",
        help="write report to FILE", default="test.file", metavar="FILE")
    parser.add_option("-d","--pcadims", dest="pca_dims", type="int",
            help="Number of dimensions to keep after PCA",
            default="2",
            metavar="DIMS")
    parser.add_option("-n","--particles", dest="num_particles", type="int",
            help="Number of particles to use", 
            default=1000,
            metavar="NUM")
    return parser.parse_args()


def logging_setup():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='debug.log',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def prepare_data(options,fn):
    data_file = load_file(fn)
    data_raw = data_file[:,2:].T
    data_time = data_file[:,1].T*1000
    # data = preprocessing.normalize(data_raw)
    # data = preprocessing.pca(data,keep=options.pca_dims)
    return data_raw,data_time

def plot_result(data,labeling,fn="out.pdf"):
    P.clf()
    plot_scatter_2d(data,labeling)
    P.savefig(fn)

def main():
    logging_setup()
    options,args = get_options()
    data,data_time = prepare_data(options,args[0])
    
    
    params = model.DiagonalConjugate.HyperParams(
            a=4,
            b=1,
            mu0=0,
            n0=0.1,
            dims=2
            )
    
    logging.info(params)
    
    inference_params = inference.InferenceParams(
            rho=0.985,
            alpha=0.001,
            p_uniform_deletion=0.99999,
            r_abs = 2
            )
    logging.info(inference_params)
    
    m = model.DiagonalConjugate(params)
    
    #data = R.random_sample((2,100))
    #data_time = N.cumsum(R.rand(100))
    
    pf = inference.ParticleFilter(
            m,
            data,
            data_time,
            inference_params,
            options.num_particles
            )
    pf.run()
    labeling = pf.get_labeling()
    N.savetxt('labeling.txt',labeling)
    print labeling
    plot_result(data,labeling[0,:])

if __name__ == '__main__':
    #cProfile.run("main()")
    main()
