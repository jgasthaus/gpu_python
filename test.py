#!/usr/bin/python
import numpy.random as R
import numpy as N
import pylab as P
from matplotlib.colors import no_norm
import matplotlib.cm
import logging
from optparse import OptionParser
import cProfile
import cPickle,pickle

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
    parser = OptionParser(usage="%prog [options] filename")
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
    parser.add_option("-a","--algorithm",dest="algorithm",type="choice",
            default="pf",metavar="ALG",choices=("pf","gibbs"),
            help="Inference algorithm to use; either pf or gibbs.")
    return parser.parse_args()


def logging_setup():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='debug.log',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    #console.setLevel(logging.DEBUG)
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

def get_model(options):
    params = model.DiagonalConjugateHyperParams(
            a=4,
            b=1,
            mu0=0,
            n0=0.1,
            dims=2
            )
    logging.info(params)

    m = model.DiagonalConjugate(
            hyper_params=params,
            kernelClass=model.CaronIndependent,
            kernelParams=tuple([50,0.5])
            )
    return m

def main():
    logging_setup()
    options,args = get_options()
    data,data_time = prepare_data(options,args[0])
    if options.algorithm == "pf":
        pf_test(data,data_time,options)
    elif options.algorithm == "gibbs":
        gibbs_test(data,data_time,options)
    

def pf_test(data,data_time,options):
    m = get_model(options)
    inference_params = inference.InferenceParams(
            rho=0.985,
            alpha=0.1,
            p_uniform_deletion=0.99999,
            r_abs = 2
            )
    logging.info(inference_params)
    
    
    #data = R.random_sample((2,100))
    #data_time = N.cumsum(R.rand(100))
    map_collector = MAPCollector() 
    pf = inference.ParticleFilter(
            m,
            data,
            data_time,
            inference_params,
            options.num_particles,
            storage_class = FixedSizeStore,
            before_resampling_callback=map_collector
            )
    pf.run()
    labeling = pf.get_labeling()
    firstmu = N.zeros(pf.T)
    firstlam = N.zeros(pf.T)
    for t in range(pf.T):
        firstmu[t] = pf.particles[0].U.get_array(t)[0].mu[0]
        firstlam[t] = pf.particles[0].U.get_array(t)[0].lam[0]
    print firstmu
    P.subplot(2,1,1)
    P.plot(firstmu)
    P.subplot(2,1,2)
    P.plot(firstlam)
    P.show()
    N.savetxt('labeling.txt',labeling)
    print labeling
    plot_result(data,labeling[0,:])
    outf = open('aparticle.pkl', 'wb')
    cPickle.dump(pf.particles[0],outf)
    outf.close()


def gibbs_test(data,data_time,options):
    """Simple testbed for the Gibbs sampler development."""
    BURN_IN = 2
    m = get_model(options)
    state = model.GibbsState(cPickle.load(open('aparticle.pkl','rb')))
    print "mstore:0,0", state.mstore[0,0]
    state.U[0,0] = None
    state.check_consistency()
    raw_input()
    sampler = inference.GibbsSampler(data,data_time,model,state) 
    for n in range(BURN_IN):
        logging.info("Burn-in sweep %i of %i" % (n+1,BURN_IN))
        sampler.sweep()
    # print sampler.state



if __name__ == '__main__':
    #cProfile.run("main()")
    main()
