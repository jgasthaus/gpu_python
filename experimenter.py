#!/usr/bin/python
import numpy.random as R
import numpy as N
import pylab as P
import logging
from optparse import OptionParser
from cfgparse import ConfigParser
import cPickle

from utils import *
import model
import inference
import preprocessing
from plotting import *


def parse_array_string(str):
    """Parse a string option into a numpy array of type double."""
    return (N.fromstring(str,dtype=float64,sep=' '),None)

def parse_tuple_string(str):
    return tuple(float(s) for s in str[1:-1].split(','))

def get_kernel_class(str):
    return (
        {'caron':model.CaronIndependent,
         'metropolis':model.MetropolisWalk}[str],
        None
        )

def get_storage_class(str):
    return (
        {'fixed':FixedSizeStore,
         'dynamic':ArrayOfLists,
         'ring':FixedSizeStoreRing}[str],
        None
        )

def get_resampling_function(str):
    return (
        {'multinomial':inference.multinomial_resampling,
         'residual':inference.residual_resampling,
         'stratified':inference.stratified_resampling,
         'systematic':inference.systematic_resampling}[str],
        None
        )


def handle_options():
    """Handle command line and config file options."""
    o = OptionParser()
    c = ConfigParser()
    c.add_optparse_help_option(o)
    c.add_optparse_files_option(o)
    c.add_file('default.cfg')
    o.add_option("-v","--verbose", action="store_true", dest="verbose",
            help="Be verbose.",default=False)
    o.add_option("--debug", action="store_true", dest="debug",
            help="Print extra debugging info to STDOUT", default=False)
    o.add_option("-i", "--identifier", type="string",
            help="Unique identifier for this run.", metavar="ID")
    c.add_option("identifier")
    o.add_option("-f", "--filename", type="string",
            help="File name of the data file", metavar="FILE")
    c.add_option("filename")
    o.add_option("-n","--particles", type="int",
            help="Number of particles to use", 
            metavar="NUM")
    c.add_option('particles')
    o.add_option("-a","--algorithm",dest="algorithm",type="choice",
            metavar="ALG",choices=("pf","gibbs"),
            help="Inference algorithm to use; either pf or gibbs.")
    c.add_option('algorithm')
    o.add_option("-d","--dims",dest="use_dims",type="int",
            help="Number of dimensions to use. If 0, all data dimensions will"+
            " be used.")
    c.add_option("dims",dest="use_dims",default=0)
    o.add_option("--rows",dest="use_rows",type="int",
            help="Number of rows of data to be used. If 0, all are used.")
    c.add_option("rows",dest="use_rows",default=0)

    ### Model options
    c.add_option('a',check=parse_array_string,
            help="Alpha parameter of the Gamma prior")
    c.add_option('b',check=parse_array_string,
            help="Beta parameter of the Gamma prior")
    c.add_option('mu0',check=parse_array_string,
            help="mu0 parameter of the prior")
    c.add_option('n0',check=parse_array_string,
            help="n0 parameter of the prior")

    c.add_option("kernel_class",type="choice",
            choices=("caron","metropolis"), check=get_kernel_class,
            help="Transition kernel; either 'caron' or 'metropolis'")
    c.add_option("aux_vars",type="int",
            help="Number of auxiliary variables to use in the Caron kernel.")
    c.add_option("variance_factor",type="float",
            help="Variance scaling factor to use in the Caron kernel.")
    c.add_option("mean_sd",type="float",
            help="Standard deviation of the proposal for the mean in the" +
                 " Metropolis kernel.")
    c.add_option("precision_sd",type="float",
            help="Standard deviation of the proposal for the precision in the" +
                 " Metropolis kernel.")

    ### Other model options
    o.add_option("--alpha",type="float",
            help="The DP concentration parameter alpha.")
    c.add_option("alpha")
    o.add_option("--rho",type="float",
            help="Probability of survival of an allocation variable")
    c.add_option("rho")
    c.add_option("p_uniform_deletion",type="float",
            help="Probability of uniform deletion.")
    o.add_option("--rp",type="float",
            help="Length of the absolute refractory period (in ms)")
    c.add_option("rp")
   
    ### Particle Filter Options
    c.add_option("storage_class",type="choice",check=get_storage_class,
            choices=("dynamic","fixed","ring"),
            help="Storage container to use; one of ('dynamic','fixed','ring')")
    c.add_option("resampling_method", type="choice",
            check=get_resampling_function,
            choices=("multinomial","residual","stratified","systematic"),
            help="Resampling scheme; one of (multinomial,residual,stratified,"+
                 "systematic)")


    (options,args) = c.parse(o)
    set_kernel_parameters(options)
    return (options,args)


def set_kernel_parameters(options):
    """Construct and set the kernel paramters tuple from the given options."""
    if options.kernel_class==model.CaronIndependent:
        options.kernel_params = (
                options.aux_vars,
                options.variance_factor
                )
    elif options.kernel_class == model.MetropolisWalk:
        options.kernel_params = (
                options.mean_sd,
                options.precision_sd
                )


def load_data(options):
    """Load the data file specified in the options."""
    # also, make sure that use_dims and use_rows is adhered to
    # and set to the correct value (if 0)
    logging.info("Loading data from " + options.filename)
    data_file = load_file(options.filename)
    data_raw = data_file[:,2:].T
    data_time = data_file[:,1].T
    labels = array(data_file[:,0].T,dtype=int32)
    D,N = data_raw.shape
    if options.use_dims == 0:
        options.use_dims = D
    if options.use_rows == 0:
        options.use_rows = N
    D = min(options.use_dims,D)
    N = min(options.use_rows,N)
    data_raw = data_raw[0:D,0:N]
    data_time = data_time[0:N]
    labels = labels[0:N]
    return data_raw,data_time,labels

def get_model(options):
    params = model.DiagonalConjugateHyperParams(
            a=options.a,
            b=options.b,
            mu0=options.mu0,
            n0=options.n0,
            dims=options.use_dims
            )
    logging.info(params)

    m = model.DiagonalConjugate(
            hyper_params=params,
            kernelClass=options.kernel_class,
            kernelParams=options.kernel_params
            )
    return m

def logging_setup(options):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='debug.log',
                        filemode='w')
    if options.verbose:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        if options.debug:
            console.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def get_inference_params(options):
    inference_params = inference.InferenceParams(
            rho=options.rho,
            alpha=options.alpha,
            p_uniform_deletion=options.p_uniform_deletion,
            r_abs = options.rp
            )
    logging.info(inference_params)
    return inference_params

def run_pf(data,data_time,model,inference_params,options):
    pf = inference.ParticleFilter(
            model,
            data,
            data_time,
            inference_params,
            options.particles,
            storage_class = options.storage_class,
            resample_fun = options.resampling_method,
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


def main():
    opts, args = handle_options()
    logging_setup(opts)
    data,data_time,labels = load_data(opts)
    print opts
    model = get_model(opts)
    ip = get_inference_params(opts)
    if opts.algorithm == "pf":
        run_pf(data,data_time,model,ip,opts)


if __name__ == "__main__":
    main()
