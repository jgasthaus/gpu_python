#!/usr/bin/python
import numpy.random as R
from numpy import float64, fromstring, zeros, savetxt, vstack
import logging
from optparse import OptionParser
from cfgparse import ConfigParser
import cPickle
import os.path as P
import os
from os.path import abspath, exists
from utils import *
import model
import inference
import preprocessing
import plotting


def parse_array_string(str):
    """Parse a string option into a numpy array of type double."""
    return (fromstring(str,dtype=float64,sep=' '),None)

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
    f = c.add_file('default.cfg')
    o.add_option("-v","--verbose", action="store_true", dest="verbose",
            help="Be verbose.",default=False)
    o.add_option("--debug", action="store_true", dest="debug",
            help="Print extra debugging info to STDOUT", default=False)
    o.add_option("-i", "--identifier", type="string",
            help="Unique identifier for this run.", metavar="ID")
    c.add_option("identifier")
    o.add_option("--suffix", type="string",default="",
            help="Suffix to add to the identifier")
    o.add_option("-f", "--filename", type="string",
            help="File name of the data file", metavar="FILE")
    c.add_option("filename")
    o.add_option("--data-dir", type="string",dest="data_dir",
            help="Directory containing the data files. This will be"+
                 " concatenated with filename to yield the comple path.", metavar="DIR")
    c.add_option("data_dir",dest="data_dir")
    o.add_option("--output", type="string", dest="output_dir",
            help="Directory for the output files (default: output)", metavar="DIR")
    c.add_option("output_dir",dest="output_dir",default="output")
    o.add_option("--force",action="store_true",dest="overwrite",default=False,
            help="Run even if the output directory already exists.")
    o.add_option("-n","--particles", type="int",
            help="Number of particles to use", 
            metavar="NUM")
    c.add_option('particles')
    o.add_option("--max-clusters", type="int", dest="max_clusters",
            help="Maximum possible number of clusters.", 
            metavar="NUM")
    c.add_option('max_clusters',dest="max_clusters",default=100)

    o.add_option("-a","--algorithm",dest="algorithm",type="choice",
            metavar="ALG",choices=("pf","gibbs","mh"),
            help="Inference algorithm to use (pf,gibbs,mh).")
    c.add_option('algorithm')
    o.add_option("-d","--dims",dest="use_dims",type="int",
            help="Number of dimensions to use. If 0, all data dimensions will"+
            " be used.")
    c.add_option("dims",dest="use_dims",default=0)
    o.add_option("--rows",dest="use_rows",type="int",
            help="Number of rows of data to be used. If 0, all are used.")
    c.add_option("rows",dest="use_rows",default=0)
    o.add_option("--save-particle",dest="save_particle",type="choice",
            metavar="NUM",choices=("none","one","all"),
            help="Number of particles to save (none,one,all)")
    c.add_option('save_particle',dest="save_particle",default="none")
    o.add_option("--draw-prior", action="store_true", dest="draw_prior",
            help="Make a plot of draws from the prior and exit.", default=False)

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
    options.identifier += options.suffix
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
    fn = P.abspath(options.data_dir + options.filename)
    logging.info("Loading data from " + fn)
    data_file = load_file(fn)
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
    if options.debug:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename='debug.log',
                            filemode='w')
    else:
        logging.basicConfig(level=logging.ERROR)

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
            max_clusters = options.max_clusters,
            resample_fun = options.resampling_method,
            )
    pf.run()
    return pf

def plot_pf_output(pf):
    # TO BE REMOVED
    firstmu = zeros(pf.T)
    firstlam = zeros(pf.T)
    for t in range(pf.T):
        firstmu[t] = pf.particles[0].U.get_array(t)[0].mu[0]
        firstlam[t] = pf.particles[0].U.get_array(t)[0].lam[0]
    print firstmu
    P.subplot(2,1,1)
    P.plot(firstmu)
    P.subplot(2,1,2)
    P.plot(firstlam)
    P.show()
    print labeling
    plot_result(data,labeling[0,:])
    outf = open('aparticle.pkl', 'wb')
    cPickle.dump(pf.particles[0],outf)
    outf.close()

def prepare_output_dir(options):
    """Check that the output directory exists and create it if necessary.
    Also create a directory for this identifier inside the output dir.
    """
    outdir = options.output_dir
    id = options.identifier
    outdir = P.abspath(outdir)
    if not P.exists(outdir):
        os.mkdir(outdir)
    full_dir = outdir + "/" + id
    if not P.exists(full_dir):
        os.mkdir(full_dir)
    else:
        if options.overwrite:
            logging.warning("Output directory " + full_dir + " already exists.")
        else:
            raise RuntimeError, "Output directory already exists. Aborting."
    return full_dir

def write_pf_output(pf,outdir,options):
    id = options.identifier
    prefix = outdir + "/" + id
    
    # save labeling for all particles
    labeling = pf.get_labeling()
    savetxt(prefix + '.label',labeling,fmt="%i")
    if options.save_particle == "one":
        # save a pickled version of the first particle
        outf = open(prefix + '.0.particle', 'wb')
        cPickle.dump(pf.particles[0],outf)
        outf.close()
    if options.save_particle == "all":
        # save pickled version of particles array
        outf = open(prefix + '.particles', 'wb')
        cPickle.dump(pf.particles,outf)
        outf.close()

    # save effictive sample size
    savetxt(prefix + '.ess',
            vstack((pf.effective_sample_size,
                    pf.unique_particles,
                    pf.filtering_entropy)))

def make_prior_plot(model,ip,opts):
    import plotting
    import pylab
    NUM_CLUSTERS = 5
    NUM_SAMPLES = 30
    NUM_SUBPLOTS = 6
    WALK_LENGTH = 1000
    data = zeros((model.dims,NUM_CLUSTERS*NUM_SAMPLES))
    labels = zeros(NUM_CLUSTERS*NUM_SAMPLES,dtype=int32)
    for i in range(NUM_SUBPLOTS):
        pylab.subplot(3,2,i+1)
        for n in range(NUM_CLUSTERS):
            params = model.sample_prior()
            for s in range(NUM_SAMPLES):
                data[:,n*NUM_SAMPLES + s] = rnorm(params.mu,params.lam)
                labels[n*NUM_SAMPLES + s] = n
            color = pylab.get_cmap("flag")(n*3)
            plotting.plot_diagonal_gaussian(params.mu,params.lam,color)
        plotting.plot_scatter_2d(data,labels)
        pylab.grid()
        pylab.axis([-5,5,-5,5])
    F = pylab.gcf()
    F.set_size_inches(8.3,11.7)
    pylab.savefig("prior_draw.eps")

    pylab.clf()
    import explore_kernel
    explore_kernel.diagnostic_plots(model,WALK_LENGTH)
    pylab.savefig("walk_diagnostics.eps")

    pylab.clf()
    params = model.sample_prior()
    data = zeros((model.dims,WALK_LENGTH))
    for t in range(0,1000):
        data[:,t] = rnorm(params.mu,params.lam)
        params = model.walk(params)
    pylab.plot(data[0,:],data[1,:],'x')
    pylab.savefig("walk_draw.eps")


def load_particle(options):
    prefix = abspath(options.output_dir + "/" + options.identifier + "/" +
                 options.identifier)
    fn1 = prefix + ".0.particle"
    fn2 = prefix + ".particles"
    if exists(fn1):
        return cPickle.load(open(fn1,'rb'))
    elif exists(fn2):
        return cPickle.load(open(fn2,'rb'))[0]
    else:
        return None

def run_mh(data,data_time,m,ip,options):
    """Run Metropolis-Hastings sampler."""
    # load partcile for initialization
    import pylab
    particle = load_particle(options)
    if particle == None:
        raise RuntimeError, "Particle not found -- run particle filter first!"
    state = model.GibbsState(particle,m)
    state.check_consistency(data_time)
    sampler = inference.GibbsSampler(
            data=data,
            data_time=data_time,
            params = ip,
            model=m,
            state=state) 
    lnps = []
    prefix = abspath(options.output_dir + "/" + options.identifier + "/" +
                 options.identifier)
    f = open(prefix + ".mh_labels","w")
    pylab.ion()
    for t in range(1,2000):
        print "t = %i / %i" % (t,2000)
        # pylab.clf()
        #plotting.plot_sampler_params(sampler.state)
        #pylab.draw()
        sampler.mh_sweep()
        sampler.state.check_consistency(data_time)
        lnps.append(sampler.p_log_joint(False))
        #pylab.plot(sampler.state.mstore)
        sampler.state.c.tofile(f,sep=' ')
        f.write('\n')
    f.close()
    pylab.clf()
    pylab.plot(array(lnps))
    pylab.show()



def main():
    opts, args = handle_options()
    logging_setup(opts)
    data,data_time,labels = load_data(opts)
    #print opts
    model = get_model(opts)
    ip = get_inference_params(opts)
    if opts.draw_prior:
        make_prior_plot(model,ip,opts)
        return

    if opts.algorithm == "pf":
        outdir = prepare_output_dir(opts)
        pf = run_pf(data,data_time,model,ip,opts)
        write_pf_output(pf,outdir,opts)
    elif opts.algorithm == "mh":
        run_mh(data,data_time,model,ip,opts)
    
    print "Done"


if __name__ == "__main__":
    main()
