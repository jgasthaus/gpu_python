#!/usr/bin/python
"""Tool for analyzing the results obtained using the experimenter script.

This script supports quantitative analysis of the results as well as
qualitative analysis in the form of plots.
"""

from numpy import *
from pylab import *
from scipy.misc import comb
import locale
from os.path import abspath, exists
from os import mkdir
from optparse import OptionParser
from cfgparse import ConfigParser
import cPickle
import experimenter,plotting,utils

# I have no idea what causes the locale to be changed, but this fixes it ...
locale.setlocale(locale.LC_NUMERIC,"C")

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
    o.add_option("--draw-prior", action="store_true", dest="draw_prior",
            help="Make a plot of draws from the prior and exit.", default=False)
    o.add_option("--noplot", action="store_true", dest="no_plot",
            help="Disable creation of plots entirely.", default=False)
    o.add_option("--nostats", action="store_true", dest="no_stats",
            help="Do not compute any statistics for the result.", default=False)
    o.add_option("-i", "--identifier", type="string",
            help="Unique identifier for this run.", metavar="ID")
    c.add_option("identifier")
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
    o.add_option("-d","--dims",dest="use_dims",type="int",
            help="Number of dimensions to use. If 0, all data dimensions will"+
            " be used.")
    c.add_option("dims",dest="use_dims",default=0)
    o.add_option("--rows",dest="use_rows",type="int",
            help="Number of rows of data to be used. If 0, all are used.")
    c.add_option("rows",dest="use_rows",default=0)

    (options,args) = c.parse(o)
    return (options,args)

def draw_prior(options):
    # make plot for draws from the prior
    pass # May  have to move this to experimenter

def load_labels(options):
    fn = abspath(options.output_dir + "/" + options.identifier + "/" +
                 options.identifier + ".label")
    labels = loadtxt(fn,dtype=int32)
    return labels

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

def load_ess(options):
    fn = abspath(options.output_dir + "/" + options.identifier + "/" +
                 options.identifier + ".ess")
    return loadtxt(fn)
    


def compute_rand_index(labeling1,labeling2):
    """Compute the (adjusted) rand index between the two labelings.
    
    Code based on: 
    "Cluster Validation Toolbox for estimating the number of clusters"
    Kaijun Wang
    http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do
    ?objectId=13916&objectType=File

    Returns: a tuple consisting of
        the Hubert & Arabie adjusted Rand index
        the unadjusted Rand index, 
        the Mirkin index,
        the Hubert(1977) index
    """
    if labeling1.shape[0] != labeling2.shape[0]:
        raise RuntimeError, "Labeling do not have the same lenghts!"
    N = labeling1.shape[0]
    C = zeros((max(labeling1)+1,max(labeling2)+1),dtype=float64)
    for i in range(N):
        C[labeling1[i],labeling2[i]] += 1
    n = sum(C)
    nis = sum(sum(C,1)**2)
    njs = sum(sum(C,0)**2)
    t1 = comb(n,2,1)
    t2 = sum(C**2)
    t3 = 0.5 * (nis + njs)
    nc=(n*(n**2+1)-(n+1)*nis-(n+1)*njs+2*(nis*njs)/n)/(2*(n-1))
    A=t1+t2-t3  # no. agreements
    D=  -t2+t3  # no. disagreements
    if t1==nc:
        AR=0			#avoid division by zero; if k=1, define Rand = 0
    else:
        AR=(A-nc)/(t1-nc)		#adjusted Rand - Hubert & Arabie 1985

    RI=A/t1     # Rand 1971	Probability of agreement
    MI=D/t1     # Mirkin 1970	%p(disagreement)
    HI=(A-D)/t1 # Hubert 1977   %p(agree)-p(disagree)
    return (AR,RI,MI,HI)




def do_plotting(options):
    print "Generating Plots ..."
    plot_dir = options.output_dir + "/" + options.identifier + "/plots"
    if not exists(plot_dir):
        mkdir(plot_dir)
    predicted_labels = load_labels(options)
    data,data_time,true_labels = experimenter.load_data(options)
    T = data_time.shape[0]
    
    # Labeled 2D scatter plot of the first two PCs
    clf()
    plotting.plot_scatter_2d(data[0:2,:],predicted_labels[0,:])
    grid()
    savefig(plot_dir + "/" + "scatter_predicted.eps")

    # 2D scatter plot of PCs against time with predicted labels (1st particle)
    clf()
    plotting.plot_pcs_against_time_labeled(data,data_time,predicted_labels[0,:])
    savefig(plot_dir + "/" + "pcs_vs_time_predicted.eps")

    # plot of effective sample size
    clf()
    ess = load_ess(options)
    plot(ess)
    title("Effective Sample Size")
    xlabel("Time Step")
    ylabel("ESS")
    grid()
    savefig(plot_dir + "/" + "ess.eps")
    
    particle = load_particle(options)
    if particle != None:
        ### plots requiring the information from at least one particle
        clf()
        # plot of clusters + data at fixed, equally spaced time points
        num_plots = 9
        timepoints = array(arange(num_plots)*(T-1)/(float(num_plots)-1),dtype=int32)
        for i in range(num_plots):
            subplot(3,3,i+1)
            t = timepoints[i]
            plotting.plot_state_with_data(particle,data,data_time,t)
            title("t = " + str(t))
            grid()
            savefig(plot_dir + "/" + "cluster_evolution.eps")

        # plot of cluster means and variances over time
        clf()
        plotting.plot_pcs_against_time_labeled_with_particle(
                data,data_time,predicted_labels[0,:],particle)
        savefig(plot_dir + "/" + "clusters_vs_time.eps")





def get_descriptive(a):
    """Get a tuple of descriptive statistics (mean,var,min,max) for the 
    given array."""
    return (mean(a),var(a),min(a),max(a))

def descriptive2str(desc):
    out = []
    out.append("Mean : " + str(desc[0]))
    out.append("Var  : " + str(desc[1]))
    out.append("Min  : " + str(desc[2]))
    out.append("Max  : " + str(desc[3]))
    return '\n'.join(out)


def do_statistics(options):
    predicted_labels = load_labels(options)
    data,data_time,true_labels = experimenter.load_data(options)
    out = [    "Statistics for result set: " + options.identifier]
    out.append("============================================================")
    num_particles = predicted_labels.shape[0]
    # Descriptive statistics
    num_predicted = predicted_labels.shape[1]
    num_original = true_labels.shape[0]
    unique_true = unique(true_labels)
    unique_predicted = zeros(num_particles,dtype=int32)
    for i in range(num_particles):
        unique_predicted[i] = unique(predicted_labels[i,:]).shape[0]
    out.append("")
    out.append("Data set length (processed/total): " + str(num_predicted) +
               "/" + str(num_original))
    out.append("Number of particles: " +str(num_particles))
    out.append("")
    out.append("Number of clusters")
    out.append("------------------")
    out.append(descriptive2str(get_descriptive(unique_predicted))) 

    # Rand indices
    AR,RI,MI,HI = compute_rand_index(predicted_labels[0,:],true_labels)
    out.append("")
    out.append("Rand indices")
    out.append("------------")
    out.append("Adjusted: " + str(AR))
    out.append("Unadjusted: " + str(RI))
    outstr = '\n'.join(out)
    print outstr

def main():
    options,args = handle_options()
    if options.draw_prior:
        draw_prior(options)

    if not options.no_stats:
        do_statistics(options)

    if not options.no_plot:
        do_plotting(options)



if __name__ == "__main__":
    main()


