#!/usr/bin/python
"""Tool for analyzing the results obtained using the experimenter script.

This script supports quantitative analysis of the results as well as
qualitative analysis in the form of plots.
"""

from numpy import *
import numpy
from numpy.random import shuffle
from pylab import *
from matplotlib.colors import no_norm
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

# Global variable to store the labels once loaded and preprocessed
LABELS = None

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
    o.add_option("--quiet", action="store_true", dest="quiet",
            help="Only print the bare minimum information.",default=False)
    o.add_option("--noplot", action="store_true", dest="no_plot",
            help="Disable creation of plots entirely.", default=False)
    o.add_option("--nostats", action="store_true", dest="no_stats",
            help="Do not compute any statistics for the result.", default=False)
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
    o.add_option("-d","--dims",dest="use_dims",type="int",
            help="Number of dimensions to use. If 0, all data dimensions will"+
            " be used.")
    c.add_option("dims",dest="use_dims",default=0)
    o.add_option("--rows",dest="use_rows",type="int",
            help="Number of rows of data to be used. If 0, all are used.")
    c.add_option("rows",dest="use_rows",default=0)
    o.add_option("-p","--use-particle",dest="use_particle",type="int",
                 default=0,help="Particle ID for labeling",metavar="ID")
    o.add_option("--merge-noise",dest="merge_noise",type="float",default=0.,
            help="Merge all clusters with NUM% of the data into one.",
            metavar="NUM")
    o.add_option("--subsample",type="int",default=0,metavar="NUM",
            help="Subsample data set to contain this number of points (0: all).")
    o.add_option("--binary-label",action="store_true",default=False,
            dest="binary_label",help="The label is binary: compare only the " +
            "cluster with the largest overlap to the one with label 1.")
    o.add_option("--true-labels",action="store_true",default=False,
            dest="true_labels",help="Use the true labels for plotting. ")
    o.add_option("--plot-fmt",type="choice",choices=("eps","pdf","png","jpg"),
            dest="output_format",
            default="eps",help="Plot output format (eps,pdf,png,jpg).")
    o.add_option("--label-file",type="string",dest="label_file",default="",
            help="File containing the labels.")

    (options,args) = c.parse(o)
    options.identifier = options.identifier + options.suffix
    return (options,args)

def draw_prior(options):
    # make plot for draws from the prior
    pass # May  have to move this to experimenter

def map_labels(labels):
    """Map an arbitrary labeling so that in contains labels in 0,...,K."""
    avail = unique(labels)
    mapping = -1*ones(max(avail)+1)
    j = 0
    for i in range(labels.shape[0]):
        if mapping[labels[i]]==-1:
            mapping[labels[i]] = j
            j += 1
    return take(mapping,labels)

def load_labels(options):
    global LABELS
    if LABELS != None:
        return LABELS
    if options.label_file != "":
        fn = abspath(options.label_file)
    else:
        fn = abspath(options.output_dir + "/" + options.identifier + "/" +
                     options.identifier + ".label")
    labels = loadtxt(fn,dtype=int32)
    if len(labels.shape)==1:
        labels.shape = (1,labels.shape[0])
    if options.merge_noise>0:
        print "Merging noise clusters ..."
        unique_labels = unique(labels)
        for l in range(labels.shape[0]):
            mapping = -1*ones(max(unique_labels)+1,dtype=int32)
            k = 0
            for c in unique_labels:
                if sum(labels[l,:]==c) > labels.shape[1]*options.merge_noise/100.:
                    mapping[c] = k
                    k += 1
            mapping[mapping==-1] = max(mapping)+1
            for n in range(labels.shape[1]):
                labels[l,n] = mapping[labels[l,n]]
        print "Done."
    for n in range(labels.shape[0]):
        labels[n,:] = map_labels(labels[n,:])
    LABELS = labels
    return labels


def load_ess(options):
    fn = abspath(options.output_dir + "/" + options.identifier + "/" +
                 options.identifier + ".ess")
    ess = loadtxt(fn)
    if ess.shape[0] != 3:
        tmp = zeros((3,ess.shape[1]))
        tmp[0:2,:] = ess
        ess = tmp
    return ess
    


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


def variation_of_information(labeling1,labeling2):
    """Compute the "Variation of Information" (Meila, 2003) between the two
    labelings. Each labeling must contain all labels from 0 to max(labeling).
    """
    if labeling1.shape[0] != labeling2.shape[0]:
        raise RuntimeError, "Labeling do not have the same lenghts!"
    N = labeling1.shape[0]
    K1 = max(labeling1) + 1
    K2 = max(labeling2) + 1
    
    C = zeros((K1,K2),dtype=float64)
    for i in range(N):
        C[labeling1[i],labeling2[i]] += 1
    P_kk = C / N
    P_k1 = sum(C,1)/N
    P_k2 = sum(C,0)/N
    P_k1.shape = (K1,1)
    P_k2.shape = (K2,1)
    X = P_kk * numpy.log2(P_kk/(P_k1*P_k2.T))
    # enfore 0*log 0 = 0
    X[isnan(X)] = 0
    I = sum(X)
    H1tmp = numpy.log2(P_k1)*P_k1
    H1tmp[isnan(H1tmp)] = 0
    H1 = -sum(H1tmp)
    H2tmp = numpy.log2(P_k2)*P_k2
    H2tmp[isnan(H2tmp)] = 0
    H2 = -sum(H2tmp)
    VI = H1 + H2 - 2*I
    return VI/numpy.log2(N)


def subsample(data,data_time,labels,options):
    idx = arange(data_time.shape[0])
    if options.subsample == 0:
        return data,data_time,labels,idx
    else:
        shuffle(idx)
        idx = idx[:options.subsample]
        return (data[:,idx],data_time[idx],labels[:,idx],idx)



def do_plotting(options):
    print "Generating Plots ..."
    particle_id = options.use_particle
    ext = "." + options.output_format
    plot_dir = options.output_dir + "/" + options.identifier + "/plots"
    if not exists(plot_dir):
        mkdir(plot_dir)
    data,data_time,true_labels = experimenter.load_data(options)
    if options.true_labels:
        predicted_labels = true_labels
        predicted_labels.shape = (1,predicted_labels.shape[0])
        ext = "_true" + ext
    else:
        predicted_labels = load_labels(options)

    s_data,s_data_time,s_predicted_labels,idx = subsample(
            data,data_time,predicted_labels,options)
    T = data_time.shape[0]
    ess = load_ess(options)
    
    # Labeled 2D scatter plot of the first two PCs
    clf()
    plotting.plot_scatter_2d(s_data[0:2,:],s_predicted_labels[particle_id,:])
    grid()
    savefig(plot_dir + "/" + "scatter_predicted" + ext)

    # 2D scatter plot with entropy heatmap
    clf()
    #ent = compute_label_entropy(s_predicted_labels)
    ent = ess[2,idx]
    certain = ent==0
    uncertain = logical_not(certain)
    if sum(certain)>0:
        scatter(s_data[0,certain],s_data[1,certain],10,marker="s",facecolors="none",
                linewidth=0.3)
    if sum(uncertain)>0:
        scatter(s_data[0,uncertain],s_data[1,uncertain],10,ent[uncertain],
                linewidth=0.3,cmap=cm.hot)
    grid()
    title("Label Entropy")
    savefig(plot_dir + "/" + "scatter_entropy" + ext)

    # Label entropy vs. time
    clf()
    ent = compute_label_entropy(predicted_labels)
    #subplot(2,1,1)
    axes([0.25, 0.2, 0.7, 0.7])
    plot(data_time,ent,'x',linewidth=1.5)
    #title("Label Entropy")
    #ylabel("Entropy")
    xlabel("Time")
    axis([-1,max(data_time)+2,-0.1,1.1])
    grid()
    #subplot(2,1,2)
    #plot(data_time,ess[2,:],'x',linewidth=1.5)
    #ylabel("Entropy")
    #xlabel("Time (ms)")
    #axis([0,60000,0,1])
    #title("Average Label Filtering Distribution Entropy (SMC)")
    #grid()
    F = gcf()
    F.set_size_inches(2,2)
    savefig(plot_dir + "/" + "entropy" + ext)
    
    clf()
    subplot(2,1,1)
    #axes([0.25, 0.2, 0.7, 0.7])
    plot(data_time,ent,linewidth=0.5)
    title("Label Entropy (M-H sampler)")
    ylabel("Entropy")
    #xlabel("Time")
    axis([0,60000,-0.1,1.1])
    grid()
    subplot(2,1,2)
    plot(data_time,ess[2,:],linewidth=0.5)
    ylabel("Entropy")
    xlabel("Time (ms)")
    axis([0,60000,-0.1,1.1])
    title("Average Label Filtering Distribution Entropy (SMC)")
    grid()
    F = gcf()
    F.set_size_inches(6,4)
    savefig(plot_dir + "/" + "entropy_both" + ext)


    # 2D scatter plot of PCs against time with predicted labels (1st particle)
    clf()
    plotting.plot_pcs_against_time_labeled(s_data,s_data_time,
            s_predicted_labels[particle_id,:])
    F = gcf()
    F.set_size_inches(8.3,4*data.shape[0])
    savefig(plot_dir + "/" + "pcs_vs_time_predicted" + ext)

    # 2D scatter plot of PCs against time for RPV candidates
    clf()
    isi = data_time[1:]-data_time[:-1]
    rpvs = where(isi < 2)[0] + 1
    rpvs = hstack((rpvs,rpvs-1))
    if rpvs.shape[0] > 0:
        plotting.plot_pcs_against_time_labeled(data[:,rpvs],data_time[rpvs],
                predicted_labels[particle_id,rpvs])
        F = gcf()
        F.set_size_inches(8.3,4*data.shape[0])
        savefig(plot_dir + "/" + "pcs_vs_time_rpv" + ext)


    # 2D scatter plot with binary labels
    if options.binary_label:
        clf()
        match = find_best_match(predicted_labels[particle_id,:],true_labels)
        matches = (predicted_labels[particle_id,:]==match)[idx]
        non_matches = (predicted_labels[particle_id,:]!=match)[idx]
        subplot(1,2,1)
        plot(s_data[0,matches],s_data[1,matches],'x')
        plot(s_data[0,non_matches],s_data[1,non_matches],'.')
        grid()
        title("Predicted Labels")
        axis([-5,5,-5,5])
        subplot(1,2,2)
        plot(s_data[0,true_labels[idx]==1],s_data[1,true_labels[idx]==1],'x')
        plot(s_data[0,true_labels[idx]!=1],s_data[1,true_labels[idx]!=1],'.')
        axis([-5,5,-5,5])
        grid()
        title("True Labels")
        F = gcf()
        F.set_size_inches(6,3)
        savefig(plot_dir + "/" + "scatter_binary" + ext)

    # plot of effective sample size
    clf()
    subplot(2,1,1)
    plot(ess[1,:],linewidth=0.3)
    title('Unique Particles')
    ylabel("Unique Particles")
    axis([0,650,0,1100])
    grid()
    xlabel("Time Step")
    subplot(2,1,2)
    plot(ess[0,:],linewidth=0.3)
    axis([0,650,0,1100])
    title("Effective Sample Size")
    xlabel("Time Step")
    ylabel("ESS")
    grid()
    F = gcf()
    F.set_size_inches(6,4)
    savefig(plot_dir + "/" + "ess" + ext)

    # ISI histogram for each neuron
    clf()
    l = predicted_labels[particle_id,:]
    unique_labels = unique(l)
    for i in range(unique_labels.shape[0]):
        c = unique_labels[i]
        points = data[:,l==c]
        times = data_time[l==c]
        isi = times[1:] - times[0:-1]
        subplot(unique_labels.shape[0],2,2*i+1)
        label_colors = array(unique_labels,dtype=float64)/max(unique_labels+1)
        colors = ones(sum(l==c))*label_colors[i]
        scatter(points[0,:],points[1,:],marker=plotting.markers[c],c=colors,
                cmap=matplotlib.cm.jet,
                norm=no_norm(),
                linewidths=(0.3,))
        title("Cluster %i (weight=%.2f)" % (c,sum(l==c)/float(l.shape[0])))
        grid()
        axis([-5,5,-5,5])
        subplot(unique_labels.shape[0],2,2*i+2)
        hist(isi,bins=100,range=(0,100),normed=True,facecolor='k')
        xx = arange(2,100,0.1)
        rate = 1/mean(isi-2)
        plot(xx,rate*exp(-rate*(xx-2)))
        title("ISI (mean = %.2f)" % mean(isi))
    F = gcf()
    F.set_size_inches(8.3,2*unique_labels.shape[0])
    savefig(plot_dir + "/" + "isi" + ext)
    
    
    particle = experimenter.load_particle(options)
    if particle != None:
        ### plots requiring the information from at least one particle
        clf()
        # plot of clusters + data at fixed, equally spaced time points
        num_plots = 9
        timepoints = array(arange(1,num_plots+1)*(T-1)/(float(num_plots)),dtype=int32)
        for i in range(num_plots):
            subplot(3,3,i+1)
            t = timepoints[i]
            plotting.plot_state_with_data(particle,data,data_time,t)
            title("t = " + str(t))
            grid()
        F.set_size_inches(6,6)
        savefig(plot_dir + "/" + "cluster_evolution" + ext)

        # plot of cluster means and variances over time
        clf()
        plotting.plot_pcs_against_time_labeled_with_particle(
                data,data_time,predicted_labels[0,:],particle)
        F.set_size_inches(40,8*data.shape[0])
        savefig(plot_dir + "/" + "clusters_vs_time" + ext)

        # plot of mstore for each clusters
        clf()
        plotting.plot_mstore_against_time(particle)
        savefig(plot_dir + "/" + "mstore" + ext)



def compute_label_entropy(labeling):
    N,T = labeling.shape
    ent = zeros(T)
    for t in range(T):
        possible = unique(labeling[:,t])
        p = zeros(possible.shape[0])
        for i in range(possible.shape[0]):
            p[i] = sum(labeling[:,t]==possible[i])
        p = p / sum(p)
        ent[t] = - sum(p*numpy.log2(p))
    return ent



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
    stats_fn = (options.output_dir + "/" + options.identifier
                + "/" + options.identifier + ".stats")
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
    
    out.append("")
    out.append("Label Entropy")
    out.append("------------------")
    ent = compute_label_entropy(predicted_labels)
    out.append(descriptive2str(get_descriptive(ent))) 

    # Rand indices
    rand_indices = zeros((4,num_particles))
    for i in range(num_particles):
        ind = compute_rand_index(predicted_labels[i,:],true_labels)
        rand_indices[:,i] = ind
    out.append("")
    out.append("Rand indices")
    out.append("------------")
    out.append("Adjusted: ")
    out.append(descriptive2str(get_descriptive(rand_indices[0,:]))) 
    out.append("Unadjusted: ")
    out.append(descriptive2str(get_descriptive(rand_indices[1,:]))) 
    
    # Variation of information
    vi = zeros(num_particles)
    for i in range(num_particles):
        vi[i] = variation_of_information(predicted_labels[i,:],true_labels)
    out.append("")
    out.append("Variation of Information")
    out.append("------------------------")
    out.append(descriptive2str(get_descriptive(vi))) 
    out.append("MAP: VI: %.4f; Rand: %.4f" % (vi[0], rand_indices[0,0]))
    
    if options.binary_label:
        tp = zeros(num_particles)
        fp = zeros(num_particles)
        tn = zeros(num_particles)
        fn = zeros(num_particles)
        rpvs = zeros(num_particles)
        for l in range(num_particles):
            labels = predicted_labels[l,:]
            match = find_best_match(labels,true_labels)
            times = data_time[labels==match]
            rpvs[l]= sum(times[1:]-times[:-1]<2)
            tp[l] = sum(logical_and(labels==match,true_labels==1))
            fp[l] = sum(logical_and(labels==match,true_labels!=1))
            tn[l] = sum(logical_and(labels!=match,true_labels!=1))
            fn[l] = sum(logical_and(labels!=match,true_labels==1))
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        fscore = 2*precision*recall/(precision + recall)
        accuracy = (tp + tn)/(tp + fp + tn + fn)
        out.append("\nBinary label")
        out.append("------------")
        out.append("Precision: ")
        out.append(descriptive2str(get_descriptive(precision))) 
        out.append("Recall: ")
        out.append(descriptive2str(get_descriptive(recall))) 
        out.append("Fscore: ")
        out.append(descriptive2str(get_descriptive(fscore))) 
        out.append("FP %: ")
        out.append(descriptive2str(get_descriptive(fp/labels.shape[0]))) 
        out.append("FN %: ")
        out.append(descriptive2str(get_descriptive(fn/(fn + tp)))) 
        out.append("RPVs: ")
        out.append(descriptive2str(get_descriptive(rpvs))) 
        out.append("MAP: FP: %.4f; FN: %.4f; FScore: %.4f; RPV: %i" % (fp[0]/labels.shape[0], fn[0]/(fn[0] + tp[0]),fscore[0],rpvs[0]))

    outstr = '\n'.join(out)
    if not options.quiet:
        print outstr
    else:
        print (options.identifier + ": VI: %.2f (%.1f)" % (mean(vi),mean(unique_predicted)))
    outfile = open(stats_fn,"w")
    outfile.write(outstr)
    outfile.close()

def find_best_match(labeling,true_labeling):
    """Find the label of the cluster in labeling that has the largest 
    overlap with the cluster with label "1" in true_labeling."""
    u = unique(labeling)
    counts = zeros(u.shape[0],dtype=int32)
    for i in range(u.shape[0]):
        c = u[i]
        counts[i] = sum(logical_and(labeling==c,true_labeling==1))
    best_match = u[argmax(counts)]
    print "best match is: " + str(best_match)
    return best_match 

def main():
    options,args = handle_options()

    if not options.no_stats:
        do_statistics(options)

    if not options.no_plot:
        do_plotting(options)



if __name__ == "__main__":
    main()


