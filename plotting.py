"""Various functions and classes related to plotting"""

from pylab import *
import matplotlib as M
import matplotlib.cm as cm
from matplotlib.colors import no_norm
from numpy import *
from optparse import OptionParser
from utils import *

markers = ['+','x','o','d','^','>' ,'v' ,'<' ,'s','p' ,'h' ,'8']

def plot_scatter_2d(data,labels):
    unique_labels = unique(labels)
    label_markers = array(markers)[unique_labels % len(markers)]
    label_colors = array(unique_labels,dtype=float64)/max(unique_labels)
    for i in range(len(unique_labels)):
        l = unique_labels[i]
        colors = ones(sum(labels==l))*label_colors[i]
        scatter(data[0,labels==l],data[1,labels==l],
                marker=str(label_markers[i]),
                c=colors,
                cmap=matplotlib.cm.jet,
                norm=no_norm(),
                linewidths=(0.3,)
               )

def plot_pcs_against_time(data,time):
    num_dims = data.shape[0]
    for n in range(num_dims):
        subplot(num_dims,1,n+1)
        scatter(time,data[n,:])
        grid()

def plot_pcs_against_time_labeled(data,time,labels):
    num_dims = data.shape[0]
    for n in range(num_dims):
        subplot(num_dims,1,n+1)
        plot_scatter_2d(vstack([time,data[n,:]]),labels)
        grid()

def main():
    HAVE_LABELS = False
    parser = OptionParser()
    parser.add_option("-l", "--labels", dest="label_fn",
        help="load labels from FILE", default=None, metavar="FILE")
    parser.add_option("-n", "--use-particle", dest="particle_idx",
        help="load lables of particle IDX", default=0, metavar="IDX",type="int")
    parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")
    options,args = parser.parse_args()
    fn = args[0]
    if options.label_fn != None:
        HAVE_LABELS = True
        labels = array(load_file(options.label_fn)[options.particle_idx],dtype=int32)
    data_file = load_file(fn)
    data_raw = data_file[:,2:].T
    data_time = data_file[:,1].T*1000
    num_dims = data_raw.shape[0]
    clf()
    if HAVE_LABELS:
        plot_pcs_against_time_labeled(data_raw,data_time,labels)
    else:
        plot_pcs_against_time(data_raw,data_time)
    show()

if __name__ == "__main__":
    main()
