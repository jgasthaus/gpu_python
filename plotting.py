"""Various functions and classes related to plotting"""

from pylab import *
import matplotlib as M
import matplotlib.cm as cm
from matplotlib.colors import no_norm
from numpy import *

markers = ['s','o','^','>' ,'v' ,'<' ,'d' ,'p' ,'h' ,'8' ,'+' ,'x']

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
                linewidths=(0,)
               )

