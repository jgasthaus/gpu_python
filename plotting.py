"""Various functions and classes related to plotting"""

from pylab import *
import matplotlib as M
import matplotlib.cm as cm
from matplotlib.colors import no_norm
from matplotlib.patches import Ellipse
from numpy import *
from optparse import OptionParser
import cPickle
from utils import *

markers = ['+','x','o','d','^','>' ,'v' ,'<' ,'s','p' ,'h' ,'8']*10

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
        axis([0,max(time),-3,3])

def plot_pcs_against_time_labeled_with_particle(data,time,labels,particle):
    num_dims = data.shape[0]
    mstore = particle.mstore.to_array()
    mean_cluster_size = mean(mstore[mstore>0])
    for n in range(num_dims):
        subplot(num_dims,1,n+1)
        plot_scatter_2d(vstack([time,data[n,:]]),labels)
        for c in range(particle.K):    
            start = particle.birthtime[c]
            stop = particle.deathtime[c]
            if stop == 0: stop = particle.T
            length = stop-start
            mus = N.zeros(length)
            lams = N.zeros(length)
            for i in range(length):
                t = range(start,stop)[i]
                mus[i] = particle.U.get_array(t)[c].mu[n]
                lams[i] = particle.U.get_array(t)[c].lam[n]
            #plot(time[arange(start,stop)],mus)
            lw = mean(mstore[c,start:stop])/mean_cluster_size*0.5
            errorbar(time[arange(start,stop)],mus,sqrt(1/lams),
                linewidth=lw,elinewidth=lw)
        xlabel("Time")
        ylabel("PC " + str(n+1))
        grid()


def matlab_plot_3d(data,data_time,labeling):
    from mlabwrap import mlab
    mlab.scatter3(data_time,data[0,:],data[1,:],20,labeling)


def plot_lifespan_histogram(particle_fn,rho=0.975):
    p = cPickle.load(open(particle_fn,'rb'))
    a = array(range(p.T))
    ls = p.d - a
    hist(ls,bins=100,normed=True)
    plot(a,rho**a*(1-rho))

def plot_geometric(rhos,xaxis=(0,500)):
    """Plot the geometric distribution for the given values of rho.
    p(k|rho) = (1-rho)**(k) x rho
    """
    x = range(xaxis[0],xaxis[1])
    f = figure(figsize=(5.3,3.5))
    for r in rhos:
        plot(x,(1-r)**x*r)
    grid()
    title("Geometric Distribution $(1-p)^k p$")
    legend(["p=%3.3f" % r for r in rhos])
    savefig("geometric_distribution.pdf")

def plot_gaussian(mu,sigma):
    """Plot the contour of a general bivariate gaussian with mean mu and
    covariance matrix sigma."""
    t = arange(-pi,pi,0.01)
    x = sin(t)
    y = cos(t)

    dd,vv = eig(sigma)
    A = vv*sqrt(dd)
    z = dot(vstack([x,y]).T,A)

    plot(z[:,0]+mu[0],z[:,1]+mu[1]);

def plot_diagonal_gaussian(mu,lam,color=get_cmap()(0)):
    """Plot a gaussian with mean mu and diagonal precision lam."""
    t = arange(-pi,pi,0.01)
    x = sin(t)
    y = cos(t)

    A = eye(2)*sqrt(1/lam)
    z = dot(vstack([x,y]).T,A)

    plot(z[:,0]+mu[0],z[:,1]+mu[1],'-',color=color);


def plot_state(particle,t):
    active = where(particle.mstore.get_array(t)>0)[0]
    for c in active:
        U = particle.U.get(t,c)
        plot_diagonal_gaussian(U.mu,U.lam)

def plot_state_with_data(particle,data,data_time,t):
    # TODO: Fix color plotting so Gaussian contours and data points ahve the
    # same color
    active = where(particle.mstore.get_array(t)>0)[0]
    for c in active:
        idx = where(
                logical_and(
                    particle.c==c,
                    logical_and(
                        t >= arange(particle.T),
                        particle.d>t
                    )
                    )
                )[0]
        color = get_cmap("flag")(c*3)
        plot(data[0,idx],data[1,idx],'x',color=color)
        U = particle.U.get(t,c)
        # print t,c,U.mu, U.lam
        plot_diagonal_gaussian(U.mu,U.lam,color=color)
        axis([-5, 5, -5, 5])
    
    

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
    ion()
    # clf()
    # p = cPickle.load(open("aparticle.pkl",'rb'))
    # for t in range(1,500):
    #     ioff()
    #     clf()
    #     plot_state_with_data(p,data_raw,data_time,t)
    #     axis([-6,6,-3,3])
    #     draw()
    #     raw_input()
    # return
    # for t in range(1,500):
    #     ioff()
    #     clf()
    #     plot_scatter_2d(data_raw[:,max(0,t-10):t],labels[max(0,t-10):t])
    #     plot_state(p,t)
    #     axis([-5,1,-3,3])
    #     draw()
    #     raw_input()
    # #plot_geometric(array([0.03,0.02,0.015,0.010,0.005,0.001]),(0,300))
    # #show()
    # return
    # plot_lifespan_histogram("aparticle.pkl")
    # show()
    if HAVE_LABELS:
        plot_pcs_against_time_labeled(data_raw,data_time,labels)
        matlab_plot_3d(data_raw,data_time,labels)
    else:
        plot_pcs_against_time(data_raw,data_time)
    show()

if __name__ == "__main__":
    main()
