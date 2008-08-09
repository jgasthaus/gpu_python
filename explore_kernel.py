from numpy import *
from pylab import *

import model
from utils import *


def gen_walk(themodel,start,num):
    b = empty(num,dtype=object)
    state = start
    for i in range(num):
        b[i] = state
        state = themodel.walk(state)
    return b

def plot_means(means):
    plot(means[0,:],means[1,:],'.')
    axis([-15,15,-15, 15])

def plot_lambdas(means):
    plot(means[0,:],means[1,:],'.')
    axis([0,40,0, 40])

def sample_prior(themodel,num):
    b = empty(num,dtype=object)
    for i in range(num):
        b[i] = themodel.sample_prior()
    return b

def get_means(walk):
    N = walk.shape[0]
    D = walk[0].mu.shape[0]
    means = zeros((D,N))
    for i in range(N):
        means[:,i] = walk[i].mu
    return means


def get_lambdas(walk):
    N = walk.shape[0]
    D = walk[0].lam.shape[0]
    lams = zeros((D,N))
    for i in range(N):
        lams[:,i] = walk[i].lam
    return lams

def get_probs(walk,themodel):
    N = walk.shape[0]
    ps = zeros(N)
    for i in range(N):
        ps[i] = themodel.p_prior_params(walk[i])
    return ps

def get_cov(themodel,num,steps):
    #samples = empty(num,dtype=object)
    # for i in range(num):
    #     samples[i] = themodel.walk(start)
    # means = get_means(samples)
    # m = mean(means,1)
    # print start.mu-m
    # run num chains
    means = zeros((num,steps))
    start = themodel.sample_prior()
    for i in range(num):
        w = gen_walk(themodel,themodel.walk(start),steps)
        m = get_means(w)
        means[i,:] = m[0,:]
    # print cov(means.T)
    return cov(means,rowvar=0)


def main():
    params = model.DiagonalConjugateHyperParams(
            a=4,
            b=1,
            mu0=0,
            n0=0.1,
            dims=2
            )
    # m = model.DiagonalConjugate(
    #     params,
    #     kernelClass=model.MetropolisWalk,
    #     kernelParams=(0.1,0.001)
    #     )
    m = model.DiagonalConjugate(
            hyper_params=params,
            kernelClass=model.CaronIndependent,
            kernelParams=tuple([100,1])
            )
    diagnostic_plots(m,200)
    # print get_cov(m,1000,30)[:,0]

def diagnostic_plots(m,length):
    start = m.sample_prior()
    mywalk = gen_walk(m,start,length)
    prior = sample_prior(m,length)
    walk_means = get_means(mywalk)
    walk_lambdas = get_lambdas(mywalk)
    prior_means = get_means(prior)
    prior_lambdas = get_lambdas(prior)
    # print mywalk
    subplot(3,2,1)
    plot_means(walk_means)
    subplot(3,2,2)
    plot_means(prior_means)
    subplot(3,2,3)
    plot_lambdas(walk_lambdas)
    subplot(3,2,4)
    plot_lambdas(prior_lambdas)

    subplot(3,2,5)
    plot(walk_means[0,:],)
    plot(walk_means[1,:],'r')
    subplot(3,2,6)
    plot(walk_lambdas[0,:],)
    plot(walk_lambdas[1,:],'r')

    #subplot(3,2,5)
    #plot(get_probs(mywalk,m))

    

if __name__ == "__main__":
    main()
