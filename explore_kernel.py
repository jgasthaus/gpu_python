# Copyright (c) 2008-2011, Jan Gasthaus
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.  
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

def compute_correlation(seq,diff):
    m = mean(seq)
    v = var(seq)
    sm = seq - m
    tmp = sm[:-diff]*sm[diff:]
    return mean(tmp)/v

def compute_acf(seq,N):
    tmp = zeros(N)
    for n in range(1,N):
        tmp[n] = compute_correlation(seq,n)
    return tmp


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
            kernelParams=tuple([100,0.01])
            )
    #diagnostic_plots(m,5000)
    #plot_acf(m,10000,100)
    #show()
    figure(2)
    sigmas = [(1,1),(10,1),(100,1),(1,0.1),(100,0.1),(1,10),(100,10)]
    for l in sigmas:
        print l
        m = model.DiagonalConjugate(
                hyper_params=params,
                kernelClass=model.CaronIndependent,
                kernelParams=tuple(l)
                )
        plot_acf(m,100000,100)
    legend([r"$M=%i,\xi=%.2f$" % i for i in sigmas])
    subplot(1,2,1)
    F = gcf()
    F.set_size_inches(6,3)
    savefig("acf_caron.pdf")
    #figure(3)
    #diagnostic_plots(m,10000)
    #show()
    # print get_cov(m,1000,30)[:,0]

def plot_acf(m,walk_len,acf_len):
    start = m.sample_prior()
    mywalk = gen_walk(m,start,walk_len)
    walk_means = get_means(mywalk)
    walk_lambdas = get_lambdas(mywalk)
    subplot(1,2,1)
    acf = compute_acf(walk_means[0,:],acf_len)
    plot(arange(1,acf.shape[0]+1),acf)
    #acf = compute_acf(walk_means[1,:],acf_len)
    #plot(arange(1,acf.shape[0]+1),acf)
    axis([1,acf.shape[0]+1,0,1])
    title(r"ACF of $\mu$")
    xlabel(r"$\Delta t$")
    ylabel("ACF")
    grid()
    subplot(1,2,2)
    acf = compute_acf(walk_lambdas[0,:],acf_len)
    plot(acf)
    #acf = compute_acf(walk_lambdas[1,:],acf_len)
    #plot(acf)
    axis([1,acf.shape[0]+1,0,1])
    title(r"ACF of $\lambda$")
    xlabel(r"$\Delta t$")
    grid()
    #ylabel("ACF")


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
