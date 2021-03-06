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
import analysis
from pylab import *


def labelings_from_partitions():
    txt = open("set_partitions.txt",'r').read()
    sets = eval(txt.replace("\r\n"," "))
    labelings = -1*ones((115975,10),dtype=uint8)
    for k in range(len(sets)):
        l = sets[k]
        for i in range(len(l)):
            s = l[i]
            if type(s) != tuple:
                s = tuple([s])
            for j in range(len(s)):
                labelings[k,s[j]-1] = i
    savetxt('labelings.txt',labelings,fmt="%i")

def compute_entropy(labelings,lnp):
    N,T = labelings.shape
    ent = zeros(T)
    for t in range(T):
        possible = unique(labelings[:,t])
        p = zeros(possible.shape[0])
        for i in range(possible.shape[0]):
            p[i] = sum(exp(lnp)[labelings[:,t]==possible[i]])
        p = p / sum(p)
        l = log2(p)
        l[l==-inf]=0
        ent[t] = -sum(p*l)
    return ent

def compute_kl(all_labels,lnps,samples):
    """Compute KL divergence between the discrete distribution with 
    atoms all_labels and (log-)weights lnps, and the equally weighted
    samples in samples."""
    p1 = exp(lnps)
    # use small data type for efficiency
    all_labels = array(all_labels,dtype=uint8)
    samples = array(samples,dtype=uint8)
    for i in range(samples.shape[0]):
        samples[i,:] = analysis.map_labels(samples[i,:])
    samples = array(samples,dtype=uint8)

    p2 = zeros_like(p1)
    N = p1.shape[0]
    kl1 = 0.
    kl2 = 0.
    for i in range(N):
        p2[i] = sum(all(samples==all_labels[i,:],1))/float(samples.shape[0])
    return (p2,cross_entropy(p2,p1)-cross_entropy(p2,p2))

def plot_probs(labelings,probs,num=10):
    idx = argsort(probs)
    mat = labelings[idx[-num:],:]
    ax = subplot(1,2,1)
    #axis('image')
    pcolor(mat,cmap=cm.binary)
    grid()
    setp(gca(), yticklabels=[])
    title('Labelings')
    ylabel('Labeling')
    xlabel('Time step')
    subplot(1,2,2,sharey=ax)
    barh(arange(num),probs[idx[-num:]])
    grid()
    setp(gca(), yticklabels=[])
    title('Prior probability')
    xlabel('p(labeling)')
    F = gcf()
    F.set_size_inches(6,3)
    subplots_adjust(0.1,0.15,0.95,0.9,0.1,0.2)

def cross_entropy(p1,p2):
    l2 = log2(p2)
    l2[l2==-inf] = 0
    return -sum(p1*l2)

def main():
    labels = loadtxt('labelings.txt')
    


if __name__ == "__main__":
    main()
