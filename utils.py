import numpy as N
from numpy import array, exp, log, sqrt, cumsum, empty, zeros, pi, int32
from scipy.special import gammaln
import numpy.random as R
import logging

try:
    from scipy.sandbox.montecarlo._intsampler import _intsampler
    HAVE_INTSAMPLER = True
except:
    HAVE_INTSAMPLER = False

LOG2PI = log(2*pi)

def noop(*args,**kwargs):
    pass

def isarray(a):
    return isinstance(a, N.ndarray)

def load_file(fn):
    logging.info('Loading file ' + str(fn))
    data = N.loadtxt(fn)
    logging.info('Read file, size: ' + str(data.shape))
    return data

def counts_to_index(counts):
    """Transform an array of counts to the corresponding array of indices, 
    i.e. [2,3,1] -> [0,0,1,1,1,2]
    """
    idx = empty(sum(counts),dtype=int32)
    k = 0
    for i in range(len(counts)):
        for j in range(counts[i]):
            idx[k] = i
            k += 1
    return idx

class FixedSizeStore(object):
    def __init__(self,size,max_clusters=100,dtype=object,copy=None):
        if copy != None:
            self.array = copy.array.copy()
            self.lengths = copy.lengths.copy()
            self.size = copy.size
            self.max_clusters = copy.max_clusters
            self.dtype = copy.dtype
        else:
            self.array = zeros((size,max_clusters),dtype=dtype)
            self.lengths = zeros(size,dtype=int32)
            self.size = size
            self.max_clusters = max_clusters
            self.dtype = dtype

    def get(self,t,i):
        return self.array[t,i]

    def get_array(self,t):
        return self.array[t,:].copy()
        
    def append(self,t,x):
        self.array[t,self.lengths[t]] = x
        self.lengths[t] += 1

    def set(self,t,i,x):
        self.array[t,i] = x
    
    def copy(self,fro,to):
        self.array[to,:] = self.array[fro,:]
        self.lengths[to] = self.lengths[fro]

    def shallow_copy(self):
        return FixedSizeStore(self.size,self.max_clusters,self.dtype,self)

    def __str__(self):
        return str(self.array) + "\nLengths: " + str(self.lengths)

    __repr__ = __str__

class FixedSizeStoreRing(object):
    def __init__(self,size,max_clusters=100,dtype=object,copy=None):
        if copy != None:
            self.array = copy.array.copy()
            self.lengths = copy.lengths
            self.size = copy.size
            self.max_clusters = copy.max_clusters
            self.dtype = copy.dtype
        else:
            self.array = zeros(max_clusters,dtype=dtype)
            self.lengths = 0
            self.size = size
            self.max_clusters = max_clusters
            self.dtype = dtype
    
    def get(self,t,i):
        return self.array[i]

    def get_array(self,t):
        return self.array[:].copy()
        
    def append(self,t,x):
        self.array[self.lengths] = x
        self.lengths += 1

    def set(self,t,i,x):
        self.array[i] = x
    
    def copy(self,fro,to):
        pass

    def shallow_copy(self):
        return FixedSizeStoreRing(self.size,self.max_clusters,self.dtype,self)

    def __str__(self):
        return str(self.array) + "\nLengths: " + str(self.lengths)

    __repr__ = __str__

class ArrayOfLists(object):
    def __init__(self,size,dtype=None,copy=None):
        if copy != None:
            self.size = copy.size
            self.array = copy.array.copy()
            self.dtype = copy.dtype
            for t in range(self.size):
                self.array[t] = self.array[t][:]
        else:
            self.array = empty(size,dtype=object)
            for t in range(size):
                self.array[t] = []
            self.size = size
            self.dtype = dtype

    def get(self,t,i):
        if t < self.size and i < self.len(t):
            return self.array[t][i]
        else:
            raise ValueError, 'Index out of bounds t=' + str(t) + ' i=' + str(i)
    
    def copy(self,fro,to):
        self.array[to] = self.array[fro][:]

    def get_list(self,t):
        return self.array[t]

    def get_array(self,t):
        return array(self.array[t])

    def set_array(self,t,a):
        self.array[t] = a.tolist()

    def set_list(self,t,l):
        self.array[t] = l

    def len(self,t):
        return len(self.array[t])

    def append(self,t,x):
        self.array[t].append(x)

    def set(self,t,i,x):
        self.array[t][i] = x

    def __str__(self):
        out = []
        for t in range(self.size):
            if self.array[t] != []:
                out.append(str(t) + ': ' + str(self.array[t]) + '\n')

        return ''.join(out)

    def __repr__(self):
        return self.__str__()

    def shallow_copy(self):
        """Make a shallow copy of the array and the lists, but not the
        list contents."""
        new = ArrayOfLists(self.size,self.dtype,self)
        return new

StorageType = FixedSizeStoreRing
#StorageType = ArrayOfLists

def zerofun():
    return 0

class ExtendingList(list):
    """A list type that grows if the given index is out of bounds and fills the
    new space with a given default value."""
    def __init__(self,default=zerofun):
        list.__init__(self)
        self.default = default

    def __check(self,i):
        if len(self) <= i:
            for j in range(len(self),i+1):
                self.append(self.default())

    def __getitem__(self,i):
        if len(self) <= i:
            return self.default()
        else:
            return list.__getitem__(self,i)

    def __setitem__(self,i,x):
        self.__check(i)
        list.__setitem__(self,i,x)

    def shallow_copy(self):
        # FIXME: This may be inefficient
        new = ExtendingList(self.default)
        new.extend(self)
        return new


### Probability distributions 
# Normal distribution
def logpnorm(x, mu, lam):
    """Log of PDF of univariate Normal with mean mu and precision lam.
    
    If x, mean, and sigma are vectors, each dimension is treated independently.
    """
    return 0.5*(log(lam) - LOG2PI - lam*(x-mu)**2)

def pnorm(x, mu, lam):
    """PDF of univariate Normal with mean mu and precision lam.
    
    If x, mean, and sigma are vectors, each dimension is treated independently.
    """
    return exp(logpnorm(x, mu, lam))
        
def rnorm(mu, lam):
    sd = sqrt(1/lam)
    return mu + sd*R.standard_normal(mu.shape)

# Student-t distribution
def logpstudent(x, mu, lam, alpha):
    lnc = gammaln(0.5*(alpha + 1)) - gammaln(0.5*alpha) + 0.5*(log(lam) - log(alpha) - log(pi))
    lnp = lnc - (alpha+1)/2 * log(1+(lam*(x-mu)**2)/alpha)
    return lnp

def pstudent(x, mu, lam, alpha):
    return exp(logpstudent(x, mu, lam, alpha))
    
def rstudent(mu, lam, alpha):
    X = R.chisquare(alpha, mu.shape);
    Z = R.standard_normal(mu.shape);
    return mu + Z*sqrt(alpha/X)/sqrt(lam);
    
# Gamma distribution

def logpgamma(x,a,b):
    """Log gamma PDF. 

    The parameter b is the INVERSE scale, i.e. reciprocal of the corresponding
    parameter of the MATLAB gampdf function.
    """
    lnc = a*log(b)-gammaln(a)
    return lnc + (a-1)*log(x) - b*x

def pgamma(x,a,b):
    return exp(logpgamma(x,a,b))
    
def rgamma(a,b):
    return R.gamma(a,1/b)
    
# Binomial distribution
def rbinom(p, n):
    if isarray(p):
        return R.binomial(p, n, p.shape)
    else:
        return R.binomial(p, n)
        
# Multinomial distribution
def rmultinomial(p, N):
    return R.multinomial(N, p)
    
# Discrete distribution
def rdiscrete(pvals, numsamples=1):
    """Sample from discrete distribution with probabilities given by pvals.
    
    If size is given, a matrix of the given size is filled with samples, 
    otherwise a single value (between 0 and len(pvals) is returned.
    """
    if HAVE_INTSAMPLER and numsamples > 100:
        sampler = _intsampler(pvals)
        return sampler.sample(numsamples)
    else:
        cdf = cumsum(pvals)
        unif = R.random_sample(size=numsamples)
        return cdf.searchsorted(unif)


### Callback functions for use with the particle filter
class MAPCollector(object):
    """Class for collecting values from the MAP particle of the particle filter
    before resampling."""
    def __init__(self):
        self.paramlist = []

    def __call__(self,pf,t):
        map_idx = N.argmax(pf.weights)
        map_particle = pf.particles[map_idx]
        self.paramlist.append(map_particle.U.get_array(t))

    def __str__(self):
        print self.paramlist

    def __repr__(self):
        self.__str__()

    def get_first_mu(self):
        mus = zeros(len(self.paramlist))
        for n in range(len(self.paramlist)):
            mus[n] = self.paramlist[n][0].mu[0]
        return mus
