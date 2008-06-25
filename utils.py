import numpy as N
from numpy import array, exp, log, sqrt, cumsum,  pi
from scipy.special import gammaln
import numpy.random as R

try:
    from scipy.sandbox.montecarlo._intsampler import _intsampler
    HAVE_INTSAMPLER = True
except:
    HAVE_INTSAMPLER = False

LOG2PI = log(2*pi)

def isarray(a):
    return isinstance(a, N.ndarray)

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
    Z = R.standard_nomal(mu.shape);
    return mu + Z*sqrt(alpha/X)/sqrt(lam);
    
# Gamma distribution
def pgamma():
    pass
    
def rgamma():
    pass
    
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
        unif = R.uniform(size=numsamples)
        return cdf.searchsorted(unif)

    
