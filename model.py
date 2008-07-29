from utils import *
from numpy import *
import numpy.random as R

class TransitionKernel(object):
    def __init__(self,model,params):
        self.params = params
        self.model = model

    def p_walk(self,old_mu,old_lam,mu,lam):
       raise NotImplementedError 

    def walk(self,mu,lam,tau=None,p_old=None):
       raise NotImplementedError 

class MetropolisWalk(TransitionKernel):

    def walk(self,params,tau=None,p_old=None):
        if p_old == None:
            p_old = exp(self.model.p_log_prior_params(params));
        
        # random walk on mean
        n_mu = params.mu + self.params[0] * R.standard_normal(self.model.dims)
        
        n_lam = params.lam + self.params[1] * R.standard_normal(self.model.dims)

        # keep values that are about to become negative the same
        if self.model.dims > 1:
            idx = n_lam <= 0
            n_lam[idx] = params.lam[idx]
        else:
            if n_lam <= 0:
                n_lam = params.lam
        
        # Metropolis update rule
        new_params = self.model.get_storage(n_mu,n_lam)
        p_new = exp(self.model.p_log_prior_params(new_params))
        # print p_new,p_old,p_new/p_old      
        if R.rand() > p_new/p_old: # not accepted -> keep old values
            new_params = params
        return new_params

class Model(object):
    pass

class DiagonalConjugate(Model):
    
    def __init__(self,hyper_params,kernelClass=MetropolisWalk,kernelParams=(0.1,0.001)):
        self.params = hyper_params
        self.dims = self.params.dims
        self.empty = True
        self.kernel = kernelClass(self,kernelParams)
        self.walk = self.kernel.walk

    def set_data(self,data):
        # if data.shape[1]!=0:
        #     self.empty = True
        # else:
        #     self.empty = False;
        if len(data.shape) <= 1:
            # just one data point
            self.mean = data
            self.nvar =  zeros_like(data)
            self.nk = 1
            self.nn = self.params.n0 + 1
            self.mun = (self.params.n0 * self.params.mu0 + self.mean)/self.nn
            self.bn = self.params.b  + 0.5/self.nn*self.params.n0* \
                        (self.params.mu0 - self.mean)**2
            self.ibn = 1/self.bn;
        else:
            self.mean = mean(data,1)
            # column vector of variances
            self.nvar =  (data - samplemean)**2
            self.nk = data.shape[1]
            self.nn = self.params.n0 + self.nk
            self.mun = (self.params.n0 * self.params.mu0 + self.nk * self.mean)/(self.nn)
            self.bn = self.params.b + 0.5*self.nvar + 0.5/self.nn*self.nk*self.params.n0* \
                        (self.params.mu0 - self.mean)**2;
            self.ibn = 1/self.bn;
        self.empty = False

    def p_log_likelihood(self,x,params):
        """Compute log p(x|params)"""
        return sum(logpnorm(x,params.mu,params.lam))
    
    def p_likelihood(self,x,params):
        return exp(self.p_log_likelihood(x,params))

    def p_log_predictive(self,x):
        """Compute log p(x|z)."""
        if self.empty:
            p = self.p_log_prior(x)
        else:
            p = sum(logpstudent(
                x,
                self.mun,
                self.nn*(self.params.a + 0.5*self.nk)/(self.nn + 1)*self.ibn,
                2*self.params.a+self.nk))
        return p

    def p_predictive(self,x):
        return exp(self.p_log_predictive(x))

    def p_log_posterior_mean(self,mu):
        """Compute log p(mu|z)."""
        if self.empty:
            p = 0;
        else:
            p = sum(logpstudent(mu,self.mun,
                                self.nn*(self.params.a + 0.5*self.nk)*self.ibn,
                                2*self.params.a+self.nk));
        return p

    def p_log_posterior_precision(self,lam):
        if self.empty:
            p = 0;
        else:
            p = sum(logpgamma(lam,self.params.a+0.5*self.nk,self.bn));
        return p
    
    def p_posterior(self,params):
        return exp(self.p_log_posterior_mean(params.mu) +
                   self.p_log_posterior_precision(params.lam))

    def p_log_prior(self,x):
        """Compute log p(x) (i.e. \int p(x|theta)p(theta) dtheta)."""
        return sum(logpstudent(x,self.params.mu0,
            self.params.n0/(self.params.n0+1)*self.params.a/self.params.b,
            2.*self.params.a))

    def p_prior(self,x):
        return exp(self.p_log_prior(x))
    
    def p_log_prior_params(self,params):
        return sum(logpnorm(params.mu,self.params.mu0,self.params.n0 * params.lam)) + \
               sum(logpgamma(params.lam,self.params.a,self.params.b));

    def p_prior_params(self,params):
        return exp(self.p_log_prior_params(params))

    def sample_posterior(self):
        if self.empty:
            return self.sample_prior()
        mu = rstudent(
                self.mun,
                self.nn*(self.params.a + 0.5*self.nk)*self.ibn,
                2*self.params.a+self.nk)
        lam = R.gamma(self.params.a+0.5*self.nk,self.ibn)
        return self.get_storage(mu,lam)

    def sample_prior(self):
        lam = rgamma(self.params.a,self.params.b)
        mu = rnorm(self.params.mu0,self.params.n0 * lam)
        return self.get_storage(mu,lam)

    def sample_Uz(self,mu,lam,data,num_sir_samples=10):
        """Sample from p(U|U_old,z)=p(U|U_old)p(z|U)/Z."""
        if self.empty:
            #TODO: Dependence of walk on tau
            return (self.walk(mu,lam),1)

        # SIR: sample from P(U|U_old), compute weights P(x|U), then 
        # sample from the discrete distribution.
        mu_samples = zeros((self.dims,num_sir_samples))
        lam_samples = zeros((self.dims,num_sir_samples))
        sir_weights = zeros(num_sir_samples)
        p_old = self.p_log_prior_params(mu,lam);
        for s in range(num_sir_samples):
            # TODO: dependence of walk on tau
            tmp = walk(mu,lam,p_old=p_old);
            mu_samples[:,s] = tmp.mu
            lam_samples[:,s] = tmp.lam
            sir_weights[s] = self.p_posterior(tmp.mu,tmp.lam)
        sir_weights = sir_weights / sum(sir_weights);
        s = rdiscrete(sir_weights)
        new_mu = mu_samples[:,s]
        new_lam = lam_samples[:,s]
        weight = sir_weights[s]
        return (self.get_storage(new_mu,new_lam),weight)

    def get_storage(self,mu=None,lam=None):
        """Get a new parameter storage object."""
        return DiagonalConjugate.Storage(mu,lam)
    
    class HyperParams(object):
        def __init__(self,a,b,mu0,n0,dims=None):
            if dims != None:
                self.a = ones(dims) * a
                self.b = ones(dims) * b
                self.mu0 = ones(dims) * mu0
            else:
                self.a = a
                self.b = b
                self.mu0 = mu0
            self.n0 = n0
            if self.a.shape != self.b.shape:
                raise ValueError, "shape mismatch: a.shape: " + str(a.shape) +\
                    "b.shape: " + str(b.shape)
            elif self.a.shape != self.mu0.shape:
                raise ValueError, "shape mismatch: a.shape: " + str(a.shape) +\
                    "mu0.shape: " + str(mu0.shape)
            if len(self.a.shape)!= 0:
                self.dims = self.a.shape[0]
            else: 
                self.dims = 1


        def __str__(self):
            out = ['Model hyperparameters:\n']
            out.append('a: ' + str(self.a) + '\n')
            out.append('b: ' + str(self.b) + '\n')
            out.append('mu0: ' + str(self.mu0) + '\n')
            out.append('n0: ' + str(self.n0) + '\n')
            return ''.join(out)

    class Storage(object):
        """Class for storing the parameter values of a single component."""
        def __init__(self,mu=None,lam=None):
            self.mu = mu
            self.lam = lam
        def __str__(self):
            return 'mu: ' + str(self.mu) + '\nlambda: ' + str(self.lam)



class Particle(object):
    """The Particle class stores the state of the particle filter / Gibbs
    sampler.
    """
    def __init__(self,T,copy=None,storage_class=FixedSizeStoreRing):
        if copy != None:
            self.T = copy.T
            self.c = copy.c.copy()
            self.d = copy.d.copy()
            self.K = copy.K
            self.mstore = copy.mstore.shallow_copy()
            self.lastspike = copy.lastspike.shallow_copy()
            self.U = copy.U.shallow_copy()
            self.birthtime = copy.birthtime.shallow_copy()
            self.deathtime = copy.deathtime.shallow_copy()
            self.storage_class = copy.storage_class
        else:
            self.T = T
            self.storage_class = storage_class
            # allocation variables for all time steps
            self.c = -1*ones(T,dtype=int16)
            # death times of allocation variables (assume they don't die until they do)
            self.d = (T+1) * ones(T,dtype=uint32)
            
            # current time
            # p.t = 0;
            
            # total number of clusters in this particle up to the current time
            self.K = 0
            
            # column vector containing the sizes of the current non-empty clusters
            # p.m = zeros(T*Nt,1);
            
            # array to store class counts at each time step
            self.mstore = self.storage_class(T,dtype=int32)
            
            self.lastspike = self.storage_class(T,dtype=float64)
            
            # cell array to store the sampled values of rho across time
            # self.rhostore = zeros(T);
            
            # Parameter values of each cluster 1...K at each time step 1...T
            # each entry should be a struct with p.U{t}{k}.m and p.U{t}{k}.C
            self.U = self.storage_class(T,dtype=object);
            
            # vector to store the birth times of clusters
            self.birthtime = ExtendingList()
            
            # vector to store the death times of clusters (0 if not dead)
            self.deathtime = ExtendingList() 

    def shallow_copy(self):
        """Make a shallow copy of this particle.

        In essence, copies of lists are created, but the list contents are not
        copied. This is useful for making copies of particles during
        resampling, such that the resulting particles share the same history,
        but can be moved forward independently.
        """
        return Particle(self.T,self)

        


    def __str__(self):
        out = []
        out.append('c: ' + str(self.c)+'\n')
        out.append('d: ' + str(self.d)+'\n')
        out.append('K: ' + str(self.K)+'\n')
        out.append('mstore: ' + str(self.mstore)+'\n')
        out.append('lastspike: ' + str(self.lastspike)+'\n')
        out.append('U: ' + str(self.U)+'\n')
        return ''.join(out)

    __repr__ = __str__
