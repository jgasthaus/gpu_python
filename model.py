from utils import *
from numpy import *
import numpy.random as R

class TransitionKernel(object):
    def __init__(self,model,params):
        self.params = params
        self.model = model

    def p_walk(self,old_mu,old_lam,mu,lam):
       raise NotImplementedError 

    def walk(self,mu,lam,p_old=None):
       raise NotImplementedError 

class MetropolisWalk(TransitionKernel):

    def walk(self,mu,lam,p_old=None):
        if p_old == None:
            p_old = exp(self.model.p_log_prior_params(mu,lam));
        
        # random walk on mean
        n_mu = mu + self.params[0] * R.standard_normal(self.model.dims)
        
        n_lam = lam + self.params[1] * R.standard_normal(self.model.dims)

        # keep values that are about to become negative the same
        if self.model.dims > 1:
            idx = n_lam <= 0
            n_lam[idx] = lam[idx]
        else:
            if n_lam <= 0:
                n_lam = lam
        
        # Metropolis update rule
        p_new = exp(self.model.p_log_prior_params(n_mu,n_lam))
        
        if R.rand() > p_new/p_old: # not accepted -> keep old values
            n_mu = mu
            n_lam = lam
        return self.model.get_storage(mu,lam)

class Model(object):
    pass

class DiagonalConjugate(Model):
    
    def __init__(self,hyper_params,kernelClass=MetropolisWalk,kernel_params=(0.001,0.001)):
        self.params = hyper_params
        if len(self.params.b.shape) > 0:
            self.dims = self.params.b.shape[0]
        else:
            self.dims = 1
        self.empty = True
        self.kernel = kernelClass(self,kernel_params)
        self.walk = self.kernel.walk

    def set_data(self,data):
        if isempty(data):
            self.empty = True;
        else:
            self.empty = False;

        self.mean = mean(data,1)
        # column vector of variances
        self.nvar =  (data - samplemean)**2
        self.nk = data.shape[1]
        self.nn = self.params.n0 + self.nk
        self.mun = (self.params.n0 * self.params.mu0 + self.nk * self.mean)/(self.nn)
        self.bn = self.params.b + 0.5*self.nvar + 0.5/self.nn*self.nk*self.params.n0* \
                    (self.params.mu0 - self.mean)**2;
        self.ibn = 1/self.bn;

    def p_log_likelihood(self,x,params):
        """Compute log p(x|params)"""
        return sum(logpnorm(x,params.mu,params.lam))

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

    def p_log_posterior_mean(self,mu):
        """Compute log p(mu|z)."""
        if self.empty:
            p = 0;
        else:
            p = sum(logpstudent(x,self.mun,
                                self.nn*(self.params.a + 0.5*self.nk)*self.ibn,
                                2*self.params.a+self.nk));
        return p

    def p_log_posterior_precision(self,lam):
        if self.empty:
            p = 0;
        else:
            p = sum(logpgamma(lam,self.params.a+0.5*self.nk,self.bn));
        return p
    
    def p_posterior(self,mu,lam):
        return exp(p_log_posterior_mean(mu) +
                   p_log_posterior_precision(lam))

    def p_log_prior(self,x):
        return sum(logpstudent(x,self.params.mu0,
            self.params.n0/(self.params.n0+1)*self.params.a/self.params.b,
            2.*self.params.a))
    
    def p_log_prior_params(self,mu,lam):
        return sum(logpnorm(mu,self.params.mu0,self.params.n0 * lam)) + \
               sum(logpgamma(lam,self.params.a,self.params.b));

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
            return (self.walk(mu,lam),1)

        # SIR: sample from P(U|U_old), compute weights P(x|U), then 
        # sample from the discrete distribution.
        mu_samples = zeros((self.dims,num_sir_samples))
        lam_samples = zeros((self.dims,num_sir_samples))
        sir_weights = zeros(num_sir_samples)
        p_old = self.p_log_prior_params(mu,lam);
        for s in range(num_sir_samples):
            tmp = walk(mu,lam,p_old);
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
        def __init__(self,a,b,mu0,n0):
            self.a = array(a,copy=False)
            self.b = array(b,copy=False)
            self.mu0 = array(mu0)
            self.n0 = n0
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
    pass

