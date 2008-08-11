import numpy.random as R
from collections import deque

from utils import *
from numpy import *

class TransitionKernel(object):
    def __init__(self,model,params):
        self.params = params
        self.model = model

    def p_walk(self,old_mu,old_lam,mu,lam,tau=None):
       raise NotImplementedError 

    def walk(self,params,tau=None,p_old=None):
       raise NotImplementedError 
    
    def walk_with_data(self,params,data,tau=None):
        """Sample from the walk given some observersion. This fallback 
        implementation just samples from the walk ignoring the data.""" 
        return (self.walk(params,tau),1)

    def walk_backwards(self,params,tau=None):
        # FIXME: Not always true ...
        return self.walk(params,tau) 

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


class CaronIndependent(TransitionKernel):

    def __init__(self,model,params):
        TransitionKernel.__init__(self,model,params)
        self.num_aux = params[0]
        self.rho = params[1]
        self.D = model.params.mu0.shape[0]
        n0 = self.model.params.n0
        mu0 = self.model.params.mu0
        alpha = self.model.params.a
        beta = self.model.params.b
        self.beta_up = n0/(2*(n0+1))
        self.np = n0 + 1
        self.mu_up1 = (n0*mu0)/self.np
        self.mu_up2 = self.np * (alpha+0.5)
        self.mu_up3 = 2*alpha + 1
        self.gam_up = alpha+0.5

    def walk(self,params,tau=None):
        return self.__general_walk(params,data=None,tau=tau)

    def __general_walk(self,params,data=None,tau=None):
        """General version of the random walk allowing for an arbitrary
        number of auxiliary variables and/or data points.
        """
        return self.sample_posterior(self.sample_aux(params,tau),data,tau)

    def p_posterior(self,params,aux_vars,data=None):
        n0 = self.model.params.n0
        mu0 = self.model.params.mu0
        alpha = self.model.params.a
        beta = self.model.params.b
        num_aux = aux_vars.shape[1]
        if data != None:
            N = num_aux + 1
            nn = num_aux/self.rho + 1 
        else:
            N = num_aux
            nn = num_aux/self.rho 
        if data != None:
            aux_vars = c_[aux_vars,data]
        data_mean = mean(aux_vars,1)
        # make data_mean a rank-2 D-by-1 array so we can use broadcasting
        data_mean.shape = (data_mean.shape[0],1)
        nvar = sum((aux_vars-data_mean)**2,1)
        data_mean.shape = (data_mean.shape[0],)
        mu_star = (n0*mu0 + nn*data_mean)/(n0+nn)
        beta_star = beta + 0.5*nvar + (nn*n0*(mu0-data_mean)**2)/(2*(n0+nn))
        p1 = sum(logpgamma(params.lam,alpha+0.5*nn,beta_star))
        p2 = sum(logpnorm(params.mu,mu_star,(nn+n0)*params.lam))
        return exp(p1+p2)
        

    def sample_posterior(self,aux_vars,data,tau=None):
        """Sample from the posterior given the auxiliary variables and data."""
        n0 = self.model.params.n0
        mu0 = self.model.params.mu0
        alpha = self.model.params.a
        beta = self.model.params.b
        num_aux = aux_vars.shape[1]
        if data != None:
            N = num_aux + 1
            nn = num_aux/self.rho + 1 
        else:
            N = num_aux
            nn = num_aux/self.rho 
        if data != None:
            #print aux_vars.shape,data.shape
            aux_vars = c_[aux_vars,data]
        data_mean = mean(aux_vars,1)
        # make data_mean a rank-2 D-by-1 array so we can use broadcasting
        data_mean.shape = (data_mean.shape[0],1)
        nvar = sum((aux_vars-data_mean)**2,1)
        data_mean.shape = (data_mean.shape[0],)
        mu_star = (n0*mu0 + nn*data_mean)/(n0+nn)
        beta_star = beta + 0.5*nvar + (nn*n0*(mu0-data_mean)**2)/(2*(n0+nn))
        n_lam = rgamma(alpha+0.5*nn,beta_star)
        n_mu = rnorm(mu_star,(nn+n0)*n_lam)
        return self.model.get_storage(n_mu,n_lam)


    def sample_aux(self,params,tau=None):
        """Sample auxiliary variables given the current state."""
        return rnorm_many(params.mu,params.lam*self.rho,self.num_aux)

    
    def walk_with_data(self,params,data,tau=None):
        aux_vars = self.sample_aux(params,tau)
        params = self.sample_posterior(aux_vars,data,tau)
        p1 = self.p_posterior(params,aux_vars,None)
        p2 = self.p_posterior(params,aux_vars,data)
        return (params,p1/p2)




class Model(object):
    pass

class DiagonalConjugate(Model):
    
    def __init__(self,hyper_params,kernelClass=MetropolisWalk,kernelParams=(0.1,0.001)):
        self.params = hyper_params
        self.dims = self.params.dims
        self.empty = True
        self.kernel = kernelClass(self,kernelParams)
        self.walk = self.kernel.walk
        self.walk_with_data = self.kernel.walk_with_data

    def read_params(self):
        """Read the parameters of the model from a file.
        """
        # Maybe use the ConigParser class?
        pass # TODO

    def write_params(self):
        """Write the parameters of this model to a file, so it can be 
        reconstructed using read_params.
        """
        pass # TODO

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

    def p_log_posterior_mean(self,mu,lam):
        """Compute log p(mu|z)."""
        if self.empty:
            p = 0;
        else:
            p = sum(logpnorm(mu,self.mun,lam*self.nn))
        return p

    def p_log_posterior_precision(self,lam):
        if self.empty:
            p = 0;
        else:
            p = sum(logpgamma(lam,self.params.a+0.5*self.nk,self.bn));
        return p
    
    def p_posterior(self,params):
        return exp(self.p_log_posterior_mean(params.mu,params.lam) +
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
        lam = rgamma(self.params.a+0.5*self.nk,self.bn)
        mu = rnorm(self.mun,lam*self.nn)
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
        return DiagonalConjugateStorage(mu,lam)
    
class DiagonalConjugateHyperParams(object):
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

class DiagonalConjugateStorage(object):
    """Class for storing the parameter values of a single component."""
    def __init__(self,mu=None,lam=None):
        self.mu = mu
        self.lam = lam
    
    def __str__(self):
        return 'mu: ' + str(self.mu) + '\nlambda: ' + str(self.lam)
    
    def __repr__(self):
        return self.__str__()




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
            self.d = T * ones(T,dtype=uint32)
            
            # total number of clusters in this particle up to the current time
            self.K = 0
            
            # array to store class counts at each time step
            self.mstore = self.storage_class(T,dtype=int32)
           
            # storage object for the spike time of the last spike associated 
            # with each cluster for each time step. 
            self.lastspike = self.storage_class(T,dtype=float64)
            
            # cell array to store the sampled values of rho across time
            # self.rhostore = zeros(T);
            
            # Parameter values of each cluster 1...K at each time step 1...T
            self.U = self.storage_class(T,dtype=object);
            
            # vector to store the birth times of clusters
            self.birthtime = ExtendingList()
            
            # vector to store the death times of allocation variables (0 if not dead)
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

class GibbsState():
    """Class representing the state of the Gibbs sampler. This is similar to
    a particle in many respects. However, as in the Gibbs sampler we only 
    need to hold one state object in memory at any given time, we can trade
    off speed and memory consumption differently.
    
    If a particle object is passed to the constructor it will be used to
    initialize the state.
    """
    def __init__(self,particle=None,model=None,max_clusters=100):
        self.max_clusters = max_clusters
        if particle != None and model != None:
            self.from_particle(particle,model)
        else:
            self.__empty_state()

    def from_particle(self,particle,model):
        """Construct state from the given particle object."""
        self.T = particle.T
        # allocation variables for all time steps
        self.c = particle.c.copy()
        # death times of allocation variables
        self.d = particle.d.copy()
        # make sure the maximum death time is T
        self.d[self.d>self.T] = self.T
        # total number of clusters in the current state
        self.K = particle.K
       
        # array to store class counts at each time step
        self.mstore = zeros((self.max_clusters,self.T),dtype=int32)
        self.lastspike = zeros((self.max_clusters,self.T),dtype=float64)
        self.U = empty((self.max_clusters,self.T),dtype=object)

        self.aux_vars = zeros(
            (self.T,
            self.max_clusters,
            model.kernel.D,
            model.kernel.num_aux))
        print self.aux_vars.shape
        for t in range(self.T):
            m = particle.mstore.get_array(t)
            n = m.shape[0]
            self.mstore[0:n,t] = m

            m = particle.lastspike.get_array(t)
            n = m.shape[0]
            self.lastspike[0:n,t] = m
            
            m = particle.U.get_array(t)
            n = m.shape[0]
            self.U[0:n,t] = m
        
        # vector to store the birth times of clusters
        self.birthtime = particle.birthtime.to_array(self.max_clusters,dtype=int32)
        
        # vector to store the death times of clusters (0 if not dead)
        self.deathtime = particle.deathtime.to_array(self.max_clusters,dtype=int32) 
        self.deathtime[self.deathtime==0] = self.T # TODO: Needed?
        
        # determine active clusters
        active = where(sum(self.mstore,1)>0)[0]

        # compute free labels
        self.free_labels = deque(reversed(list(set(range(self.max_clusters))-set(active))))

        # all clusters must have parameters from time 0 to their death
        # -> sample them from their birth backwards
        for c in active:
            for t in reversed(range(0,self.birthtime[c])):
                logging.debug("sampling params for cluster %i at time %i" % (c,t))
                self.U[c,t] = model.kernel.walk_backwards(
                        self.U[c,t+1])


    def __empty_state(self):
        """Set all fields to represent an empty state."""
        pass # TODO -> do we really need this?

    def check_consistency(self,data_time):
        """Check consistency of the Gibbs sampler state.

        In particular, perform the following checks:
        
            1) if m(c,t) > 0 then U(c,t) != None
            2) m(c,birth:death-1)>0 and m(c,0:birth)==0 and m(c,death:T)==0
            3) m matches the information in c and deathtime
            4) birthtime matches c
            5) no parameter is NaN
            6) check that lastspike is correct

        """
        errors = 0
        # check 1) we have parameter values for all non-empty clusters
        idx = where(self.mstore>0)
        if any(isNone(self.U[idx])):
            logging.error("Consitency error: Some needed parameters are None!"+
                    str(where(isNone(self.U[idx]))))
            errors += 1
        # check 1b) we need parameter values from 0 to the death of each cluster
        active = where(sum(self.mstore,1)>0)[0]
        for c in active:
            d = self.deathtime[c]
            if any(isNone(self.U[c,0:d])):
                logging.error("Consitency error: Parameters not avaliable " +
                        "from the start")


        # check 2) There are no "re-births", assuming birthtime and deathtime
        # are correct
        active = where(sum(self.mstore,1)>0)[0]
        for c in active:
            # the death time of _cluster_ c is the first zero after its birth
            birth = self.birthtime[c]
            active_birth_to_end = where(self.mstore[c,birth:]==0)[0]
            if active_birth_to_end.shape[0] == 0:
                death = self.T
            else:
                death = birth + active_birth_to_end[0]
            if death != self.deathtime[c]:
                logging.error("deatime does not contain the first zero after "+
                        "birth of cluster %i" % c)
            if (any(self.mstore[c,birth:death]==0)):
                logging.error(("Consistency error: mstore 0 while cluster %i is " +
                        "alive") % c)
            if any(self.mstore[c,0:birth]>0):
                logging.error(("Consistency error: mstore > 0 while cluster %i is " +
                        "not yet born") % c)
            if any(self.mstore[c,death:]>0):
                logging.error(("Consistency error: mstore > 0 while cluster %i is " +
                        "already dead!") % c)

        # check 3) we can reconstruct mstore from c and d
        new_ms = self.reconstruct_mstore(self.c,self.d)
        if any(self.mstore != new_ms):
            logging.error("Consitency error: Cannot reconstruct mstore from c and d")

        # check 4) 
        # birth = where(self.mstore[c,:]>0)[0][0]
        # print birth

        # check 5)

        # check 6)
        # lastspike[c,t] is supposed to contain the last spike time for all
        # clusters _after_ the observation at time t
        lastspike = zeros(self.max_clusters)
        for t in range(self.T):
            lastspike[self.c[t]] = data_time[t]
            if any(self.lastspike[:,t]!=lastspike):
                logging.error("Consitency error:lastspike incorrect at time %i"
                        % t)


    def reconstruct_mstore(self,c,d):
        new_ms = zeros_like(self.mstore)
        for t in range(self.T):
            if t > 0:
                new_ms[:,t] = new_ms[:,t-1]
            new_ms[c[t],t] += 1
            dying = where(d == t)[0]
            for tau in dying:
                new_ms[c[tau],t] -= 1
        return new_ms


    def __str__(self,include_U=True):
        out = []
        out.append('c: ' + str(self.c)+'\n')
        out.append('d: ' + str(self.d)+'\n')
        out.append('K: ' + str(self.K)+'\n')
        out.append('mstore: ' + str(self.mstore)+'\n')
        out.append('lastspike: ' + str(self.lastspike)+'\n')
        if include_U:
            out.append('U: ' + str(self.U)+'\n')
        return ''.join(out)



