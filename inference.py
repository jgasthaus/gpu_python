from utils import *
from model import *
from numpy import *
from numpy.random import rand, random_sample
from scipy.maxentropy.maxentutils import logsumexp
import logging
import time

### RESAMPLING SCHEMES
def multinomial_resampling(weights):
    return (1/ones(len(weights)),
            counts_to_index(rmultinomial(weights,len(weights)))
           )

class InferenceParams(object):
    def __init__(self,rho,alpha,p_uniform_deletion,r_abs):
        self.rho = rho
        self.alpha = alpha
        self.p_uniform_deletion = p_uniform_deletion
        self.r_abs = r_abs

    def __str__(self):
        out = []
        out.append('Inference parameters:')
        out.append('rho: ' + str(self.rho))
        out.append('alpha: ' + str(self.alpha))
        out.append('p_uniform_deletion: ' + str(self.p_uniform_deletion))
        out.append('r_abs: ' + str(self.r_abs))
        return '\n'.join(out)

class Inference:
    pass

class ParticleFilter(Inference):
    def __init__(self,model,data,data_time,params,num_particles,
                 storage_class=FixedSizeStoreRing,
                 resample_fun=multinomial_resampling,
                 before_resampling_callback=noop):
        self.model = model
        self.data = data
        self.data_time = data_time
        self.num_particles = num_particles
        self.resample_fun = resample_fun
        self.params = params
        self.before_resampling_callback = before_resampling_callback
        self.T = data.shape[1]
        self.particles = empty(num_particles,dtype=object)
        for i in range(num_particles):
            self.particles[i] = Particle(self.T,None,storage_class)
        self.weights = ones(num_particles)/num_particles
        self.__check()

    def __check(self):
        """Check whether dimensions of parameters and data are consistent."""
        if not len(self.data.shape) == 2:
            raise ValueError, \
                  "Data should be a 2D array with data points as columns!"
        if not self.model.dims == self.data.shape[0]:
            raise ValueError, "Model dimension does not match data dimension: "+\
                    str(self.model.dims) + " != " + str(self.data.shape[0])

    def run(self):
        for t in range(self.T):
            start_t = time.time()
            logging.info('t = ' + str(t) + '/' + str(self.T))
            x = self.data[:,t]
            tau = self.data_time[t]
            # the probability under the prior is the same for all particles
            p_prior = self.model.p_prior(x)
            self.model.set_data(x);
            # move particles forward
            for n in range(self.num_particles):
                p = self.particles[n]
                # perform deletion step / compute new cluster sizes {{{
                if t > 0:
                    p.mstore.copy(t-1,t)
                    p.lastspike.copy(t-1,t)
                    m = p.mstore.get_array(t)
                    old_zero = m == 0;
                    if rand() < self.params.p_uniform_deletion: # uniform deletion
                        # TODO: Speed this up by sampling only from surviving allocations
                        U = random_sample(p.c.shape);
                        # delete from alive allocations with prob. 1-p.rho
                        # We assume that for non-assigned x we have c<0
                        idx = logical_and(logical_and(U<1-self.params.rho,p.d>=t), p.c>=0)
                    else: # size-biased deletion
                        i = rdiscrete(m/float(sum(m)),1)
                        idx = logical_and(logical_and(p.c == i, p.d>=t), p.c >= 0)
                    print "deleting",idx
                
                    p.d[idx] = t
                     # compute current alive cluster sizes p.m; TODO: vectorize this?
                    for k  in range(p.K):
                        nm = sum(logical_and(p.c[0:t] == k,p.d[0:t]>t))
                        m[k] = nm
                        p.mstore.set(t,k,nm)
                
                    new_zero = m == 0;
                    died = logical_and(new_zero,logical_not(old_zero)).nonzero()[0]
                    for d in died:
                        p.deathtime[d] = t
                else:
                    m = array([],dtype=int32)
                ### sample new labels for all data points in data {{{ 
                # We use q(c_t|m,U,z) = p(z_k|U) x p(c|m) as proposal distribution, 
                # i.e. the product of the CRP and the probability of the data point under
                # that class assignment with the current class parameters (or the prior if we
                # generate a new class).
                active_idx = m>0
                active = where(active_idx)[0]
                
                # number of clusters before we see new data
                Kt = len(active)
                
                # Generalized Poly Urn / CRP
                p_crp = hstack((m[active_idx],self.params.alpha))
                p_crp = p_crp/sum(p_crp)
                
                # Vector for storing the likelihood values
                p_lik = zeros(Kt+1);
                
                # compute probability of data point under all old clusters
                for i in range(Kt):
                    isi = self.data_time[t] - p.lastspike.get(t,active[i])
                    if isi < self.params.r_abs:
                        p_crp[i] = 0
                    p_lik[i] = self.model.p_likelihood(x,p.U.get(t-1,active[i]))
                
                # likelihood for new cluster
                p_lik[Kt] = p_prior
                logging.debug("x: " + str(x))
                logging.debug("p_lik: " + str(p_lik)) 
                logging.debug("p_crp: " + str(p_crp/sum(p_crp)))
                # propsal distribution: CRP x likelihood
                q = p_crp * p_lik
                
                # normalize to get a proper distribution
                q = q / sum(q)
                
                # sample a new label from the discrete distribution q
                c = rdiscrete(q,1)[0]
                Z_qc = p_crp[c]/q[c]
                # update data structures if we propose a new cluster
                if c == Kt:
                    # set birthtime of cluster K to the current time
                    p.birthtime[p.K] = t
                    active = hstack((active,p.K))
                    p.mstore.append(t,0)
                    p.lastspike.append(t,0)
                    # update number-of-clusters counts
                    p.K += 1
                active_c = active[c]
                p.mstore.set(t,active_c,p.mstore.get(t,active_c)+1)
                # assign data point to cluster
                p.c[t] = active_c
                p.lastspike.set(t,active_c,self.data_time[t])
                ### sample parameters U for all alive clusters {{{ 
                # 
                # This samples from q(U|...), for each of the three conditions:
                #   - new cluster created at this time step
                #       - sample from prior updated with the data from this time step
                #   - old cluster, and data assigned to it in this time step
                #       - sample from distribution given old value and new data
                #   - old cluster, but no new data assigned to it
                #       - sample from transition kernel
                # 
                pU_U = ones(Kt)
                qU_Uz = ones(Kt)
                p.U.copy(t-1,t)
                for i in range(len(active)):  # for all active clusters
                    cabs = active[i]
                    if i >= Kt:  # cluster newly created at this time step
                        new_params = self.model.sample_posterior()
                        p.U.append(t,new_params)
                        # compute probability of this sample for use in weight
                        qU_z = self.model.p_posterior(new_params)
                        # compute probability of this sample under G_0
                        G0 = self.model.p_prior_params(new_params)
                    else:  # old cluster
                        if cabs == c: # new data associated with cluster at this time step
                            new_params = self.model.walk_with_data(p.U.get(t,cabs),x)
                        else: # no new data
                            new_params = self.model.walk(p.U.get(t,cabs))
                        p.U.set(t,cabs,new_params) 
                        
                # %%% compute incremental weight for this update step {{{ 
                # %
                # % The weight is computed from the following components:
                # %   - prod(Z_qc) -- the normalizer of q(c|m,...); first line of (9)
                # %   - prod(G0)   -- G0(U); num. of third line of (9)
                # %   - prod(qU_z) -- q(U|{z|c=k}); denom. of third line of (9)
                # %   - prod(pU_U) -- p(U|U_old); num. of second line of (9)
                # %   - prod(qU_Uz)-- q(U|U_old,z); denom. of second line of (9)
                # 
                # w_inc = prod(Z_qc).*prod(pU_U)./prod(qU_Uz).*prod(G0)./prod(qU_z);
                # compute probability of current data point under new parameters
                pz_U = self.model.p_likelihood(x,p.U.get(t,active_c))
                w_inc = pz_U*Z_qc*G0/qU_z
                self.weights[n] *= w_inc
                # 
                # if isnan(w_inc) % bad weight -- can happen if underflow occurs
                #     w_inc = 0; % discard the particle by giving it weight 0
                # end

            ### resample
            # normalize weights
            self.weights = self.weights / sum(self.weights)
            Neff = 1/sum(self.weights**2)
            self.before_resampling_callback(self,t)
            self.weights, resampled_indices = self.resample_fun(self.weights)
            new_particles = empty(self.num_particles,dtype=object)
            used = set()
            for i in range(len(resampled_indices)):
                j = resampled_indices[i]
                if j in used:
                    new_particles[i] = self.particles[j].shallow_copy()
                else:
                    new_particles[i] = self.particles[j]
                    used.add(j)
            self.particles = new_particles
            logging.info("Effective sample size: " + str(Neff))
            end_t = time.time()
            elapsed = end_t - start_t
            remaining = elapsed * (self.T-t)
            logging.info("One step required " + str(elapsed) + " seconds, " +
                    str(remaining) + " secs remaining.")
            logging.debug(str(self.particles[0].mstore.get_array(t)))
            logging.debug(self.particles[0].U.get_array(t))
    
    def get_labeling(self):
        labeling = empty((self.num_particles,self.T),dtype=int32)
        for p in range(self.num_particles):
            labeling[p,:] = self.particles[p].c
        return labeling

class GibbsSampler(Inference):
    def __init__(self,data,data_time,model,state=None):
        self.data = data
        self.data_time = data_time
        self.model = model
        self.T = data.shape[1]
        if state != None:
            self.state = state
            if state.T != self.T:
                logging.error("State length does not match data length!")
        else:
            self.state = self.__init_state()

    def __init_state(self):
        """Initialize the state of the Gibbs sampler."""
        self.num_clusters = 1 # TODO

    def sweep(self):
        """Do one Gibbs sweep though the data."""
        for t in range(self.T):
            logging.info("t=%i/%i" % (t,self.T))
            self.sample_label(t)
            self.sample_death_time(t)
            self.sample_params(t)
            self.state.check_consistency()
            #print self.state
            raw_input()

    
    def sample_death_time(self,t):
        """Sample a new death time for the allocation variable at time t."""
        pass

    def sample_label(self,t):
        """Sample a new label for the data point at time t."""
        logging.debug("Sampling new label at time %i" % t)
        state = self.state
        print self.p_label_posterior(t)

    def sample_params(self,t):
        """Sample new parameters for the clusters at time t."""
        # for c in cluster(t):
        # self.sample_param(t,c)
        pass

    def sample_param(self,t,c):
        """Sample new parameters for cluster c at time. The cluster may be
        an old cluster or newly created."""

    def sample_walk(self,c,start,stop):
        """Sample new parameters from the walk for cluster c between time 
        steps start and stop. This is necessary if we extend the life of a
        cluster by sampling a new death time.
        """
        pass

    def p_label_posterior(self,t):
        """Compute the posterior probability over allocation variables given
        all other allocation variables and death times.
        
        Returns:
            None            if the allocation cannot be changed due to DCW
            (possible,p)    where possible is an array of cluster labels
                            the we can assign to, and p is an array of 
                            the respective probabilities.
        """
        # 2) temporarily remove the current allocation from the counts m
        # 3) for each possible label:
        #       - temporarily assign to this cluster and update m
        #       - compute joint of seating arrangement up to d_t
        state = self.state
        ms = state.mstore.copy() # local working copy
        c_old = state.c[t]
        d = min(state.d[t],state.T+1) # TODO: min needed?
        # remove from old cluster
        ms[c_old,t:d] = ms[c_old,t:d] - 1
        # Check for "dying customer's wish": 
        # If removing the current allocation causes the cluster to die, but
        # data is assigned to it _after_ its death, then we can't move the
        # allocation
        tmp = where(ms[c_old,t:]==0)[0]
        if tmp.shape[0] == 0:
            new_death = state.T+1
        else:
            new_death = t + tmp[0]
        if any(where(state.c==c_old)[0]>=new_death):
            # dying customers wish
            logging.debug("DCW at time %i, %i=>%i" % 
                    (t,state.deathtime[c_old],new_death))
            return None
        
        # 1) Determine which clusters we can potentially assign to:
        #       - any cluster that is alive at any point from now until this
        #         alloc dies
        possible = where(sum(state.mstore[:,t:d],1)>0)[0]
        # remove allocation c_t from ms[c_t,t]
        for tau in range(t+1,d):
            ms[state.c[tau],tau] -= 1
        p_crp = zeros(possible.shape[0],dtype=float64)
        for i in range(possible.shape[0]):
            ms_tmp = ms.copy()
            c_new = possible[i]
            # temporarily allocate to c_new
            ms_tmp[c_new,t:d] +=1
            if ms_tmp[c_new,t] > 0:
                p_crp[i] = log(ms_tmp[c_new,t])
            for tau in range(t+1,d):
                if ms_tmp[c_new,tau] > 0:
                    p_crp[i] += log(ms_tmp[c_new,tau])
        Z = 0.
        for tau in range(t+1,d):
            if ms[state.c[tau],tau] > 0:
                Z += log(ms[state.c[tau],tau])
        return (possible,exp(p_crp - logsumexp(p_crp)))
        

                








    def get_free_label(self,t):
        """Return a label that is currently "free", i.e. can be used for
        starting a new cluster."""
        return self.num_clusters + 1 # TODO

