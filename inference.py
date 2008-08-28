from numpy import *
from numpy.random import rand, random_sample
from scipy.maxentropy.maxentutils import logsumexp
import logging
import time
from pylab import *
import sys

from utils import *
from model import *

### RESAMPLING SCHEMES
def multinomial_resampling(weights):
    """Multinomial resampling of the given weights. The counts for each class
    are simply drawn from a multinomial distribution with the given weights.
    """
    return counts_to_index(rmultinomial(weights,len(weights)))

def residual_resampling(weights):
    """Residual resampling. The counts in each bin are floor(w*N) + N' where
    N' is sampled from a multinomial with the residual weights."""
    N = weights.shape[0]
    counts = floor(weights*N)
    R = int(sum(counts))
    new_weights = (weights*N - counts)/(N-R)
    counts += rmultinomial(new_weights,N-R)
    return counts_to_index(array(counts,dtype=int32))

def stratified_resampling(weights):
    N = weights.shape[0]
    # obtain u_i drawn from U(i/N,(i+1)/N)
    us = 1./N*arange(N) + 1./N*rand(N) 
    return inverseCDF(cumsum(weights),us)

def systematic_resampling(weights):
    N = weights.shape[0]
    u = 1./N*rand(N)
    us = arange(N,dtype=double)/N+u
    return inverseCDF(cumsum(weights),us)

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
                 max_clusters = 100,
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
            self.particles[i] = Particle(self.T,None,storage_class,max_clusters)
        self.weights = ones(num_particles)/float(num_particles)
        self.effective_sample_size = zeros(self.T)
        self.filtering_entropy = zeros(self.T)
        self.current_entropy = zeros(num_particles)
        self.unique_particles = zeros(self.T,dtype=uint32)
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
                        U = random_sample(p.c.shape);
                        # delete from alive allocations with prob. 1-p.rho
                        # We assume that for non-assigned x we have c<0
                        idx = logical_and(logical_and(U<1-self.params.rho,p.d>=t), p.c>=0)
                    else: # size-biased deletion
                        i = rdiscrete(m/float(sum(m)),1)
                        idx = logical_and(logical_and(p.c == i, p.d>=t), p.c >= 0)
                
                    p.d[idx] = t
                     # compute current alive cluster sizes p.m
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
                # propsal distribution: CRP x likelihood
                q = p_crp * p_lik
                
                # normalize to get a proper distribution
                q = q / sum(q)
                self.current_entropy[n] = entropy(q)
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
                qU_z = 1
                G0 = 1
                p_ratio = 1
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
                            (new_params,p_ratio) = self.model.walk_with_data(p.U.get(t,cabs),x)
                        else: # no new data
                            new_params = self.model.walk(p.U.get(t,cabs))
                        p.U.set(t,cabs,new_params) 
                        
                # compute incremental weight for this update step
                #
                # The weight is computed from the following components:
                #   - prod(Z_qc) -- the normalizer of q(c|m,...); first line of (9)
                #   - prod(G0)   -- G0(U); num. of third line of (9)
                #   - prod(qU_z) -- q(U|{z|c=k}); denom. of third line of (9)
                #   - prod(pU_U) -- p(U|U_old); num. of second line of (9)
                #   - prod(qU_Uz)-- q(U|U_old,z); denom. of second line of (9)
                # 
                # compute probability of current data point under new parameters
                pz_U = self.model.p_likelihood(x,p.U.get(t,active_c))
                w_inc = pz_U*Z_qc*G0*p_ratio/qU_z
                self.weights[n] *= w_inc

            ### resample
            # normalize weights
            self.weights = self.weights / sum(self.weights)
            Neff = 1/sum(self.weights**2)
            self.effective_sample_size[t] = Neff
            self.filtering_entropy[t] = mean(self.current_entropy)
            self.before_resampling_callback(self,t)
            self.unique_particles[t] = self.num_particles
            # resample if Neff too small or last time step
            if (Neff < (self.num_particles / 2.)) or (t == self.T-1):
                resampled_indices = self.resample_fun(self.weights)
                self.unique_particles[t] = unique(resampled_indices).shape[0]
                # assume weights are uniform after resampling
                self.weights = 1./self.num_particles * ones(self.num_particles)
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
            end_t = time.time()
            elapsed = end_t - start_t
            remaining = elapsed * (self.T-t)
            finish_time = time.strftime("%a %H:%M:%S",
                    time.localtime(time.time()+remaining))
            print "Status: %i/%i -- %.1f => %s" % (t,self.T,elapsed,finish_time)
            sys.stdout.flush()
            logging.info("One step required " + str(elapsed) + " seconds, " +
                    str(remaining) + " secs remaining.")
    
    def get_labeling(self):
        labeling = empty((self.num_particles,self.T),dtype=int32)
        for p in range(self.num_particles):
            labeling[p,:] = self.particles[p].c
        return labeling

class GibbsSampler(Inference):
    def __init__(self,data,data_time,params,model,state=None):
        self.data = data
        self.data_time = data_time
        self.model = model
        self.params = params
        self.T = data.shape[1]
        if state != None:
            self.state = state
            if state.T != self.T:
                logging.error("State length does not match data length!")
        else:
            self.state = self.__init_state()
        self.old_lnp = self.p_log_joint()
        self.lnps = []
        self.num_accepted = 0
        self.num_rejected = 0

    def __init_state(self):
        """Initialize the state of the Gibbs sampler."""
        self.num_clusters = 1 

    def p_log_joint(self,inc_walk=True,inc_likelihood=True,inc_death=True):
        """Compute the log-joint probability of the current state."""
        state = self.state
        ms = zeros_like(self.state.mstore)
        lnp = 0
        active = set()
        for t in range(self.T):
            # construct m up to time t
            if t > 0:
                ms[:,t] = ms[:,t-1]
            ms[state.c[t],t] += 1
            dying = where(state.d == t)[0]
            for tau in dying:
                ms[state.c[tau],t] -= 1

            if inc_walk:
                for k in where(ms[:,t]>0)[0]:
                    theta = self.state.U[k,t]
                    if t > 0 and ms[k,t-1]>0:
                        # old cluster that is still alive
                        # aux | previous theta
                        old_theta = self.state.U[k,t-1]
                        aux_vars = self.state.aux_vars[t-1,k,:,:]
                        lnp += self.model.kernel.p_log_aux_vars(old_theta,aux_vars) 

                        # theta | aux
                        lnp += self.model.kernel.p_log_posterior(theta,aux_vars)
                    else:
                        # new cluster
                        # G0(theta)
                        lnp += self.model.p_log_prior_params(theta)

            # c | m
            lnp += log(self.p_crp(t,ms,active))
            active.add(state.c[t])

            # x | c, theta
            if inc_likelihood:
                c = self.state.c[t]
                theta = self.state.U[c,t]
                lnp += sum(logpnorm(self.data[:,t],theta.mu,theta.lam)) 

            # d_t
            if inc_death:
                lnp += self.p_log_deathtime(t)
       
        return lnp

    def p_log_joint_cs(self):
        return self.p_log_joint(False,False,False)

    def mh_sweep(self):
        """Do one MH sweep through the data, i.e. propose for all parameters
        once."""
        for t in range(self.T):
            self.sample_label(t)
            self.propose_death_time(t)
            self.sample_params(t)
            self.propose_auxs(t)

    def p_log_deathtime(self,t):
        """Compute the log probability of the death time of the allocation
        at time step t."""
        alive = self.state.d[t] - t - 1
        return alive*log(self.params.rho) + log(1-self.params.rho)

    def p_crp(self,t,ms,active):
        """Compute the conditional probability of the allocation at time 
        t given the table sizes m (and the spike times tau).
        """
        if t == 0:
            return 1
        state = self.state
        active = array(list(active))
        num_active = active.shape[0]
        p_crp = zeros(num_active+1)
        p_crp[-1] = self.params.alpha
        for i in range(num_active):
            c = active[i]
            if (self.data_time[t] - self.get_last_spike_time(c,t-1) 
                    < self.params.r_abs):
                p_crp[i] = 0
            else:
                p_crp[i] = ms[c,t-1]
        p_crp = normalize(p_crp)
        idx = where(active==self.state.c[t])[0]
        if len(idx) > 0:
            pos = idx[0]
        else:
            pos = num_active
        return p_crp[pos]
    
    def get_last_spike_time(self,c,t):
        """Returns the occurence time of the last spike associated with cluster c 
        before time t."""
        return self.state.lastspike[c,t]

    def propose_death_time(self,t):
        log_joint_before = self.p_log_joint(False,False)
        old_d = self.state.d[t]
        new_d = t + 1 + rgeometric(self.params.rho)
        if new_d > self.T:
            new_d = self.T
        self.state.d[t] = new_d
        log_joint_after = self.p_log_joint(False,False,False)
        A = min(1,exp(log_joint_after - log_joint_before))
        if random_sample() < A:
            # accept
            # if we extended the life of the cluster, sample new params
            logging.debug("Accepted new death time %i" % new_d)
            self.num_accepted += 1
            if new_d > self.state.deathtime[self.state.c[t]]:
                self.sample_walk(
                        self.state.c[t],
                        self.state.deathtime[self.state.c[t]],
                        new_d
                        )
            self.state.deathtime[self.state.c[t]] = max(
                    self.state.d[self.state.c == self.state.c[t]])
            self.state.mstore = self.state.reconstruct_mstore(
                        self.state.c,
                        self.state.d)
        else:
            # reject
            self.num_rejected += 1
            self.state.d[t] = old_d

    def sample_label(self,t):
        """Sample a new label for the data point at time t.
        The conditional probability of p(c_t|rest) is proportional to
        p(c_t|seating) x p(x_t|c_t)
        """
        logging.debug("Sampling new label at time %i" % t)
        state = self.state
        c_old = state.c[t]
        res =  self.log_p_label_posterior(t)
        possible, p_crp = res
        num_possible = possible.shape[0]
        p_lik = empty(num_possible+1,dtype=float64)
        for i in range(num_possible):
            p_lik[i] = self.model.p_log_likelihood(self.data[:,t],state.U[possible[i],t])
        p_lik[num_possible] = self.model.p_log_prior(self.data[:,t])
        q = p_crp + p_lik
        q = exp(q - logsumexp(q))
        # sample new label
        choice = rdiscrete(q) 
        # map choice to actual label
        if choice < num_possible:
            c = possible[choice]
            new_cluster = False
        elif choice == num_possible:
            if sum(state.c == c_old)==1:
                # singleton, keep label
                logging.debug("Keeping label for singleton %i" % c_old)
                c = c_old
                new_cluster = False
            else:
                c = self.get_free_label()
                new_cluster = True
        if c != c_old:
            logging.debug("New label t=%i: %i=>%i" % (t,c_old,c))
            state.c[t] = c
            # update mstore
            state.mstore[c_old,t:state.d[t]] -= 1
            state.mstore[c,t:state.d[t]] += 1

            # update birthtime
            if new_cluster or (t < state.birthtime[c]):
                state.birthtime[c] = t
            if state.birthtime[c_old] == t:
                assocs = where(state.c == c_old)[0]
                if assocs.shape[0] > 0:
                    state.birthtime[c_old] = assocs[0]
                else:
                    state.birthtime[c_old] = self.T
                
            # update deathtime
            if new_cluster:
                state.deathtime[c] = state.d[t]
            else:
                state.deathtime[c] = max(state.deathtime[c],state.d[t])

            # update lastspike
            self.state.reconstruct_lastspike(self.data_time)

            deaths_c_old = state.d[state.c==c_old]
            if len(deaths_c_old)==0:  
                logging.debug("Cluster %i died, recycling label" % c_old)
                # cluster died completely
                state.deathtime[c_old] = self.T
                self.add_free_label(c_old)
            else:
                state.deathtime[c_old] = max(deaths_c_old)
                logging.debug("New deathtime for %i: %i"
                        % (c_old,state.deathtime[c_old]))

            # sample parameters for new cluster
            if new_cluster:
                self.model.set_data(self.data[:,t])
                self.state.U[self.state.c[t],t] = self.model.sample_posterior()
                self.sample_walk(self.state.c[t],t+1,self.state.d[t]) 
                self.sample_walk_backwards(self.state.c[t],t,0)

    def propose_auxs(self,t):
        active = self.get_active(t) 
        for c in active:
            self.propose_aux(t,c)
    
    def propose_aux(self,t,c):
        """Propose new values for the auxiliary variables after time t."""
        # forward proposal
        params = self.state.U[c,t]
        old_aux = self.state.aux_vars[t,c,:,:]
        new_aux = self.model.kernel.sample_aux(params)
        # we can speed this up by only computing the joint for these params
        if self.state.c[t] == c:
            data = self.data[:,t]
        else:
            data = None
        if t < self.T-1 and self.state.deathtime[c]!=t+1:
            p_old = self.model.kernel.p_log_posterior(
                    self.state.U[c,t+1],
                    old_aux,
                    data)
            p_new = self.model.kernel.p_log_posterior(
                    self.state.U[c,t+1],
                    new_aux,
                    data)
            A = min(1,exp(p_new - p_old))
        else:
            A = 1.0
        if random_sample() < A:
            # accept! 
            self.num_accepted += 1
            self.state.aux_vars[t,c,:,:] = new_aux
        else:
            # reject
            self.num_rejected += 1
            


    def sample_params(self,t):
        """Sample new parameters for the clusters at time t."""
        active = where(sum(self.state.mstore,1)>0)[0] 
        for c in active:
            if self.state.deathtime[c] > t:
                self.sample_param(t,c)

    def sample_param(self,t,c):
        """Sample new parameters for cluster c at time. The cluster may be
        an old cluster or newly created. The auxiliary variables at this 
        time step have already been sampled."""
        logging.debug("New parameter for cluster %i at time %i" % (c,t))
        data = None
        if self.state.c[t] == c:
            # there is data associated with this cluster at this time step
            data = self.data[:,t]
        
        previous = zeros((self.model.dims,0))
        next = zeros((self.model.dims,0))
        
        if t > 0 and self.state.birthtime[self.state.c[t]] < t:
            previous = self.state.aux_vars[t-1,c,:,:]
        next = self.state.aux_vars[t,c,:,:]

        aux_vars = hstack((previous,next))
        self.state.U[c,t] = self.model.kernel.sample_posterior(
                aux_vars,
                data
                )


    def sample_walk(self,c,start,stop):
        """Sample new parameters from the walk for cluster c between time 
        steps start and stop. This is necessary if we extend the life of a
        cluster by sampling a new death time.
        """
        logging.debug("Sampling walk forward for %i: %i=>%i" % (c,start,stop))
        for tau in range(start,stop):
            self.state.aux_vars[tau-1,c,:,:] = self.model.kernel.sample_aux(
                    self.state.U[c,tau-1])
            self.state.U[c,tau] = self.model.kernel.sample_posterior(
                    self.state.aux_vars[tau-1,c,:,:])
        self.state.aux_vars[stop-1,c,:,:] = self.model.kernel.sample_aux(
                self.state.U[c,stop-1])


    def sample_walk_backwards(self,c,start,stop):
        """Sample backwards from walk starting at start-1 to stop (inclusive).
        """
        logging.debug("Sampling walk backwards for %i: %i=>%i" % (c,start,stop))
        for tau in reversed(range(stop,start)):
            self.state.aux_vars[tau,c,:,:] = self.model.kernel.sample_aux(
                    self.state.U[c,tau+1])
            self.state.U[c,tau] = self.model.kernel.sample_posterior(
                    self.state.aux_vars[tau,c,:,:])

    def sample_aux_vars(self,t):
        """Sample the auxiliary variables at time step t."""
        active = where(sum(self.state.mstore,1)>0)[0] 
        for c in active:
            if self.deathtime[c] > t and self.state.birthtime[c] != t:
                self.sample_aux_var(t,c)

    def sample_aux_var(self,t,c):
        """Sample the auxiliary variable(s) for cluster c at time t.
        We can assume that the cluster has existed at the previous time step.
        """
        logging.debug("Sampling aux vars for cluster %i at time %i" % (c,t)) 
        self.state.aux_vars[t,c,:,:] = self.model.kernel.sample_aux(
                self.state.U[c,t-1])

    def log_p_label_posterior(self,t):
        """Compute the conditional probability over allocation variables at
        time t."""
        state = self.state
        d = min(state.d[t],state.T) 
        possible = where(sum(state.mstore[:,t:d],1)>0)[0]
        lnp = zeros(possible.shape[0]+1)
        old_c = self.state.c[t]
        for i in range(possible.shape[0]):
            c = possible[i]
            self.state.c[t] = c
            lnp[i] = self.p_log_joint_cs()
        self.state.c[t] = self.state.free_labels.pop()
        lnp[possible.shape[0]] = self.p_log_joint_cs()
        self.state.free_labels.append(self.state.c[t])
        self.state.c[t] = old_c
        # normalize
        return (possible,lnp - logsumexp(lnp))

    def get_active(self,t):
        """Return a list of active clusters at time t."""
        return where(self.state.mstore[:,t]>0)[0]

    def add_free_label(self,label):
        self.state.free_labels.append(label)

    def get_free_label(self):
        """Return a label that is currently "free", i.e. can be used for
        starting a new cluster."""
        return self.state.free_labels.pop()
