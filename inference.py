from utils import rdiscrete
from model import *
from numpy import *
from numpy.random import rand, random_sample
import logging

### RESAMPLING SCHEMES
def multinomial_resampling(weights):
    return (1/ones(len(weights)),
            counts_to_index(rmultinomial(weights,len(weights)))
           )

class InferenceParams(object):
    def __init__(self,rho,alpha,p_uniform_deletion):
        self.rho = rho
        self.alpha = alpha
        self.p_uniform_deletion = p_uniform_deletion

    def __str__(self):
        out = []
        out.append('Inference parameters:')
        out.append('rho: ' + str(self.rho))
        out.append('alpha: ' + str(self.alpha))
        out.append('p_uniform_deletion: ' + str(self.p_uniform_deletion))
        return '\n'.join(out)

class Inference:
    pass

class ParticleFilter(Inference):
    def __init__(self,model,data,data_time,params,num_particles,
                 resample_fun=multinomial_resampling):
        self.model = model
        self.data = data
        self.data_time = data_time
        self.num_particles = num_particles
        self.resample_fun = resample_fun
        self.params = params
        self.T = data.shape[1]
        self.particles = empty(num_particles,dtype=object)
        for i in range(num_particles):
            self.particles[i] = Particle(self.T)
        self.weights = ones(num_particles)/num_particles

    def run(self):
        for t in range(self.T):
            logging.info('t = ' + str(t) + '/' + str(self.T))
            x = self.data[:,t]
            tau = self.data_time[t]
            # move particles forward
            for n in range(self.num_particles):
                p = self.particles[n]
                # perform deletion step / compute new cluster sizes {{{
                if t > 0:
                    p.mstore.copy_list(t-1,t)
                    p.lastspike.copy_list(t-1,t)
                    m = p.mstore.get_array(t)
                    old_zero = m == 0;
                    if rand() < self.params.p_uniform_deletion: # uniform deletion
                        # TODO: Speed this up by sampling only from surviving allocations
                        U = random_sample(p.c.shape);
                        # delete from alive allocations with prob. 1-p.rho
                        idx = logical_and(logical_and(U<1-self.params.rho,p.d>=t), p.c>0)
                    else: # size-biased deletion
                        i = rdiscrete(m/float(sum(m)),1);
                        idx = logical_and(logical_and(p.c == i, p.d>=t), p.c > 0)
                
                    p.d[idx] = t
                     # compute current alive cluster sizes p.m; TODO: vectorize this?
                    for k  in range(p.K):
                        m[k] = sum(logical_and(p.c == k,p.d>t))
                
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
                Kbefore = len(active)
                 
                # current number of active clusters
                Kt = Kbefore
                
                # Generalized Poly Urn / CRP
                p_crp = hstack((m[active_idx],self.params.alpha))
                
                # Vector for storing the likelihood values
                p_lik = zeros(Kt+1);
                
                # compute probability of data point under all old clusters
                for i in range(Kbefore):
                    isi = self.data_time[t] - p.lastspike.get(t,active[i])
                    p_lik[i] = self.model.p_likelihood(x,p.U.get(t-1,active[i]))
                
                # likelihood for new cluster
                p_lik[Kt] = self.model.p_prior(x)
                
                # propsal distribution: CRP x likelihood
                q = p_crp * p_lik
                
                # normalize to get a proper distribution
                Z_qc = sum(q)
                q = q / Z_qc
                
                logging.debug('q(c|z): ' + str(q))
                # sample a new label from the discrete distribution q
                c = rdiscrete(q,1)
                
                # update data structures if we propose a new cluster
                if c == Kt:
                    logging.info('Starting new cluster')
                    # update number-of-clusters counts
                    p.K += 1
                    # set birthtime of cluster K to the current time
                    p.birthtime[Kt] = t
                    active = hstack((active,Kt))
                    p.mstore.append(t,0)
                    p.lastspike.append(t,0)
                
                # increment old cluster size
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
                # % preallocate vectors to store probabilities for the weight computation
                #qU_z = ones(Kt - Kbefore,1);
                #G0 = ones(Kt - Kbefore,1);
                pU_U = ones(Kbefore)
                qU_Uz = ones(Kbefore)
                p.U.copy_list(t-1,t)
                for i in range(len(active)):  # for all active clusters
                    cabs = active[i]
                    # find associated data points 
                    #idx = logical_and(p.c == cabs) & (data_idx == p.t);
                    # nk = sum(idx);
                    # extract data points
                    # points = data_full(:,idx);

                    if i >= Kbefore:  # cluster newly created at this time step
                        self.model.set_data(x);
                        new_params = self.model.sample_posterior()
                        p.U.append(t,new_params)

                        # compute probability of this sample for use in weight
                        qU_z = self.model.p_posterior(new_params)
                        # compute probability of this sample under G_0
                        G0 = self.model.p_prior_params(new_params)
                    else:  # old cluster
                        if cabs == c: # new data associated with cluster at this time step
                            new_params = self.model.walk(p.U.get(t,cabs))
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
                w_inc = Z_qc*G0/qU_z
                self.weights[n] *= w_inc
                # 
                # if isnan(w_inc) % bad weight -- can happen if underflow occurs
                #     w_inc = 0; % discard the particle by giving it weight 0
                # end
                # 
                # %%% }}}
                # 
                # %%%  store values of p.rho {{{
                # p.rhostore(p.t) = p.rho;
                # 
                # p.mstore(:,p.t) = m;
                # 
                # %%% }}}

            ### resample
            # normalize weights
            self.weights = self.weights / sum(self.weights)
            self.weights, resampled_indices = self.resample_fun(self.weights)
            new_particles = empty(self.num_particles,dtype=object)
            for i in range(len(resampled_indices)):
                new_particles[i] = self.particles[resampled_indices[i]].shallow_copy()
            self.particles = new_particles

class GibbsSampler(Inference):
    def __init__(self,data,model):
        pass

