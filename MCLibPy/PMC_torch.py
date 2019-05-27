from scipy.stats import multivariate_normal as mvn
import torch.distributions.multivariate_normal as mvnt
import torch


def logw2w(logw):
    '''
    convert log weightd to normalized weights

    logw: tensor vector
    '''
    logw = logw - torch.max(logw)
    w = torch.exp(logw)
    sumw = torch.sum(w)
    w = w / sumw
    return w


def logmean(logw):
    '''
    return the log of mean of weights given the log weights

    logw: tensor vector
    return: float
    '''
    log_scale = torch.max(logw)
    logw = logw - log_scale
    w = torch.exp(logw)
    sumw = torch.sum(w)
    w = w / sumw
    log_scale = log_scale + torch.log(sumw)
    m = torch.mean(w)
    return torch.log(m) + log_scale

def weightedsum(x,w):
    '''
    Weighted sum of vectors
    
    x: M*D, tensor
    w: M, tensor
    
    accu: D, tensor
    '''
    accu = torch.zeros(x.shape[1])
    for i in range(len(w)):
        accu += w[i]*x[i,:]
    return accu

class PMC:
    '''
    Population Monte Carlo class
    
    The class only record particles from the last sampling step.
    History particles are not recorded inside the class.
    This means the shape of particle group is not changed.
    To get samples from multiple sampling steps,
    recording is needed outside the class.
    '''
    
    def __init__(self, mu0, K, logtarget):
        '''
        construction funciton
        
        inputs:
            mu0: N*D, tensor
            K: scaler, int
        class properties:
            N: int: number of populations
            D: int: dimension of sampling space
            K: int: number of samples per-population
            mu: N*D tensor: population means
            C: D*D tensor: covariance of proposal distribution
            rho: float: tempering of target distribution
            resample_method: 'global' or 'local'
            x: N*K*D tensor: samples
            w: N*K tensor: sample weights
            
            logp = logtarget(x) : lambda: log target distribution
                x: M*D, tensor
                logp: M, tensor
                
        '''
        # passing the parameters
        self.mu = mu0
        self.N, self.D = mu0.shape
        self.K = K
        self.logtarget = logtarget
        # default parameters
        self.C = torch.eye(self.D)
        self.rho = 1.0
        self.resample_method = 'global'
        # initial sampling
        self.x = torch.zeros(size=(self.N,self.K,self.D)) # space allocation
        self.w = torch.zeros(size=(self.N,self.K)) # space allocation
        for n in range(self.N):
            self.x[n,:,:] = mvnt.MultivariateNormal(
                    loc=self.mu[n,:],
                    covariance_matrix=self.C
                    ).sample(sample_shape=torch.Size([self.K]))
            self.w[n,:] = torch.ones(self.K)/self.K
        return
    def setSigma(self, sig):
        '''
        set the proposal covariance by the std of each dimension
        
        sig: std of each dimension
        '''
        self.C = torch.eye(self.D)*(sig**2)
        return
    def setTemp(self, rho):
        '''
        set target distribution rempering
        '''
        self.rho = rho
        return
    def sample(self):
        '''
        main procesdure
        '''
        logw_n = torch.ones([self.N,self.K]) # log weights of samples for current population
        logTw_n = torch.ones([self.N,self.K]) # log tempered weights of samples for current population
        for n in range(self.N): # for each population
            logprop = torch.ones([self.K]) # log proposal probability
            for k in range(self.K): # for each particle in the population
                self.x[n,k,:] = mvnt.MultivariateNormal(
                        loc=self.mu[n,:],
                        covariance_matrix=self.C
                        ).sample()  # sampling from proposal
                logprop[k] = logmean(
                        mvnt.MultivariateNormal(
                                loc=self.x[n,k,:],
                                covariance_matrix=self.C
                                ).log_prob(self.mu)
                        )  # DM-weights
            logw_n[n,:] = self.logtarget(self.x[n,:,:]) - logprop # weights
            logTw_n[n,:] = self.logtarget(self.x[n,:,:])**self.rho - logprop # tempered witghts
        # prepare global particles for output
        outx = torch.reshape(self.x,(-1,self.D))
        outlogw = torch.flatten(logw_n)
        outlogTw = torch.flatten(logTw_n)
        # resampling
        if self.resample_method == 'global':
            ind = torch.multinomial(logw2w(outlogTw), self.N, replacement=True)
            self.mu = outx[ind,:]
        elif self.resample_method == 'local':
            for n in range(self.N):
                ind = torch.multinomial(logw2w(logTw_n[n,:]), 1, replacement=True)
                self.mu[n,:] = self.x[n,ind,:]
        else:
            print('wrong resample type')
        return outx, outlogw

if __name__=="__main__":
    # example
    def lognormal(x,D):
        '''
        Target distribution
        
        x: M*D, tensor
        logp: M, tensor
        '''
        logp = mvnt.MultivariateNormal(
                loc=10*torch.ones(D),
                covariance_matrix=torch.eye(D)
                ).log_prob(x)
        return logp
    
    D = 10 # number of dimension of sampling space
    N = 300 # number of particles per population
    K = 90 # number of populations
    mu0 = mvnt.MultivariateNormal(
        loc=torch.zeros(D),
        covariance_matrix=torch.eye(D) * 3
        ).sample(sample_shape=torch.Size([N])) # initial mean of each population
    logtarget = lambda x: lognormal(x,D)
    pmc = PMC(mu0,K,logtarget) # define pmc object
    pmc.resample_method = 'local'
    for i in range(20):    
        outx, outlogw = pmc.sample()
        print(weightedsum(outx, logw2w(outlogw)))
        
# The conclusion, pure torch code is much slower (3 times) than numpy code.
        