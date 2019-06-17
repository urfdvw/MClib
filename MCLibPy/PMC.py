import functions as fn
import numpy as np
from scipy.stats import multivariate_normal as mvn
from tqdm import tqdm
import matplotlib.pyplot as plt


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
            mu0: N*D, nparray
            K: scaler, int
        class properties:
            N: int: number of populations
            D: int: dimension of sampling space
            K: int: number of samples per-population
            mu: N*D nparray: population means
            C: D*D nparray: covariance of proposal distribution
            rho: float: tempering of target distribution
            resample_method: 'global' or 'local'
            x: N*K*D nparray: samples
            w: N*K nparray: sample weights
            logp = logtarget(x) : lambda: log target distribution
                x: M*D, nparray
                logp: M, nparray
        '''
        # passing the parameters
        self.mu = mu0
        self.N, self.D = np.shape(mu0)
        self.K = K
        self.logtarget = logtarget
        # default parameters
        self.C = np.eye(self.D)
        self.rho = 1.0
        self.resample_method = 'global'
        # initial sampling
        self.x = np.zeros(shape=(self.N, self.K, self.D))  # space allocation
        self.w = np.zeros(shape=(self.N, self.K))  # space allocation
        for n in range(self.N):
            self.x[n, :, :] = mvn.rvs(
                    mean=self.mu[n, :],
                    cov=self.C,
                    size=self.K)
            self.w[n, :] = np.ones(shape=self.K) / self.K
        return

    def setSigma(self, sig):
        '''
        set the proposal covariance by the std of each dimension
        sig: std of each dimension
        '''
        self.C = np.eye(self.D)*(sig**2)
        return

    def setRho(self, rho):
        '''
        set target distribution rempering
        '''
        self.rho = rho
        return

    def sample(self):
        '''
        main procesdure
        '''
        # log weights of samples for current population
        logw_n = np.ones([self.N, self.K])
        # log tempered weights of samples for current population
        logTw_n = np.ones([self.N, self.K])
        for n in range(self.N):  # for each population
            logprop = np.ones([self.K])  # log proposal probability
            for k in range(self.K):  # for each particle in the population
                self.x[n, k, :] = mvn.rvs(
                        mean=self.mu[n, :],
                        cov=self.C,
                        size=1)  # sampling from proposal
                logprop[k] = fn.logmean(
                        mvn.logpdf(
                                x=self.mu,
                                mean=self.x[n, k, :],
                                cov=self.C))  # DM-weights
            # weights
            logw_n[n, :] = self.logtarget(self.x[n, :, :]) - logprop
            # tempered witghts
            logTw_n[n, :] = self.logtarget(self.x[n, :, :])*self.rho - logprop
        # prepare global particles for output
        outx = np.reshape(self.x, (-1, self.D))
        outlogw = np.reshape(logw_n, (-1))
        outlogTw = np.reshape(logTw_n, (-1))
        # resampling
        if self.resample_method == 'global':
            ind = np.random.choice(
                    a=np.arange(self.N*self.K),
                    p=fn.logw2w(outlogTw),
                    size=self.N)
            self.mu = outx[ind, :]
        elif self.resample_method == 'local':
            for n in range(self.N):
                ind = np.random.choice(
                        a=np.arange(self.K),
                        p=fn.logw2w(logTw_n[n, :]))
                self.mu[n, :] = self.x[n, ind, :]
        else:
            print('wrong resample type')
        return outx, outlogw


if __name__ == "__main__":
    # example

    D = 6  # number of dimension of sampling space
    N = 50  # number of particles per population
    K = 20  # number of populations
    M = 20  # number of iterations
    mu0 = mvn.rvs(mean=np.zeros(shape=D),
                  cov=np.eye(D) * 3,
                  size=N)  # initial mean of each population
    logtarget = lambda x: fn.logbanana(x, D)
    pmc = PMC(mu0, K, logtarget)  # define pmc object
    pmc.resample_method = 'local'

    sig_plan = np.linspace(2, 0.1, num=M)
    rho_plan = np.linspace(0.1, 1, num=M)

    x = np.zeros((0, D))
    logw = np.zeros((0))
    plt.figure()
    for i in tqdm(range(M)):
        pmc.setSigma(sig_plan[i])
#        pmc.setRho(rho_plan[i])
        outx, outlogw = pmc.sample()
        x = np.concatenate((x, outx))
        logw = np.concatenate((logw, outlogw))
        plt.clf()
        fn.plotsamples(outx, fn.logw2w(outlogw))
        plt.pause(0.01)
    plt.clf()
    fn.plotsamples(x, fn.logw2w(logw))
    plt.show()