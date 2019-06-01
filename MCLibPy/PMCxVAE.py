import numpy as np
from scipy.stats import multivariate_normal as mvn
from VAE import VAE
import functions as fn
import matplotlib.pyplot as plt


class PMC:
    # only record last particles, no record of history
    # which means the shape of particle group is not changed
    def __init__(self, mu0, logtarget, Dz, Dh):
        '''
        mu0: N*D, nparray
        K: scaler, int
        '''
        self.N, self.D = np.shape(mu0)
        self.x = mu0
        self.Dz = Dz
        self.Dh = Dh

        self.logpi = logtarget

        self.rhopi = 1.0
        self.rhoz = 1.2
        self.rhox = 1.2
        self.resample_method = 'global'

        if self.N > 10000:
            xpu = "cuda"
        else:
            xpu = "cpu"
        self.ae = VAE(self.D, self.Dh, self.Dz, xpu=xpu)
        self.ae.train(self.x, 10000)
        print('init done.\n')
        return

    def setTemp(self, rho):
        self.rhopi = rho
        return

    def sample(self):
        outx, log_prop = self.ae.reconstruct(self.x, self.rhoz, self.rhox)
        log_like = self.logpi(outx)
        outlogw = log_like - log_prop
        outlogTw = log_like*self.rhopi - log_prop
        ind = np.random.choice(a=np.arange(self.N),
                               p=fn.logw2w(outlogTw),
                               size=self.N)
        self.x = outx[ind, :]
        self.ae.train(self.x, 1000)
        return outx, outlogw


if __name__ == "__main__":
    D = 6  # number of dimension of sampling space
    Dh = 4
    Dz = 2
    N = 1000  # number of particles per population
    M = 20  # number of AIS iterations
    mu0 = mvn.rvs(mean=np.zeros(shape=D),
                  cov=np.eye(D) * 3,
                  size=N)  # initial mean of each population
    logtarget = lambda x: fn.logbanana(x, D)

    pmc = PMC(mu0, logtarget, Dz, Dh)  # define pmc object
    plt.figure()

    rhoxz_plan = np.linspace(1.2, 1, num=M)
    rho_plan = np.linspace(0.1, 1, num=M)

    x = np.zeros((0, D))
    logw = np.zeros((0))
    for i in range(M):
        pmc.rhoz = rhoxz_plan[i]
        pmc.rhox = rhoxz_plan[i]
        pmc.rhopi = rho_plan[i]
        outx, outlogw = pmc.sample()
        x = np.concatenate((x, outx))
        logw = np.concatenate((logw, outlogw))
        plt.clf()
        fn.plotsamples(outx, fn.logw2w(outlogw))
        plt.pause(0.1)
    plt.clf()
    fn.plotsamples(x, fn.logw2w(logw))
    plt.show()

