from scipy.stats import multivariate_normal as mvn
import numpy as np
import matplotlib.pyplot as plt

# globals
D = 2
N = 10000


def plot(sample, measure, pi, tau):
    """ plot the trend at different tau
    sample(tau): function handle: sampling process
        tau: scaler: sampling parameter
    measure(x, pi, w=None): function handle
        x: N*D array: samples
        pi(x): function handle: target distribution
            x: N*D array
        w: N array: sample weights
    pi(x): function handle: target distribution
        x: N*D array
    tau: vector array: all taus we want to test
    """
    log_measures = []
    for i in range(len(tau)):
        x = sample(tau[i])
        log_measures.append(np.log(measure(x, pi)))
    plt.figure()
    plt.plot(tau, log_measures)
    plt.show()
    return


def hist(sample, measure, pi, tau):
    """ plot the distribution at a specific tau
    sample(tau): function handle: sampling process
        tau: scaler: sampling parameter
    measure(x, pi, w=None): function handle
        x: N*D array: samples
        pi(x): function handle: target distribution
            x: N*D array
        w: N array: sample weights
    pi(x): function handle: target distribution
        x: N*D array
    tau: scaler: the tau we want to test
    """
    log_measures = []
    for i in range(10000):
        x = sample(tau)
        log_measures.append(np.log(measure(x, pi)))
    plt.figure()
    plt.hist(log_measures, 100)
    plt.show()
    return


# sampling experiments


def sampleshift(tau):
    """
    differnt in mu
    """
    mu = np.zeros(shape=D)
    mu[0] = tau
    x = mvn.rvs(mean=mu, cov=np.eye(D), size=N)
    return x


def samplescale(tau):
    """
    different in sigma
    """
    mu = np.zeros(shape=D)
    x = mvn.rvs(mean=mu, cov=tau * np.eye(D), size=N)
    return x


def sampleshape(tau):
    """
    different in shape of distribution
    """
    N2 = int(tau * N)
    N1 = N - N2
    mu = np.zeros(shape=D)
    x1 = mvn.rvs(mean=mu, cov=np.eye(D), size=N1)
    x2 = np.random.uniform(low=-3.0, high=3.0, size=(N2, D))
    xout = np.vstack((x1, x2))
    return xout


def samplesize(N):
    """
    different in number of samples
    """
    x = mvn.rvs(mean=np.zeros(shape=D), cov=np.eye(D), size=N)
    return x