import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
'''
Sample related
'''


def logw2w(logw):
    '''
    convert log weightd to normalized weights

    logw: nparray vector
    '''
    logw = logw - np.max(logw)
    w = np.exp(logw)
    sumw = np.sum(w)
    w = w / sumw
    return w


def logmean(logw):
    '''
    return the log of mean of weights given the log weights

    logw: nparray vector
    return: float
    '''
    log_scale = np.max(logw)
    logw = logw - log_scale
    w = np.exp(logw)
    sumw = np.sum(w)
    w = w / sumw
    log_scale = log_scale + np.log(sumw)
    m = np.mean(w)
    return np.log(m) + log_scale


def weightedsum(x, w):
    '''
    Weighted sum of vectors

    x: M*D, nparray
    w: M, nparray

    accu: D, nparray
    '''
    accu = np.ones(np.size(x, 1))*0
    for i in range(len(w)):
        accu += w[i] * x[i, :]
    return accu


def plotsamples(x, w):
    alpha = w / np.max(w)
    color = np.array([0.8, 0.2, 0.6])
    rgba = np.zeros((len(w), 4))
    for i in range(len(w)):
        rgba[i, 0:3] = color
        rgba[i, 3] = alpha[i]
    plt.scatter(x[:, 0], x[:, 1], c=rgba, marker='o', s=100)


'''
Target Distributions
'''


def logbanana(x, D):
    p = 0
    for d in range(D-1):
        p += 2*(x[:, d+1]-x[:, d]**2)**2 + (1-x[:, d])**2
    return -p


def lognormal(x, D):
    '''
    Target distribution

    x: M*D, nparray
    logp: M, nparray
    '''
    logp = mvn.logpdf(x, mean=10*np.ones(shape=D), cov=np.eye(D))
    return logp
