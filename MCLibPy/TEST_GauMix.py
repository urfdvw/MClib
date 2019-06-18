"""
10 D Gaussian mixture example
"""


import numpy as np
from scipy.stats import multivariate_normal as mvn
from VAE import VAE
import functions as fn
from tqdm import tqdm
import matplotlib.pyplot as plt

import PMC
import PMCxVAE

def GauMix(x):
    D = 10
    Lambda = np.eye(D) * 3
    v1 = np.ones(D) * 6
    v2 = np.ones(D) * (-5)
    v3 = np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1])
    p1 = mvn.pdf(x, mean=v1, cov=Lambda)
    p2 = mvn.pdf(x, mean=v2, cov=Lambda)
    p3 = mvn.pdf(x, mean=v3, cov=Lambda)
    return np.log((p1 + p2 + p3)/3)


v1 = np.ones(10) * 6
v2 = np.ones(10) * (-5)
v3 = np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1])
ExTheory = (v1 + v2 + v3)/3

#%% PMC
D = 10  # number of dimension of sampling space
N = 50  # number of particles per population
K = 20  # number of populations
M = 1000  # number of iterations
mu0 = mvn.rvs(mean=np.zeros(shape=D),
              cov=np.eye(D) * 3,
              size=N)  # initial mean of each population
pmc = PMC.PMC(mu0, K, GauMix)  # define pmc object
pmc.resample_method = 'local'

sig_plan = np.linspace(2, 0.1, num=M)
rho_plan = np.linspace(0.1, 1, num=M)

x = np.zeros((0, D))
logw = np.zeros((0))
plt.figure()
for i in tqdm(range(M)):
    pmc.setSigma(sig_plan[i])
    pmc.setRho(rho_plan[i])
    outx, outlogw = pmc.sample()
    x = np.concatenate((x, outx))
    logw = np.concatenate((logw, outlogw))
    plt.clf()
    fn.plotsamples(outx, fn.logw2w(outlogw))
    plt.pause(0.01)
plt.clf()
fn.plotsamples(x, fn.logw2w(logw))
plt.show()
error = fn.weightedsum(x, fn.logw2w(logw)) - ExTheory
print(error)
print(np.sqrt(np.sum(error**2)))

#%% VAExPMC
Dh = 6
Dz = 3
N = N * K  # number of particles per population
mu0 = mvn.rvs(mean=np.zeros(shape=D),
              cov=np.eye(D) * 3,
              size=N)  # initial mean of each population

pmc = PMCxVAE.PMC(mu0, GauMix, Dz, Dh)  # define pmc object
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
plt.show
error = fn.weightedsum(x, fn.logw2w(logw)) - ExTheory
print(error)
print(np.sqrt(np.sum(error**2)))
