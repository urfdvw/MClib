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

def TwoDlogbanana(x):
    B = 10
    ng1 = 4
    ng2 = 3.5
    ng3 = 3.5
    p = 1/(2*ng1*ng1)\
        *(4 - B*x[:, 0] - x[:, 1]*x[:, 1])\
        *(4 - B*x[:, 0] - x[:, 1]*x[:, 1])\
        +x[:, 0]*x[:, 0]/(2*ng2*ng2)\
        +x[:, 1]*x[:, 1]/(2*ng3*ng3)
    return -p

#%% fine grid
N = 100
maxx = 10
minx = -10

x = np.linspace(minx, maxx, N)
w = np.zeros([N, N])
y = np.zeros([N*N, 2])

iy = 0
for i in range(N):
    for j in range(N):
        x_now = np.zeros([1, 2])
        x_now[0, 0] = x[i]
        x_now[0, 1] = x[j]
        y[iy, :] = x_now
        iy += 1

logw = TwoDlogbanana(y)
w = fn.logw2w(logw)

print(fn.weightedsum(y, w))

#%% PMC
D = 2  # number of dimension of sampling space
N = 50  # number of particles per population
K = 20  # number of populations
M = 20  # number of iterations
mu0 = mvn.rvs(mean=np.zeros(shape=D),
              cov=np.eye(D) * 3,
              size=N)  # initial mean of each population
pmc = PMC.PMC(mu0, K, TwoDlogbanana)  # define pmc object
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
error = fn.weightedsum(x, fn.logw2w(logw)) - np.array([-0.4845, 0])
print(error)
print(np.sqrt(np.sum(error**2)))

#%% VAExPMC
Dh = 2
Dz = 2
N = 1000  # number of particles per population
M = 20  # number of AIS iterations
mu0 = mvn.rvs(mean=np.zeros(shape=D),
              cov=np.eye(D) * 3,
              size=N)  # initial mean of each population

pmc = PMCxVAE.PMC(mu0, TwoDlogbanana, Dz, Dh)  # define pmc object
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
error = fn.weightedsum(x, fn.logw2w(logw)) - np.array([-0.4845, 0])
print(error)
print(np.sqrt(np.sum(error**2)))
