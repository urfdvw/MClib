import torch
import torch.optim as optim
import torch.distributions.multivariate_normal as mvn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


D = 1
M = 1000


def zero_init(size):
    return torch.zeros(size, requires_grad=True)


mu = zero_init(size=D)
logsig = zero_init(size=D)
params = [mu, logsig]


def logbanana(x, D):
    p = 0
    for d in range(D-1):
        p += 2*(x[:, d+1]-x[:, d]**2)**2 + (1-x[:, d])**2
    return -p


def lognormal(x, D):
    return mvn.MultivariateNormal(
        loc=torch.ones(D)*2,
        covariance_matrix=torch.eye(D)
    ).log_prob(x)


def pi(x): return torch.exp(lognormal(x, D))

#def f(t): return t * torch.log(t)
def f(t): return -torch.log(t)

solver = optim.Adam(params, lr=1e-3)

N = 10000
for i in tqdm(range(N)):
    esp = torch.randn((M, D))
    x = mu + torch.exp(logsig) * esp
    hx = torch.exp(
            mvn.MultivariateNormal(
                loc=mu,
                covariance_matrix=torch.eye(
                    D) * torch.exp(logsig * 2)
            ).log_prob(x))
    Df = torch.mean(f(pi(x) / hx))
    Df.backward()
    solver.step()
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
print(mu)
print(torch.exp(logsig))
