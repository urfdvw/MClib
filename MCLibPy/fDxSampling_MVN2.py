import torch
import torch.optim as optim
import torch.distributions.multivariate_normal as mvn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


Dz, Dh, Dx = 20, 10, 2
M = 1000
nIter = 3000
lr = 1e-2


def xavier_init(size):
    # https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    out = torch.randn(*size) * xavier_stddev
    return out.detach().requires_grad_(True)


def zero_init(size):
    return torch.zeros(size, requires_grad=True)


wh1 = xavier_init(size=[Dz, Dh])
bh1 = zero_init(size=[Dh])
wh2 = xavier_init(size=[Dh, Dh])
bh2 = zero_init(size=[Dh])
wh3 = xavier_init(size=[Dh, Dh])
bh3 = zero_init(size=[Dh])
wx = xavier_init(size=[Dh, Dx])
bx = zero_init(size=[Dx])
wlogsig = xavier_init(size=[Dh, Dx])
blogsig = zero_init(size=[Dx])

params = [wh1, bh1,
          wh2, bh2,
          wh3, bh3,
          wx, bx,
          wlogsig, blogsig]

def logbanana(x, D):
    p = torch.zeros(x.shape[0])
    for d in range(D-1):
        p += 2*(x[:, d+1]-x[:, d]**2)**2 + (1-x[:, d])**2
    return -p


def lognormal(x, D):
    return mvn.MultivariateNormal(
        loc=torch.ones(D)*2,
        covariance_matrix=torch.eye(D) * 0.2**2
    ).log_prob(x)


def pi(x): return torch.exp(logbanana(x, Dx))

#def f(t): return t * torch.log(t + 1e-10)
def f(t): return -torch.log(t + 1e-10)
#def f(t): return t*t - t
solver = optim.Adam(params, lr=lr)

for i in range(nIter):
    z = torch.randn((M, Dz))
    h1 = torch.sigmoid(z @ wh1 + bh1)
    h2 = torch.sigmoid(h1 @ wh2 + bh2)
    h3 = torch.tanh(h2 @ wh3 + bh3)
    fz = h3 @ wx + bx
    logsig = h3 @ wlogsig + blogsig
    x = fz + torch.randn(fz.shape) * torch.exp(logsig)
    
    hx_list = []
    for m in range(M):
        hx_list.append(torch.mean(torch.exp(
            mvn.MultivariateNormal(
                loc=x[m, :],
                covariance_matrix=torch.eye(Dx)*torch.exp(logsig[m, :]+1e-10)
            ).log_prob(fz))))
    Df = torch.mean(f(pi(x) / torch.stack(hx_list)))
    Df.backward()
    solver.step()
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    print(i,Df)
    print(torch.mean(x),torch.std(x))
    print(torch.mean(fz),torch.std(fz))
    print("")
#%%
    x_np = fz.detach().numpy()
    plt.clf()
    plt.plot(x_np[:, 0], x_np[:, 1], '.')
    plt.pause(0.01)