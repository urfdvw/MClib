import torch
import torch.optim as optim
import torch.distributions.multivariate_normal as mvn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import functions as fn

D = 6
M = 10000
nIter = 1000
lr = 1e-1

def xavier_init(size):
    # https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    out = torch.randn(*size) * xavier_stddev
    return out.detach().requires_grad_(True)


def zero_init(size):
    return torch.zeros(size, requires_grad=True)


wh1 = xavier_init(size=[D, D])
bh1 = zero_init(size=[D])
wh2 = xavier_init(size=[D, D])
bh2 = zero_init(size=[D])
wh3 = xavier_init(size=[D, D])
bh3 = zero_init(size=[D])
wh4 = xavier_init(size=[D, D])
bh4 = zero_init(size=[D])
wh5 = xavier_init(size=[D, D])
bh5 = zero_init(size=[D])
wh6 = xavier_init(size=[D, D])
bh6 = zero_init(size=[D])
wh7 = xavier_init(size=[D, D])
bh7 = zero_init(size=[D])
wx = xavier_init(size=[D, D])
bx = zero_init(size=[D])

params = [wh1, bh1,
          wh2, bh2,
          wh3, bh3,
          wh4, bh4,
          wh5, bh5,
          wh6, bh6,
          wh7, bh7,
          wx, bx]


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


def pi(x): return torch.exp(logbanana(x, D))

#def f(t): return t * torch.log(t + 1e-10)
def f(t): return -torch.log(t + 1e-10)
solver = optim.Adam(params, lr=lr)

for i in range(nIter):
    z = torch.randn((M, D), requires_grad=True)
    h1 = torch.sigmoid(z @ wh1 + bh1)
    h2 = torch.tanh(h1 @ wh2 + bh2)
    h3 = torch.tanh(h2 @ wh3 + bh3)
    h4 = torch.tanh(h3 @ wh4 + bh4)
    h5 = torch.tanh(h4 @ wh5 + bh5)
    h6 = torch.tanh(h5 @ wh6 + bh6)
    h7 = torch.tanh(h6 @ wh7 + bh7)
    x = h7 @ wx + bx
    
    p_z = torch.exp(mvn.MultivariateNormal(
            loc=torch.zeros(D),
            covariance_matrix=torch.eye(D)
            ).log_prob(z))
    
    def jacobian(inputs, outputs):
        # https://discuss.pytorch.org/t/calculating-jacobian-in-a-differentiable-way/13275/4
        return torch.stack([
                torch.autograd.grad(
                    [outputs[:, i].sum()],
                    [inputs],
                    retain_graph=True,
                    create_graph=True)[0]
                for i in range(outputs.size(1))
                ],dim=-1)
    def jacobian_det(inputs, outputs):
        J = jacobian(inputs, outputs)
        return torch.stack(
                [ torch.det(J[i, :, :]) for i in range(outputs.size(0))])
        
    q_x = p_z.detach().clone()\
        / torch.abs(jacobian_det(z, h1))\
        / torch.abs(jacobian_det(h1, h2))\
        / torch.abs(jacobian_det(h2, h3))\
        / torch.abs(jacobian_det(h3, h4))\
        / torch.abs(jacobian_det(h4, h5))\
        / torch.abs(jacobian_det(h5, h6))\
        / torch.abs(jacobian_det(h6, h7))\
        / torch.abs(jacobian_det(h7, x))
        
    Df = torch.mean(f(pi(x)/q_x))
    
    Df.backward()
    solver.step()
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    print(Df)
    x_np = x.detach().numpy()
    plt.clf()
    plt.plot(x_np[:, 0], x_np[:, 1], '.')
    plt.pause(0.001)
    