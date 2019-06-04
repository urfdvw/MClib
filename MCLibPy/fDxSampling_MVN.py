import torch
import torch.optim as optim
import torch.distributions.multivariate_normal as mvn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class MLP():
    """
    class properties:
        # dimensions
        D: sample space dimensions
        M: int: number of samples
        # samples
        x: M * Dx tensor: HD samples
        # functions
        pi(x): fucntion handle: target pdf
            x: M * Dx tensor: HD samples
            return: M tensor: likelihoods
        f(t): function handle: f-function for f-divergence
            t: tensor scaler
            return: tensor scaler
            default: t*log(t)
        # forward mapping
        logsig: scaler: the SD of last layer Gaussian
        (and all NN parameters and nodes)
    """

    def xavier_init(self, size):
        # https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/
        in_dim = size[0]
        xavier_stddev = 1. / np.sqrt(in_dim / 2.)
        out = torch.randn(*size, device=self.xpu) * xavier_stddev
        return out.detach().requires_grad_(True)

    def zero_init(self, size):
        return torch.zeros(size, requires_grad=True, device=self.xpu)

    def __init__(
            self,
            D,
            pi,
            M=1000,
            f=None,
            lr=1e-2,
            xpu='cpu'):
        # dimensions
        self.D = D
        self.M = M
        # functions
        self.pi = pi
        if not f:
            def f(t): return t * torch.log(t)
        self.f = f
        # control
        self.xpu = xpu
        # forward parameters
        self.mu = self.zero_init(size=D)
        self.logsig = self.zero_init(size=D)
        self.params = [self.mu, self.logsig]
        # solver
        self.solver = optim.Adam(self.params, lr=lr)
        return

    def forward(self):
        """ forward mapping function
        updated:
            x
        """
        self.esp = torch.randn((self.M, self.D))
        self.x = self.mu + torch.exp(self.logsig) * self.esp
        return

    def h_x(self):
        """ The likelihood function of the computation network
        op-input:
            x: ? * Dx tensor: external data
        output:
            return: M tensor: the likelihoods
        """
        hx = []
        for m in range(self.M):
            hx.append(
                torch.exp(
                    mvn.MultivariateNormal(
                        loc=self.mu,
                        covariance_matrix=torch.eye(self.D) * torch.exp(self.logsig * 2)
                    ).log_prob(self.x[m, :])
                )
            )
        return torch.stack(hx)

    def train(self, N=1):
        """ train the network
        input:
            N: int: number of iterations
        updates:
            forward parameters
        """
        for i in range(N):
            self.forward()
            Df = torch.mean(
                self.pi(self.x) / self.h_x()
            )
            print(Df)
            Df.backward()
            self.solver.step()
            for p in self.params:
                if p.grad is not None:
                    p.grad.zero_()
        return


if __name__ == "__main__":
    def logbanana(x, D):
        p = 0
        for d in range(D-1):
            p += 2*(x[:, d+1]-x[:, d]**2)**2 + (1-x[:, d])**2
        return -p

    def lognormal(x, D):
        return mvn.MultivariateNormal(
                loc=torch.ones(D)*2,
                covariance_matrix=torch.eye(D)*2
                ).log_prob(x)

    D = 2
    def pi(x): return torch.exp(lognormal(x, D))
    def f(t): return -torch.log(t)
    model = MLP(
            D,
            pi
            )
    # train
    plt.figure()
    for i in range(1000):
        model.train()
        x = model.x.cpu().detach().numpy()
        plt.clf()
        plt.plot(x[:, 0], x[:, 1], '.')
        plt.pause(0.01)
