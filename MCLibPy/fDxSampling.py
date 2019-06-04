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
        Dz, Dh, Dx: ints: dimension of layers
        M: int: number of samples
        # samples
        z: M * Dz tensor: LD samples
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
        return torch.zeros(size[0], requires_grad=True, device=self.xpu)

    def __init__(
            self,
            Dz, Dh, Dx,
            pi,
            M=1000,
            f=None,
            lr=1e-2,
            xpu='cpu'):
        # dimensions
        self.Dz, self.Dh, self.Dx = Dz, Dh, Dx
        self.M = M
        # functions
        self.pi = pi
        if not f:
            def f(t): return t * torch.log(t)
        self.f = f
        # control
        self.xpu = xpu
        # forward parameters
        self.wh = self.xavier_init(size=[Dz, Dh])
        self.bh = self.zero_init(size=[Dh])
        self.wx = self.xavier_init(size=[Dh, Dx])
        self.bx = self.zero_init(size=[Dx])
        self.logsig = self.zero_init(size=[Dx])
        self.params = [self.wh, self.bh, self.wx, self.bx, self.logsig]
        # solver
        self.solver = optim.Adam(self.params, lr=lr)
        return

    def forward(self, z=None):
        """ forward mapping function
        op-input:
            z: ? * Dx tensor: external data
        updated:
            x
        """
        if not z:
            # if no x input, use object data
            z = self.z
        self.h = torch.tanh((z @ self.wh + self.bh) * 0.5)
        self.fz = self.h @ self.wx + self.bx
        self.x = self.fz + torch.randn(
            self.fz.shape, device=self.xpu
        ) * torch.exp(self.logsig)
        return

    def h_x(self, x=None):
        """ The likelihood function of the computation network
        op-input:
            x: ? * Dx tensor: external data
        output:
            return: M tensor: the likelihoods
        """
        if not x:
            # if no x input, use object data
            x = self.x
        h_x = []
        for m in range(self.M):
            h_x.append(torch.mean(
                torch.exp(
                    mvn.MultivariateNormal(
                        loc=x[m, :],
                        covariance_matrix=torch.eye(Dx) * torch.exp(self.logsig * 2)
                    ).log_prob(self.fz))))
        return torch.stack(h_x)

    def train(self, N=1):
        """ train the network
        input:
            N: int: number of iterations
        updates:
            forward parameters
        """
        self.z = torch.randn([self.M, self.Dz], device=self.xpu)
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

    def sample(self):
        self.z = torch.randn([self.M, self.Dz], device=self.xpu)
        self.forward()
        return self.x


if __name__ == "__main__":
    def logbanana(x, D):
        p = 0
        for d in range(D-1):
            p += 2*(x[:, d+1]-x[:, d]**2)**2 + (1-x[:, d])**2
        return -p

    Dz, Dh, Dx = 2, 4, 6
    def pi(x): return torch.exp(logbanana(x, Dx))
    def f(t): return -torch.log(t)
    model = MLP(
            Dz, Dh, Dx,
            pi,
            M=1000,
            f=f,
            lr=1e-2
            )
    # plot inittial
    model.sample()
    x = model.x.cpu().detach().numpy()
    h = model.h.cpu().detach().numpy()
    z = model.z.cpu().detach().numpy()
    plt.figure()
    plt.plot(x[:, 0], x[:, 1], '.')
    plt.figure()
    plt.plot(h[:, 0], h[:, 1], '.')
    plt.figure()
    plt.plot(z[:, 0], z[:, 1], '.')
    # train
    plt.figure()
    for i in range(1000):
        model.train()
        x = model.sample().cpu().detach().numpy()
        plt.clf()
        plt.plot(x[:, 0], x[:, 1], '.')
        plt.pause(0.01)
