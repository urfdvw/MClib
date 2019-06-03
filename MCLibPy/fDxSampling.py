import torch
import torch.optim as optim
import torch.distributions.multivariate_normal as mvn
import functions as fn
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
        sig: Dx tensor: the SD of last layer Gaussian
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

    def one_init(self, size):
        return torch.ones(size[0], requires_grad=True, device=self.xpu)

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
        else:
            self.f = f
        # control
        self.xpu = xpu
        # forward parameters
        self.wh = self.xavier_init(size=[Dz, Dh])
        self.bh = self.zero_init(size=[Dh])
        self.wx = self.xavier_init(size=[Dh, Dx])
        self.bx = self.zero_init(size=[Dx])
        self.sig = self.one_init(size=[Dx])
        self.params = [self.wh, self.bh, self.wx, self.bx, self.sig]
        # solver
        self.solver = optim.Adam(self.params, lr=lr)
        return

    def forward(self):
        """ forward mapping function
        mentioned:
            z: M * Dz tensor: LD samples
        updated:
            x: M * Dx tensor: HD samples
        """
        self.h = torch.tanh(self.z @ self.wh + self.bh) * 3
        self.fz = torch.tanh(self.h @ self.wx + self.bx) * 3
        self.x = self.fz + torch.randn(
            self.fz.shape, device=self.xpu
        ) * self.sig
        return

    def train(self):
        """ train the network
        mentioned:
            pi(x): function handle: the target distribution
                x: M * Dx tensor: HD samples
                return: M tensor: HD sample likelihoods
        updates:
        """
        # sample LD and HD samples
        self.z = torch.randn([self.M, self.Dz], device=self.xpu)
        self.forward()
        h_x = []
        for m in range(self.M):
            h_x.append(torch.mean(
                torch.exp(
                    mvn.MultivariateNormal(
                        loc=self.x[m, :],
                        covariance_matrix=torch.diag(self.sig)
                    ).log_prob(self.fz))))
        Df = torch.mean(
            self.pi(self.x) / torch.stack(h_x)
        )
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
    model = MLP(Dz, Dh, Dx, pi)
    for i in tqdm(range(1000)):
        model.train()
        x = model.sample().detach().numpy()
        plt.clf()
        plt.plot(x[:,0],x[:,1],'.')
        plt.pause(0.01)
