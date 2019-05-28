import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class VAE:
    def xavier_init(self, size):
        # https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/
        in_dim = size[0]
        xavier_stddev = 1. / np.sqrt(in_dim / 2.)
        out = torch.randn(*size, device=self.xpu) * xavier_stddev
        return out.detach().requires_grad_(True)

    def zero_init(self, size):
        return torch.zeros(size[0], requires_grad=True, device=self.xpu)

    def __init__(self, Dx, Dh, Dz, xpu="cpu", lr=1e-2):
        self.Dx, self.Dh, self.Dz = Dx, Dh, Dz
        self.xpu = xpu

        self.wen = self.xavier_init(size=[Dx, Dh])
        self.ben = self.zero_init(size=[Dh])

        self.wz_mu = self.xavier_init(size=[Dh, Dz])
        self.bz_mu = self.zero_init(size=[Dz])

        self.wz_logvar = self.xavier_init(size=[Dh, Dz])
        self.bz_logvar = self.zero_init(size=[Dz])

        self.wde = self.xavier_init(size=[Dz, Dh])
        self.bde = self.zero_init(size=[Dh])

        self.wx_mu = self.xavier_init(size=[Dh, Dx])
        self.bx_mu = self.zero_init(size=[Dx])

        self.wx_logvar = self.xavier_init(size=[Dh, Dx])
        self.bx_logvar = self.zero_init(size=[Dx])

        self.params = [self.wen,
                       self.ben,
                       self.wz_mu,
                       self.bz_mu,
                       self.wz_logvar,
                       self.bz_logvar,
                       self.wde,
                       self.bde,
                       self.wx_mu,
                       self.bx_mu,
                       self.wx_logvar,
                       self.bx_logvar]

        self.solver = optim.Adam(self.params, lr=lr)
        return

    def sample(self, mu, log_logvar, rho=1):
        samplesize = mu.shape
        eps = torch.randn(samplesize, device=self.xpu)
        samples = mu + torch.exp(log_logvar / 2) * rho * eps
        logp = torch.distributions.normal.Normal(
                loc=0, scale=1).log_prob(eps).sum(dim=1)
        return samples, logp

    def foward(self, X, rho_z=1):
        self.hen = torch.tanh(X @ self.wen + self.ben)
        self.z_mu = torch.tanh(self.hen @ self.wz_mu + self.bz_mu)
        self.z_logvar = torch.tanh(self.hen @ self.wz_logvar + self.bz_logvar)

        self.z, self.logpz = self.sample(self.z_mu, self.z_logvar, rho_z)

        self.hde = torch.tanh(self.z @ self.wde + self.bde)
        self.x_mu = torch.tanh(self.hde @ self.wx_mu + self.bx_mu)*10
        self.x_logvar = torch.tanh(
                self.hde @ self.wx_logvar + self.bx_logvar) * 3
        return

    def train(self, x, n=1):
        x = torch.tensor(x, device=self.xpu, dtype=torch.float32)
        for i in tqdm(range(n)):
            self.foward(x)
            recon_loss = -torch.distributions.normal.Normal(
                    loc=self.x_mu, scale=torch.exp(self.x_logvar/2)
                    ).log_prob(x).mean()
            kl_loss = torch.mean(
                    0.5 * torch.sum(
                            torch.exp(self.z_logvar)
                            + self.z_mu**2 - 1. - self.z_logvar, 1
                            )
                    )
            loss = recon_loss + kl_loss
            loss.backward()
            self.solver.step()
            for p in self.params:
                if p.grad is not None:
                    p.grad.zero_()
        return loss

    def reconstruct(self, x, rho_z=1, rho_x=1):
        x = torch.tensor(x, device=self.xpu, dtype=torch.float32)
        self.foward(x, rho_z)
        x_rec, logpx = self.sample(self.x_mu, self.x_logvar, rho=rho_x)
        return x_rec.cpu().detach().numpy(), (logpx).cpu().detach().numpy()

#%%
if __name__ == "__main__":
    xpu = "cuda"
    ae = VAE(8, 4, 2, xpu=xpu)
    X = torch.randn(200000, 8, device=xpu).mul(3).add(5).detach()

    ae.train(X, 1000)

#%%
    x_rec, p_rec = ae.reconstruct(X)
    print(np.mean(x_rec, axis=0))
    print(np.cov(x_rec, rowvar=0))
    import seaborn as sns
    plt.figure()
    sns.heatmap(np.cov(x_rec, rowvar=0))

    print("std(x_)")
    print(torch.exp(ae.x_logvar / 2).cpu().detach().numpy())
    print("mean(x_)")
    print(ae.x_mu.cpu().detach().numpy())
