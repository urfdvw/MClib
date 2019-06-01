import torch
import torch.distributions.multivariate_normal as mvn



def Df(pi, fl2h, sig, z, f=None):
    """
    input:
        pi(x): function handle: the target distribution
            x: M * Dx tensor: HD samples
            return: M tensor: HD sample likelihoods
        x = fl2h(z): function handle: the mapping function
            z: M * Dz tensor: LD samples
            x: M * Dx tensor: HD samples
        sig: tensor scaler: the SD of last layer Gaussian
        z: M * Dz tensor: LD samples
    op-input:
        f(t): function handle: f-function for f-divergence
            t: tensor scaler
            return: tensor scaler
            default: t*log(t)
    output:
        return: tensor scaler: the divergence
    """
    if not f:
        def f(t): return t * torch.log(t)

    x = fl2h(z)
    x += torch.randn(x.shape) * sig
    pi_x = pi(x)
    h_x = torch.mean()
    return


class MLP():
    def __init__(self):
        return

    def forward(self, z=None):
        if not z:
            z = self.z
        """ forward mapping function
        op-input:
            z: M * Dz tensor: LD samples
        output:
            x: M * Dx tensor: HD samples
        """
        return x

    def train(self, pi):
        """ train the network
        input:
            pi(x): function handle: the target distribution
                x: M * Dx tensor: HD samples
                return: M tensor: HD sample likelihoods
        output:
            no output, only update the object itself
        """
        return