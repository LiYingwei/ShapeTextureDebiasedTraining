import torch
import torch.nn as nn
import numpy as np


class _ModeNormalization(nn.Module):
    def __init__(self, dim, n_components, eps):
        super(_ModeNormalization, self).__init__()
        self.eps = eps
        self.dim = dim
        self.n_components = n_components

        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.phi = lambda x: x.mean(3).mean(2)


class ModeNorm(_ModeNormalization):
    """
    An implementation of mode normalization. Input samples x are allocated into individual modes (their number is controlled by n_components) by a gating network; samples belonging together are jointly normalized and then passed back to the network.
    args:
        dim:                int
        momentum:           float
        n_components:       int
        eps:                float
    """
    def __init__(self, dim, momentum, n_components, eps=1.e-5):
        super(ModeNorm, self).__init__(dim, n_components, eps)

        self.momentum = momentum

        self.x_ra = torch.zeros(n_components, 1, dim, 1, 1).cuda()
        self.x2_ra = torch.zeros(n_components, 1, dim, 1, 1).cuda()

        self.W = torch.nn.Linear(dim, n_components)
        self.W.weight.data = torch.ones(n_components, dim) / n_components + .01 * torch.randn(n_components, dim)
        self.softmax = torch.nn.Softmax(dim=1)

        self.weighted_mean = lambda w, x, n: (w * x).mean(3, keepdim=True).mean(2, keepdim=True).sum(0, keepdim=True) / n


    def forward(self, x):
        g = self._g(x)
        n_k = torch.sum(g, dim=1).squeeze()

        if self.training:
            self._update_running_means(g.detach(), x.detach())

        x_split = torch.zeros(x.size()).cuda().to(x.device)

        for k in range(self.n_components):
            if self.training:
                mu_k = self.weighted_mean(g[k], x, n_k[k])
                var_k = self.weighted_mean(g[k], (x - mu_k)**2, n_k[k])
            else:
                mu_k, var_k = self._mu_var(k)
                mu_k = mu_k.to(x.device)
                var_k = var_k.to(x.device)

            x_split += g[k] * ((x - mu_k) / torch.sqrt(var_k + self.eps))

        x = self.alpha * x_split + self.beta

        return x


    def _g(self, x):
        """
        Image inputs are first flattened along their height and width dimensions by phi(x), then mode memberships are determined via a linear transformation, followed by a softmax activation. The gates are returned with size (k, n, c, 1, 1).
        args:
            x:          torch.Tensor
        returns:
            g:          torch.Tensor
        """
        g = self.softmax(self.W(self.phi(x))).transpose(0, 1)[:, :, None, None, None]
        return g


    def _mu_var(self, k):
        """
        At test time, this function is used to compute the k'th mean and variance from weighted running averages of x and x^2.
        args:
            k:              int
        returns:
            mu, var:        torch.Tensor, torch.Tensor
        """
        mu = self.x_ra[k]
        var = self.x2_ra[k] - (self.x_ra[k] ** 2)
        return mu, var


    def _update_running_means(self, g, x):
        """
        Updates weighted running averages. These are kept and used to compute estimators at test time.
        args:
            g:              torch.Tensor
            x:              torch.Tensor
        """
        n_k = torch.sum(g, dim=1).squeeze()

        for k in range(self.n_components):
            x_new = self.weighted_mean(g[k], x, n_k[k])
            x2_new = self.weighted_mean(g[k], x**2, n_k[k])

            # ensure that tensors are on the right devices
            self.x_ra = self.x_ra.to(x_new.device)
            self.x2_ra = self.x2_ra.to(x2_new.device)
            self.x_ra[k] = self.momentum * x_new + (1-self.momentum) * self.x_ra[k]
            self.x2_ra[k] = self.momentum * x2_new + (1-self.momentum) * self.x2_ra[k]

