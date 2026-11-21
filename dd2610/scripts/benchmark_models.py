"""Project baselines and an auditable extension of the authors' continuous SCFM."""
import math
import torch
from torch import nn
from torch.distributions import Normal


def mlp(n_in, n_out, width, depth, activation='softplus'):
    act = nn.Tanh if activation == 'tanh' else nn.Softplus
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(n_in, width), act()]; n_in = width
    return nn.Sequential(*layers, nn.Linear(n_in, n_out))


class FM(nn.Module):
    def __init__(self, dim, width, depth, activation='softplus'):
        super().__init__()
        self.net = mlp(dim+4, dim, width, depth, activation)

    def forward(self, t, x):
        tx = t[:, None] * x.new_tensor([math.pi, 2*math.pi])
        emb = torch.cat([tx.sin(), tx.cos()], -1)
        return self.net(torch.cat([x.flatten(1), emb], -1)).view_as(x)


class Autoencoder(nn.Module):
    def __init__(self, dim, z_dim, width, depth, variational=False, activation='softplus'):
        super().__init__()
        self.variational = variational
        self.encoder = mlp(dim, z_dim*(2 if variational else 1), width, depth, activation)
        self.decoder = mlp(z_dim, dim*(2 if variational else 1), width, depth, activation)

    def encode(self, x):
        out = self.encoder(x.flatten(1))
        if not self.variational:
            return out
        mu, log_std = out.chunk(2, -1)
        return Normal(mu, torch.nn.functional.softplus(log_std)+1e-4)

    def decode(self, z):
        out = self.decoder(z)
        if not self.variational:
            return out
        mu, log_std = out.chunk(2, -1)
        return Normal(mu, torch.nn.functional.softplus(log_std)+1e-4)

    def loss(self, x, beta=1.0):
        if not self.variational:
            return (self.decode(self.encode(x))-x.flatten(1)).square().mean()
        q = self.encode(x); z = q.rsample(); p = self.decode(z)
        nll = -p.log_prob(x.flatten(1)).sum(-1).mean()
        kl = 0.5*(q.loc.square()+q.scale.square()-1-2*q.scale.log()).sum(-1).mean()
        return (nll+beta*kl)/x[0].numel()


def bridge(x, sigma=1e-4, generator=None):
    # Author convention r=0 data, r=1 noise; use separate endpoint noise for every sample.
    t = torch.rand(len(x), device=x.device, dtype=x.dtype,generator=generator)
    shape = (-1,) + (1,)*(x.ndim-1)
    tt = t.view(shape); noise = torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=generator)
    return t, (1-tt)*x+(sigma+(1-sigma)*tt)*noise, (1-sigma)*noise-x


def scfm_loss(vt, rt, x, estimator='k1', alpha=0.0, sigma=1e-4, fixed_bridge=None):
    t, xt, u = fixed_bridge if fixed_bridge is not None else bridge(x, sigma)
    q = rt(xt, t)
    count = {'k1': 1, 'k4': 4, 'cross': 2, 'validation': 16}[estimator]
    # Separate reparameterization draws from the SAME q(z|xt,t); both branches retain gradients.
    z = [q.rsample() for _ in range(count)]
    v = [vt(t, xt, zz) for zz in z]
    if estimator == 'cross':
        loss = ((v[0]-u)*(v[1]-u)).mean()
    else:
        loss = (torch.stack(v).mean(0)-u).square().mean()
    # Same expected regularizer for every estimator, never norm(mean(z)).
    reg = torch.stack([zz.square().mean() for zz in z]).mean()
    return loss + alpha*reg
