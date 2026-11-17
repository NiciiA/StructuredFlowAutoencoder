"""Small MLP baselines: VAE, autoencoder + latent FM, and data-space FM."""
import math

import torch
from torch import nn

from models.nn import MLP


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_features):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = MLP(input_dim, 2 * latent_dim, hidden_features, fct=nn.SiLU())
        self.decoder = MLP(latent_dim, input_dim, hidden_features[::-1], fct=nn.SiLU())

    def encode(self, x):
        return self.encoder(x.flatten(1)).chunk(2, dim=-1)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = mean + torch.exp(.5 * log_var) * torch.randn_like(mean)
        return self.decoder(z), mean, log_var

    def loss(self, x):
        reconstruction, mean, log_var = self(x)
        # Unit-variance Gaussian decoder and a standard Gaussian latent prior.
        reconstruction_loss = .5 * ((reconstruction - x.flatten(1)).square()
                                    + math.log(2 * math.pi)).sum(-1).mean()
        kl = .5 * (mean.square() + log_var.exp() - 1 - log_var).sum(-1).mean()
        return reconstruction_loss + kl, reconstruction_loss, kl


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_features):
        super().__init__()
        self.encoder = MLP(input_dim, latent_dim, hidden_features, fct=nn.SiLU())
        self.decoder = MLP(latent_dim, input_dim, hidden_features[::-1], fct=nn.SiLU())

    def forward(self, x):
        return self.decoder(self.encoder(x.flatten(1)))


class FlowMatching(nn.Module):
    def __init__(self, dim, hidden_features):
        super().__init__()
        self.dim = dim
        self.velocity = MLP(dim + 1, dim, hidden_features, fct=nn.SiLU())

    def forward(self, x, t):
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype).reshape(-1, 1)
        return self.velocity(torch.cat((x, t.expand(len(x), 1)), dim=-1))

    def loss(self, target):
        noise = torch.randn_like(target)
        t = torch.rand(len(target), 1, device=target.device, dtype=target.dtype)
        xt = (1 - t) * noise + t * target
        return (self(xt, t) - (target - noise)).square().mean()

    @torch.no_grad()
    def sample(self, n, steps=100):
        if steps < 1:
            raise ValueError('sample.steps must be positive')
        reference = next(self.parameters())
        x = torch.randn(n, self.dim, device=reference.device, dtype=reference.dtype)
        dt = 1. / steps
        # Fixed-step midpoint integration, from Gaussian noise at t=0 to data at t=1.
        for i in range(steps):
            t = i * dt
            midpoint = x + .5 * dt * self(x, t)
            x = x + dt * self(midpoint, t + .5 * dt)
        return x


class LatentFM(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_features, flow_hidden_features):
        super().__init__()
        self.autoencoder = Autoencoder(input_dim, latent_dim, hidden_features)
        self.flow = FlowMatching(latent_dim, flow_hidden_features)
