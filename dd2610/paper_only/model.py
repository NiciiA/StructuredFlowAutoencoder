"""Continuous-latent SFA derived only from the supplied paper.

Forward time runs from Gaussian observation noise at t=0 to data at t=1.
See PROVENANCE.md for source references and underspecified choices.
"""

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


def _mlp(input_dim, output_dim, width, depth, activation):
    """depth counts hidden layers; direct concatenation is a local choice."""
    activations = {"tanh": nn.Tanh, "softplus": nn.Softplus}
    if activation not in activations:
        raise ValueError("activation must be tanh or softplus")
    layers = []
    for _ in range(depth):
        layers.extend([nn.Linear(input_dim, width), activations[activation]()])
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class PaperSFA(nn.Module):
    """Gaussian posterior and conditional observation velocity, jointly trained.

    The constructor's optional activation selects the paper's Pinwheel tanh
    or MNIST softplus convention. Width, depth, diagonal covariance, direct
    concatenation, and the positive-scale transform are replication choices.
    No KL penalty, imposed Gaussian latent prior, or latent ODE is introduced.
    """

    def __init__(self, x_dim, z_dim, width=128, depth=3, activation="tanh"):
        super().__init__()
        if any(not isinstance(value, int) or value < 1
               for value in (x_dim, z_dim, width, depth)):
            raise ValueError("x_dim, z_dim, width, and depth must be positive integers")
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.width = width
        self.depth = depth
        self.activation = activation
        self.posterior_net = _mlp(x_dim + 1, 2 * z_dim, width, depth, activation)
        self.field_net = _mlp(x_dim + z_dim + 1, x_dim, width, depth, activation)

    def _check_x(self, x):
        if x.ndim != 2 or x.shape[1] != self.x_dim or x.shape[0] == 0:
            raise ValueError(f"x must have nonempty shape [B, {self.x_dim}]")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")

    def _time_column(self, t, x):
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        if t.shape == (x.shape[0],):
            t = t[:, None]
        if t.shape != (x.shape[0], 1):
            raise ValueError("t must be scalar, [B], or [B, 1]")
        return t

    def posterior(self, x, t):
        """Return coordinatewise Normal q_t(z | x), with shape [B, z_dim]."""
        self._check_x(x)
        params = self.posterior_net(torch.cat([x, self._time_column(t, x)], dim=-1))
        mean, raw_scale = params.chunk(2, dim=-1)
        scale = F.softplus(raw_scale) + 1e-4
        return Normal(mean, scale)

    def velocity(self, t, x, z):
        """Evaluate v_t(x | z); t=0 is noise and t=1 is data."""
        self._check_x(x)
        if z.shape != (x.shape[0], self.z_dim):
            raise ValueError(f"z must have shape [B, {self.z_dim}]")
        return self.field_net(torch.cat([x, z, self._time_column(t, x)], dim=-1))

    def loss(self, x):
        """Single-draw approximation of Eq. (6), as suggested on paper p. 6.

        x0 ~ N(0,I), t ~ U[0,1), xt=(1-t)*x0+t*x, z~q_t(z|xt).
        The exact interpolant velocity is x-x0, avoiding division by 1-t.
        One rsample() retains gradients through both posterior parameters.
        Sum observation coordinates, then average independent batch samples.
        This one-draw squared loss includes conditional-velocity variance;
        it is not an unbiased estimator of the exact nested-expectation risk.
        """
        self._check_x(x)
        t = torch.rand(x.shape[0], device=x.device, dtype=x.dtype)
        x0 = torch.randn_like(x)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x
        z = self.posterior(xt, t).rsample()
        error = self.velocity(t, xt, z) - (x - x0)
        return error.square().sum(dim=-1).mean()

    @torch.no_grad()
    def sample_given_latent(self, z, steps=100, x0=None):
        """Integrate Eq. (7) from 0 to 1 with fixed z and RK4 (local choice).

        z can be sampled from the aggregated learned posterior. This method
        does not silently assume a standard-normal latent prior.
        """
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError(f"z must have shape [B, {self.z_dim}]")
        if x0 is None:
            x = torch.randn(z.shape[0], self.x_dim, device=z.device, dtype=z.dtype)
        else:
            self._check_x(x0)
            x = x0.clone()
        dt = 1.0 / steps
        for i in range(steps):
            t = i * dt
            k1 = self.velocity(t, x, z)
            k2 = self.velocity(t + dt / 2, x + dt * k1 / 2, z)
            k3 = self.velocity(t + dt / 2, x + dt * k2 / 2, z)
            k4 = self.velocity(t + dt, x + dt * k3, z)
            x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return x
