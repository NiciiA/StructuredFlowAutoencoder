"""Local dtype/device handling for the runners and ODE solvers."""

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from torch.distributions import Normal
from torchdiffeq import odeint as _odeint, odeint_adjoint as _odeint_adjoint


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def floating_dtype(device, dtype=torch.float64):
    # The existing experiments use double precision on CPU and CUDA.
    return torch.float32 if torch.device(device).type == "mps" else dtype


def trainer_device_kwargs():
    device = select_device()
    if device.type == "cuda":
        return dict(accelerator="gpu", devices="auto",
                    strategy=DDPStrategy(find_unused_parameters=True))
    if device.type == "mps":
        return dict(accelerator="mps", devices=1, strategy="auto", precision="32-true")
    return dict(accelerator="cpu", devices="auto", strategy="auto")


def prepare_models(runner):
    """Retain the runners' double precision without changing torch defaults."""
    dtype = floating_dtype(runner.config.device)
    for name in ("vt", "rt", "Rt", "priorz", "prior", "embx"):
        model = getattr(runner, name, None)
        if isinstance(model, torch.nn.Module):
            model.to(dtype=dtype)


class DeviceLightningModule(pl.LightningModule):
    def setup(self, stage=None):
        # Lightning calls setup before moving the model to the accelerator.
        device = self._trainer.strategy.root_device if self._trainer else self.device
        dtype = next(self.parameters()).dtype
        self.to(device=device, dtype=floating_dtype(device, dtype))
        # Distribution parameters are not nn.Module buffers and do not follow .to().
        prior = getattr(self, "priorz", None)
        if isinstance(prior, Normal):
            self.priorz = Normal(prior.loc.to(device=device, dtype=self.dtype),
                                 prior.scale.to(device=device, dtype=self.dtype))

    def standard_normal(self, *shape):
        return Normal(torch.zeros(shape, device=self.device, dtype=self.dtype),
                      torch.ones(shape, device=self.device, dtype=self.dtype))

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Cast before transferring: numpy float64 batches cannot first move to MPS.
        def transfer(value):
            if isinstance(value, torch.Tensor):
                dtype = self.dtype if value.is_floating_point() else value.dtype
                return value.to(device=device, dtype=dtype)
            if isinstance(value, dict):
                return {key: transfer(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(transfer(item) for item in value)
            return value
        return transfer(batch)


def _ode_kwargs(y0, kwargs, adjoint=False):
    state = y0[0] if isinstance(y0, (tuple, list)) else y0
    if state.device.type == "mps":
        # Adaptive torchdiffeq solvers otherwise create float64 time tensors.
        kwargs["options"] = {**(kwargs.get("options") or {}), "dtype": torch.float32}
        if adjoint:
            kwargs["adjoint_options"] = {
                **(kwargs.get("adjoint_options") or kwargs["options"]), "dtype": torch.float32}
    return kwargs


def odeint(func, y0, t, **kwargs):
    return _odeint(func, y0, t, **_ode_kwargs(y0, kwargs))


def odeint_adjoint(func, y0, t, **kwargs):
    state = y0[0] if isinstance(y0, (tuple, list)) else y0
    if state.device.type == "mps" and torch.is_grad_enabled():
        # torchdiffeq's flattened adjoint state triggers a native MPS view
        # assertion on PyTorch 2.2. Differentiate the same ODE solve directly
        # on MPS; this uses more memory but retains the vector field, solver,
        # tolerances and objective, and never transfers computations to CPU.
        kwargs = {key: value for key, value in kwargs.items()
                  if not key.startswith("adjoint_")}
        return odeint(func, y0, t, **kwargs)
    return _odeint_adjoint(func, y0, t, **_ode_kwargs(y0, kwargs, adjoint=True))
