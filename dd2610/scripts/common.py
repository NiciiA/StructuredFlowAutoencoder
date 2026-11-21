import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import numpy as np
import torch


def seed_all(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str)+'\n')
    tmp.replace(p)


def atomic_save(path, obj):
    p = Path(path); tmp = p.with_suffix(p.suffix+'.tmp'); torch.save(obj, tmp); tmp.replace(p)


def provenance():
    root = Path(__file__).resolve().parents[1]
    packages = {}
    for p in ['torch', 'torchvision', 'torchdiffeq', 'numpy', 'scipy', 'scikit-learn', 'scikit-image', 'pandas']:
        try: packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: pass
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in ['scripts', 'paper_only'] for p in sorted((root/folder).rglob('*.py'))}
    upstream = root/'upstream/StructuredFlowAutoencoder'
    author_sources = {str(p.relative_to(upstream)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(upstream.rglob('*.py'))}
    try:
        revision = subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return dict(python=sys.version, platform=platform.platform(), packages=packages, source_sha256=sources,
                author_revision=revision, author_source_sha256=author_sources,
                installed_packages={d.metadata.get('Name','unknown'):d.version for d in importlib.metadata.distributions()},
                device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                cuda=torch.version.cuda)


def preserve_manifest(path, value, keys=None):
    """Never erase training provenance on resume or when evaluating another split."""
    p = Path(path)
    if p.exists():
        old = json.loads(p.read_text())
        keys = keys or sorted(set(old)|set(value))
        changed = [k for k in keys if old.get(k)!=value.get(k)]
        if changed:
            raise ValueError(f'{p.name} changed ({changed}); use a new run and document the change.')
    else:
        write_json(p,value)


@torch.no_grad()
def heun(field, x, steps=100, reverse=True):
    """Fixed-step Heun with exactly 2*steps function evaluations."""
    dt = (-1 if reverse else 1)/steps
    for i in range(steps):
        t = 1 + i*dt if reverse else i*dt
        tt = torch.full((len(x),), t, device=x.device, dtype=x.dtype)
        a = field(tt, x)
        b = field(tt+dt, x+dt*a)
        x = x + dt*(a+b)/2
        if not torch.isfinite(x).all():
            raise FloatingPointError('ODE generated nonfinite samples; report failure, do not clip silently.')
    return x
