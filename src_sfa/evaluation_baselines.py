"""Shared, bounded evaluation for baselines and existing SFA checkpoints."""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torchmetrics.functional.image import structural_similarity_index_measure
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataloader.dataloader_mnist import get_mnist, inv_transform
from dataloader.dataloader_pinwheel import get_dataset


@contextmanager
def evaluation_seed(seed):
    """Restore training RNGs after evaluation, including the MPS RNG."""
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def evaluation_data(config, seed, limit, data_root='data', return_indices=False):
    if limit < 2:
        raise ValueError('Evaluation needs at least two examples')
    with evaluation_seed(seed):
        if config.data.dataset in ('RNA', 'PENDULUM'):
            from dataloader.dataloader_baselines import extended_datasets
            datasets = extended_datasets(config)
            split = getattr(config.evaluation, 'split', 'validation')
            if split not in ('validation', 'test') or split not in datasets:
                raise ValueError(f'Held-out split {split!r} is unavailable; configure an independent test_root for Pendulum test evaluation')
            dataset = datasets[split]
            indices = np.random.permutation(len(dataset))[:limit].tolist()
            batch = next(iter(DataLoader(dataset, batch_size=len(indices), sampler=indices, num_workers=0)))
            if return_indices and config.data.dataset == 'PENDULUM':
                return batch[0], batch[1], batch[2]
            return batch[0], batch[1]
        if config.data.dataset == 'MNIST':
            _, val, _, sampler = get_mnist(config.data.n_classes, data_root,
                                            config.data.samplesize, config.data.test_samplesize,
                                            logtran=config.data.logit_transform)
            indices = list(sampler)[:limit]
            loader = DataLoader(val, batch_size=len(indices), sampler=indices, num_workers=0)
        elif config.data.dataset == 'PinWheel':
            # Generate the same train-then-validation draws as SFA without overwriting its CSVs.
            with tempfile.TemporaryDirectory() as root:
                train_dir = str(Path(root) / 'data')
                Path(train_dir).mkdir()
                Path(train_dir + '_test').mkdir()
                _, val = get_dataset(config.data.n_classes, train_dir, config.data.samplesize,
                                      config.data.test_samplesize)
            indices = np.random.permutation(len(val))[:limit].tolist()
            loader = DataLoader(val, batch_size=len(indices), sampler=indices, num_workers=0)
        else:
            raise ValueError('Evaluation currently supports MNIST and PinWheel')
        return next(iter(loader))


def observation_space(x, config):
    if config.data.dataset in ('MNIST', 'PENDULUM') and config.data.logit_transform:
        return inv_transform(x)
    return x


def save_samples(samples, config, path, reference=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = observation_space(samples.detach().cpu(), config)
    if config.data.dataset == 'MNIST':
        save_image(x, str(path), nrow=10)
    elif config.data.dataset == 'PENDULUM':
        rows, columns = min(len(x), 4), min(x.shape[1], 12)
        times = np.linspace(0, x.shape[1] - 1, columns, dtype=int)
        fig, axes = plt.subplots(rows, columns, figsize=(columns, 1.4 * rows), squeeze=False)
        for i in range(rows):
            for j, t in enumerate(times):
                axes[i, j].imshow(x[i, t, 0], cmap='gray', vmin=0, vmax=1)
                axes[i, j].set_title(f't={t}', fontsize=8)
                axes[i, j].axis('off')
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    elif config.data.dataset == 'RNA':
        fig, ax = plt.subplots(figsize=(6, 5))
        if reference is None:
            image = ax.imshow(x[:32, :64].numpy(), aspect='auto')
            ax.set(xlabel='Gene index (first 64)', ylabel='Generated cell (first 32)')
            fig.colorbar(image, ax=ax, label='Processed expression')
        else:
            real = reference.detach().cpu().flatten(1).numpy()
            pca = PCA(n_components=2, svd_solver='full').fit(real)
            real_xy, fake_xy = pca.transform(real), pca.transform(x.flatten(1).numpy())
            ax.scatter(*real_xy.T, s=8, alpha=.5, label='Held-out cells')
            ax.scatter(*fake_xy.T, s=8, alpha=.5, label='Generated cells')
            ax.set(xlabel='PC1 (reference PCA)', ylabel='PC2 (reference PCA)')
            ax.legend()
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(x[:, 0].numpy(), x[:, 1].numpy(), s=5, alpha=.6)
        ax.set_aspect('equal')
        fig.savefig(path)
        plt.close(fig)


def vendi_score(features):
    """Cosine-kernel Vendi, equivalent to vendi.score_dual(X, normalize=True).

    Use the smaller Gram matrix; nonzero eigenvalues are identical. Reference:
    https://github.com/vertaix/Vendi-Score/blob/main/vendi_score/vendi.py
    """
    x = normalize(np.asarray(features, dtype=np.float64), axis=1)
    gram = x @ x.T if len(x) <= x.shape[1] else x.T @ x
    values = np.linalg.eigvalsh(gram / len(x))
    positive = values[values > 0]
    return float(np.exp(-(positive * np.log(positive)).sum()))


def save_latent_plot(module, x, y, path, seed=123):
    if module.config.data.dataset != 'RNA':
        return
    with evaluation_seed(seed), torch.no_grad():
        z = module.encode(x.to(device=module.device, dtype=module.dtype))
    if z is None:
        return
    z = z.detach().cpu().numpy()
    if z.shape[1] == 1:
        xy = np.column_stack((z[:, 0], np.zeros(len(z))))
    else:
        xy = PCA(n_components=2, svd_solver='full').fit_transform(z)
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = y.numpy()
    for label in np.unique(labels):
        name = module.config.data.cell_types[int(label)]
        ax.scatter(*xy[labels == label].T, s=12, alpha=.7, label=name)
    ax.set(xlabel='Latent PC1', ylabel='Latent PC2')
    ax.legend(title='Cell type', fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def evaluate_baseline(module, x, y, seed=123, n_gen=512, projections=128):
    """Common observable-space metrics. Objective losses are deliberately not ranked."""
    if n_gen < 2 or projections < 1:
        raise ValueError('Use n_gen >= 2 and projections >= 1')
    config = module.config
    reference = observation_space(x, config).to(torch.float32).cpu()
    fingerprint = hashlib.sha256(x.contiguous().numpy().tobytes() + y.numpy().tobytes()).hexdigest()
    with evaluation_seed(seed), torch.no_grad():
        device_x = x.to(device=module.device, dtype=module.dtype)
        latent = module.encode(device_x)
        reconstructed = module.reconstruct(device_x)
        generated = module.generate(n_gen).detach().cpu()
    for value in (generated, latent, reconstructed):
        if value is not None and not torch.isfinite(value.reshape(-1)).all():
            raise ValueError('Evaluation produced non-finite model outputs')
    metrics = dict(dataset=config.data.dataset, evaluation_seed=seed, n_eval=len(x), n_gen=n_gen,
                   evaluation_fingerprint=fingerprint, projections=projections,
                   reconstruction_mse=None, reconstruction_ssim=None, latent_nmi=None, latent_ari=None,
                   reconstruction_rmse=None, generated_vendi=None, reference_vendi=None, latent_vendi=None,
                   gene_mean_rmse=None, gene_std_rmse=None, generated_negative_fraction=None, generated_frame_delta_rms=None,
                   reference_frame_delta_rms=None, frame_delta_wasserstein=None,
                   generated_acceleration_rms=None, reference_acceleration_rms=None,
                   acceleration_wasserstein=None, latent_recovery_rmse=None,
                   latent_recovery_note=('No coordinate alignment assumed; trajectory baselines expose global codes'
                                         if config.data.dataset == 'PENDULUM' else None),
                   evaluation_split=getattr(config.evaluation, 'split', 'validation') if hasattr(config, 'evaluation') else 'validation')
    if reconstructed is not None:
        reconstruction = observation_space(reconstructed.detach().cpu(), config).to(torch.float32)
        metrics['reconstruction_mse'] = (reconstruction - reference).square().mean().item()
        metrics['reconstruction_rmse'] = float(np.sqrt(metrics['reconstruction_mse']))
        if config.data.dataset == 'MNIST':
            metrics['reconstruction_ssim'] = structural_similarity_index_measure(
                reconstruction, reference, data_range=1.).item()
    if latent is not None and config.data.dataset != 'PENDULUM':
        n_classes = config.data.n_classes
        if len(x) < n_classes:
            raise ValueError('Use at least n_classes evaluation examples for clustering')
        # The same unsupervised KMeans protocol for every model with an encoder.
        clusters = KMeans(n_clusters=n_classes, random_state=seed, n_init=10).fit_predict(
            latent.detach().cpu().numpy())
        metrics['latent_nmi'] = float(normalized_mutual_info_score(y.numpy(), clusters))
        metrics['latent_ari'] = float(adjusted_rand_score(y.numpy(), clusters))
    if config.data.dataset == 'RNA':
        metrics['reference_vendi'] = vendi_score(reference.numpy())
        metrics['generated_vendi'] = vendi_score(generated.flatten(1).numpy())
        if latent is not None:
            metrics['latent_vendi'] = vendi_score(latent.detach().cpu().numpy())
        real_expression, fake_expression = reference, generated.to(torch.float32)
        metrics['generated_negative_fraction'] = (fake_expression < 0).float().mean().item()
        metrics['gene_mean_rmse'] = (real_expression.mean(0) - fake_expression.mean(0)).square().mean().sqrt().item()
        metrics['gene_std_rmse'] = (real_expression.std(0) - fake_expression.std(0)).square().mean().sqrt().item()
    if config.data.dataset == 'PENDULUM':
        fake_observations = observation_space(generated, config).to(torch.float32)
        for order, label in [(1, 'frame_delta'), (2, 'acceleration')]:
            real_motion = torch.diff(reference, n=order, dim=1).flatten(1).square().mean(1).sqrt()
            fake_motion = torch.diff(fake_observations, n=order, dim=1).flatten(1).square().mean(1).sqrt()
            metrics[f'reference_{label}_rms'] = real_motion.square().mean().sqrt().item()
            metrics[f'generated_{label}_rms'] = fake_motion.square().mean().sqrt().item()
            metrics[f'{label}_wasserstein'] = float(wasserstein_distance(real_motion.numpy(), fake_motion.numpy()))
    real = reference.flatten(1).numpy()
    fake = observation_space(generated, config).to(torch.float32).flatten(1).numpy()
    directions = np.random.default_rng(seed).normal(size=(real.shape[1], projections))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    real_projected, fake_projected = real @ directions, fake @ directions
    metrics['sliced_wasserstein'] = float(np.mean([
        wasserstein_distance(real_projected[:, i], fake_projected[:, i]) for i in range(projections)]))
    return metrics, generated


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    elif torch.device(device).type == 'mps':
        torch.mps.synchronize()


class TrainingTimer(pl.Callback):
    """Measure the same fit-hook interval for SFA and each baseline stage."""
    def __init__(self, report_path=None, metadata=None):
        self.report_path = report_path
        self.metadata = metadata or {}
        self.seconds = 0.

    def on_fit_start(self, trainer, pl_module):
        synchronize(pl_module.device)
        self.started = time.perf_counter()

    def on_fit_end(self, trainer, pl_module):
        synchronize(pl_module.device)
        self.seconds = time.perf_counter() - self.started
        if self.report_path is not None and trainer.is_global_zero:
            checkpoint = trainer.checkpoint_callback
            report = dict(self.metadata, training_seconds=self.seconds,
                          parameters=sum(p.numel() for p in pl_module.parameters()),
                          checkpoint=checkpoint.best_model_path if checkpoint else None,
                          timing_scope='on_fit_start to on_fit_end; includes validation and callback IO')
            Path(self.report_path).write_text(json.dumps(report, indent=2) + '\n')
