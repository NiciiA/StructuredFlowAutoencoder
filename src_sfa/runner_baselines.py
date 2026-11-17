"""Baseline runners following the project's Lightning/configuration conventions."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from callbacks import PlotLossCallback
from compat import DeviceLightningModule, floating_dtype, trainer_device_kwargs
from dataloader.dataloader_mnist import get_mnist
from dataloader.dataloader_pinwheel import get_dataset
from models.baselines import VAE, LatentFM, FlowMatching
from utils import dict2namespace


def config_dict(value):
    if hasattr(value, '__dict__'):
        return {key: config_dict(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: config_dict(item) for key, item in value.items()}
    if isinstance(value, torch.device):
        return str(value)
    return value


def baseline_dataloaders(config, data_root='data'):
    """Use SFA's dataset definitions, per-class counts, transforms and samplers."""
    data = config.data
    if data.dataset in ('RNA', 'PENDULUM'):
        from dataloader.dataloader_baselines import extended_dataloaders
        return extended_dataloaders(config)
    Path(data_root).mkdir(parents=True, exist_ok=True)
    Path(data_root + '_test').mkdir(parents=True, exist_ok=True)
    if data.dataset == 'MNIST':
        train, val, train_sampler, val_sampler = get_mnist(
            data.n_classes, data_root, data.samplesize, data.test_samplesize,
            logtran=data.logit_transform)
        train_loader = DataLoader(train, batch_size=config.training.batch_size,
                                  sampler=train_sampler, num_workers=data.num_workers)
        val_loader = DataLoader(val, batch_size=config.training.batch_size,
                                sampler=val_sampler, num_workers=data.num_workers, drop_last=True)
    elif data.dataset == 'PinWheel':
        train, val = get_dataset(data.n_classes, data_root, data.samplesize, data.test_samplesize)
        train_loader = DataLoader(train, batch_size=config.training.batch_size, shuffle=True,
                                  num_workers=data.num_workers)
        val_loader = DataLoader(val, batch_size=config.training.batch_size, shuffle=True,
                                num_workers=data.num_workers, drop_last=True)
    else:
        raise ValueError('Baselines support PinWheel, MNIST, RNA and PENDULUM')
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError('Dataset counts must provide at least one train/validation batch')
    return train_loader, val_loader


class BaselineLightningModule(DeviceLightningModule):
    def __init__(self, config, kind, stage=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = dict2namespace(config)
        self.kind = kind
        self.stage = stage or ('autoencoder' if kind == 'latentfm' else kind)
        if kind == 'latentfm' and self.stage not in ('autoencoder', 'flow'):
            raise ValueError('LatentFM stage must be autoencoder or flow')
        data, model = self.config.data, self.config.model
        if data.dataset == 'PENDULUM':
            input_dim = data.S * data.channel * data.p ** 2
        else:
            input_dim = data.size ** 2 * data.channel if data.dataset == 'MNIST' else data.size
        if kind == 'vae':
            self.model = VAE(input_dim, model.latent_dim, model.hidden_features)
        elif kind == 'latentfm':
            self.model = LatentFM(input_dim, model.latent_dim, model.hidden_features,
                                  model.flow_hidden_features)
            self.model.autoencoder.requires_grad_(self.stage == 'autoencoder')
            self.model.flow.requires_grad_(self.stage == 'flow')
        elif kind == 'fm':
            self.model = FlowMatching(input_dim, model.flow_hidden_features)
        else:
            raise ValueError('Baseline kind must be vae, latentfm or fm')
        self.to(dtype=floating_dtype(self.config.device))

    def forward(self, x):
        if self.kind == 'vae':
            return self.model(x)
        if self.kind == 'latentfm':
            return self.model.autoencoder(x)
        raise ValueError('Unconditional FM has no reconstruction forward pass; use generate()')

    def compute_loss(self, x):
        if self.kind == 'vae':
            loss, reconstruction, kl = self.model.loss(x)
            return loss, {'reconstruction_loss': reconstruction, 'kl_loss': kl}
        if self.kind == 'latentfm':
            if self.stage == 'autoencoder':
                loss = (self.model.autoencoder(x) - x.flatten(1)).square().sum(-1).mean()
            else:
                with torch.no_grad():
                    z = self.model.autoencoder.encoder(x.flatten(1))
                loss = self.model.flow.loss(z)
        else:
            loss = self.model.loss(x.flatten(1))
        return loss, {}

    def training_step(self, batch, batch_idx):
        loss, components = self.compute_loss(batch[0])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for name, value in components.items():
            self.log('train_' + name, value, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, components = self.compute_loss(batch[0])
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        for name, value in components.items():
            self.log('val_' + name, value, on_epoch=True, sync_dist=True)
        if batch_idx == 0:
            self.validation_examples = batch[0][:64].detach()
        return loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or not self.trainer.is_global_zero:
            return
        if self.current_epoch % self.config.training.snapshot_freq != 0:
            return
        # Keep diagnostic sampling from consuming the training RNG stream.
        from evaluation_baselines import evaluation_seed, save_samples
        with evaluation_seed(self.config.seed):
            samples = self.generate(min(self.config.sample.n_gen, 100))
            directory = Path(self.trainer.default_root_dir) / 'samples'
            save_samples(samples, self.config, directory / f'{self.stage}_{self.current_epoch:03d}.png')

    def configure_optimizers(self):
        return torch.optim.AdamW((p for p in self.parameters() if p.requires_grad),
                                 lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)

    def encode(self, x):
        if self.kind == 'vae':
            return self.model.encode(x)[0]
        if self.kind == 'latentfm':
            return self.model.autoencoder.encoder(x.flatten(1))
        return None

    def reconstruct(self, x):
        z = self.encode(x)
        if z is None:
            return None
        decoder = self.model.decoder if self.kind == 'vae' else self.model.autoencoder.decoder
        return decoder(z).reshape_as(x)

    @torch.no_grad()
    def generate(self, n):
        if self.kind == 'vae':
            z = torch.randn(n, self.config.model.latent_dim, device=self.device, dtype=self.dtype)
            x = self.model.decoder(z)
        elif self.kind == 'latentfm':
            if self.stage != 'flow':
                # AE-stage snapshots show reconstructions; its latent prior is not fitted yet.
                return self.model.autoencoder(self.validation_examples).reshape_as(self.validation_examples)
            z = self.model.flow.sample(n, self.config.sample.steps)
            x = self.model.autoencoder.decoder(z)
        else:
            x = self.model.sample(n, self.config.sample.steps)
        if self.config.data.dataset == 'PENDULUM':
            data = self.config.data
            return x.reshape(n, data.S, data.channel, data.p, data.p)
        if self.config.data.dataset == 'MNIST':
            return x.reshape(n, self.config.data.channel, self.config.data.size, self.config.data.size)
        return x


class BaselineRunner:
    kind = None

    def __init__(self, args, config):
        self.args = args
        self.config = copy.deepcopy(config)
        self.config.seed = args.seed
        if self.config.data.dataset in ('RNA', 'PENDULUM'):
            split = getattr(args, 'evaluation_split', None)
            if split is not None or getattr(args, 'test', False):
                self.config.evaluation.split = split or 'test'
        expected_dataset = {'mnist': 'MNIST', 'pinwheel': 'PinWheel', 'hvg': 'RNA',
                            'pendulum': 'PENDULUM'}[args.runner.split('_')[0]]
        if self.config.data.dataset != expected_dataset:
            raise ValueError('Runner name and configuration dataset must match')
        self.log_path = Path(args.log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        if getattr(args, 'smoke_test', False):
            if self.config.data.dataset in ('PinWheel', 'MNIST'):
                self.config.data.samplesize = 8
                self.config.data.test_samplesize = 4
            self.config.training.batch_size = 16
            self.config.training.n_epochs = 1
            self.config.training.ae_epochs = 1
            self.config.training.flow_epochs = 1
            self.config.sample.n_gen = 16
            self.config.sample.steps = 4
        self.config_dict = config_dict(self.config)
        self.phase_seconds = {}

    def fit_stage(self, module, stage, epochs, train_loader, val_loader, resume=None):
        directory = self.log_path / 'autoencoder' if stage == 'autoencoder' else self.log_path
        from evaluation_baselines import TrainingTimer
        timer = TrainingTimer()
        checkpoint = ModelCheckpoint(dirpath=str(directory), filename='best-{epoch:03d}-{val_loss:.4f}',
                                     monitor='val_loss', mode='min', save_top_k=1, save_last=True)
        kwargs = dict(max_epochs=epochs, default_root_dir=str(directory),
                      callbacks=[checkpoint, PlotLossCallback(str(directory / 'loss.png')), timer],
                      logger=CSVLogger(str(directory), name='metrics'),
                      **trainer_device_kwargs())
        if getattr(self.args, 'smoke_test', False):
            kwargs.update(limit_train_batches=2, limit_val_batches=1, num_sanity_val_steps=0,
                          enable_progress_bar=False, log_every_n_steps=1)
        trainer = pl.Trainer(**kwargs)
        trainer.fit(module, train_loader, val_loader, ckpt_path=resume)
        self.phase_seconds[stage] = timer.seconds
        return checkpoint.best_model_path, trainer

    def train(self):
        data_root = 'data'
        if getattr(self.args, 'smoke_test', False) and self.config.data.dataset == 'PinWheel':
            data_root = str(self.log_path / 'smoke_data')
        train_loader, val_loader = baseline_dataloaders(self.config, data_root)
        self.config_dict = config_dict(self.config)
        final_last = self.log_path / 'last.ckpt'
        resume = bool(self.args.resume_training)
        if resume and not final_last.exists() and not (self.log_path / 'autoencoder' / 'last.ckpt').exists():
            raise FileNotFoundError('No last.ckpt found for --resume_training')
        if self.kind == 'latentfm' and not (resume and final_last.exists()):
            ae_last = self.log_path / 'autoencoder' / 'last.ckpt'
            module = BaselineLightningModule(self.config_dict, self.kind, stage='autoencoder')
            best_ae, _ = self.fit_stage(module, 'autoencoder', self.config.training.ae_epochs,
                                       train_loader, val_loader,
                                       str(ae_last) if resume and ae_last.exists() else None)
            # Reload the best AE, then freeze it and use a separate optimizer for the flow.
            module = BaselineLightningModule.load_from_checkpoint(best_ae, map_location='cpu')
            module.stage = 'flow'
            module.hparams.stage = 'flow'
            module.model.autoencoder.requires_grad_(False)
            module.model.flow.requires_grad_(True)
        else:
            stage = 'flow' if self.kind == 'latentfm' else self.kind
            module = BaselineLightningModule(self.config_dict, self.kind, stage=stage)
        epochs = self.config.training.flow_epochs if self.kind == 'latentfm' else self.config.training.n_epochs
        best, trainer = self.fit_stage(module, module.stage, epochs, train_loader, val_loader,
                                       str(final_last) if resume and final_last.exists() else None)
        if trainer.is_global_zero:
            previous_seconds = {}
            summary_file = self.log_path / 'summary.json'
            if resume and summary_file.exists():
                previous_seconds = json.loads(summary_file.read_text()).get('phase_seconds', {})
            times = dict(previous_seconds)
            for stage, seconds in self.phase_seconds.items():
                times[stage] = times.get(stage, 0.) + seconds
            summary = dict(kind=self.kind, dataset=self.config.data.dataset, seed=self.args.seed,
                           config=self.config_dict, checkpoint=str(Path(best).resolve()),
                           parameters=sum(p.numel() for p in module.parameters()),
                           phase_seconds=times, training_seconds=sum(times.values()),
                           timing_scope='on_fit_start to on_fit_end; includes validation and callback IO',
                           device=str(self.config.device), smoke_test=getattr(self.args, 'smoke_test', False))
            summary_file.write_text(json.dumps(summary, indent=2) + '\n')
            self.sample(best)

    def sample(self, checkpoint=None):
        from evaluation_baselines import evaluate_baseline, evaluation_data, save_samples, save_latent_plot
        checkpoint = checkpoint or getattr(self.args, 'checkpoint', None)
        if checkpoint is None:
            summary = json.loads((self.log_path / 'summary.json').read_text())
            checkpoint = summary['checkpoint']
        module = BaselineLightningModule.load_from_checkpoint(checkpoint, map_location='cpu')
        if module.kind != self.kind or module.config.data.dataset != self.config.data.dataset:
            raise ValueError('Checkpoint kind/dataset does not match the requested runner')
        if module.kind == 'latentfm' and module.stage != 'flow':
            raise ValueError('Use the final flow-stage checkpoint to generate LatentFM samples')
        device = self.config.device
        module.to(device=device, dtype=floating_dtype(device, next(module.parameters()).dtype)).eval()
        limit = 32 if getattr(self.args, 'smoke_test', False) else self.config.evaluation.n_samples
        # Use checkpoint preprocessing/schema and explicit split; keep the current data paths.
        evaluation_config = copy.deepcopy(module.config)
        for name in ('root', 'val_root', 'test_root'):
            if hasattr(self.config.data, name):
                setattr(evaluation_config.data, name, getattr(self.config.data, name))
        evaluation_config.evaluation = self.config.evaluation
        module.config.evaluation = self.config.evaluation
        x, y = evaluation_data(evaluation_config, self.args.seed, limit)
        metrics, samples = evaluate_baseline(module, x, y, self.args.seed,
                                             self.config.sample.n_gen,
                                             self.config.evaluation.projections)
        save_samples(samples, self.config, self.log_path / 'samples' / 'generated.png', reference=x)
        np.savez(self.log_path / 'samples' / 'generated.npz', samples=samples.cpu().numpy())
        save_latent_plot(module, x, y, self.log_path / 'samples' / 'latent.png', self.args.seed)
        (self.log_path / 'evaluation.json').write_text(json.dumps(metrics, indent=2) + '\n')
        print(json.dumps(metrics, indent=2))


class VAERunner(BaselineRunner):
    kind = 'vae'


class LatentFMRunner(BaselineRunner):
    kind = 'latentfm'


class FMRunner(BaselineRunner):
    kind = 'fm'
