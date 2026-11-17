"""Evaluate baseline and SFA checkpoints on exactly the same held-out examples."""
import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml

from compat import select_device, floating_dtype
from evaluation_baselines import evaluation_data, evaluate_baseline, evaluation_seed, save_latent_plot
from runner_baselines import BaselineLightningModule, config_dict
from utils import dict2namespace


class SFAEvaluation:
    """Adapt the existing inference methods without modifying SFA models/runners."""
    def __init__(self, module, config, mixture):
        self.module = module
        self.config = config
        self.mixture = mixture
        self.device, self.dtype = module.device, module.dtype
        self.indices = None

    def trajectory_latent(self, x):
        sequence = x.transpose(0, 1).contiguous()
        # Under no_grad, rsample has the same Gaussian law; its cloned recurrence
        # also avoids the older MPS backend view issue in the legacy sample path.
        prior = self.module.prior.rsample(len(x), x.shape[1], device=self.device)
        indices = self.indices.to(device=self.device, dtype=self.dtype).T.contiguous() if self.indices is not None else None
        if indices is not None and self.device.type == 'mps':
            # Older MPS cannot embed nonzero-offset row views of the time grid.
            indices = [step.clone() for step in indices]
        observations = sequence if self.config.flow.cnn else sequence.flatten(2)
        return self.module.rt.decode_sequence(prior, observations, indices=indices)

    def encode(self, x):
        if self.config.data.dataset == 'PENDULUM':
            return self.trajectory_latent(x).transpose(0, 1).flatten(1)
        flat = x.flatten(1)
        if self.mixture:
            logits, _ = self.module.Rt.rsample(flat, x.new_tensor(0.))
            return self.module.rt.sample(logits, flat)
        if self.config.flow.fix_z:
            return self.module.rt.sample(flat)
        return self.module.rt.decode(self.module.priorz.sample((len(x),)), flat)

    def reconstruct(self, x):
        if self.config.data.dataset == 'PENDULUM':
            z = self.trajectory_latent(x)
            indices = self.indices.to(device=self.device, dtype=self.dtype).T.contiguous() if self.indices is not None else None
            if indices is not None and self.device.type == 'mps':
                indices = [step.clone() for step in indices]
            return self.module.vt.decode_sequence(torch.randn_like(x.transpose(0, 1)), z,
                                                   indices=indices).transpose(0, 1)
        return self.module.vt.decode(torch.randn_like(x), self.encode(x))

    def generate(self, n):
        if self.config.data.dataset == 'PENDULUM':
            data = self.config.data
            z = self.module.prior.rsample(n, data.S, device=self.device)
            noise = torch.randn(data.S, n, data.channel, data.p, data.p,
                                device=self.device, dtype=self.dtype)
            return self.module.vt.decode_sequence(noise, z).transpose(0, 1)
        if self.mixture:
            logits = self.module.priorpi.sample((n,))
            z = self.module.priorz.sample(logits, (1,))
        else:
            z = self.module.priorz.sample((n,))
        data = self.config.data
        shape = (n, data.channel, data.size, data.size) if data.dataset == 'MNIST' else (n, data.size)
        noise = torch.randn(shape, device=self.device, dtype=self.dtype)
        return self.module.vt.decode(noise, z)


def load_sfa(checkpoint, config, runner_name):
    from main import _get_runner_cls
    # Runner constructors also build their usual Trainer. Its temporary paths
    # keep evaluation from touching a training run's logs or checkpoints.
    with tempfile.TemporaryDirectory() as directory:
        args = argparse.Namespace(log_path=directory)
        runner = _get_runner_cls(runner_name)(args, config)
        if runner_name == 'pinwheel_mixture':
            from runner_mixture import PinWheelMixtureLightningModule
            module = PinWheelMixtureLightningModule(runner.vt, runner.Rt, runner.rt, runner.priorz, config, args)
        elif runner_name == 'mnist_mixture':
            from runner_mixture import FlowMatchingLightningModule
            module = FlowMatchingLightningModule(runner.vt, runner.Rt, runner.rt, runner.priorz, runner.embx, config, args)
        elif runner_name == 'pendulum':
            from runner_pendulum import FlowMatchingLightningModule
            module = FlowMatchingLightningModule(runner.vt, runner.rt, runner.prior, config, args)
        else:
            from runner_continuous import LightningModule
            module = LightningModule(runner.vt, runner.rt, runner.priorz, config, args)
        module._trainer = runner.trainer
        module.setup('test')
        module._trainer = None
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        module.load_state_dict(saved['state_dict'], strict=True)
        module.eval()
    return SFAEvaluation(module, config, 'mixture' in runner_name)


def compare(checkpoints, output, seed=123, max_samples=512, n_gen=512, projections=128,
            sfa_checkpoint=None, sfa_config=None, sfa_runner=None, sfa_training_seconds=None, split=None):
    if not checkpoints and not sfa_checkpoint:
        raise ValueError('Provide at least one baseline or SFA checkpoint')
    device = select_device()
    modules = []
    for checkpoint in checkpoints:
        module = BaselineLightningModule.load_from_checkpoint(checkpoint, map_location='cpu')
        if module.kind == 'latentfm' and module.stage != 'flow':
            raise ValueError('LatentFM comparison needs the final flow-stage checkpoint')
        module.to(device=device, dtype=floating_dtype(device, next(module.parameters()).dtype)).eval()
        report = Path(checkpoint).parent / 'summary.json'
        summary = json.loads(report.read_text()) if report.exists() else {}
        modules.append((module.kind, module, checkpoint, summary))
    if sfa_checkpoint:
        if sfa_config is None or sfa_runner is None:
            raise ValueError('SFA evaluation needs --sfa-config and --sfa-runner')
        with open(sfa_config) as handle:
            config = dict2namespace(yaml.safe_load(handle))
        config.device = device
        if not hasattr(config, 'evaluation'):
            config.evaluation = argparse.Namespace(split='validation')
        if not hasattr(config.model, 'cnn'):
            config.model.cnn = config.data.dataset == 'MNIST'
        if sfa_runner == 'hvg':
            config.data.dataset = 'RNA'
            config.model.cnn = False
            # The legacy hvg YAML labels its data PinWheel and sets an unused
            # logit flag. The RNA loader actually reads processed adata.X.
            config.data.logit_transform = False
            for key, value in dict(root='data', label_key='cell_type', split_key='split_B',
                                   train_split='train', val_split='valid', test_split='ood').items():
                if not hasattr(config.data, key):
                    setattr(config.data, key, value)
        if sfa_runner == 'pendulum':
            for key, value in dict(root='data', val_root='data_test', test_root=None, logit_transform=True).items():
                if not hasattr(config.data, key):
                    setattr(config.data, key, value)
        with evaluation_seed(seed):
            adapter = load_sfa(sfa_checkpoint, config, sfa_runner)
        report = Path(sfa_checkpoint).parent / 'comparison_summary.json'
        summary = json.loads(report.read_text()) if report.exists() else {}
        if sfa_training_seconds is not None:
            summary['training_seconds'] = sfa_training_seconds
        modules.append((sfa_runner, adapter, sfa_checkpoint, summary))
    config = modules[-1][1].config if sfa_checkpoint else modules[0][1].config
    if split is not None:
        if config.data.dataset not in ('RNA', 'PENDULUM'):
            raise ValueError('--split is for explicit RNA/Pendulum held-out splits')
        config.evaluation.split = split
    def signature(cfg):
        data = cfg.data
        return (data.dataset, getattr(data, 'size', None), getattr(data, 'channel', 1),
                getattr(data, 'n_classes', None), data.logit_transform,
                getattr(data, 'S', None), getattr(data, 'Stotal', None), getattr(data, 'p', None),
                getattr(data, 'stride', None))
    if any(signature(module.config) != signature(config) for _, module, _, _ in modules):
        raise ValueError('Compare checkpoints for the same dataset, dimensions, classes and transform')
    batch = evaluation_data(config, seed, max_samples, return_indices=True)
    x, y = batch[:2]
    for _, module, _, _ in modules:
        if config.data.dataset in ('RNA', 'PENDULUM'):
            module.config.evaluation = config.evaluation
        if isinstance(module, SFAEvaluation) and len(batch) == 3:
            module.indices = batch[2]
        if config.data.dataset in ('RNA', 'PENDULUM'):
            old_manifest = config_dict(getattr(module.config.data, 'split_manifest', {}))
            shared_manifest = config_dict(config.data.split_manifest)
            if any(old_manifest[key] != shared_manifest[key] for key in old_manifest.keys() & shared_manifest.keys()):
                raise ValueError('Checkpoint split membership/source differs from shared evaluation files')
        if config.data.dataset == 'RNA':
            if getattr(module.config.data, 'gene_names_hash', config.data.gene_names_hash) != config.data.gene_names_hash:
                raise ValueError('Checkpoint RNA gene order differs from evaluation data')
            module.config.data.cell_types = config.data.cell_types
    rows = []
    for name, module, checkpoint, summary in modules:
        metrics, _ = evaluate_baseline(module, x, y, seed, n_gen, projections)
        parameters = module.module.parameters() if isinstance(module, SFAEvaluation) else module.parameters()
        row = dict(model=name, checkpoint=str(Path(checkpoint).resolve()),
                   parameters=sum(p.numel() for p in parameters),
                   training_seconds=summary.get('training_seconds'),
                   training_seed=summary.get('seed', getattr(module.config, 'seed', None)),
                   timing_scope=summary.get('timing_scope'),
                   training_device=summary.get('device'),
                   configured_training_epochs=(module.config.training.ae_epochs + module.config.training.flow_epochs
                                               if name == 'latentfm' else module.config.training.n_epochs),
                   training_samples_per_class=(module.config.data.samplesize if config.data.dataset in ('PinWheel', 'MNIST') else None),
                   validation_samples_per_class=(module.config.data.test_samplesize if config.data.dataset in ('PinWheel', 'MNIST') else None),
                   split_manifest=json.dumps(config_dict(getattr(module.config.data, 'split_manifest', {})), sort_keys=True),
                   **metrics)
        save_latent_plot(module, x, y, Path(output).parent / (Path(output).stem + f'_{len(rows)}_{name}_latent.png'), seed)
        rows.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix('.json').write_text(json.dumps(rows, indent=2) + '\n')
    for row in rows:
        print(json.dumps(row, indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='*', default=[])
    parser.add_argument('--sfa-checkpoint')
    parser.add_argument('--sfa-config', help='Original YAML used for the SFA checkpoint')
    parser.add_argument('--sfa-runner', choices=['pinwheel', 'mnist', 'pinwheel_mixture', 'mnist_mixture', 'hvg', 'pendulum'])
    parser.add_argument('--sfa-training-seconds', type=float, help='Optional measured time for an older SFA run')
    parser.add_argument('--split', choices=['validation', 'test'], help='Explicit RNA/Pendulum held-out split')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max-samples', type=int, default=512)
    parser.add_argument('--n-gen', type=int, default=512)
    parser.add_argument('--projections', type=int, default=128)
    parser.add_argument('--output', default='exp/comparison.csv')
    compare(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
