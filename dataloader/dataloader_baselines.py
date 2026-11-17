"""Thin baseline entry points for the existing RNA and trajectory datasets.

No dummy-data fallback, new normalization, or window-level split is introduced.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset


def _digest(values):
    return hashlib.sha256('\n'.join(map(str, values)).encode()).hexdigest()


def _record_manifest(data, manifest):
    previous = getattr(data, 'split_manifest', {})
    previous = json.loads(json.dumps(previous, default=vars))
    for split in previous.keys() & manifest.keys():
        if previous[split] != manifest[split]:
            raise ValueError(f'{split} membership/source differs from the training checkpoint')
    data.split_manifest = manifest


def rna_datasets(config):
    data = config.data
    path = Path(data.root) / 'kang_normalized_hvg.h5ad'
    if not path.is_file():
        raise FileNotFoundError(f'Kang data missing: {path}. Supply the processed H5AD; dummy data is not used.')
    # Import only for RNA runs. Reuse the existing X/layer conversion and label mapping.
    from dataloader.dataloader_hvg import AnnDataFeaturesLabels, TorchDatasetFromPrepared
    prepared = AnnDataFeaturesLabels(str(path), data.label_key,
                                    feature_layer=getattr(data, 'feature_layer', None))
    if prepared.features.shape[1] != data.size:
        raise ValueError(f'RNA data.size={data.size}, but the file has {prepared.features.shape[1]} genes')
    if len(prepared.label_to_int) != data.n_classes:
        raise ValueError('RNA data.n_classes does not match the file cell types')
    if not np.isfinite(prepared.features).all():
        raise ValueError('RNA expression contains non-finite values')
    gene_hash = _digest(prepared.adata.var_names)
    if getattr(data, 'gene_names_hash', gene_hash) != gene_hash:
        raise ValueError('RNA gene order differs from the training checkpoint')
    data.gene_names_hash = gene_hash
    cell_types = [str(prepared.int_to_label[i]) for i in range(len(prepared.label_to_int))]
    if hasattr(data, 'cell_types') and data.cell_types != cell_types:
        raise ValueError('RNA cell-type encoding differs from the training checkpoint')
    if not prepared.adata.obs_names.is_unique:
        raise ValueError('RNA observation identifiers must be unique across split memberships')
    data.cell_types = [str(prepared.int_to_label[i]) for i in range(len(prepared.label_to_int))]
    if data.split_key not in prepared.adata.obs:
        raise KeyError(f'RNA split column {data.split_key!r} is absent')
    split = np.asarray(prepared.adata.obs[data.split_key])
    values = [data.train_split, data.val_split, data.test_split]
    if len(set(values)) != 3:
        raise ValueError('RNA train/validation/test split values must be distinct')
    dataset = TorchDatasetFromPrepared(prepared.features, prepared.labels)
    datasets, manifest = {}, {}
    for name, value in zip(['train', 'validation', 'test'], values):
        indices = np.flatnonzero(split == value)
        if len(indices) == 0:
            raise ValueError(f'RNA {name} split {value!r} is empty')
        datasets[name] = Subset(dataset, indices.tolist())
        manifest[name] = dict(count=len(indices), ids_hash=_digest(prepared.adata.obs_names[indices]))
    # Membership is always taken from split_B (or the configured existing column).
    _record_manifest(data, manifest)
    return datasets


def pendulum_datasets(config):
    from dataloader.dataloader_pendulum import PENDULUM, transform
    data = config.data
    if getattr(data, 'gen', False):
        raise ValueError('Baselines read cached trajectory splits. Generate independent files first; set data.gen=false.')
    roots = dict(train=data.root, validation=data.val_root)
    counts = dict(train=data.samplesize, validation=data.test_samplesize)
    if getattr(data, 'test_root', None):
        roots['test'] = data.test_root
        counts['test'] = getattr(data, 'test_count', data.test_samplesize)
    datasets, hashes, manifest = {}, {}, {}
    for split, root in roots.items():
        path = Path(root) / 'pendulum.npz'
        if not path.is_file():
            raise FileNotFoundError(f'Pendulum {split} data missing: {path}. Supply the cached trajectory split.')
        with np.load(path) as loaded:
            images, states = loaded['data'], loaded['states']
            expected = (data.Stotal, data.channel, data.p, data.p)
            if tuple(images.shape[1:]) != expected or tuple(states.shape[1:]) != (data.Stotal, data.n):
                raise ValueError(f'Pendulum {split} shape differs from configured Stotal/channel/p/state dimension')
            if len(images) < counts[split] or len(states) != len(images):
                raise ValueError(f'Pendulum {split} does not contain the requested number of trajectories')
            if not np.isfinite(images).all() or not np.isfinite(states).all():
                raise ValueError('Pendulum cache contains non-finite values')
            selected = images[:counts[split]]
            # Detect duplicated source trajectories, not just identical windows.
            hashes[split] = {hashlib.sha256(frame.tobytes() + state.tobytes()).hexdigest()
                             for frame, state in zip(selected, states[:counts[split]])}
        dataset = PENDULUM(counts[split], data.p, data.Stotal, str(root), window_size=data.S,
                           stride=data.stride, use_sliding_window=True,
                           batch_from_same_trajectory=False, state_dim=data.n, gen=False,
                           transform=transform if data.logit_transform else None)
        if data.S < 3 or data.S > data.Stotal or len(dataset) == 0:
            raise ValueError('Pendulum requires 3 <= data.S <= data.Stotal and a nonempty window set')
        datasets[split] = dataset
        manifest[split] = dict(trajectories=counts[split], windows=len(dataset),
                               source_hash=_digest(sorted(hashes[split])))
    names = list(hashes)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if hashes[left] & hashes[right]:
                raise ValueError(f'Pendulum source trajectories overlap between {left} and {right}')
    _record_manifest(data, manifest)
    return datasets


def extended_datasets(config):
    if config.data.dataset == 'RNA':
        return rna_datasets(config)
    if config.data.dataset == 'PENDULUM':
        return pendulum_datasets(config)
    raise ValueError('Expected RNA or PENDULUM')


def extended_dataloaders(config):
    datasets = extended_datasets(config)
    return tuple(DataLoader(datasets[split], batch_size=config.training.batch_size,
                            shuffle=split == 'train', num_workers=config.data.num_workers)
                 for split in ['train', 'validation'])
