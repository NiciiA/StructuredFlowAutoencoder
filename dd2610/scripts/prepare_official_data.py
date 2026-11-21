"""Deterministic, leakage-aware datasets for the author's SFA modules.

All labels are returned solely for evaluation; training code uses x only.
The released loaders are deliberately not invoked: they mix split roles or reuse
simulator seeds. This is an audited adaptation, not a bitwise paper reproduction.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def logit(x, lam=1e-6):
    x = lam + (1 - 2 * lam) * np.asarray(x, dtype=np.float32)
    return np.log(x) - np.log1p(-x)


def inverse_image(x, lam=1e-6):
    # Stable sigmoid with exact inverse of the input squeeze.
    x = np.clip(np.asarray(x), -80, 80)
    return np.clip((1 / (1 + np.exp(-x)) - lam) / (1 - 2 * lam), 0, 1)


def _pinwheel(n, seed):
    """Author dataloader formula, with exactly n total points (not n/class)."""
    rng = np.random.RandomState(seed)
    y = np.arange(n) % 5
    features = rng.randn(n, 2) * np.array([.15, .05])
    features[:, 0] += 1
    angle = 2 * np.pi * y / 5 + .25 * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angle), -np.sin(angle), np.sin(angle), np.cos(angle)], axis=1).reshape(-1, 2, 2)
    x = 10 * np.einsum('ti,tij->tj', features, rotations)
    perm = rng.permutation(n)
    return x[perm].astype('float32'), y[perm].astype('int64')


def _pendulum(n, seed, p, frames, repo):
    spec = importlib.util.spec_from_file_location('official_pendulum_simulator', Path(repo) / 'dataloader/Pendulum.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = mod.Pendulum
    params = cls.pendulum_default_params()
    params[cls.FRICTION_KEY] = .001
    params[cls.LENGTH_KEY] = .12
    sim = cls(p, observation_mode=cls.OBSERVATION_MODE_LINE,
              transition_noise_std=0., observation_noise_std=1e-5,
              seed=seed, pendulum_params=params, state_dim=2)
    obs, states, _, _ = sim.sample_data_set(n, frames, full_targets=False)
    return logit(obs[:, :, None].astype('float32') / 255.), np.asarray(states, dtype='float32')


def load_arrays(dataset, data_root='data', split_seed=2026, profile='pilot', rna_file=None, **options):
    """Return train/val/test (numpy x,y) pairs and a JSON-compatible metadata dict.

    Profiles: pilot uses subsets; full keeps proposal splits. Explicit *_limit
    changes are recorded. Split seed must stay constant across training seeds.
    For pendulum, one example is a complete trajectory window; split happens
    before any window extraction. For RNA use prepare_rna() output, never dummy.
    """
    dataset = dataset.replace('_mixture', '')
    if profile not in ('pilot', 'full'):
        raise ValueError('profile must be pilot or full')
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    meta = {'dataset': dataset, 'profile': profile, 'split_seed': split_seed,
            'labels_for_training': False, 'protocol': 'proposal_disjoint_splits_v1'}
    arrays = {}
    ids = {}
    if dataset == 'pinwheel':
        counts = (15000, 2000, 2000) if profile == 'full' else (2000, 500, 500)
        for i, (name, n) in enumerate(zip(('train', 'val', 'test'), counts)):
            n = int(options.get(name + '_limit') or n)
            arrays[name] = _pinwheel(n, split_seed + i * 100003)
            ids[name] = np.arange(n, dtype='int64') + i * 100000000
        meta.update(preprocessing='none; author raw scale 10', source='author pinwheel formula', n_classes=5)
    elif dataset in ('mnist', 'emnist'):
        import torch
        from torchvision.datasets import MNIST, EMNIST
        from torch.nn.functional import interpolate
        resolution = int(options.get('resolution', 28))
        if dataset == 'emnist':
            parts = []; labels = []; index = []
            limit = int(options.get('test_limit') or 1000)
            if limit < 20:
                raise ValueError('EMNIST protocol requires at least one sample from each of 20 classes.')
            rng = np.random.default_rng(split_seed + 1999)
            for offset, split_name in [(0, 'digits'), (10, 'letters')]:
                ds = EMNIST(str(root), split=split_name, train=False, download=True)
                yy = ds.targets.numpy()
                for cls in range(10):
                    source_label = cls if split_name == 'digits' else cls+1
                    available = np.flatnonzero(yy == source_label)
                    count = limit//20 + (int(offset+cls < limit%20))
                    pick = rng.permutation(available)[:count]
                    parts.append(ds.data.numpy()[pick]); labels.append(np.full(len(pick),offset+cls))
                    index.append(pick + (1000000 if split_name=='letters' else 0))
            x, y = np.concatenate(parts), np.concatenate(labels)
            order = rng.permutation(len(y))
            # EMNIST's stored orientation is transposed relative to MNIST.
            x = x.transpose(0, 2, 1)
            ids['test'] = np.concatenate(index)[order]
            arrays['test'] = (x[order], y[order])
            meta['orientation'] = 'transpose height/width to match upright MNIST'
            meta['source'] = 'EMNIST digits official test 0..9 + letters official test 1..10; stratified fixed subset'
            meta['label_mapping'] = 'digits 0..9 unchanged; letters 1..10 mapped to 10..19'
        else:
            tra = MNIST(str(root), train=True, download=True)
            tes = MNIST(str(root), train=False, download=True)
            perm = np.random.default_rng(split_seed).permutation(60000)
            ids['train'], ids['val'] = perm[:54000], perm[54000:]
            ids['test'] = np.random.default_rng(split_seed + 1).permutation(10000)
            arrays['train'] = (tra.data.numpy()[ids['train']], tra.targets.numpy()[ids['train']])
            arrays['val'] = (tra.data.numpy()[ids['val']], tra.targets.numpy()[ids['val']])
            arrays['test'] = (tes.data.numpy()[ids['test']], tes.targets.numpy()[ids['test']])
            meta['official_test_never_used_for_validation'] = True
        defaults = {'train': 2000, 'val': 500, 'test': 1000} if profile == 'pilot' else {}
        for name, (x, y) in list(arrays.items()):
            n = options.get(name + '_limit') or defaults.get(name) or len(x)
            x = torch.from_numpy(x[:n].copy()).float()[:, None] / 255.
            if resolution != 28:
                x = interpolate(x, size=(resolution, resolution), mode='bilinear', align_corners=False)
            arrays[name] = (logit(x.numpy()), np.asarray(y[:n], dtype='int64'))
            ids[name] = ids[name][:n]
        meta.update(preprocessing='uint8/255; lambda=1e-6 squeezed logit', resolution=resolution,
                    fixed_test_subset='first 1000 of seeded test permutation', n_classes=10 if dataset == 'mnist' else 20)
    elif dataset == 'pendulum':
        p = int(options.get('resolution', 10))
        frames = int(options.get('trajectory_length', 100 if profile == 'full' else 40))
        sequence = int(options.get('sequence_length', 40 if profile == 'full' else 10))
        if sequence > frames:
            raise ValueError('sequence_length cannot exceed trajectory_length')
        counts = (200, 50, 50) if profile == 'full' else (32, 8, 8)
        repo = options.get('repo') or ROOT / 'upstream/StructuredFlowAutoencoder'
        for i, (name, n) in enumerate(zip(('train', 'val', 'test'), counts)):
            n = int(options.get(name + '_limit') or n)
            # Independent simulator RNGs avoid the official loader's repeated seed=42.
            x, y = _pendulum(n, split_seed + i * 100003, p, frames, repo)
            # Full profile uses sliding windows; IDs make their trajectory lineage explicit.
            stride = int(options.get('window_stride', sequence))
            starts = list(range(0, frames - sequence + 1, stride))
            arrays[name] = (np.concatenate([x[:, j:j + sequence] for j in starts]),
                            np.concatenate([y[:, j:j + sequence] for j in starts]))
            ids[name] = np.tile(np.arange(n) + i * 100000000, len(starts))
        meta.update(preprocessing='author simulator uint8/255; lambda=1e-6 squeezed logit',
                    resolution=p, trajectory_length=frames, sequence_length=sequence,
                    window_stride=stride, split_unit='whole trajectory',
                    simulator_seeds=[split_seed + i * 100003 for i in range(3)],
                    source='author Pendulum.py; friction .001; length .12; transition noise 0')
    elif dataset == 'hvg':
        if rna_file is None:
            raise FileNotFoundError('RNA requires --rna-file prepared.npz. Run scripts/prepare_official_data.py rna first; no synthetic fallback is used.')
        path = Path(rna_file)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix != '.npz':
            raise ValueError('Use the explicit RNA preparation command to create .npz with disjoint splits.')
        with np.load(path, allow_pickle=False) as f:
            for name in ('train', 'val', 'test'):
                x, y = f['x_' + name], f['y_' + name]
                n = options.get(name + '_limit') or (min(len(x), 2000 if name == 'train' else 500) if profile == 'pilot' else len(x))
                arrays[name] = (x[:n].astype('float32'), y[:n].astype('int64'))
                ids[name] = f['id_' + name][:n]
        manifest = path.with_suffix('.json')
        if not manifest.is_file():
            raise FileNotFoundError('RNA manifest missing: ' + str(manifest))
        source_meta = json.loads(manifest.read_text())
        if source_meta.get('synthetic', False):
            raise ValueError('Synthetic RNA cannot be labelled as Kang reproduction.')
        meta.update(source_meta)
        meta.update(dataset='hvg', profile=profile, prepared_file=str(path.resolve()))
    else:
        raise ValueError('Unknown dataset: ' + dataset)
    for name, (x, y) in arrays.items():
        if len(x) == 0 or len(x) != len(y) or not np.isfinite(x).all():
            raise ValueError(f'Invalid {name} data')
    meta['counts'] = {k: len(v[0]) for k, v in arrays.items()}
    meta['input_shape'] = list(next(iter(arrays.values()))[0].shape[1:])
    meta['split_id_sha256'] = {k: digest(v) for k, v in ids.items()}
    meta['data_sha256'] = {k: digest(v[0]) for k, v in arrays.items()}
    meta['label_sha256'] = {k: digest(v[1]) for k, v in arrays.items()}
    arrays['metadata'] = meta
    arrays['indices'] = ids
    return arrays


def prepare_rna(input_file, output, label_key, patient_key=None, input_state='counts',
                layer=None, n_hvg=5000, split_seed=2026, split_key=None, keep_features=False):
    """Train-only HVG selection; explicit raw/log-normalized input state.

    This supplies a documented protocol, since the release does not include its
    prepared Kang file. Do not claim exact Table 4 reproduction from this alone.
    """
    import anndata as ad
    import scanpy as sc
    from scipy import sparse
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import LabelEncoder
    path = Path(input_file)
    if not path.is_file():
        raise FileNotFoundError(path)
    a = ad.read_h5ad(path)
    if label_key not in a.obs:
        raise KeyError(f'{label_key} absent; available obs keys: {list(a.obs.columns)}')
    if a.obs[label_key].isna().any():
        raise ValueError('Missing cell-type labels: explicitly filter/annotate before preparation.')
    if layer:
        a.X = a.layers[layer].copy()
    values = a.X.data if sparse.issparse(a.X) else np.asarray(a.X).ravel()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('RNA input must be finite nonnegative counts or log-normalized expression.')
    if input_state == 'counts' and np.any(np.abs(values - np.round(values)) > 1e-4):
        raise ValueError('Noninteger X is not raw counts. Select a counts layer or use --input-state lognorm.')
    rng = np.random.default_rng(split_seed)
    all_ids = np.arange(a.n_obs)
    if split_key:
        if split_key not in a.obs:
            raise KeyError(split_key)
        split = a.obs[split_key].astype(str).to_numpy()
        inds = {'train': all_ids[split == 'train'], 'val': all_ids[np.isin(split, ['valid', 'val'])],
                'test': all_ids[np.isin(split, ['test', 'ood'])]}
        protocol = 'provided split column ' + split_key
    elif patient_key:
        if patient_key not in a.obs or a.obs[patient_key].isna().any():
            raise ValueError('Patient key must exist without missing values.')
        groups = a.obs[patient_key].astype(str).to_numpy()
        if len(set(groups)) < 3:
            raise ValueError('At least three patients required for train/val/test grouped split.')
        tv, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=split_seed).split(all_ids, groups=groups))
        train, val = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=split_seed + 1).split(tv, groups=groups[tv]))
        inds = {'train': tv[train], 'val': tv[val], 'test': test}
        protocol = 'patient-grouped split'
    else:
        order = rng.permutation(a.n_obs)
        n1, n2 = int(.8 * a.n_obs), int(.9 * a.n_obs)
        inds = {'train': order[:n1], 'val': order[n1:n2], 'test': order[n2:]}
        protocol = 'random cell 80/10/10; donor leakage not controlled'
    if any(len(v) == 0 for v in inds.values()):
        raise ValueError('All three RNA splits must be nonempty.')
    if input_state == 'counts':
        sc.pp.normalize_total(a, target_sum=10000)
        sc.pp.log1p(a)
    if keep_features:
        mask = np.ones(a.n_vars,dtype=bool)
    else:
        fit = a[inds['train']].copy()
        sc.pp.highly_variable_genes(fit, n_top_genes=min(n_hvg, a.n_vars), flavor='seurat')
        mask = fit.var['highly_variable'].to_numpy()
    if not mask.any():
        raise ValueError('HVG selection found no usable features.')
    a = a[:, mask].copy()
    enc = LabelEncoder().fit(a.obs[label_key].astype(str))
    y = enc.transform(a.obs[label_key].astype(str))
    out = {}
    for name, ii in inds.items():
        x = a.X[ii]
        x = x.toarray() if sparse.issparse(x) else np.asarray(x)
        out['x_' + name] = x.astype('float32')
        out['y_' + name] = y[ii].astype('int64')
        out['id_' + name] = np.asarray(a.obs_names[ii], dtype=str)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(output, **out)
    meta = {'dataset': 'hvg', 'source': str(path.resolve()), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'synthetic': False, 'protocol': protocol, 'input_state': input_state, 'layer': layer,
            'normalization': 'library size 10000 + log1p' if input_state == 'counts' else 'already log-normalized; unchanged',
            'hvg_method': 'keep preselected input genes; upstream selection may have used all cells' if keep_features else 'scanpy seurat; fit on training cells only within provided gene universe', 'n_hvg': int(mask.sum()),
            'gene_names': list(map(str, a.var_names)), 'label_classes': enc.classes_.tolist(),
            'n_classes': len(enc.classes_), 'patient_key': patient_key, 'split_seed': split_seed,
            'counts': {k: len(v) for k, v in inds.items()},
            'paper_exactness': 'adapted protocol; original prepared h5ad is absent from repository'}
    meta['classes_by_split']={k:dict(zip(*[v.tolist() for v in np.unique(y[ii],return_counts=True)])) for k,ii in inds.items()}
    if split_key:
        meta['provided_split_counts']=a.obs[split_key].astype(str).value_counts().to_dict()
        meta['provided_split_mapping']={'train':['train'],'val':['valid','val'],'test':['test','ood']}
    if patient_key:
        meta['patients_by_split'] = {k: sorted(set(a.obs[patient_key].astype(str).iloc[v])) for k, v in inds.items()}
        if split_key:
            sets = [set(v) for v in meta['patients_by_split'].values()]
            meta['patient_overlap'] = any(sets[i] & sets[j] for i in range(3) for j in range(i))
    output.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('rna', help='Prepare a real Kang/PBMC h5ad; no dummy replacement')
    r.add_argument('--input', required=True)
    r.add_argument('--output', default='data/kang_prepared.npz')
    r.add_argument('--label-key', default='cell_type')
    r.add_argument('--patient-key')
    r.add_argument('--split-key')
    r.add_argument('--layer')
    r.add_argument('--input-state', choices=['counts', 'lognorm'], default='counts')
    r.add_argument('--n-hvg', type=int, default=5000)
    r.add_argument('--keep-features', action='store_true',help='Retain provided preselected genes; disclose upstream feature-selection leakage.')
    r.add_argument('--split-seed', type=int, default=2026)
    r.add_argument('--download-url', help='Optional verified direct URL; download only if --input does not exist')
    args = p.parse_args()
    if args.download_url and not Path(args.input).exists():
        Path(args.input).parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(args.download_url, args.input)
    meta = prepare_rna(args.input, args.output, args.label_key, args.patient_key,
                       args.input_state, args.layer, args.n_hvg, args.split_seed, args.split_key,args.keep_features)
    print(json.dumps({k: v for k, v in meta.items() if k != 'gene_names'}, indent=2))


if __name__ == '__main__':
    main()
