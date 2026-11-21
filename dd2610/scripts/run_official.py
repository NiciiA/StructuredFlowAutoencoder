"""Run frozen AUTHOR core modules through an audited Colab-friendly harness.

The architecture/loss classes are loaded from the official checkout. Data splits,
checkpointing, device handling and evaluation are replaced explicitly. See
adaptations.json in each run; this is not an unmodified-release claim.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_official_data import load_arrays, inverse_image
from scripts.common import preserve_manifest, provenance, write_json

AUTHOR_SHA = '05e89071f6bae502eedfd3464fad4e8f75f506c1'
_AUTHOR = None


def load_author_modules(repo_path=None):
    """Load actual upstream classes, omitting unused data/plot imports only."""
    global _AUTHOR
    if repo_path is None and _AUTHOR is not None:
        return _AUTHOR
    repo = Path(repo_path or ROOT / 'upstream/StructuredFlowAutoencoder').resolve()
    if _AUTHOR is not None:
        if _AUTHOR['repo'] != str(repo):
            raise RuntimeError('One author checkout per Python process; restart to change revision.')
        return _AUTHOR
    if not (repo / 'src_sfa/sfa.py').is_file():
        raise FileNotFoundError(f'Official checkout missing at {repo}; run bootstrap first.')
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    removed, hashes, modules = {}, {}, {}
    for name in ('sfa', 'sfa_llk', 'sfa_discrete', 'sfa_lds'):
        path = repo / 'src_sfa' / (name + '.py')
        source = path.read_text()
        hashes[name] = hashlib.sha256(source.encode()).hexdigest()
        lines = source.splitlines(keepends=True)
        removed[name] = [line.strip() for line in lines
                         if line.startswith('from dataloader.') or line.startswith('from utils import')
                         or line.startswith('from tqdm import') or line.startswith('import matplotlib.pyplot')]
        source = ''.join('\n' if line.strip() in removed[name] else line for line in lines)
        mod = types.ModuleType('official_' + name)
        mod.__file__ = str(path)
        sys.modules[mod.__name__] = mod
        exec(compile(source, str(path), 'exec'), mod.__dict__)
        modules[name] = mod
    try:
        revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unknown'
    modules.update(repo=str(repo), revision=revision, omitted_unused_imports=removed, source_sha256=hashes)
    _AUTHOR = modules
    return modules


def build_continuous(input_shape, z_dim, width=128, depth=3, posterior='gaussian',
                     freqs=4, posterior_freqs=2, posterior_depth=0,
                     model_type='mlp', hidden_dim=None, activation='silu'):
    """Return author's vt and rt. Reverse time: observed x is at r=0.

    mlp requires hidden_dim==flat input dim because its x embedding is identity.
    Passing this explicitly repairs the released 2D default shape mismatch.
    """
    author = load_author_modules()
    image = len(input_shape) == 3
    flat = int(np.prod(input_shape))
    spatial = input_shape[-1] if image else flat
    act = {'silu': nn.SiLU, 'tanh': nn.Tanh, 'softplus': nn.Softplus}[activation]
    if model_type == 'mlp':
        hidden_dim = flat
    elif hidden_dim is None:
        hidden_dim = width
    vt = author['sfa_llk'].LLK(spatial, z_dim, model=model_type, is_image=image,
                              in_ch=input_shape[0] if image else 1,
                              hidden_dim=hidden_dim, freqs=freqs,
                              hidden_features=[width] * depth, fct=act())
    if posterior == 'gaussian':
        rt = author['sfa'].GaussianPrior(flat, z_dim, freqs=posterior_freqs,
                                         hidden_features=[width] * posterior_depth, fct=nn.Tanh())
    elif posterior == 'cnf':
        rt = author['sfa'].CNF(flat, z_dim, freqs=posterior_freqs,
                              hidden_features=[width] * posterior_depth)
    else:
        raise ValueError(posterior)
    return vt, rt


def defaults(dataset, profile):
    base = dict(width=128, depth=3, epochs=3, batch_size=128, lr=.0005, weight_decay=.0001,
                z_dim=32, freqs=4, posterior_freqs=2, posterior_depth=0,
                posterior='gaussian', alpha=.01, sig_min=1e-4, clip_norm=1.,
                model_type='mlp', activation='silu', ode_steps=50, ode_method='heun',
                eval_samples=500, eval_batch_size=128, probe_train_samples=2000,
                resolution=28, beta=1., sequence_length=10, trajectory_length=40,
                window_stride=10, dtype='float32')
    base['marginal_latent'] = 'aggregate_train'
    if dataset.startswith('pinwheel'):
        base.update(z_dim=1, alpha=0., lr=.001, resolution=2, activation='tanh')
    if dataset == 'hvg':
        base.update(model_type='mlp_high', activation='softplus', lr=.001, alpha=.001)
    if dataset == 'pendulum':
        base.update(z_dim=2, resolution=10, batch_size=16, depth=2, alpha=0., lr=.0008,
                    posterior='sequence', eval_samples=8, eval_batch_size=8,
                    posterior_depth=0, freqs=10, posterior_freqs=10)
    if profile == 'full':
        base.update(eval_samples=1000, ode_steps=100)
        if dataset == 'pinwheel':
            base.update(width=200, depth=5, epochs=1000, batch_size=256,
                        posterior='cnf', posterior_freqs=0, freqs=2)
        elif dataset == 'pinwheel_mixture':
            base.update(width=200, depth=5, epochs=500, batch_size=256, alpha=.01, lr=.0001, freqs=2)
        elif dataset in ('mnist', 'mnist_mixture'):
            base.update(width=800, depth=3, epochs=201, batch_size=256, freqs=20, posterior_freqs=4)
        elif dataset == 'hvg':
            base.update(width=800, depth=3, epochs=200, batch_size=256, freqs=20, posterior_freqs=2)
        elif dataset == 'pendulum':
            base.update(width=1024, epochs=301, batch_size=64, sequence_length=40,
                        trajectory_length=100, window_stride=1, eval_samples=50, eval_batch_size=8)
    return base


def make_model(dataset, input_shape, cfg):
    author = load_author_modules()
    if dataset == 'pendulum':
        p, S, d = cfg['resolution'], cfg['sequence_length'], cfg['z_dim']
        vt = author['sfa_llk'].LLK(p, d, model='mlp', S=S, F=cfg['trajectory_length'],
                                 is_image=True, hidden_dim=p*p, freqs=cfg['freqs'], dsemb=0,
                                 hidden_features=[cfg['width']] * cfg['depth'], fct=nn.SiLU())
        rt = author['sfa_lds'].fullGauss(p*p, d, S, cfg['trajectory_length'], dsemb=2,
                                       num_hidden=64, num_layers=2, freqs=cfg['posterior_freqs'],
                                       hidden_features=[cfg['width']] * cfg['posterior_depth'], fct=nn.Tanh())
        model = nn.ModuleDict({'vt': vt, 'rt': rt})
        prior = author['sfa_lds'].LatentDynamicalSystem(d)
        loss = author['sfa_lds'].FlowMatchingLossSeq(vt, rt, prior,
                                                   alpha=cfg['alpha'], sig_min=cfg['sig_min'],
                                                   const=True, flowcnn=False)
    else:
        vt, rt = build_continuous(input_shape, cfg['z_dim'], width=cfg['width'], depth=cfg['depth'],
                                  posterior=cfg['posterior'], freqs=cfg['freqs'],
                                  posterior_freqs=cfg['posterior_freqs'], posterior_depth=cfg['posterior_depth'],
                                  model_type=cfg['model_type'], activation=cfg['activation'])
        model = nn.ModuleDict({'vt': vt, 'rt': rt})
        prior = torch.distributions.Normal(torch.zeros(cfg['z_dim']), torch.ones(cfg['z_dim']))
        if dataset.endswith('_mixture'):
            k = 5 if dataset.startswith('pinwheel') else 10
            flat = int(np.prod(input_shape))
            # Release identity pi embedding has width k, not 784.
            rt = author['sfa_discrete'].GaussianMixtureComponent(k, cfg['z_dim'], flat,
                        hidden_dim=k, freqs=cfg['posterior_freqs'], hidden_features=[], fct=nn.Tanh())
            Rt = author['sfa_discrete'].CatNF_fixed(flat, k, temp=cfg['beta'], freqs=cfg['posterior_freqs'],
                                                   hidden_features=[], fct=nn.Tanh())
            model = nn.ModuleDict({'vt': vt, 'rt': rt, 'Rt': Rt})
            loss = author['sfa_discrete'].FlowMatchingLossMixture(vt, Rt, rt, priorpi=None, priorz=None,
                        k=k, beta=cfg['beta'], alpha=cfg['alpha'], sig_min=cfg['sig_min'])
        else:
            loss = author['sfa'].FlowMatchingLoss(vt, rt, prior, alpha=cfg['alpha'],
                        sig_min=cfg['sig_min'], fixz=cfg['posterior'] == 'gaussian')
    return model, loss


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def atomic_save(obj, path):
    temp = Path(str(path) + '.tmp')
    torch.save(obj, temp)
    os.replace(temp, path)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def batch_x(x, device, dtype, dataset):
    x = x.to(device=device, dtype=dtype)
    # transpose, never view: the release's .view(S,B,...) mixes trajectory order.
    return x.transpose(0, 1).contiguous() if dataset == 'pendulum' else x


def integrate(vt, noise, z, steps, method='heun'):
    """Reverse author's data-to-noise convention, noise r=1 to data r=0."""
    if method == 'dopri5':
        from torchdiffeq import odeint
        count = [0]
        def vf(t, x):
            count[0] += 1
            return vt(t, x, z)
        out = odeint(vf, noise, noise.new_tensor([1., 0.]), atol=1e-5, rtol=1e-5)[-1]
        return out, count[0]
    x = noise
    dt = -1. / steps
    for j in range(steps):
        t = x.new_tensor(1. - j / steps)
        first = vt(t, x, z)
        if method == 'euler':
            x = x + dt * first
        else:
            second = vt(t + dt, x + dt * first, z)
            x = x + dt * (first + second) / 2
    return x, steps * (2 if method == 'heun' else 1)


def infer(model, x, dataset, cfg, sample=False):
    if dataset == 'pendulum':
        shape = (x.shape[0], x.shape[1], cfg['z_dim'])
        return model['rt'].decode_sequence(x.new_zeros(shape), x.flatten(start_dim=2),
                                           t=x.new_zeros(()), deterministic=not sample)
    if dataset.endswith('_mixture'):
        Rt, rt = model['Rt'], model['rt']
        t = x.new_zeros(len(x))
        # Same logits used in upstream mixture loss, including its sorted-logit rt behavior.
        temp = Rt.freqs * t[..., None]
        temb = torch.cat((temp.cos(), temp.sin()), -1)
        logits = Rt.fc(torch.cat((temb, Rt.embx(x.flatten(start_dim=1))), -1))
        q = rt(logits, x, t)
        return q.rsample() if sample else q.mean
    if cfg['posterior'] == 'cnf':
        z = torch.randn(len(x), cfg['z_dim'], device=x.device, dtype=x.dtype)
        # CNF has no analytic mean; seed fixes this posterior draw for diagnostics.
        return model['rt'].decode(z, x.flatten(start_dim=1), 0.)
    q = model['rt'](x, x.new_zeros(len(x)))
    return q.rsample() if sample else q.mean


def evaluate(model, arrays, dataset, cfg, device, dtype, outdir, split, seed):
    from scripts.metrics import evaluate_arrays,soft_nmi_variants
    from sklearn.linear_model import Ridge
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    model.eval()
    seed_all(seed + 500000)
    n = min(cfg['eval_samples'], len(arrays[split][0]))
    ntrain = min(cfg['probe_train_samples'], len(arrays['train'][0]))
    batch = cfg['eval_batch_size']
    all_z = {}
    artifacts = {}
    nfe = 0
    start = time.perf_counter()
    with torch.no_grad():
        for key, nn_ in (('train', ntrain), (split, n)):
            zs, pp, sp, gp, probs = [], [], [], [], []
            for start_i in range(0, nn_, batch):
                x = batch_x(torch.from_numpy(arrays[key][0][start_i:start_i+batch][:nn_-start_i]), device, dtype, dataset)
                zmean = infer(model, x, dataset, cfg)
                z = infer(model, x, dataset, cfg, sample=True)
                zs.append(zmean.transpose(0, 1).cpu().numpy() if dataset == 'pendulum' else zmean.cpu().numpy())
                if key == split:
                    noise = torch.randn_like(x)
                    if dataset == 'pendulum':
                        ss, bb = x.shape[:2]
                        # Same base noise, z shuffled only across independent trajectories.
                        permutation = torch.roll(torch.arange(bb, device=device), 1)
                        zshuf = z[:, permutation]
                        pred, calls = integrate(model['vt'], noise.flatten(0, 1), z.flatten(0, 1), cfg['ode_steps'], cfg['ode_method'])
                        shuf, calls2 = integrate(model['vt'], noise.flatten(0, 1), zshuf.flatten(0, 1), cfg['ode_steps'], cfg['ode_method'])
                        pred, shuf = pred.view_as(x), shuf.view_as(x)
                        pp.append(pred.transpose(0, 1).cpu().numpy())
                        sp.append(shuf.transpose(0, 1).cpu().numpy())
                    else:
                        permutation = torch.roll(torch.arange(len(x), device=device), 1)
                        pred, calls = integrate(model['vt'], noise, z, cfg['ode_steps'], cfg['ode_method'])
                        shuf, calls2 = integrate(model['vt'], noise, z[permutation], cfg['ode_steps'], cfg['ode_method'])
                        pp.append(pred.cpu().numpy())
                        sp.append(shuf.cpu().numpy())
                        # The learned posterior is not forced to aggregate to N(0,I).
                        if cfg['marginal_latent'] == 'aggregate_train':
                            pick = np.random.randint(ntrain, size=len(x))
                            references = torch.as_tensor(arrays['train'][0][pick], device=device, dtype=dtype)
                            zg = infer(model,references,dataset,cfg,sample=True)
                        else:
                            zg = torch.randn_like(z)
                        gen, calls3 = integrate(model['vt'], torch.randn_like(x), zg, cfg['ode_steps'], cfg['ode_method'])
                        gp.append(gen.cpu().numpy())
                        nfe += calls3
                        if dataset.endswith('_mixture'):
                            probs.append(model['Rt'](x, x.new_zeros(len(x))).cpu().numpy())
                    nfe += calls + calls2
            all_z[key] = np.concatenate(zs)
            if key == split:
                artifacts.update(posterior_pred=np.concatenate(pp), shuffled_pred=np.concatenate(sp))
                if gp:
                    artifacts['generated'] = np.concatenate(gp)
                if probs:
                    artifacts['mixture_probs'] = np.concatenate(probs)
    true = arrays[split][0][:n]
    y = arrays[split][1][:n]
    train_y = arrays['train'][1][:ntrain]
    if dataset == 'pendulum':
        pix = inverse_image(true)
        pred = inverse_image(artifacts['posterior_pred'])
        shuf = inverse_image(artifacts['shuffled_pred'])
        ztra, ztest = all_z['train'], all_z[split]
        # Linear calibration is learned ONLY on training trajectories.
        reg = Ridge(alpha=1.).fit(ztra.reshape(-1, ztra.shape[-1]), train_y.reshape(-1, train_y.shape[-1]))
        state_pred = reg.predict(ztest.reshape(-1, ztest.shape[-1])).reshape(y.shape)
        metrics = {'observed_pixel_rmse': float(np.sqrt(np.mean((pix-pred)**2))),
                   'observed_frame_l2_rmse': float(np.sqrt(np.mean(np.sum((pix-pred)**2, axis=(-1,-2,-3))))),
                   'latent_train_calibrated_rmse': float(np.sqrt(np.mean((state_pred-y)**2))),
                   'latent_raw_coordinate_rmse': float(np.sqrt(np.mean((ztest-y)**2))),
                   'shuffled_pixel_rmse': float(np.sqrt(np.mean((pix-shuf)**2))),
                   'matched_noise_latent_change_rms': float(np.sqrt(np.mean((pred-shuf)**2))),
                   'latent_calibration': 'Ridge(alpha=1), fit training latent means to simulator states',
                   'task': 'posterior reconstruction of heldout complete windows; not forecasting'}
        artifacts['state_prediction'] = state_pred
    else:
        metrics = evaluate_arrays(true_x=true, generated=artifacts.get('generated'),
                  posterior_pred=artifacts['posterior_pred'], shuffled_pred=artifacts['shuffled_pred'],
                  train_z=all_z['train'], train_y=train_y, test_z=all_z[split], test_y=y,
                  dataset=dataset.replace('_mixture',''),
                  n_clusters=arrays['metadata'].get('n_classes', len(np.unique(train_y))), seed=seed)
        if 'mixture_probs' in artifacts:
            assignment = artifacts['mixture_probs'].argmax(-1)
            metrics.update(mixture_hard_nmi=float(normalized_mutual_info_score(y, assignment)),
                           mixture_hard_ari=float(adjusted_rand_score(y, assignment)))
            metrics.update(soft_nmi_variants(artifacts['mixture_probs'],y))
    metrics.update(evaluation_split=split, n_evaluation=n, n_probe_train=ntrain,
                   ode_method=cfg['ode_method'], ode_steps=cfg['ode_steps'],
                   total_vector_field_calls=nfe, evaluation_seconds=time.perf_counter()-start,
                   generation_latent_source=cfg['marginal_latent'] if dataset!='pendulum' else 'not evaluated; posterior predictive only',
                   posterior_diagnostic='fixed posterior draw' if cfg['posterior']=='cnf' else 'posterior mean')
    artifacts.update(true_x=true, y=y, z=all_z[split], train_z=all_z['train'], train_y=train_y)
    np.savez_compressed(outdir / ('evaluation_' + split + '.npz'), **artifacts)
    (outdir / ('metrics_' + split + '.json')).write_text(json.dumps(metrics, indent=2, default=float))
    plot_artifacts(artifacts,dataset,outdir/('evaluation_'+split+'.png'))
    return metrics


def plot_artifacts(data,dataset,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if dataset!='pendulum' and dataset!='hvg':
        from scripts.run_experiment import plot_samples
        from types import SimpleNamespace
        plot_samples(data,SimpleNamespace(dataset=dataset.replace('_mixture','')),path)
        return
    if dataset=='pendulum':
        count=min(6,data['true_x'].shape[1]);frames=np.linspace(0,data['true_x'].shape[1]-1,count).astype(int)
        fig,axes=plt.subplots(2,count,figsize=(2*count,4),squeeze=False)
        for row,key in enumerate(['true_x','posterior_pred']):
            for col,idx in enumerate(frames):
                axes[row,col].imshow(inverse_image(data[key][0,idx]).squeeze(),cmap='gray',vmin=0,vmax=1)
                axes[row,col].axis('off')
            axes[row,0].set_title(key)
    else:
        from sklearn.decomposition import PCA
        proj=PCA(n_components=2).fit(data['train_z']).transform(data['z'])
        fig,ax=plt.subplots(figsize=(6,5));ax.scatter(proj[:,0],proj[:,1],c=data['y'],s=8,cmap='tab10')
        ax.set_title('Held-out latent means; PCA fit on train')
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['pinwheel','pinwheel_mixture','mnist','mnist_mixture','hvg','pendulum'], required=True)
    p.add_argument('--profile', choices=['pilot','full'], default='pilot')
    p.add_argument('--config', help='JSON overrides, all saved to resolved_config.json')
    p.add_argument('--repo', default=str(ROOT / 'upstream/StructuredFlowAutoencoder'))
    p.add_argument('--data-root', default='data')
    p.add_argument('--rna-file')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split-seed', type=int, default=2026)
    p.add_argument('--participant', default='dwei')
    p.add_argument('--output-root', default='runs')
    p.add_argument('--run-id', help='Name inside output-root; existing nonempty dir requires --resume')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--eval-split', choices=['val','test'], default='val', help='Use test only after design/checkpoint choices are frozen')
    p.add_argument('--no-eval', action='store_true')
    p.add_argument('--device', choices=['auto','cpu','cuda','mps'], default='auto')
    p.add_argument('--epochs', type=int)
    p.add_argument('--train-limit', type=int)
    p.add_argument('--val-limit', type=int)
    p.add_argument('--test-limit', type=int)
    p.add_argument('--max-batches', type=int, help='Smoke test only; recorded in config')
    args = p.parse_args()
    author = load_author_modules(args.repo)
    if author['revision'] != AUTHOR_SHA:
        raise RuntimeError(f'Expected frozen author revision {AUTHOR_SHA}, found {author["revision"]}. Checkout the frozen commit before running.')
    cfg = defaults(args.dataset, args.profile)
    if args.config:
        cfg.update(json.loads(Path(args.config).read_text()))
    for key in ('epochs','train_limit','val_limit','test_limit','max_batches'):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    cfg.update(dataset=args.dataset, profile=args.profile, seed=args.seed, split_seed=args.split_seed,
               participant=args.participant, author_revision=author['revision'])
    if cfg['epochs'] < 1 or cfg['ode_steps'] < 1:
        raise ValueError('epochs and ode_steps must be positive')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else torch.device(args.device)
    dtype = getattr(torch, cfg['dtype'])
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('dtype must be float32 or float64')
    torch.set_default_dtype(dtype)
    seed_all(args.seed)
    run_id = args.run_id or f'{args.participant}_official_{args.dataset}_{args.profile}_s{args.seed}'
    out = Path(args.output_root) / run_id
    if out.exists() and any(out.iterdir()) and not (args.resume or args.eval_only):
        raise FileExistsError(f'{out} exists. Use --resume, --eval-only, or a new --run-id.')
    out.mkdir(parents=True, exist_ok=True)
    old_cfg_file = out / 'resolved_config.json'
    if old_cfg_file.exists() and (args.resume or args.eval_only):
        old_cfg = json.loads(old_cfg_file.read_text())
        changed = {k for k in set(old_cfg)|set(cfg) if old_cfg.get(k)!=cfg.get(k)}
        permitted = {'epochs','eval_samples','eval_batch_size','ode_steps','ode_method','probe_train_samples'}
        if changed - permitted:
            raise ValueError('Resume/eval config differs for: ' + ','.join(sorted(changed-permitted)))
    if not old_cfg_file.exists():
        old_cfg_file.write_text(json.dumps(cfg, indent=2))
    write_json(out/'latest_invocation.json',dict(config=cfg,args=vars(args)))
    options = {k:cfg[k] for k in ('resolution','sequence_length','trajectory_length','window_stride','train_limit','val_limit','test_limit') if k in cfg}
    arrays = load_arrays(args.dataset,args.data_root,args.split_seed,args.profile,args.rna_file,repo=args.repo,**options)
    preserve_manifest(out/'data_manifest.json',arrays['metadata'])
    if not (out/'split_indices.npz').exists():
        np.savez_compressed(out/'split_indices.npz',**arrays['indices'])
    adaptations = {'classification':'author core modules with explicit audited harness adaptations',
        'author_revision':author['revision'],'source_sha256':author['source_sha256'],
        'omitted_unused_imports':author['omitted_unused_imports'],
        'changes':['independent train/val/test and fixed split seed; labels excluded from training',
                   'float32 default instead of original runner float64; configurable',
                   'AdamW with cosine schedule; no released WarmUpScheduler',
                   'clip gradients before optimizer step',
                   'explicit LLK mlp hidden_dim=flat input dim repairs vector dimension mismatch',
                   'mixture posterior hidden_dim=number of logits repairs released dimension mismatch',
                   'pendulum transpose B,S to S,B and independent simulator RNG per split',
                   'sequence original squared-summed residual loss retained (not sum of per-step squares)',
                   'validation-selected best.pt, complete RNG/optimizer last.pt, unique run folder',
                   'explicit ODE solver; Heun default instead of adaptive dopri5; configurable',
                   'heldout metrics, raw Euclidean W1, train-fit probes; no batch-dependent distance rescaling',
                   'marginal samples default to aggregate TRAIN posterior samples; normal-prior variant is explicit',
                   'RNA requires real prepared data, refuses silent synthetic fallback'],
        'claim_limit':'pilot/full are harness presets; full uses release-scale hyperparameters but altered protocol, not exact paper result'}
    preserve_manifest(out/'adaptations.json',adaptations)
    environment=provenance()
    preserve_manifest(out/'environment.json',environment,keys=['source_sha256','author_revision','author_source_sha256'])
    model, criterion = make_model(args.dataset, arrays['train'][0].shape[1:], cfg)
    model.to(device=device,dtype=dtype)
    criterion.to(device=device,dtype=dtype)
    if args.dataset not in ('pendulum','pinwheel_mixture','mnist_mixture'):
        criterion.prior = torch.distributions.Normal(torch.zeros(cfg['z_dim'],device=device,dtype=dtype),torch.ones(cfg['z_dim'],device=device,dtype=dtype))
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'])
    loader_gen = torch.Generator().manual_seed(args.seed+100)
    def loader(key,shuffle):
        x,y = arrays[key]
        return DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=cfg['batch_size'],
                          shuffle=shuffle,generator=loader_gen if shuffle else None,num_workers=0,drop_last=False)
    train_loader, val_loader = loader('train',True), loader('val',False)
    start_epoch, best, elapsed = 0,float('inf'),0.
    checkpoint = out/'last.pt'
    if args.resume:
        state=torch.load(checkpoint,map_location=device,weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        start_epoch,best,elapsed=state['epoch']+1,state['best_val_loss'],state.get('training_seconds',0.)
        restore_rng(state['rng']);loader_gen.set_state(state['loader_rng'].cpu())
        if state['config']['epochs'] != cfg['epochs']:
            raise ValueError('Cosine schedule horizon is fixed; resume with the original --epochs value.')
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    if not args.eval_only:
        for epoch in range(start_epoch,cfg['epochs']):
            tick=time.perf_counter(); model.train(); total=count=0
            for bi,(xb,_) in enumerate(train_loader):
                if cfg.get('max_batches') and bi>=cfg['max_batches']:
                    break
                x=batch_x(xb,device,dtype,args.dataset)
                optimizer.zero_grad(set_to_none=True)
                loss=criterion(x)
                if not torch.isfinite(loss):
                    (out/'failure.json').write_text(json.dumps({'epoch':epoch,'batch':bi,'reason':'nonfinite training loss'}))
                    raise FloatingPointError('Nonfinite loss; failure saved, no false completion.')
                loss.backward(); norm=nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_norm'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite gradient norm')
                optimizer.step(); total+=float(loss.detach())*len(xb);count+=len(xb)
            model.eval(); val_total=val_count=0
            # Fixed validation Monte Carlo randomness; preserve training RNG progression.
            saved_rng=rng_state();seed_all(args.seed+90000)
            with torch.no_grad():
                for bi,(xb,_) in enumerate(val_loader):
                    if cfg.get('max_batches') and bi>=cfg['max_batches']:
                        break
                    val=criterion(batch_x(xb,device,dtype,args.dataset))
                    val_total+=float(val)*len(xb);val_count+=len(xb)
            restore_rng(saved_rng)
            val_loss=val_total/val_count
            if not np.isfinite(val_loss):
                raise FloatingPointError('Nonfinite validation loss')
            scheduler.step()
            if device.type=='cuda':
                torch.cuda.synchronize(device)
            seconds=time.perf_counter()-tick;elapsed+=seconds
            row={'epoch':epoch,'train_loss':total/count,'val_loss':val_loss,'seconds':seconds,'lr':optimizer.param_groups[0]['lr']}
            log=out/'history.csv'
            with log.open('a',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=row)
                if f.tell()==0:writer.writeheader()
                writer.writerow(row)
            improve=val_loss<best;best=min(best,val_loss)
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                   'epoch':epoch,'best_val_loss':best,'training_seconds':elapsed,'rng':rng_state(),'loader_rng':loader_gen.get_state(),'config':cfg}
            atomic_save(state,checkpoint)
            if improve:atomic_save(state,out/'best.pt')
            print(json.dumps(row),flush=True)
    best_state=torch.load(out/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(best_state['model'])
    last_state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    info={'device':str(device),'torch':torch.__version__,'numpy':np.__version__,
          'python':sys.version,'parameters':sum(p.numel() for p in model.parameters()),
          'training_seconds':last_state.get('training_seconds',0.),'best_epoch':best_state['epoch'],
          'best_val_loss':best_state['best_val_loss'],'author_revision':author['revision'],
          'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}
    try:info['pip_freeze']=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()
    except subprocess.CalledProcessError:pass
    (out/'run_metadata.json').write_text(json.dumps(info,indent=2))
    if not args.no_eval:
        print(json.dumps(evaluate(model,arrays,args.dataset,cfg,device,dtype,out,args.eval_split,args.seed),indent=2,default=float))
    print('Saved:',out.resolve())


if __name__=='__main__':
    main()
