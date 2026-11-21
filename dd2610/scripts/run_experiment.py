"""Controlled continuous-latent experiments, competitive baselines, and K ablations.

This is an adapted research harness, not a claim of exact paper reproduction.
Run from the project root: python -m scripts.run_experiment --help
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_models import FM, Autoencoder, bridge, scfm_loss
from scripts.common import seed_all, rng_state, restore_rng, write_json, atomic_save, provenance, heun, preserve_manifest
from scripts.metrics import evaluate_arrays
from scripts.prepare_official_data import load_arrays


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['pinwheel', 'mnist', 'hvg'], default='pinwheel')
    p.add_argument('--method', choices=['sfa_k1', 'sfa_k4', 'sfa_cross', 'fm', 'vae', 'latent_fm', 'paper_only'], default='sfa_k1')
    p.add_argument('--profile', choices=['pilot', 'full'], default='pilot')
    p.add_argument('--participant', default='me')
    p.add_argument('--tag', default='v1')
    p.add_argument('--output', default='results')
    p.add_argument('--run-dir', default=None)
    p.add_argument('--data-root', default='data')
    p.add_argument('--repo', default='upstream/StructuredFlowAutoencoder')
    p.add_argument('--rna-file')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split-seed', type=int, default=2026)
    p.add_argument('--epochs', type=int, default=10, help='Per stage; latent_fm has TWO stages.')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--z-dim', type=int, default=None)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--alpha', type=float, default=0.0)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--sigma', type=float, default=1e-4)
    p.add_argument('--clip-grad', type=float, default=1.0)
    p.add_argument('--ode-steps', type=int, default=100)
    p.add_argument('--eval-samples', type=int, default=500)
    p.add_argument('--train-limit', type=int)
    p.add_argument('--val-limit', type=int)
    p.add_argument('--test-limit', type=int)
    p.add_argument('--resolution', type=int, default=28)
    p.add_argument('--grad-repeats', type=int, default=8)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--evaluate-test', action='store_true', help='Only after all choices are frozen.')
    return p


def build(args, shape):
    d = int(np.prod(shape)); act = 'tanh' if args.dataset == 'pinwheel' else 'softplus'
    modules = {}
    if args.method.startswith('sfa'):
        from scripts.run_official import load_author_modules, build_continuous
        load_author_modules(args.repo)
        vt, rt = build_continuous(shape, args.z_dim, width=args.width, depth=args.depth,
                                  posterior='gaussian', freqs=4, posterior_freqs=2, activation=act)
        modules.update(vt=vt, rt=rt)
    elif args.method == 'paper_only':
        from scripts.verify_freeze import verify
        verify()
        from paper_only.model import PaperSFA
        modules['paper'] = PaperSFA(x_dim=d, z_dim=args.z_dim, width=args.width, depth=args.depth,activation=act)
    elif args.method == 'fm':
        modules['fm'] = FM(d, args.width, args.depth, act)
    else:
        modules['ae'] = Autoencoder(d, args.z_dim, args.width, args.depth,
                                    variational=args.method == 'vae', activation=act)
        if args.method == 'latent_fm':
            modules['fm'] = FM(args.z_dim, args.width, args.depth, act)
    return nn.ModuleDict(modules)


def objective(models, x, args, stage, validation=False, bridge_generator=None):
    if args.method.startswith('sfa'):
        est = 'validation' if validation else args.method.replace('sfa_', '')
        fixed=bridge(x,args.sigma,generator=bridge_generator)
        return scfm_loss(models['vt'], models['rt'], x, est, args.alpha, args.sigma,fixed_bridge=fixed)
    if args.method == 'paper_only':
        return models['paper'].loss(x.flatten(1))
    if args.method == 'vae' or (args.method == 'latent_fm' and stage == 0):
        return models['ae'].loss(x, args.beta)
    if args.method == 'latent_fm':
        with torch.no_grad():
            x = (models['ae'].encode(x)-models.z_mean)/models.z_std
    t, xt, u = bridge(x, args.sigma,generator=bridge_generator)
    return (models['fm'](t, xt)-u).square().mean()


@torch.no_grad()
def encode(models, x, args, sample=False):
    if args.method.startswith('sfa'):
        q = models['rt'](x, x.new_zeros(len(x)))
        return q.rsample() if sample else q.mean
    if args.method == 'paper_only':
        q = models['paper'].posterior(x.flatten(1), x.new_ones(len(x)))
        return q.rsample() if sample else q.mean
    if args.method in ('vae', 'latent_fm'):
        q = models['ae'].encode(x)
        return (q.rsample() if sample else q.mean) if args.method == 'vae' else q
    return None


@torch.no_grad()
def generate(models, noise, args, z=None, observation_noise=None):
    if args.method.startswith('sfa'):
        return heun(lambda t,x: models['vt'](t,x,z), noise, args.ode_steps)
    if args.method == 'paper_only':
        shape = noise.shape
        return heun(lambda t,x: models['paper'].velocity(t,x,z), noise.flatten(1),
                    args.ode_steps, reverse=False).view(shape)
    if args.method == 'fm':
        return heun(models['fm'], noise, args.ode_steps)
    if args.method == 'vae':
        p = models['ae'].decode(z)
        eps = noise.flatten(1) if observation_noise is None else observation_noise.flatten(1)
        return (p.mean+p.stddev*eps).view_as(noise)
    return models['ae'].decode(z).view_as(noise)


def latent_statistics(models, x, args, device):
    with torch.no_grad():
        z = torch.cat([models['ae'].encode(b.to(device)).cpu() for b in x.split(args.batch_size)])
    models.z_mean.copy_(z.mean(0).to(device))
    models.z_std.copy_(z.std(0).clamp_min(1e-3).to(device))


def gradient_diagnostic(models, x, args):
    if not args.method.startswith('sfa') or args.grad_repeats < 2:
        return {}
    fixed = bridge(x, args.sigma)
    mean = m2 = None
    norms = []; losses = []
    for r in range(args.grad_repeats):
        models.zero_grad(set_to_none=True)
        loss = scfm_loss(models['vt'], models['rt'], x, args.method.replace('sfa_', ''),
                         args.alpha, args.sigma, fixed_bridge=fixed)
        grads = torch.autograd.grad(loss, tuple(models.parameters()), allow_unused=True)
        g = torch.cat([g.detach().flatten().cpu().double() for g in grads if g is not None])
        if mean is None:
            mean = g.clone(); m2 = torch.zeros_like(g)
        else:
            delta = g-mean; mean += delta/(r+1); m2 += delta*(g-mean)
        losses.append(loss.item()); norms.append(g.norm().item())
    return dict(gradient_variance_trace_fixed_bridge=float(m2.sum()/(args.grad_repeats-1)),
                gradient_mean_squared_norm=float(mean.square().sum()), gradient_norms=norms,
                estimator_losses_fixed_bridge=losses, repeats=args.grad_repeats,
                protocol='Same x, t, endpoint noise; independent posterior draws; before clipping; checkpoint-specific')


@torch.no_grad()
def evaluate(models, arrays, args, device, out):
    split = 'test' if args.evaluate_test else 'val'
    x_np, y_np = arrays[split]
    n = min(args.eval_samples, len(x_np)); x_np = x_np[:n]; y_np = y_np[:n]
    tx_np, ty_np = arrays['train']; tx_np = tx_np[:5000]; ty_np = ty_np[:5000]
    x = torch.as_tensor(x_np, dtype=torch.float32); train = torch.as_tensor(tx_np, dtype=torch.float32)
    seed_all(args.split_seed+9000)
    zs, zmeans, gen, pp, sp = [], [], [], [], []
    tz = None
    has_latent = args.method != 'fm'
    if has_latent:
        tz = torch.cat([encode(models,b.to(device),args).cpu() for b in train.split(args.batch_size)])
        for b in x.split(args.batch_size):
            zs.append(encode(models,b.to(device),args,True).cpu())
            zmeans.append(encode(models,b.to(device),args).cpu())
        z = torch.cat(zs); zmean = torch.cat(zmeans)
        # A non-identity cyclic shuffle for every n>1, same noise for both generations.
        shift = max(1, n//2); shuffled = torch.roll(z, shifts=shift, dims=0)
    noise = torch.randn_like(x)
    for start in range(0,n,args.batch_size):
        end = min(start+args.batch_size,n); base = noise[start:end].to(device); size=end-start
        if has_latent:
            pp.append(generate(models,base,args,z[start:end].to(device)).cpu())
            sp.append(generate(models,base,args,shuffled[start:end].to(device)).cpu())
        if args.method == 'vae':
            zg = torch.randn(size,args.z_dim,device=device)
        elif args.method == 'latent_fm':
            zg = heun(models['fm'],torch.randn(size,args.z_dim,device=device),args.ode_steps)
            zg = zg*models.z_std+models.z_mean
        elif has_latent:
            idx = torch.randint(len(train),(size,))
            zg = encode(models,train[idx].to(device),args,True)
        else:
            zg = None
        gen.append(generate(models,base,args,zg).cpu())
    saved = dict(true_x=x.numpy(), test_y=y_np, generated=torch.cat(gen).numpy())
    if has_latent:
        saved.update(train_z=tz.numpy(),train_y=ty_np,test_z=zmean.numpy(),
                     posterior_pred=torch.cat(pp).numpy(),shuffled_pred=torch.cat(sp).numpy())
    results = evaluate_arrays(**saved,dataset=args.dataset,seed=args.split_seed,
                              n_clusters=arrays['metadata'].get('n_classes'))
    results.update(evaluation_split=split,ode_solver='Heun',ode_steps=args.ode_steps,
                   nfe_per_flow_sample=2*args.ode_steps if args.method!='vae' else 0,
                   marginal_source='training_aggregated_posterior' if args.method.startswith('sfa') or args.method=='paper_only'
                   else ('standard_normal_prior' if args.method=='vae' else 'learned_flow'),
                   latent_fm_protocol='frozen_AE_then_FM; no joint fine tuning' if args.method=='latent_fm' else None)
    if has_latent:
        results['train_latent_dimension_variances'] = tz.var(0).tolist()
        if args.dataset=='pinwheel':
            from sklearn.linear_model import Ridge
            from sklearn.metrics import r2_score
            a = np.arctan2(tx_np[:,1],tx_np[:,0]); b = np.arctan2(x_np[:,1],x_np[:,0])
            reg = Ridge(alpha=1.).fit(tz.numpy(),np.stack([np.sin(a),np.cos(a)],1))
            results['angular_probe_r2'] = float(r2_score(np.stack([np.sin(b),np.cos(b)],1),reg.predict(zmean.numpy())))
    stem = f'eval_{split}_steps{args.ode_steps}'
    np.savez_compressed(out/(stem+'.npz'),**saved)
    write_json(out/(stem+'.json'),results)
    plot_samples(saved,args,out/(stem+'.png'))
    return results


def plot_samples(saved,args,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.special import expit
    if args.dataset=='pinwheel':
        fig,axs=plt.subplots(1,3,figsize=(12,4))
        for ax,key,title in zip(axs,['true_x','generated','posterior_pred'],['Held-out data','Marginal samples','Posterior predictive']):
            if key in saved:
                x=saved[key]; ax.scatter(x[:,0],x[:,1],s=4,alpha=.6)
            ax.set_title(title); ax.set_aspect('equal')
    elif args.dataset=='mnist':
        keys=[k for k in ['true_x','posterior_pred','shuffled_pred','generated'] if k in saved]
        fig,axs=plt.subplots(len(keys),8,figsize=(12,2*len(keys)),squeeze=False)
        for r,k in enumerate(keys):
            for c in range(8):
                if c<len(saved[k]): axs[r,c].imshow(expit(saved[k][c]).squeeze(),cmap='gray',vmin=0,vmax=1)
                axs[r,c].axis('off')
            axs[r,0].set_title(k,fontsize=8)
    else:
        return
    fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)


def main(argv=None):
    args=parser().parse_args(argv)
    if args.z_dim is None: args.z_dim=1 if args.dataset=='pinwheel' else 32
    if args.epochs<1 or args.ode_steps<1 or args.eval_samples<2: raise ValueError('Positive epochs/steps and >=2 evaluation samples required.')
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable.')
    device=torch.device('cuda' if args.device!='cpu' and torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    out=Path(args.run_dir) if args.run_dir else Path(args.output)/args.participant/f'{args.dataset}_{args.method}_{args.profile}_s{args.seed}_{args.tag}'
    out.mkdir(parents=True,exist_ok=True)
    config_path=out/'config.json'
    if config_path.exists():
        if not (args.resume or args.eval_only): raise FileExistsError(f'{out} exists: use --resume or a new --tag; never overwrite a run.')
        old=json.loads(config_path.read_text())
        mutable={'resume','eval_only','evaluate_test','ode_steps','eval_samples','grad_repeats','device','run_dir','output'}
        for k,v in old.items():
            if k not in mutable and getattr(args,k)!=v: raise ValueError(f'Resume config mismatch: {k}. Reuse exact command or a new directory.')
    else:
        if args.resume or args.eval_only: raise FileNotFoundError('No run config to resume/evaluate.')
        write_json(config_path,vars(args))
    seed_all(args.seed)
    data=load_arrays(args.dataset,data_root=args.data_root,split_seed=args.split_seed,profile=args.profile,
                     rna_file=args.rna_file,train_limit=args.train_limit,val_limit=args.val_limit,test_limit=args.test_limit,
                     resolution=args.resolution)
    preserve_manifest(out/'data_manifest.json',data['metadata'])
    if not (out/'split_indices.npz').exists():
        np.savez_compressed(out/'split_indices.npz',**data['indices'])
    train=torch.as_tensor(data['train'][0],dtype=torch.float32)
    val=torch.as_tensor(data['val'][0],dtype=torch.float32)
    model=build(args,tuple(train.shape[1:])).to(device)
    if args.method=='latent_fm':
        model.register_buffer('z_mean',torch.zeros(args.z_dim,device=device))
        model.register_buffer('z_std',torch.ones(args.z_dim,device=device))
    metadata=provenance(); metadata.update(parameter_count=sum(p.numel() for p in model.parameters()),
                                         device=str(device),time_convention='author: data0→noise1; paper_only: noise0→data1')
    metadata['loss_reduction']='sum_observation_coordinates' if args.method=='paper_only' else 'mean_observation_coordinates'
    metadata['selection_score']='fixed-seed K16 squared averaged velocity + same latent regularizer' if args.method.startswith('sfa') else 'fixed-seed own validation objective'
    preserve_manifest(out/'environment.json',metadata,
                      keys=['source_sha256','author_revision','author_source_sha256','parameter_count'])
    write_json(out/('evaluation_environment.json' if args.eval_only else 'latest_environment.json'),metadata)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    loader_gen=torch.Generator().manual_seed(args.seed+100)
    bridge_gen=torch.Generator(device=device).manual_seed(args.seed+101)
    loader=DataLoader(TensorDataset(train),batch_size=args.batch_size,shuffle=True,num_workers=0,generator=loader_gen)
    val_loader=DataLoader(TensorDataset(val),batch_size=args.batch_size,shuffle=False,num_workers=0)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    total=args.epochs*(2 if args.method=='latent_fm' else 1)
    start=0; best=float('inf'); stage_prev=0; history=[]; elapsed=0
    if args.resume and (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(ck['model'])
        if args.method=='latent_fm' and ck['stage']==1:
            for p in model['ae'].parameters(): p.requires_grad_(False)
            optimizer=torch.optim.Adam(model['fm'].parameters(),lr=args.lr,weight_decay=args.weight_decay)
        optimizer.load_state_dict(ck['optimizer'])
        start=ck['epoch']+1; best=ck['best']; stage_prev=ck['stage']; history=ck['history']; elapsed=ck['elapsed']
        restore_rng(ck['rng'])
        loader_gen.set_state(ck['loader_rng'].cpu());bridge_gen.set_state(ck['bridge_rng'].cpu())
        if args.method=='latent_fm' and args.epochs!=json.loads(config_path.read_text())['epochs']:
            raise ValueError('Do not change latent_fm per-stage epochs during resume; stage schedule is fixed.')
    if not args.eval_only:
        for epoch in range(start,total):
            stage=int(args.method=='latent_fm' and epoch>=args.epochs)
            if stage and not stage_prev:
                ck=torch.load(out/'best.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model'])
                latent_statistics(model,train,args,device)
                for p in model['ae'].parameters():
                    p.requires_grad_(False);p.grad=None
                optimizer=torch.optim.Adam(model['fm'].parameters(),lr=args.lr,weight_decay=args.weight_decay)
                best=float('inf')
            elif stage and stage_prev and epoch==start:
                # Recreate optimizer parameter groups before loading a phase-two checkpoint.
                for p in model['ae'].parameters(): p.requires_grad_(False)
            stage_prev=stage; tic=time.perf_counter(); model.train(); losses=[]; norms=[]
            for (x,) in loader:
                x=x.to(device); optimizer.zero_grad(set_to_none=True)
                loss=objective(model,x,args,stage,bridge_generator=bridge_gen)
                if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite objective at epoch {epoch}.')
                loss.backward()
                norm=nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],args.clip_grad,error_if_nonfinite=True)
                optimizer.step(); losses.append((loss.item(),len(x))); norms.append(float(norm))
            state=rng_state(); seed_all(args.split_seed+1234); model.eval(); values=[]
            with torch.no_grad():
                for (x,) in val_loader:
                    x=x.to(device); values.append((objective(model,x,args,stage,True).item(),len(x)))
            restore_rng(state)
            score=sum(a*b for a,b in values)/sum(b for _,b in values)
            if not math.isfinite(score): raise FloatingPointError('Nonfinite validation objective.')
            if device.type=='cuda': torch.cuda.synchronize()
            duration=time.perf_counter()-tic; elapsed+=duration
            record=dict(epoch=epoch,stage=stage,train_loss=sum(a*b for a,b in losses)/sum(b for _,b in losses),
                        val_loss=score,epoch_seconds=duration,gradient_norm_preclip_mean=float(np.mean(norms)),
                        negative_minibatch_fraction=float(np.mean([a<0 for a,_ in losses])))
            history.append(record); print(json.dumps(record),flush=True)
            improved=score<best
            if improved: best=score
            ck=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,
                    stage=stage,rng=rng_state(),loader_rng=loader_gen.get_state(),bridge_rng=bridge_gen.get_state(),history=history,elapsed=elapsed)
            atomic_save(out/'last.pt',ck)
            if improved: atomic_save(out/'best.pt',ck)
            write_json(out/'history.json',history)
    best_ck=torch.load(out/'best.pt',map_location=device,weights_only=False); model.load_state_dict(best_ck['model']); model.eval()
    seed_all(args.split_seed+5678)
    diag=gradient_diagnostic(model,val[:min(64,len(val))].to(device),args)
    write_json(out/'gradient_diagnostic.json',diag)
    tic=time.perf_counter(); scores=evaluate(model,data,args,device,out)
    if device.type=='cuda': torch.cuda.synchronize()
    last_ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    summary=dict(status='complete',best_epoch=best_ck['epoch'],training_seconds=last_ck['elapsed'],
                 best_checkpoint_training_seconds=best_ck['elapsed'],
                 evaluation_seconds=time.perf_counter()-tic,
                 peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
                 metrics=scores,parameter_count=metadata['parameter_count'])
    write_json(out/'summary.json',summary); print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
