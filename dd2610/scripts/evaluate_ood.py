"""Frozen benchmark MNIST encoder on EMNIST digits + first ten letters (20 classes)."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_official_data import load_arrays
from scripts.run_experiment import build,encode,generate,plot_samples
from scripts.metrics import paired_ssim,pixels,vendi_cosine
from scripts.common import seed_all,write_json,provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--samples',type=int,default=1000)
    p.add_argument('--ode-steps',type=int,default=100)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args(); out=Path(a.run_dir)
    cfg=json.loads((out/'config.json').read_text()); args=argparse.Namespace(**cfg)
    if args.dataset!='mnist' or args.method=='fm': raise ValueError('Requires a frozen MNIST latent model from run_experiment.')
    args.ode_steps=a.ode_steps
    device=torch.device('cuda' if a.device!='cpu' and torch.cuda.is_available() else 'cpu')
    model=build(args,(1,args.resolution,args.resolution)).to(device)
    if args.method=='latent_fm':
        model.register_buffer('z_mean',torch.zeros(args.z_dim,device=device));model.register_buffer('z_std',torch.ones(args.z_dim,device=device))
    ck=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model']);model.eval()
    data=load_arrays('emnist',data_root=args.data_root,split_seed=args.split_seed,profile='full',test_limit=a.samples,resolution=args.resolution)
    x,y=data['test'];seed_all(args.split_seed+9988);zs=[];pp=[]
    with torch.no_grad():
        for start in range(0,len(x),args.batch_size):
            b=torch.from_numpy(x[start:start+args.batch_size]).to(device)
            zs.append(encode(model,b,args).cpu().numpy())
            z=encode(model,b,args,True)
            pp.append(generate(model,torch.randn_like(b),args,z).cpu().numpy())
    z=np.concatenate(zs);pred=np.concatenate(pp)
    # A transductive clustering diagnostic: labels only used for scoring, never k-means fitting.
    zz=StandardScaler().fit_transform(z)
    cl=KMeans(n_clusters=20,n_init=10,random_state=args.split_seed).fit_predict(zz)
    scores=dict(n=len(y),n_classes=20,nmi_ood_testfit=float(normalized_mutual_info_score(y,cl)),
                ari_ood_testfit=float(adjusted_rand_score(y,cl)),posterior_predictive_ssim=paired_ssim(pixels(x),pixels(pred)),
                vendi_posterior_predictive_cosine=vendi_cosine(pixels(pred)),
                protocol='EMNIST test-only digits0..9 + lettersA..J; unsupervised kmeans fit on OOD; no OOD training or tuning',
                class_counts={str(k):int((y==k).sum()) for k in np.unique(y)},ode_steps=a.ode_steps)
    stem=f'ood_emnist_steps{a.ode_steps}'
    write_json(out/(stem+'.json'),scores);write_json(out/(stem+'_data.json'),data['metadata'])
    write_json(out/(stem+'_environment.json'),provenance())
    np.savez_compressed(out/(stem+'.npz'),true_x=x,y=y,z=z,posterior_pred=pred)
    plot_samples(dict(true_x=x,posterior_pred=pred),args,out/(stem+'.png'))
    print(json.dumps(scores,indent=2))


if __name__=='__main__':main()
