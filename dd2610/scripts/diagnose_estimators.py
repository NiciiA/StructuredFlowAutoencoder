"""Compare posterior gradient variance of all estimators at one fixed SFA checkpoint."""
import argparse
import copy
import json
from pathlib import Path
import sys
import torch
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_experiment import build,gradient_diagnostic
from scripts.prepare_official_data import load_arrays
from scripts.common import seed_all,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--repeats',type=int,default=32);a=p.parse_args();out=Path(a.run_dir)
    cfg=json.loads((out/'config.json').read_text());args=argparse.Namespace(**cfg)
    if not args.method.startswith('sfa'):raise ValueError('Choose an SFA benchmark checkpoint.')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=load_arrays(args.dataset,data_root=args.data_root,split_seed=args.split_seed,profile=args.profile,
                     rna_file=args.rna_file,train_limit=args.train_limit,val_limit=args.val_limit,test_limit=args.test_limit,resolution=args.resolution)
    model=build(args,data['train'][0].shape[1:]).to(device)
    ck=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model']);model.eval()
    args.grad_repeats=a.repeats;x=torch.from_numpy(data['val'][0][:64]).to(device)
    rows={}
    for estimator in ['k1','k4','cross']:
        local=copy.copy(args);local.method='sfa_'+estimator
        seed_all(args.split_seed+123456)
        rows[estimator]=gradient_diagnostic(model,x,local)
    result=dict(checkpoint=str(out/'best.pt'),best_epoch=ck['epoch'],protocol='One parameter setting, same validation batch/t/base noise; only posterior estimator differs',results=rows)
    write_json(out/'same_checkpoint_estimator_variance.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
