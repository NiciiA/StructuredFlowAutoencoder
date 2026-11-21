"""Print or execute bounded experiment batches. Pilot is for profiling, not final results."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def jobs(args):
    base=[sys.executable,'-m']
    all_jobs=[]
    for seed in args.seeds.split(','):
        if args.phase in ('pilot','breadth'):
            for dataset in ['pinwheel','mnist','mnist_mixture','hvg','pendulum']:
                cmd=base+['scripts.run_official','--dataset',dataset,'--profile','pilot',
                          '--participant',args.participant,'--seed',seed,'--output-root',str(Path(args.output)/'official'),
                          '--run-id',f'{args.participant}_{args.phase}_{dataset}_s{seed}',
                          '--epochs',str(args.epochs or (3 if args.phase=='pilot' else 50))]
                if args.rna_file and dataset=='hvg':cmd+=['--rna-file',args.rna_file]
                all_jobs.append(dict(name=f'official_{dataset}_s{seed}',argv=cmd,
                                     blocked=('RNA preparation required; use download_kang.py and prepare_official_data.py') if dataset=='hvg' and not args.rna_file else None))
        if args.phase in ('pilot','core','mnist_ablation'):
            datasets=['pinwheel','mnist'] if args.phase!='mnist_ablation' else ['mnist']
            for dataset in datasets:
                methods=['sfa_k1','sfa_k4','sfa_cross','fm','vae','latent_fm','paper_only'] if dataset=='pinwheel' else ['sfa_k1','vae','latent_fm','paper_only']
                if args.phase=='mnist_ablation':methods=['sfa_k1','sfa_k4','sfa_cross']
                for method in methods:
                    cmd=base+['scripts.run_experiment','--dataset',dataset,'--method',method,
                              '--profile','pilot' if args.phase=='pilot' else 'full',
                              '--participant',args.participant,'--seed',seed,'--output',str(Path(args.output)/'benchmarks'),
                              '--tag',args.phase,'--epochs',str(args.epochs or (3 if args.phase=='pilot' else (100 if dataset=='pinwheel' else 50)))]
                    if args.phase!='pilot':cmd+=['--eval-samples','1000','--ode-steps','100']
                    all_jobs.append(dict(name=f'{dataset}_{method}_s{seed}',argv=cmd,blocked=None))
    if args.resume:
        for job in all_jobs:job['argv'].append('--resume')
    return all_jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--participant',required=True)
    p.add_argument('--phase',choices=['pilot','breadth','core','mnist_ablation'],default='pilot')
    p.add_argument('--seeds',default='0',help='Use 0 first, then 0,1,2 after profiling.')
    p.add_argument('--epochs',type=int)
    p.add_argument('--rna-file')
    p.add_argument('--output',default='results')
    p.add_argument('--only',help='Optional comma-separated job names from printed plan')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();plan=jobs(args)
    if args.only:
        wanted=set(args.only.split(','));plan=[j for j in plan if j['name'] in wanted]
        if {j['name'] for j in plan}!=wanted:raise ValueError('Unknown --only job; print the full plan first.')
    for job in plan:
        print(job['name'],':',('BLOCKED '+job['blocked']) if job['blocked'] else shlex.join(job['argv']))
    if not args.execute:
        print('\nPreview only. Add --execute to run selected jobs.');return
    folder=Path(args.output)/'campaigns'/f'{args.participant}_{args.phase}';folder.mkdir(parents=True,exist_ok=True)
    reports=[]
    for job in plan:
        row=dict(job)
        if job['blocked']:row['status']='blocked'
        else:
            with (folder/(job['name']+'.log')).open('a') as log:
                result=subprocess.run(job['argv'],stdout=log,stderr=subprocess.STDOUT)
            row.update(status='complete' if result.returncode==0 else 'failed',exit_code=result.returncode)
        reports.append(row);(folder/'status.json').write_text(json.dumps(reports,indent=2))
        print(row['name'],row['status'],flush=True)
    if any(r['status']!='complete' for r in reports):raise SystemExit('Some selected jobs failed or require input; see campaign logs. Completed jobs were preserved.')


if __name__=='__main__':main()
