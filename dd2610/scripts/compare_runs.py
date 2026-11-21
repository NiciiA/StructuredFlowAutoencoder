"""Collect results without pooling different protocols or treating duplicate seeds as new evidence."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()[:12]


def collect(roots):
    rows=[];seen=set()
    exclude={'participant','seed','tag','run_dir','output','output_root','resume','eval_only','evaluate_test',
             'device','repo','data_root','rna_file','eval_samples','ode_steps','ode_method','eval_batch_size',
             'probe_train_samples','grad_repeats','method'}
    for root in roots:
        for f in sorted(Path(root).rglob('*config.json')):
            if f.resolve() in seen: continue
            seen.add(f.resolve());run=f.parent;cfg=json.loads(f.read_text())
            official=f.name=='resolved_config.json'
            if f.name not in ('config.json','resolved_config.json'):continue
            files=sorted(run.glob('metrics_*.json' if official else 'eval_*_steps*.json'))
            if not files:continue
            manifest=json.loads((run/'data_manifest.json').read_text())
            env=json.loads((run/'environment.json').read_text())
            meta_file=run/('run_metadata.json' if official else 'summary.json')
            meta=json.loads(meta_file.read_text()) if meta_file.exists() else {}
            config_fingerprint=digest({k:v for k,v in cfg.items() if k not in exclude})
            for ff in files:
                scores=json.loads(ff.read_text())
                protocol=digest(dict(data=manifest.get('data_sha256'),labels=manifest.get('label_sha256'),config=config_fingerprint,
                    n=scores.get('n_eval',scores.get('n_evaluation')),steps=scores.get('ode_steps'),
                    solver=scores.get('ode_solver',scores.get('ode_method')),
                    source=scores.get('marginal_source',scores.get('generation_latent_source')),
                    code=env.get('source_sha256'),upstream=env.get('author_source_sha256')))
                base=dict(participant=cfg.get('participant'),dataset=cfg.get('dataset'),
                          method=('official_'+cfg['dataset']) if official else cfg['method'],
                          profile=cfg.get('profile'),seed=cfg.get('seed'),split=scores.get('evaluation_split'),
                          protocol=protocol,config_id=config_fingerprint,run=str(run),metric_file=str(ff),
                          training_seconds=meta.get('training_seconds'),parameters=meta.get('parameter_count',meta.get('parameters')))
                for k,v in scores.items():
                    if isinstance(v,(float,int)) and not isinstance(v,bool):
                        rows.append(dict(base,metric=k,value=v))
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots',nargs='+');p.add_argument('--output',default='comparison')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    df=collect(args.roots)
    if df.empty:raise SystemExit('No completed evaluation JSON files found.')
    df.to_csv(out/'per_run.csv',index=False)
    group=['participant','dataset','method','profile','split','protocol','metric']
    summaries=[]
    for key,g in df.groupby(group,dropna=False):
        # Two runs with identical seed/protocol are repeats, not additional independent seeds.
        unique=g.groupby('seed').value.mean();n=len(unique);sd=unique.std(ddof=1) if n>1 else np.nan
        half=float(student_t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else np.nan
        row=dict(zip(group,key));row.update(n_independent_seeds=n,n_runs=len(g),mean=unique.mean(),sd=sd,
                                           ci95_low=unique.mean()-half,ci95_high=unique.mean()+half)
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(out/'seed_summary.csv',index=False)
    paired=[]
    for key,g in df.groupby(['dataset','method','profile','split','protocol','metric','seed'],dropna=False):
        # Explicit comparison of members at the same seed, never pooled into seed CI.
        member=g.groupby('participant').value.mean()
        for a,b in itertools.combinations(member.index,2):
            row=dict(zip(['dataset','method','profile','split','protocol','metric','seed'],key))
            row.update(participant_a=a,participant_b=b,value_a=member[a],value_b=member[b],
                       difference_b_minus_a=member[b]-member[a],absolute_difference=abs(member[b]-member[a]))
            paired.append(row)
    pd.DataFrame(paired).to_csv(out/'member_comparison.csv',index=False)
    (out/'README.txt').write_text('per_run.csv: all measured values.\nseed_summary.csv: separate participant/protocol; n=1 has no CI.\n'
        'member_comparison.csv: same seed, data, config, solver, and code only.\n'
        'No member pairs usually means protocol/config/source hashes differ; inspect per_run first.\n'
        'Three-seed t intervals are descriptive and fragile; retain per-seed values.\n'
        'Do not compare likelihood to training loss or aggregate test and validation.\n')
    print('Saved',out.resolve(),'; evaluated metric rows:',len(df),'; matched member rows:',len(paired))


if __name__=='__main__':main()
