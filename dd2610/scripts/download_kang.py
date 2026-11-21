"""Get the real preprocessed Kang data linked by the primary CPA tutorial.

Source: https://cpa-tools.readthedocs.io/en/latest/tutorials/Kang.html
This is an upstream-preselected 5000-gene dataset, NOT raw all-gene data.
"""
import argparse
import hashlib
import json
from pathlib import Path

FILE_ID = '1z8gGKQ6oDoi2blCU2IVihKA38h5fORRp'
SOURCE = 'https://cpa-tools.readthedocs.io/en/latest/tutorials/Kang.html'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='data/kang_normalized_hvg.h5ad')
    args=p.parse_args(); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():
        import gdown
        temp=out.with_suffix('.download')
        result=gdown.download(id=FILE_ID,output=str(temp),quiet=False)
        if result is None: raise RuntimeError('Download unavailable; preserve the failure and get the same file manually from the source tutorial.')
        with temp.open('rb') as f:
            if f.read(8)!=b'\x89HDF\r\n\x1a\n': raise ValueError('Downloaded response is not HDF5. Do not use it as data.')
        temp.replace(out)
    import anndata as ad
    data=ad.read_h5ad(out,backed='r')
    info=dict(source_page=SOURCE,google_drive_file_id=FILE_ID,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
              shape=list(data.shape),obs_columns=list(data.obs.columns),layers=list(data.layers.keys()),
              cell_type_counts=data.obs['cell_type'].astype(str).value_counts().to_dict() if 'cell_type' in data.obs else {},
              input_state='X is documented normalized/log-transformed; counts layer is separate',
              caveat='Genes already selected upstream; cannot claim all-gene train-only feature selection.')
    data.file.close(); out.with_suffix('.source.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))


if __name__=='__main__': main()
