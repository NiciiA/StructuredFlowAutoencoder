"""Verify the paper-only implementation before importing or comparing it."""
import hashlib
import json
from pathlib import Path


def verify():
    root=Path(__file__).resolve().parents[1]
    folder=root/'paper_only'
    data=json.loads((folder/'FROZEN_MANIFEST.json').read_text())
    for name,expected in data['files'].items():
        path=folder/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Paper-only frozen artifact changed or missing: '+name)
    paper=root/'references/12981_Structured_Flow_Autoenco.txt'
    if paper.exists() and hashlib.sha256(paper.read_bytes()).hexdigest()!=data['input_paper']['sha256']:
        raise ValueError('Input paper extraction changed.')
    return data


if __name__=='__main__':
    print('Frozen files verified:',len(verify()['files']))
