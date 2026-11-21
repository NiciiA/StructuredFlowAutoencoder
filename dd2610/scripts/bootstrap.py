"""Download the frozen public author checkout without overwriting a changed checkout."""
import argparse
from pathlib import Path
import subprocess

URL='https://github.com/edenx/StructuredFlowAutoencoder.git'
SHA='05e89071f6bae502eedfd3464fad4e8f75f506c1'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',default='upstream/StructuredFlowAutoencoder')
    args=p.parse_args();dest=Path(args.destination)
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--no-checkout',URL,str(dest)],check=True)
        subprocess.run(['git','-C',str(dest),'checkout','--detach',SHA],check=True)
    current=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
    if current!=SHA:raise RuntimeError('Existing checkout is a different revision. Choose a new destination; it was not changed.')
    changed=subprocess.check_output(['git','-C',str(dest),'status','--porcelain'],text=True)
    if changed.strip():raise RuntimeError('Author checkout contains edits/untracked files; keep source clean and place changes in wrapper scripts.')
    print('Verified official source:',SHA)


if __name__=='__main__':main()
