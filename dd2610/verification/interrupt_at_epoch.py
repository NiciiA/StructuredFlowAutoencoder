import sys
from scripts import run_experiment as r
original=r.atomic_save
def save(path,obj):
 original(path,obj)
 if str(path).endswith('best.pt') and obj['epoch']==2:
  raise SystemExit(77)
r.atomic_save=save
r.main(sys.argv[1:])
