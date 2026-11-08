"""Independent-process check against exact pre-change source (4317bf5)."""
from pathlib import Path
import os,sys,subprocess,json,csv
O=Path(__file__).resolve().parents[1];script=O/'diagnostics/experiment.py';rows=[]
roots={'old':'/home/xieminhui_pubuse/hytorch-before-candidate-review','new':'/home/xieminhui_pubuse/hytorch'}
for rep in range(10):
 result={}
 for label in (['old','new'] if rep%2==0 else ['new','old']):
  env=dict(os.environ,HYTORCH_REVIEW_ROOT=roots[label]);dest=O/'raw'/f'process_{rep}_{label}.json'
  mode='baseline' if label=='old' else 'relu_prune'
  with (O/'raw'/f'process_{rep}_{label}.log').open('w') as f:
   subprocess.run([sys.executable,str(script),'--worker',mode,'--output',str(dest)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=roots[label])
  r=json.loads(dest.read_text());result[label]=r;r.update(source=label,pair=rep);rows.append(r)
 for key in ['weights_sha256','gradients_sha256','losses_sha256']:assert result['old'][key]==result['new'][key],(rep,key)
 with (O/'ORIGINAL_COMPARISON.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('Pair',rep,'old',result['old']['wall_s'],'new',result['new']['wall_s'],flush=True)
