"""Publish actual per-parameter comparisons, losses and phase timings; leave NPZ local."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
runs=sorted(f.parent for f in a.input.glob('*/summary.json'))
with np.load(a.input/'cupy_r1/final_weights.npz') as d: expected={k:d[k] for k in d.files}
with np.load(a.input/'cupy_r1/initial_weights.npz') as d: expected_initial={k:d[k] for k in d.files}
params=[];epochs=[];phases=[]
fig,ax=plt.subplots(figsize=(10,5))
for run in runs:
    cfg=json.loads((run/'config.json').read_text());s=json.loads((run/'summary.json').read_text())
    initial=np.load(run/'initial_weights.npz');final=np.load(run/'final_weights.npz')
    for key,ref in expected.items():
        np.testing.assert_array_equal(initial[key],expected_initial[key]);delta=final[key].astype(np.float64)-ref.astype(np.float64)
        params.append(dict(run=run.name,parameter=key,initial_max_abs_error=float(np.max(np.abs(initial[key]-expected_initial[key]))),final_max_abs_error=float(np.max(np.abs(delta))),final_relative_l2_error=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(ref.astype(np.float64).ravel()),1e-30)),atol=1e-4,rtol=1e-3,correct=bool(np.allclose(final[key],ref,atol=1e-4,rtol=1e-3))))
    initial.close();final.close()
    for e in s['epochs']: epochs.append(dict(run=run.name,**e))
    losses=[json.loads(line) for line in (run/'losses.jsonl').read_text().splitlines()]
    ax.plot(range(1,len(losses)+1),[r['loss'] for r in losses],'.-',alpha=.65,label=run.name)
    steps=[json.loads(line) for line in (run/'steps.jsonl').read_text().splitlines()]
    if cfg['mode']=='phases':
        row=dict(run=run.name)
        for phase in ('data','transfer','forward','backward','update','other','step'):
            row[phase+'_mean_ms']=float(np.mean([r[phase+'_s'] for r in steps])*1000)
        phases.append(row)
for name,rows in [('parameter-errors.csv',params),('epoch-results.csv',epochs),('phase-means.csv',phases)]:
    if rows:
        with (a.output/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
ax.set(xlabel='Training step',ylabel='Actual MSE loss',title=cfg['model']+' / SYNTHETIC / '+cfg['mode']);ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=3)
fig.tight_layout();fig.savefig(a.output/'loss.png',dpi=180);plt.close(fig)
if phases:
    fig,ax=plt.subplots(figsize=(10,5));left=np.zeros(len(phases))
    for key in ('data','transfer','forward','backward','update','other'):
        values=np.array([r[key+'_mean_ms'] for r in phases]);ax.barh([r['run'] for r in phases],values,left=left,label=key);left+=values
    ax.set(xlabel='Mean synchronized phase time (ms)',title=cfg['model']+' separate PHASES experiment');ax.legend(ncol=3);fig.tight_layout();fig.savefig(a.output/'phases.png',dpi=160)
