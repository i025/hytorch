import csv,json,statistics,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path.cwd()))
from scripts.report_cuda import compare_training
root=Path(__file__).resolve().parent
out=Path('outputs/a100_20261106');out.mkdir(parents=True,exist_ok=True)
paths=[root/f'{g}_{r}' for g in ['native_baseline','native_optimized','cupy','torch'] for r in range(1,4)]
# Validate matched data, initialization, optimizer, precision and run conditions.
compare_training(paths)
with np.load(root/'native_baseline_1/final_weights.npz') as f: reference={k:f[k] for k in f.files}
rows=[]
for group in ['native_baseline','native_optimized','cupy','torch']:
 times=[];maxerr=0.;close=True;equal=True;losses=[];step=[]
 for r in range(1,4):
  path=root/f'{group}_{r}';s=json.loads((path/'summary.json').read_text())
  times.append(sum(e['train_wall_s'] for e in s['epochs']));losses.append(s['epochs'][-1]['train_loss']);step.append(s['timing']['mean_step_s']*1000)
  with np.load(path/'final_weights.npz') as f:
   for k,v in reference.items():
    maxerr=max(maxerr,float(np.max(np.abs(f[k]-v))))
    close &= bool(np.allclose(f[k],v,atol=1e-4,rtol=1e-3));equal &= bool(np.array_equal(f[k],v))
 rows.append(dict(implementation=group,repeats=3,mean_train_wall_s=statistics.mean(times),std_train_wall_s=statistics.stdev(times),samples_per_second=768/statistics.mean(times),mean_step_ms=statistics.mean(step),weight_max_abs_difference=maxerr,weights_close=close,weights_bitwise_equal=equal,final_epoch_loss=statistics.mean(losses),wall_repeats_s=times))
for r in rows:r['speedup_vs_original']=rows[0]['mean_train_wall_s']/r['mean_train_wall_s']
(out/'training_summary.json').write_text(json.dumps(rows,indent=2))
with (out/'training_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
gemm=list(csv.DictReader((root/'gemm_paired/comparison.csv').open()))
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
selected=[r for r in gemm if r['tile']=='32'];x=np.arange(len(selected));width=.35
axes[0].bar(x-width/2,[float(r['baseline_median_ms']) for r in selected],width,label='Original shared GEMM')
axes[0].bar(x+width/2,[float(r['candidate_median_ms']) for r in selected],width,label='Register-tiled GEMM')
axes[0].set_xticks(x,[r['size'] for r in selected]);axes[0].set(ylabel='Median CUDA event time (ms)',xlabel='Square matrix size',title='A100 FP32 GEMM, K tile 32')
axes[0].legend(fontsize=8)
for i,r in enumerate(selected):axes[0].text(i,float(r['baseline_median_ms'])+.6,f"{float(r['speedup']):.2f}x",ha='center')
axes[1].bar(['Original','Optimized','CuPy','PyTorch'],[r['samples_per_second'] for r in rows],color=['#6c7f91','#2b9a88','#7194d2','#b781cf'])
axes[1].set(ylabel='Training samples / second',title='Synthetic DonkeyNet, batch 4, 3 runs')
fig.suptitle('Measured results: GEMM faster; no end-to-end training gain')
fig.text(.5,.01,'GEMM outputs bitwise preserved; original 2048/4096 strict NumPy gate still fails. Synthetic data only.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(out/'performance.png',dpi=180)
print(json.dumps(rows,indent=2))
assert all(r['weights_close'] for r in rows)
