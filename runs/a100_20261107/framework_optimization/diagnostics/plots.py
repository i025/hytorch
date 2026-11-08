from pathlib import Path
import csv,json,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parents[1]
def read(n):return list(csv.DictReader((O/n).open()))
s=json.loads((O/'SUMMARY.json').read_text());ops=read('OPERATOR_COMPARISON.csv')
plt.rcParams.update({'font.size':10,'figure.dpi':140,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
colors=['#4169a1','#15a27b','#c48932']
def save(fig,name,note):
 fig.text(.02,.015,note,fontsize=8,color='#444');fig.tight_layout(rect=[0,.09,1,.96])
 for ext in ['png','svg']:fig.savefig(O/'figures'/f'{name}.{ext}',bbox_inches='tight')
 plt.close(fig)
# 1: descriptive median ratios, explicitly not frequency weighted model cost.
cats=['ConvForward','dWeight','dInput','dBias','FullBackward','GEMM','Activation','Pooling','PoolBackward','Linear','SGD'];gv=[];wv=[]
for cat in cats:
 gr=[];wr=[]
 for case in dict.fromkeys(r['case'] for r in ops if r['category']==cat):
  d={r['implementation']:r for r in ops if r['case']==case};h=d.get('HyTorch',d.get('shared'));t=d['PyTorch'];gr.append(float(h['kernel_us'])/float(t['kernel_us']));wr.append(float(h['sync_us_median'])/float(t['sync_us_median']))
 gv.append(statistics.median(gr));wv.append(statistics.median(wr))
fig,ax=plt.subplots(figsize=(12,5));x=np.arange(len(cats));ax.bar(x-.19,gv,.38,label='GPU kernel median ratio',color=colors[0]);ax.bar(x+.19,wv,.38,label='Synchronized full-call median ratio',color=colors[1]);ax.axhline(1,color='gray',ls='--');ax.set_xticks(x,cats,rotation=30,ha='right');ax.set_ylabel('HyTorch / PyTorch time (lower is better)');ax.set_ylim(bottom=0);ax.legend();ax.set_title('Operator comparison: kernel speed is not full-call speed')
save(fig,'01_operators','A100-PCIE-40GB | Donkey + generic cases, N=1–4 | 15 balanced repeats; medians across shapes (not model-weighted)\nSources: OPERATOR_COMPARISON.csv, OPERATOR_KERNEL_SAMPLES.csv; pure kernels via CUPTI, full calls without profiler')
# 2
fig,axes=plt.subplots(2,3,figsize=(13,7));maps=[('forward','Forward'),('dweight','dWeight'),('dinput','dInput'),('dbias','dBias'),('backward','Full backward')]
for ax,(op,title) in zip(axes.flat,maps):
 for j,label in enumerate(['HyTorch','PyTorch']):
  vals=[float(next(r for r in ops if r['case']==f'conv{i}/{op}' and r['implementation']==label)['kernel_us']) for i in range(1,6)]
  ax.bar(np.arange(5)+(j-.5)*.35,vals,.35,label=label,color=colors[j])
 ax.set_xticks(range(5),[f'C{i}' for i in range(1,6)]);ax.set_title(title);ax.set_ylabel('GPU kernel sum per call (us)');ax.set_ylim(bottom=0)
axes.flat[0].legend();axes.flat[-1].axis('off');axes.flat[-1].text(.03,.8,'Isolated dInput: input requires grad.\nFull backward: x, W and b require grad.\n\nThe real Torch network omits input-image\ngradients in conv1; do not sum these\nisolated timings to predict training.',va='top',wrap=True)
save(fig,'02_conv_layers','A100-PCIE-40GB | DonkeyNet batch=4 | CUPTI sum per call, median of 15 alternating measurements\nSources: CONV_FORWARD.csv, CONV_BACKWARD.csv; TF32 off, normal cuDNN; separate gradient requirements')
# 3 frozen stage A phase timeline
rr=read('stage_a/TRAINING_HOTSPOTS.csv');fig,axes=plt.subplots(1,2,figsize=(11,4.8));ps=['forward','backward','update']
for ax,col,title in zip(axes,['gpu_kernel_us_per_step','cpu_range_elapsed_us_per_step'],['GPU kernel cumulative time','CPU phase elapsed (includes CUDA API / waiting)']):
 for j,m in enumerate(['B0','torch']):
  vals=[float(next(r for r in rr if r['mode']==m and r['phase']==p)[col])/1000 for p in ps];ax.bar(np.arange(3)+(j-.5)*.35,vals,.35,color=colors[j],label=m)
 ax.set_xticks(range(3),ps);ax.set_ylabel('ms / step');ax.set_title(title);ax.set_ylim(bottom=0);ax.legend()
save(fig,'03_gpu_cpu','A100-PCIE-40GB | DonkeyNet batch=4 | Independent 10-step Nsight traces, frozen stage A\nSource: stage_a/TRAINING_HOTSPOTS.csv. CPU and GPU overlap; these panels must NOT be added together.')
# 4 main
fig,ax=plt.subplots(figsize=(8,4.8));modes=['B0','F1','torch'];vals=[s[m]['mean_s'] for m in modes];sd=[s[m]['sd_s'] for m in modes];bars=ax.bar(modes,vals,yerr=sd,capsize=5,color=colors)
for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.02,f'{v:.6f} s',ha='center')
ax.set_ylim(0,max(vals)*1.2);ax.set_ylabel('Training wall time (s)');ax.set_title(f"F1: {s['paired']['wall_reduction_pct']:.2f}% lower training wall time")
save(fig,'04_training','A100-PCIE-40GB | DonkeyNet 256 samples, batch=4, 192 steps | 12 balanced rounds, no profiler\nSource: TRAINING_ABLATION.csv. Bars=mean; error bars=sample SD. F1=immutable shape validation cache.')
# 5 ablation one actual candidate only
fig,axes=plt.subplots(1,2,figsize=(11,4.8));pairs=read('PAIRED.csv');d=[float(r['saved_s'])*1000 for r in pairs]
axes[0].bar(range(1,13),d,color=colors[1]);axes[0].axhline(0,color='gray');axes[0].set_xlabel('Balanced round');axes[0].set_ylabel('B0 minus F1 wall time (ms)');axes[0].set_ylim(bottom=0);axes[0].set_title('12/12 paired wins')
p=s['paired'];mid=p['delta_mean_s']*1000;lo,hi=[x*1000 for x in p['ci95_s']];axes[1].errorbar([mid],[0],xerr=[[mid-lo],[hi-mid]],fmt='o',capsize=8,color=colors[1]);axes[1].set_xlim(0,hi*1.2);axes[1].set_yticks([0],['F1']);axes[1].set_xlabel('Mean saving and 95% paired t interval (ms)');axes[1].set_title('One accepted framework optimization')
save(fig,'05_ablation','A100-PCIE-40GB | DonkeyNet batch=4, 192 steps | Full 3-mode permutations repeated twice\nSources: PAIRED.csv, SUMMARY.json. No F2 implemented; final configuration is F1, not a claimed two-way combination.')
# 6 allocation / memory
r=read('MEMORY_COMPARISON.csv');fig,axes=plt.subplots(1,2,figsize=(11,4.8));batches=[1,4,16]
for j,mode in enumerate(['B0','F1']):
 vals=[statistics.mean(float(x['peak_active_bytes']) for x in r if x['mode']==mode and int(x['batch'])==b)/2**20 for b in batches]
 calls=[statistics.mean(float(x['malloc_calls']) for x in r if x['mode']==mode and int(x['batch'])==b) for b in batches]
 for ax,v in zip(axes,[vals,calls]):ax.bar(np.arange(3)+(j-.5)*.35,v,.35,color=colors[j],label=mode);ax.set_xticks(range(3),batches);ax.set_xlabel('Batch size');ax.set_ylim(bottom=0);ax.legend()
axes[0].set_ylabel('Peak active CuPy pool (MiB)');axes[1].set_ylabel('CuPy allocation requests / 768 samples');axes[0].set_title('No GPU workspace increase');axes[1].set_title('GPU allocation count unchanged')
save(fig,'06_memory','A100-PCIE-40GB | DonkeyNet 256 samples x 3 epochs | 3 runs per batch/mode, independent MemoryHook\nSource: MEMORY_COMPARISON.csv. Pool metrics exclude CUDA library/private allocations; cached metadata lives on CPU.')
print('Saved 6 PNG + 6 SVG figures')
