from pathlib import Path
import csv,json,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('runs/a100_20261106/final_training_optimization');F=O/'figures'
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
def read(p):return list(csv.DictReader((O/p).open()))
def save(fig,name):
 fig.tight_layout();fig.savefig(F/(name+'.svg'));fig.savefig(F/(name+'.png'),dpi=180);plt.close(fig)
s=json.loads((O/'raw/training_summary.json').read_text());modes=['B0','D','S','DS'];labels=['B0','D','S*','DS*'];colors=['#798b99','#167ca6','#b9bec3','#9fbac4']
fig,ax=plt.subplots(figsize=(8,4.5));ax.bar(labels,[s[m]['mean'] for m in modes],yerr=[s[m]['sd'] for m in modes],capsize=5,color=colors)
ax.set_ylim(0,1.05);ax.set_ylabel('Full training wall time (s)');ax.set_title('12 balanced paired rounds; error bars = sample SD')
for i,m in enumerate(modes):ax.text(i,s[m]['mean']+s[m]['sd']+.025,f"{s[m]['mean']:.6f}",ha='center')
fig.text(.5,.01,'* SGD candidate rejected: S uses legacy fallback; DS is D + fallback.',ha='center',fontsize=9);fig.subplots_adjust(bottom=.18);save(fig,'training')
pairs=read('PAIRED.csv');fig,axes=plt.subplots(1,2,figsize=(10,4.2))
for mode,color in zip(['D','S','DS'],colors[1:]):
 v=[1000*float(r['B0_minus_mode_s']) for r in pairs if r['mode']==mode];axes[0].plot(range(1,13),v,'o-',label=mode+('*' if mode!='D' else ''),color=color)
axes[0].axhline(0,color='black',lw=.8);axes[0].set_xlabel('Paired round');axes[0].set_ylabel('B0 minus candidate (ms)');axes[0].legend()
for i,m in enumerate(['D','S','DS']):
 mean=s[m]['delta_mean_s']*1000;lo,hi=np.array(s[m]['ci95_s'])*1000;axes[1].errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=5,color=colors[i+1])
axes[1].set_yticks([0,1,2],['D','S*','DS*']);axes[1].axvline(0,color='black',lw=.8);axes[1].set_xlabel('Paired mean saving, 95% t CI (ms)');axes[1].set_title('Positive means faster; no outliers removed');save(fig,'paired')
pr=read('PROFILE.csv');fig,ax=plt.subplots(figsize=(7,4.2));bottom=np.zeros(2)
for category,color in [('im2col','#728b9b'),('copy','#edaa4a'),('GEMM + reduction','#167ca6')]:
 values=[]
 for mode in ['B0','D']:
  rr=[r for r in pr if r['mode']==mode and r['repeat']=='0' and r['scope']=='GPU_kernel' and r['category']=='dweight']
  values.append(sum(float(r['calls_per_step']) for r in rr if ('im2col' in r['name'] if category=='im2col' else 'copy' in r['name'] if category=='copy' else 'im2col' not in r['name'] and 'copy' not in r['name'])))
 ax.bar(['B0','D'],values,bottom=bottom,label=category,color=color);bottom+=values
ax.set_ylabel('dWeight CUDA kernels per training step');ax.set_ylim(0,29);ax.legend();ax.set_title('5 layout-copy kernels removed; im2col still runs');save(fig,'dweight_kernels')
mem=read('MEMORY.csv');fig,ax=plt.subplots(figsize=(7,4.2))
for mode,col in [('B0',colors[0]),('D',colors[1])]:
 for field,style,title in [('peak_active_bytes','-','active'),('peak_reserved_bytes','--','reserved')]:
  vals=[max(float(r[field]) for r in mem if r['mode']==mode and int(r['batch'])==b)/2**20 for b in [1,4,16]];ax.plot([1,4,16],vals,'o'+style,label=mode+' '+title,color=col)
ax.set_ylim(bottom=0);ax.set_xticks([1,4,16]);ax.set_xlabel('Batch size');ax.set_ylabel('CuPy pool peak (MiB)');ax.set_title('Allocation-hook observation; excludes library-private memory');ax.legend();save(fig,'memory')
refs=read('REFERENCES.csv');fig,ax=plt.subplots(figsize=(8,4.2));means=[s['B0']['mean'],s['D']['mean']];sd=[s['B0']['sd'],s['D']['sd']]
for m in ['cupy','torch']:
 v=[float(r['wall_s']) for r in refs if r['mode']==m];means.append(statistics.mean(v));sd.append(statistics.stdev(v))
ax.bar(['HyTorch B0\n12 runs','HyTorch D\n12 runs','HyTorch CuPy\n5 runs','PyTorch FP32\n5 runs'],means,yerr=sd,capsize=5,color=[colors[0],colors[1],'#b59165','#7c5ca8']);ax.set_ylim(0,1.05);ax.set_ylabel('Training wall time (s)');ax.set_title('Same GPU window; framework reference groups are separate');save(fig,'frameworks')
r=read('NUMERICS.csv');real=read('TRAINING_DWEIGHT_NUMERICS.csv');fig,ax=plt.subplots(figsize=(7,4.2));xx=np.arange(3)
for offset,cmp,color in [(-.2,'A_R',colors[0]),(0,'B_R',colors[1]),(.2,'B_A','#edaa4a')]:
 counts=[]
 for group in ['legacy','holdout','train']:
  rr=real if group=='train' else [z for z in r if z['case'].startswith(group+'/')];counts.append(sum(int(z['failed'])>0 for z in rr if z['comparison']==cmp and z['gate']=='primary'))
 ax.bar(xx+offset,counts,.2,label=cmp,color=color)
ax.set_xticks(xx,['Original 90\n30 independent','Holdout 78\n26 independent','Real training 15']);ax.set_ylabel('Cases outside mixed tolerance');ax.set_ylim(0,15);ax.legend();ax.set_title('Only B vs independent R is the new numerical gate');save(fig,'numerics')
fig,ax=plt.subplots(figsize=(9,3.5));ax.axis('off')
for y,title,texts in [(.72,'Baseline',['6D im2col','gy packing','col packing','cuBLAS GEMM']),(.28,'Packed D',['native (T,D)','gy packing','cuBLAS GEMM','OIHW view'])]:
 ax.text(.01,y,title,va='center',weight='bold')
 for j,t in enumerate(texts):
  x=.24+j*.19;ax.text(x,y,t,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',fc='#e5f1f5',ec='#167ca6'))
  if j<3:ax.annotate('',xy=(x+.12,y),xytext=(x+.07,y),arrowprops=dict(arrowstyle='->'))
ax.set_title('Backward-only layout construction; no forward workspace reuse');save(fig,'data_path')
print('7 figures exported as SVG and PNG')
ops=read('FULL_WRAPPERS.csv');fig,axes=plt.subplots(1,2,figsize=(11,4.3));xx=np.arange(5)
for ax,field,title in zip(axes,['cpu_submit_us','synchronized_us'],['Full Function CPU submit','Full Function synchronized completion']):
 for off,mode,col in [(-.18,'B0',colors[0]),(.18,'D',colors[1])]:
  med=[];lo=[];hi=[]
  for i in range(1,6):
   v=[float(r[field]) for r in ops if r['case']==f'conv{i}' and r['layout']=='contiguous' and r['mode']==mode];q=np.percentile(v,[10,50,90]);med.append(q[1]);lo.append(q[1]-q[0]);hi.append(q[2]-q[1])
  ax.bar(xx+off,med,.36,yerr=[lo,hi],capsize=3,label=mode,color=col)
 ax.set_xticks(xx,[f'conv{i}' for i in range(1,6)]);ax.set_ylim(bottom=0);ax.set_ylabel('Microseconds');ax.set_title(title);ax.legend()
fig.suptitle('15 alternating repeats; median with p10/p90; separate from training speedup');save(fig,'operator_wrappers')
