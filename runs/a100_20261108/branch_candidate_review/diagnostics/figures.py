from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parents[1];(O/'figures').mkdir(exist_ok=True)
r=list(csv.DictReader((O/'SUMMARY.csv').open()));q=[x for x in r if x['experiment']=='same_process' and x['model']=='donkey']
order=['baseline','relu','prune','heap','relu_prune','all'];q=sorted(q,key=lambda x:order.index(x['mode']))
fig,ax=plt.subplots(1,2,figsize=(12,4.6));means=[float(x['candidate_mean_s']) for x in q];ax[0].bar(order,means,color=['#777777','#0072b2','#e69f00','#777777','#009e73','#56b4e9']);ax[0].set_ylabel('Full 192-step training wall time (s)');ax[0].tick_params(axis='x',rotation=30)
for i,y in enumerate(means):ax[0].text(i,y+.008,f'{y:.4f}',ha='center',fontsize=9)
ax[0].set_ylim(0,.84);ax[0].set_title('12 paired same-process ablations')
new=next(x for x in r if x['experiment']=='exact_4317bf5_paired_processes');ax[1].bar(['Original 4317bf5','ReLU + input pruning'],[float(new['baseline_mean_s']),float(new['candidate_mean_s'])],color=['#777777','#009e73']);ax[1].set_ylim(0,.84);ax[1].set_ylabel('Training wall time (s), initialization excluded');ax[1].set_title(f"10 paired processes: {float(new['wall_reduction_pct']):.2f}% less time")
fig.text(.02,.01,'A100-PCIE-40GB GPU 1 | DonkeyNet FP32, 256 synthetic samples, batch=4, 3 epochs | F1+O1, legacy SGD\nNo profiler/monitor in formal timings. Same-process baseline includes opt-in infrastructure; exact-source comparison uses 4317bf5.',fontsize=9)
fig.tight_layout(rect=[0,.09,1,1])
for ext in ['png','svg']:fig.savefig(O/f'figures/training_comparison.{ext}',dpi=180)
plt.close(fig)
p=list(csv.DictReader((O/'PROFILE_PHASES.csv').open()));fig,ax=plt.subplots(figsize=(8,4.6));names=['forward','loss','backward','optimizer'];x=np.arange(len(names))
for offset,mode,color in [(-.18,'original','#777777'),(.18,'candidate','#009e73')]:
 vals=[float(next(row for row in p if row['source']==mode and row['phase']==n)['kernels_per_step']) for n in names];bars=ax.bar(x+offset,vals,.36,label=mode,color=color)
 ax.bar_label(bars,fmt='%g')
ax.set_xticks(x,names);ax.set_ylabel('CUDA kernel launches per step');ax.legend();ax.set_title('Independent 20-step Nsight traces; not formal wall timing');fig.text(.03,.02,'A100 GPU 1 | Same model/data/config | Original source vs ReLU fusion + explicit input-gradient pruning',fontsize=9);fig.tight_layout(rect=[0,.06,1,1])
for ext in ['png','svg']:fig.savefig(O/f'figures/kernel_counts.{ext}',dpi=180)
print('Figures saved')
