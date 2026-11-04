"""Only measured preparation/correctness data; no unrun throughput figures."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--prepared',type=Path,required=True);p.add_argument('--cpu',type=Path,required=True)
p.add_argument('--gpu',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
def rows(path):
    with path.open() as f:return list(csv.DictReader(f))
def save(fig,name):
    fig.text(.5,.005,'Actual dataset / numerical CSV visualization; not a screenshot',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.025,1,1));fig.savefig(a.output/(name+'.png'),dpi=160);plt.close(fig)
manifest=rows(a.prepared/'manifest.csv');fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for split in ('train','validation'):
    values=np.array([float(r['steering']) for r in manifest if r['split']==split])
    axes[0].hist(values,bins=np.linspace(-1,1,41),density=True,alpha=.5,label=split)
    axes[1].bar(split,100*np.mean(values==0))
axes[0].set(xlabel='Steering',ylabel='Density',title='Temporal split, 10-second purge');axes[0].legend()
axes[1].set(ylabel='Zero steering (%)',title='Center-camera label imbalance')
save(fig,'dataset-steering')
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,folder,title in zip(axes,(a.cpu,a.gpu),('CPU real-image comparisons','GPU1 SHARED_GPU real-image comparisons')):
    data=rows(folder/'errors.csv');values=np.array([float(r['max_abs']) for r in data]);positive=values[values>0]
    ax.hist(np.log10(positive),bins=25);ax.set(title=title,xlabel='log10 max absolute error (positive values)',ylabel='Comparison count')
    ax.text(.02,.95,f'{len(data)} comparisons\n{(values==0).sum()} exact zero errors',transform=ax.transAxes,va='top')
save(fig,'real-image-correctness')
print('Two actual preparation/correctness figures; performance NOT_RUN')
