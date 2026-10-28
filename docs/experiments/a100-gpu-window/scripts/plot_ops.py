"""Five operator figures exclusively from measured ops.csv."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
with (a.input/'ops.csv').open() as f: rows=list(csv.DictReader(f))
def draw(filename,panels):
    fig,axes=plt.subplots(1,len(panels),figsize=(6*len(panels),4),squeeze=False)
    for ax,(op,key,title,methods) in zip(axes[0],panels):
        for method in sorted({r['implementation'] for r in rows if r['op']==op}):
            if methods and method not in methods: continue
            chosen=sorted([r for r in rows if r['op']==op and r['implementation']==method],key=lambda r:int(r['batch']))
            x=[int(r['batch']) for r in chosen];y=[float(r[key]) for r in chosen]
            ax.plot(x,y,'o-',label=method)
            for xx,yy,r in zip(x,y,chosen):
                if r['correct']!='True': ax.plot(xx,yy,'rx',ms=12)
        ax.set(xlabel='Batch',ylabel=key,title=title);ax.set_xticks([1,4,16]);ax.grid(alpha=.2);ax.legend(fontsize=8)
        if key=='mean_ms': ax.set_yscale('log')
    fig.suptitle('Measured operator execution including allocation/packing (red x = numerical FAIL)')
    fig.tight_layout();fig.savefig(a.output/filename,dpi=160);plt.close(fig)
draw('conv-latency.png',[('conv','mean_ms','Conv2d latency',None)])
draw('conv-speedup.png',[('conv','speedup_cupy','Conv2d speedup vs CuPy',None)])
draw('direct-vs-im2col.png',[('conv','mean_ms','Direct vs Im2col',{'cuda_direct','cuda_im2col_16','cuda_im2col_32'})])
draw('pooling.png',[('pool','mean_ms','Pooling latency',None),('pool','speedup_cupy','Pooling speedup vs CuPy',None)])
draw('conv-tile-size.png',[('conv','mean_ms','Im2col tile 16 vs 32',{'cuda_im2col_16','cuda_im2col_32'})])
