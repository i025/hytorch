"""Plot only actual CPU reference/trajectory data; no GPU acceptance implied."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--retained',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=list(csv.DictReader((a.input/'losses.csv').open()))
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for col,model in enumerate(('donkey','spatial')):
        modes={mode:np.array([float(r['loss']) for r in rows if r['model']==model and r['mode']==mode])
            for mode in ('original','cpu_model')}
        for mode,values in modes.items():axes[0,col].plot(np.arange(1,len(values)+1),values,label=mode,lw=1)
        error=np.abs(modes['cpu_model']-modes['original']);limit=1e-6+1e-5*np.abs(modes['original'])
        axes[1,col].plot(np.arange(1,len(error)+1),np.maximum(error,1e-12),label='absolute loss difference')
        axes[1,col].plot(np.arange(1,len(error)+1),limit,label='unchanged loss tolerance',color='#cc6677')
        axes[0,col].set_title(model);axes[0,col].set_ylabel('Loss');axes[0,col].legend(fontsize=8)
        axes[1,col].set_yscale('log');axes[1,col].set_xlabel('Step');axes[1,col].set_ylabel('CPU loss absolute error');axes[1,col].legend(fontsize=8)
    fig.suptitle('256-step real-input CPU Sum arithmetic-model diagnosis; GPU NOT_RUN')
    fig.tight_layout();fig.savefig(a.output/'cpu-sum-trajectory.png',dpi=160);plt.close(fig)
    diagnostic=[]
    for case in (8,23):
        data=np.load(a.retained/f'case-{case}-cancellation-False.npz');x=data['x']
        exact=float(x.astype(np.float64).sum());sum_abs=float(np.abs(x.astype(np.float64)).sum())
        for method in ('modeled','numpy_fp32'):
            value=float(data[method]);reference=float(data['fp64'])
            diagnostic.append(dict(case=case,elements=x.size,method=method,sum_abs=sum_abs,
                fp64_sum=exact,condition_sum_abs_over_abs_sum=sum_abs/abs(exact),value=value,
                abs_error_vs_fp64=abs(value-exact),relative_error_vs_fp64=abs(value-exact)/abs(exact),
                atol=1e-5,rtol=1e-4,pass_vs_fp64=abs(value-reference)<=1e-5+1e-4*abs(reference)))
    with (a.output/'retained-conditioning.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(diagnostic[0]),lineterminator='\n');w.writeheader();w.writerows(diagnostic)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,case in zip(axes,(8,23)):
        rr=[r for r in diagnostic if r['case']==case]
        ax.bar([r['method'] for r in rr],[r['abs_error_vs_fp64'] for r in rr])
        ax.set_title('N={} / condition={:.3g}'.format(rr[0]['elements'],rr[0]['condition_sum_abs_over_abs_sum']))
        ax.set_ylabel('Absolute error vs FP64 sum')
    fig.suptitle('Retained CPU cancellation cases; conditioning does not waive failed tolerances')
    fig.tight_layout();fig.savefig(a.output/'cpu-retained-conditioning.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(status='COMPLETED',gpu='NOT_RUN',rows=len(rows),conditioning_rows=len(diagnostic))))


if __name__=='__main__':main()
